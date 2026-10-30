"""決定這一輪要做什麼,跟生成那一句分開。

【為什麼要把決定獨立出來】
原本沒有「決定」這個步驟——判準寫在 prompt 裡,模型直接吐出下一句問題,
所以系統不知道自己剛剛做了什麼。`TurnResponse.is_follow_up` 因此只能反推:
早期版本用 topic 相等推論,實測五題全是追問卻給 0/5,後來改成一對一一律
預設 True。那不是判斷,是猜。

決定變成資料之後拿到三件事:

    is_follow_up   是決定本身,不用猜
    追問率         可以數 → 難度第一次驗得出來
    why            每個決定都有理由可以逐項標注

【決定是算的,不是問的】
三層,前面的優先:

    forced_decision   程式直接決定的四種,不呼叫模型
    read_facts        問模型這段話裡有什麼(三次並行)
    accepts()         程式算這些事實夠不夠放行

模型只負責供應事實。要不要追問、缺的是哪一項,都是 `AcceptRule` 算出來的。

這是量了兩輪才走到的位置:

    1d  放行條件寫進 prompt   三種難度的追問率都是 0.500,19/180 次降級
    2d  程式用事實算          0.143 / 0.429 / 0.857,一致率 1.00,0/630 次降級

2d 之後對照組那條路(`compose_decision_prompt`、`parse_decision`、
`normalize_action`、`normalize_gap`、`_GAP_ALIASES`、`_REFUSAL_MARKS`)
全部刪掉了。要翻的話在 git 裡。

【輸出一定要能壞】
Panfilova et al.(2026)實測 DeepSeek Chat V3.1 在三場面試裡累積
45 次 schema 驗證錯誤,所以判讀失敗是常態不是例外。
每一條失敗路徑都有降級,而且降級會標在 source 上讓之後查得到。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.interviewer_spec import (
    DIM_STRUGGLE,
    GAP_DIMENSION,
    STRUGGLE_FACT,
    AcceptRule,
    InterviewerSpec,
)
from app.pipeline.star_read import ASSIGN, QUESTION, AnswerFacts, read_facts
from app.prompts.probe_rules import MAX_TURNS_PER_SESSION, RESUME_PHASE_TURNS
from app.prompts.star_read import FACT_NAME
from app.prompts.turn_decision import DECISION_ACTIONS, PHASE_SWITCHES

from app.pipeline.interview_live import LLMCall

# 明確表示答不出來。與報告的情緒、可評性共用同一個判斷,改要一起改。
from app.pipeline.scorability import admitted_phrase

NEXT, FOLLOW_UP, CLARIFY = DECISION_ACTIONS
TO_RESUME, TO_JD = PHASE_SWITCHES

MIN_ANSWER_CHARS = 2
"""短於這個長度視為沒有內容。空的 timeout 輪會走到這裡。"""

MAX_SUPERSEDED_CHARS = 300
"""記錄被取代的輸出時最多留幾個字。只為了事後查得出來,不需要全文。"""

DECISION_SOURCES: tuple[str, ...] = ("code", "facts", "fallback")
CODE, FACTS, FALLBACK = DECISION_SOURCES
"""`TurnDecision.source` 的值域。

    code      程式直接決定,完全沒問模型(forced_decision 那四種)
    facts     程式用模型判讀出來的事實算的(accepts())
    fallback  判讀沒讀完,降級

宣告出來是為了報表:工具自己列一份的話,這裡多一個值而它沒跟著加,
那一整類會被漏掉而且報表照樣印得出來。

【為什麼 facts 要跟 code 分開】
兩者的判斷邏輯都是程式,但輸入不同:`code` 那幾種不碰模型,
`facts` 的事實是模型讀出來的。混成一欄的話,
「模型判讀錯導致決定錯」會被記成「程式決定的」,然後評不出來是誰的問題。

【為什麼沒有 llm】
2d 之前有一個 `llm`:模型自己挑動作的那條路。那條刪掉之後沒有生產者了,
所以值域也跟著拿掉——一個永遠是 0 的欄位比沒有欄位更容易被誤讀成
「這條路沒被走到」而不是「這條路不存在」。

CLARIFY 做完會把它加回來,那是唯一一個程式算不出來、只能問模型的動作。

【CLARIFY 現在不會發生】
`DECISION_ACTIONS` 有三個值,但這個檔只產出 NEXT 與 FOLLOW_UP。
判斷「這個詞像是辨識轉錯」程式算不出來,而那一次窄問法還沒做。
判準文字備在 `app/prompts/turn_decision.py` 的 `CLARIFY_BY_ENGINE`。

這是刻意的缺口不是忘記:乾淨逐字稿上實測 CLARIFY 出現 0 次,
裝置端逐字稿上它該不該出現還沒量。`test_the_facts_path_never_says_clarify`
釘住這件事,補上來的時候那條會紅。
"""


@dataclass(frozen=True)
class TurnDecision:
    """這一輪的決定。"""

    action: str
    why: str = ""
    gap: str = ""
    """缺的是哪一項。只會是 `interviewer_spec.GAP_VALUES` 其中一個,或空字串。"""

    source: str = FACTS
    """誰做的決定。值域見 DECISION_SOURCES。

    分開記是為了驗證時能把三者拆開看——程式直接決定的那些不該拿去
    評判讀品質,而降級的那些要看得到有多少。
    """

    superseded: str = ""
    """被這個決定取代掉的東西,截斷後留著。

    降級與(未來的)程式覆寫都會丟掉模型原本說了什麼,而那正是事後要查的。
    1e 那次就是:覆寫建了一個新的 TurnDecision,原本的 gap 沒帶過去,
    要查覆寫掉了什麼只能靠 why 裡剛好寫了推導過程——那是運氣不是設計。

    所以規則是:任何取代模型決定的程式路徑,都要把原本的東西放進這裡。
    """

    dimension: str = ""
    """缺的是哪個能力維度。只會是 `PROBE_DIMENSIONS` 其中一個,或空字串。

    派發讀這一欄:有值而且那個維度有負責人,這一輪就直接給他,不問模型。
    空字串表示「這一輪沒有維度資訊」,派發照舊做語意判斷。

    【為什麼跟 gap 分開存而不是現算】
    兩者不是一對一:缺情境與缺任務不屬於任何維度(`GAP_DIMENSION` 沒列它們),
    而自述困難那一維沒有對應的 STAR 缺口。現算的話每個讀 gap 的地方
    都要自己記得這兩條例外。

    【為什麼排在最後一個欄位】
    既有的呼叫有四個位置參數(action, why, gap, source)。插在 gap 後面會把
    source 擠掉,而且不會噴錯——只是變成 dimension 收下 "facts"。
    位置參數的順序是合約。
    """

    phase_switch: str = ""
    """程式強制換段時換到哪一段:`TO_RESUME` 或 `TO_JD`。其他情況空字串。

    生成端讀這一欄,多給一句 `PHASE_SWITCH_INSTRUCTION`。見 `phase_switch()`。
    """

    @property
    def is_follow_up(self) -> bool:
        """對應 TurnResponse.is_follow_up。CLARIFY 也算追問——沒有換題。"""
        return self.action in (FOLLOW_UP, CLARIFY)


def phase_switch(
    turn_index: int | None, depth: int, has_resume: bool
) -> TurnDecision | None:
    """這一段的最後一輪而前面都在追問,就強制換題。其他情況回 None。

    【10/30 定案:每場至少問到一題履歷、一題職缺】
    原本追問優先,階段只在面試官自己決定換題時才生效。前端 10/29 錄的一對一
    場次裡,開場之後四輪都繞著自我介紹講的那件事,選了履歷也沒有一輪問到
    履歷上的其他經歷,職缺題一題都沒有。

    兩段各有一個最後一輪:

        履歷段  followUpIdx = RESUME_PHASE_TURNS - 1(1)
        職缺段  followUpIdx = MAX_TURNS_PER_SESSION - 1(3,整場最後一輪)

    到了那一輪、上一輪還在追問,就換。換到哪一段的內容由 `interview_phase`
    照輪次給,這裡只決定「要換」,並記下換到哪一段給生成端多講一句。

    【怎麼知道上一輪換過題】
    看前端帶回來的 `depth`(probe_depth)。上一輪換題時 `routes._next_probe`
    回 0,還在追問時至少是 1。是 0 就表示這一段的題已經出現過,這一輪照常判讀,
    要追問就追問。上一輪是被切斷的話也是 0,這條不會觸發——少換一次題,
    比把人從一段沒講完的話裡拉走好。

    沒有履歷時履歷段不強制:那一段改問職缺的能力,沒有「該問到的履歷」。
    `turn_index` 是 None 時不判斷:量測工具不帶輪次,行為跟以前一樣。
    """
    if turn_index is None or depth <= 0:
        return None
    if has_resume and turn_index == RESUME_PHASE_TURNS - 1:
        return TurnDecision(
            NEXT, "履歷那一段的最後一輪,前面都在追問,換到他履歷上的經歷。",
            source=CODE, phase_switch=TO_RESUME,
        )
    if turn_index == MAX_TURNS_PER_SESSION - 1:
        return TurnDecision(
            NEXT, "整場最後一輪,職缺那一段還沒問到,換到職缺。",
            source=CODE, phase_switch=TO_JD,
        )
    return None


def forced_decision(
    *,
    spec: InterviewerSpec,
    answer: str,
    depth: int = 0,
    ended_by: str = "unknown",
    turn_index: int | None = None,
    has_resume: bool = False,
) -> TurnDecision | None:
    """程式就能決定的情況。決定不了回 None,交給模型。

    順序是有意的,前面的優先:

    1. 沒有內容       沒東西可追。群面使用者沉默送的空輪會走到這裡。
    2. 被切斷         先讓他把話講完,其他判準這一輪不適用。
                      這條排在深度上限與換段前面——被打斷不是他的問題,
                      不該因為「已經追三次了」或「該換段了」就不讓他講完。
    3. 該換段了       這一段的最後一輪,而前面都在追問。見 `phase_switch`。
    4. 到深度上限     同一個缺口追到上限就換題,不管他補完整沒有。
                      追到第三次還講不出來,繼續追只是折磨,
                      而那本來就該誠實反映在報告裡。
    5. 明確說不會     換一個他答得出來的,不要問同一件事的變體。
    """
    text = (answer or "").strip()
    if len(text) < MIN_ANSWER_CHARS:
        return TurnDecision(NEXT, "這一輪沒有內容可以追問。", source=CODE)
    if ended_by == "timeout":
        return TurnDecision(
            # gap 留空:被切斷不是 STAR 少了哪一段,而 gap 的封閉集合只描述
            # STAR 上的缺口。硬塞一個集合外的值會讓「gap 只會是 GAP_VALUES
            # 其中一個」這句話變成假的。要講的話在 why 裡。
            FOLLOW_UP,
            "這段回答被語音辨識切斷,先請他把話接完。",
            "",
            CODE,
        )
    switch = phase_switch(turn_index, depth, has_resume)
    if switch is not None:
        return switch
    if depth >= spec.accept.max_depth:
        return TurnDecision(
            NEXT, f"同一個缺口已追問 {depth} 次,到這個難度的上限。", source=CODE
        )
    # 整段太長的不算:「我不確定確切數字,但回訪率從 18% 拉到 26%」是在回答。
    # 見 scorability.ADMITS_UNKNOWN_MAX_CHARS。
    if admitted_phrase(text):
        return TurnDecision(NEXT, "他明確表示答不出來,換一個他答得出來的。", source=CODE)
    return None


def fallback_decision(reason: str, raw: str = "") -> TurnDecision:
    """決策完全失敗時的降級。

    選 NEXT 不選 FOLLOW_UP:一個解析不出來的決定不該把人釘在同一個點上,
    而且 max_depth 的保護本來就建立在決定可信之上。
    代價是這一輪的門檻被放寬了,所以 source 標成 fallback,驗證時看得到。

    `raw` 是模型吐出來的原文,截斷後放進 `superseded`。
    光看 source=fallback 只知道「失敗了」,不知道失敗長什麼樣,
    而「無法」被當成拒絕詞那次就是看不到原文才多燒一輪。
    """
    return TurnDecision(
        NEXT,
        reason,
        source=FALLBACK,
        superseded=" ".join((raw or "").split())[:MAX_SUPERSEDED_CHARS],
    )


def describe_facts(facts: AnswerFacts) -> str:
    """把判讀結果寫成一句人看得懂的話,給 `why` 用。

    決定變成程式算之後,`why` 不再是模型的說法而是事實的摘要。
    報告要逐題講評、人工要標注對錯,看的都是這一句。
    """
    named = "、".join(FACT_NAME[p] for p in facts.star_parts)
    segs = f"交代了{named}" if named else "四段都沒交代"
    return (
        f"{segs};"
        f"{'有' if facts.quantified else '沒有'}可查證的數字;"
        f"{'有' if facts.tradeoff else '沒有'}取捨"
    )


def decide_from_facts(rule: AcceptRule, facts: AnswerFacts) -> TurnDecision:
    """純函式:事實加放行條件 → 決定。不碰模型,不拋例外。

    判讀沒讀完的時候降級成 NEXT,理由跟 `fallback_decision` 一樣——
    一個不可信的判斷不該把人釘在同一個點上,而 `max_depth` 的保護
    本來就建立在決定可信之上。

    這裡的降級要看得見,所以 `superseded` 記下讀到的部分:
    「四段都沒讀到」與「讀出來四段都沒有」在 `accepts()` 眼裡一模一樣,
    只看結果分不出來,而前者該退讓、後者該追問。
    """
    if not facts.complete:
        missing = "、".join(FACT_NAME[f] for f in facts.failed) or "(沒有判讀)"
        return fallback_decision(
            f"事實判讀沒完成({missing}),不拿不完整的事實下判斷,改為推進。",
            describe_facts(facts),
        )
    desc = describe_facts(facts)
    if facts.accepted_by(rule):
        # STAR 都交代了,所以沒有缺口可追。但一段完整漂亮的成功敘事在 STAR
        # 眼裡是滿分,而那正是「他自己卡在哪」還沒問到的時候——那一維
        # STAR 看不到,要靠 star_read 多讀一件事。
        return TurnDecision(
            NEXT,
            f"{desc}。放行條件全部滿足。",
            source=FACTS,
            dimension=struggle_dimension(facts),
        )
    gap = facts.gap_for(rule)
    # gap 的值本身就帶「缺」字(缺情境、缺行動…),再寫「缺的是」會變成「缺的是缺行動」。
    return TurnDecision(
        FOLLOW_UP, f"{desc}。{gap}。", gap, FACTS,
        dimension=GAP_DIMENSION.get(gap, ""),
    )


def struggle_dimension(facts: AnswerFacts) -> str:
    """他有沒有自述困難。沒講就回那個維度名,講了或沒讀到都回空字串。

    【「沒讀到」與「讀出來沒有」不可以混】
    `AnswerFacts.read()` 對沒讀過的鍵回 `_ABSENT`,`present` 是 False——
    跟「讀過了,他確實沒講」長得一模一樣。混在一起的後果是:
    `STRUGGLE_FACT` 還沒接上的時候,每一輪都會判成「這一維是缺的」,
    於是 HR 永遠拿到所有 NEXT 輪。

    所以這裡看的是 `read_by`:那一項真的被讀過,才信它的 present。
    成員A 把那一件事加進 `FACTS` 之前,這個函式一律回空字串,
    派發行為跟現在完全一樣。
    """
    if facts.read_by.get(STRUGGLE_FACT) not in (ASSIGN, QUESTION):
        return ""
    return "" if facts.read(STRUGGLE_FACT).present else DIM_STRUGGLE


async def decide_turn(
    *,
    spec: InterviewerSpec,
    answer: str,
    llm: LLMCall,
    question: str = "",
    depth: int = 0,
    ended_by: str = "unknown",
    transcription_engine: str = "api",
    turn_index: int | None = None,
    has_resume: bool = False,
) -> TurnDecision:
    """這一輪要做什麼。永遠回一個決定,不拋例外。

    `question`、`transcription_engine` 目前沒被用到:判讀只看這一輪的回答,
    不看上一題,也不分引擎(引擎的差別在斷句,那是 `segment_text` 的事)。
    留著是因為 CLARIFY 會用到 `transcription_engine`,而呼叫端已經在傳。
    """
    forced = forced_decision(
        spec=spec, answer=answer, depth=depth, ended_by=ended_by,
        turn_index=turn_index, has_resume=has_resume,
    )
    if forced is not None:
        return forced
    facts = await read_facts(answer=answer, llm=llm)
    return decide_from_facts(spec.accept, facts)
