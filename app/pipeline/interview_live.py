"""面試現場(A1/A2/A3)管線。

【一套管線三組 persona】
    A1 一對一   mode="single"   一位面試官,不回 speaker
    A2 panel    mode="panel"    三位主管,回 speaker
    A3 群面     mode="group"    三位 AI 同儕 + 主考官,回 speaker,另有搶話

三者的輸入輸出形狀相同,差別只在 persona 設定與派發規則。
合併成一套管線是三週期程能成立的主因——不必蓋三次。

【兩個端點】
    POST /interviews              開場:主問題 + 備援追問 + 搶話台詞(群面)
    POST /interviews/{id}/turns   每輪:接話

【搶話為什麼在開場一次生成】
GroupDispatcher.interruptLine(index) 是前端偵測打字停頓時同步呼叫的,
那個時機發 HTTP 會卡住 UI。開場預生成一組、前端按 index 取用,
Kotlin 介面簽章完全不變。

【沿用 A4/A5 的紀律】
    LLM 不做它做不穩的事——這裡沒有分數,所以沒有這個問題
    speaker 是封閉值域,但用機械修復不用 Literal
    任何失敗都降級,不拋例外:面試進行中噴 500 等於整場中斷
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import uuid
import zlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from app.contracts.interview_protocols import JDInput
from app.pipeline.jd_normalize import normalize_locale
from app.pipeline.interviewer_spec import DIMENSION_OWNER
from app.pipeline.question_form import is_closed_question
from app.prompts.interview_live import (
    compose_dispatch_prompt,
    compose_interrupt_prompt,
    compose_opening_prompt,
    compose_turn_prompt,
)
from app.prompts.turn_decision import ACTION_INSTRUCTION, DECISION_ACTIONS
from app.prompts.interview_personas import (
    INTERRUPT_CAP,
    opening_intro,
    personas_for,
    speaker_names,
)
from app.prompts.probe_rules import MAX_TURNS_PER_SESSION, turn_cap_for
from app.schemas.interview import (
    ExperienceDTO,
    InterviewContext,
    Persona,
    StartInterviewResponse,
    TurnResponse,
    UtteranceDTO,
)

if TYPE_CHECKING:  # 執行期不匯入:turn_decision 匯入本檔,直匯會變成循環。
    from app.pipeline.turn_decision import TurnDecision

LLMCall = Callable[[str, str], str]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_obj(raw: str) -> dict:
    """解析 JSON 物件。失敗回空 dict,不拋例外。"""
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i : j + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_list(raw: str) -> list:
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("["):
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j > i:
            text = text[i : j + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# 輸入組裝
# ---------------------------------------------------------------------------


RESUME_MAX_CHARS = 1200
"""履歷帶進開場時的上限。

超過就截掉後面的經歷,不截單一筆的中間——半句話的經歷比沒有這一筆更糟,
模型會對著一個殘缺的句子出題。
"""


_JOINERS = re.compile(r"[\s,，。、;；:：.!！?？]+")


def _adds_to_fields(desc: str, fields: Sequence[str]) -> bool:
    """description 有沒有其他欄位以外的內容。

    只看「description 是否被某一欄包含」擋不住拼起來的那種:「action result」
    比任何一欄都長。所以把每一欄從 description 裡拿掉,剩下的只有空白與
    標點的話,它就只是其他欄位拼起來的。長的欄位先拿,免得短的那欄先把
    長的那欄切斷。

    剩下有內容的話整段列進去,不只列剩下的部分——拿掉中間幾段之後的殘句
    比重複幾個字更難讀。
    """
    if any(desc in f for f in fields):
        return False
    rest = desc
    for f in sorted(fields, key=len, reverse=True):
        rest = rest.replace(f, " ")
    return bool(_JOINERS.sub("", rest))


def resume_block(experiences: Sequence[ExperienceDTO]) -> str:
    """把經歷攤成 prompt 材料。開場與報告共用這一份。

    原本只有報告那條路有這個函式(`routes._resume_text`),而開場那邊
    根本沒用到履歷。兩邊各寫一份的話格式會分岔,而分岔的方向通常是
    「其中一邊忘了帶某個欄位」,然後沒有人發現。

    【`description` 原本沒讀,而真實資料的內容全部在那一欄】
    舊版只讀 title / role / action / result。黃金集十份的履歷共 19 筆經歷,
    **19 筆都只填 description,沒有一筆填 role、action 或 result**。於是

        前端送來   title=畢業專題 - 健康紀錄 App
                   description=以 Kotlin 開發,完成期末公開展示
        模型收到   ・畢業專題 - 健康紀錄 App(標籤:…)

    面試官只看得到經歷的標題,而履歷那一段的規則是「只問履歷上寫了的事」。
    前端的客製化履歷也是一筆經歷一段文字,形狀跟黃金集一樣。

    【報告不受這一欄影響】
    `routes._resume_text` 也呼叫這一支,但 `generate_report` 收下 `resume_text`
    之後沒有用到——9/24 進來的時候就只用 `resume_grounded`。報告的漏講點
    來自 `GapComputer`,它只讀 tags(`scan_resume_prose` 預設關,`deps.py` 沒開)。
    所以讀進 description 改變的是開場與每一輪,報告的輸出不變。

    【description 只是其他欄位拼起來的時候不再列一次】
    後端 API 回給 App 的 description 是 action 與 result 用空白接起來的
    (`careersandbox-api` 的 `routes/experiences.js` `toDto`),同時也回
    role / action / result。App 照原樣轉送的話同一段文字會列兩次,吃掉
    `RESUME_MAX_CHARS` 的額度,後面的經歷就被截掉。見 `_adds_to_fields`。
    """
    lines: list[str] = []
    used = 0
    for e in experiences:
        bits = [e.title, e.role, e.action, e.result]
        desc = (e.description or "").strip()
        if desc and _adds_to_fields(desc, [b for b in bits if b]):
            bits.append(desc)
        tags = "、".join(t for t in e.tags if str(t or "").strip())
        line = "・" + " / ".join(b for b in bits if b) + (f"(標籤:{tags})" if tags else "")
        if line == "・":
            continue
        if used + len(line) > RESUME_MAX_CHARS:
            lines.append(f"(還有 {len(experiences) - len(lines)} 筆經歷未列出)")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def context_block(
    ctx: InterviewContext,
    jd: JDInput | None = None,
    experiences: Sequence[ExperienceDTO] = (),
) -> str:
    """把設定頁的欄位攤成 prompt 材料。

    JD 是選填的。空的時候不要留一個空欄位讓模型腦補,直接寫明沒有——
    「JD:」後面接空白會讓它以為漏了什麼而去猜。履歷同理。
    """
    lines = [
        f"職稱:{ctx.custom_role or '(未填)'}",
        f"公司:{ctx.custom_company or '(未填)'}",
        f"產業:{ctx.custom_industry or '(未填)'}",
        f"年資層級:{ctx.custom_seniority or '(未填)'}",
        f"面試階段:{ctx.round}",
        f"面試類型:{ctx.type}",
        f"難度:{ctx.difficulty}",
    ]
    if jd and jd.required_skills:
        lines.append(
            "這份職缺需要的能力(依重要性排序):" + "、".join(jd.required_skills[:15])
        )
    elif ctx.custom_jd:
        lines.append(f"JD 原文:{ctx.custom_jd[:600]}")
    else:
        lines.append("JD:使用者沒有提供。只用上面的職位脈絡出題,不要假裝知道職缺內容。")

    resume = resume_block(experiences)
    if resume:
        lines += ["", "【他選的這份履歷】", resume]
    else:
        lines.append("履歷:使用者沒有選。不要假設他做過什麼。")
    return "\n".join(lines)


def _persona_dtos(
    mode: str,
    group_interviewers: int = 1,
    group_size: int = 4,
    blurbs: dict[str, str] | None = None,
) -> list[Persona]:
    """開場回應裡的角色清單。

    `blurbs` 是生成的背景敘述,按 persona id 蓋掉固定的那一句。

    【為什麼只蓋 blurb 不蓋 display_name】
    `PeerProfile.name` 有生成的中文姓名,但顯示名不能在這一層換掉:
    `repair_speaker` 與 `_interviewer_names` 是靠顯示名比對說話者的,
    而怡君那邊的 roster 目前寫死「AI-邏輯」這幾個字串。
    換成「林于哲」要同時改那條線與前端,那是接線改動,不是這裡的事。

    等前端改成從回應讀 roster 之後,這裡多一行就能換。在那之前
    生成的名字留在 `PeerProfile` 上不往外送——送出去會讓 roster 與
    每一輪的說話者名稱對不上,而那個錯是靜默的。
    """
    override = blurbs or {}
    return [
        Persona(
            id=p.id,
            display_name=p.display_name,
            role=p.role,
            blurb=override.get(p.id) or p.blurb,
        )
        for p in personas_for(mode, group_interviewers, group_size)
        if p.display_name
    ]


# ---------------------------------------------------------------------------
# 說話者的機械修復
# ---------------------------------------------------------------------------


SILENT_FORCE_AFTER_TURNS = 2
"""過了幾輪之後,還沒開口的 persona 就強制指派給他。

【為什麼由程式決定而不是寫在 prompt】
發言次數已經傳給模型、規則也寫了「某位是 0 次就給他」,實測仍然失效:
group 五輪裡 AI-強勢 掛零,而第 5 輪的回答「不如先做一個版本出來再修」
正是它的路由觸發條件。

模型可以依內容判斷派給誰,但不能可靠地執行「誰還沒講過」這種計數。
這跟分數是同一件事——計數交給程式,內容交給模型。
"""


_ADMITS_UNKNOWN = ("不知道", "不確定", "沒想過", "沒有經驗", "答不出", "不太清楚")

MAX_SALVAGE_CHARS = 120


_STOPWORDS = set("的了嗎呢你我他這那有沒是不在,。?、 ")

DUP_THRESHOLD = 0.6
"""重疊比例超過這個值就算問了同一件事。

實測校準(去掉虛詞後的字元重疊 / 較短那句的長度):

    「你過去如何處理資料清理的經驗」vs 開場的資料分析題   0.69  該算
    「你在資料分析方面的具體經驗」  vs 同上              0.71  該算
    「你怎麼跟工程團隊協調排程」    vs 同上              0.00  不該算
    「你怎麼發現需要重寫查詢」      vs 上一題            0.00  不該算

分離度很乾淨——重複的落在 0.69 以上,不重複的是 0。
門檻放 0.6 兩邊都有餘裕。
"""


REPEAT_THRESHOLD = 0.9
"""追問用的門檻。超過這個值就算「幾乎同一句」,不只是同一個話題。

【為什麼追問要另一個門檻,不能沿用 DUP_THRESHOLD】
`DUP_THRESHOLD = 0.6` 問的是「這兩題在問同一件事嗎」,那是主問題該擋的。
追問本來就該留在同一件事上,拿 0.6 去擋會把合法的追問全殺掉:

    「你當時為什麼選那個方案?」
    「這個選擇最後帶來什麼結果?」

這兩句在問同一件事,而且兩句都是合法的追問。追問該擋的只有同一句話。

【0.9 不是校準出來的,是挑的】
手上四次量測的 144 題裡,同一位對同一段回答在不同 round 的配對共 144 組,
重疊從 1.0 連續分布到 0.5,沒有乾淨的分離帶——那些配對全部來自**同一個
情境**,所以彼此本來就是改寫。拿它當校準集會訂出一個太鬆的門檻。

實際看過的配對:0.85 以上每一組都是同一句話的改寫(「那你當時是怎麼評估
這兩個需求的優先順序呢」vs「那你當下是怎麼評估這兩個需求的優先順序呢」
是 0.929);0.667 那組才開始出現真的不同的問題。0.9 落在「看過的都算重複」
的區間內,留了餘裕。

要真的校準需要**同一場裡不同輪**的配對,那要等真實對話的資料。
`DUP_THRESHOLD` 上面那張表是主問題的校準,不適用在這裡。

【誤判的代價很低,所以寧可多擋】
擋到了就重生成一次,重生成還是重複就照原樣送出。誤判成本是多一次呼叫,
使用者看到的東西不變;漏判的成本是他看到同一句話問兩次。
"""


def is_verbatim_repeat(question: str, asked: Sequence[str]) -> str | None:
    """找出與先前問題幾乎同一句的那一題。回傳那一題,沒有就 None。

    跟 `is_near_duplicate` 同一套比法,只有門檻不同——那個擋「同一件事」,
    這個擋「同一句話」。
    """
    return _overlapping(question, asked, REPEAT_THRESHOLD)


def is_near_duplicate(question: str, asked: Sequence[str]) -> str | None:
    """找出與先前問題高度重疊的一題。回傳那一題,沒有就 None。

    純字面重疊,不做語意。門檻訂得嚴,因為問句本來就會共用很多虛詞。

    【為什麼要機械檢查】
    已問問題原文已經傳給模型、規則也寫了不要重複,實測仍然會問近似題:
    群面第 5 輪主考官問「你過去如何處理資料清理的經驗」,
    而開場題是「你在資料分析方面的經驗,特別是如何處理數據或報表」。

    這跟派發平衡是同一類:資訊給了,模型不可靠地執行比對。
    """
    return _overlapping(question, asked, DUP_THRESHOLD)


def _overlapping(
    question: str, asked: Sequence[str], threshold: float
) -> str | None:
    """字面重疊超過門檻的第一題。兩個門檻共用同一套比法。

    `shared >= 4` 那條是給短問句的下限:「結果呢?」去掉虛詞只剩兩三個字,
    任何長句都可能把它包住,比例會衝到 1.0 而兩句其實不同。
    """
    q = set(question) - _STOPWORDS
    if not q:
        return None
    for a in asked:
        av = set(a) - _STOPWORDS
        if not av:
            continue
        shared = len(q & av)
        if shared >= 4 and shared / min(len(q), len(av)) >= threshold:
            return a
    return None


_CLARIFY = "CLARIFY"
"""`turn_decision.CLARIFY` 的值。

直接匯入 `app.pipeline.turn_decision` 會循環——它匯入本檔。所以這裡複寫一份,
下面那條守衛負責讓兩邊不會分岔。寫 `DECISION_ACTIONS[2]` 也可以,但那是位置
依賴:有人重排那個 tuple 的話會靜默指到別的動作。
"""

if _CLARIFY not in DECISION_ACTIONS:
    raise RuntimeError(
        f"{_CLARIFY} 不在 DECISION_ACTIONS {DECISION_ACTIONS} 裡。"
        "動作名稱改過了,而追問查重那一關靠它排除 CLARIFY——"
        "不一致的話 CLARIFY 會被當成追問去重生成,方向是錯的。"
    )


REGENERATE = """【剛剛產出的那一句有問題,重新問一句】

{fixes}

其餘照原本的指示,輸出格式不變。"""

_FIX_REPEAT = (
    "那句跟這場問過的幾乎一樣。重複的是:{repeat}\n"
    "留在同一個話題上——這是追問,不是換題——但換一個切入點:"
    "問他剛剛那段裡還沒講清楚的另一個部分。"
)

_FIX_GROUP_REPEAT = (
    "那句跟前面講過的幾乎一樣。重複的是:{repeat}\n"
    "換一句新的:反駁前面某個說法、補它沒講到的部分,或接著往下推一步。"
)

_FIX_CLOSED = (
    "那句可以用「有」或「沒有」答完,他答一個字就結束了。\n"
    "改成非得講一段才答得完的問法:問他當時實際怎麼做、哪一段最難、"
    "為什麼那樣選。不要用「有沒有」「是不是」開頭,也不要用「嗎」結尾。"
)


def _question_faults(
    question: str,
    asked: Sequence[str],
    *,
    is_follow_up: bool,
    clarify: bool,
    group: bool = False,
) -> list[tuple[str, str]]:
    """送出前擋得住的毛病。回傳 (notice 的說法, 重生成要給的指示)。

    兩種毛病一起算,因為它們共用同一次重生成——面試進行中,
    每一次呼叫都是使用者在等,一輪擋兩種毛病不該花兩次。

    【CLARIFY 兩關都不進】
    重複:它問的是「那個詞再講一次」,辨識連續轉錯兩次的話同一句本來就該再問。
    封閉:「你是說 Angular 嗎?」就是一個 yes/no 問句,而那正是 CLARIFY 的形狀。

    【群面只看重複】
    群面的一句話不是面試官的問句:競爭者先表態再丟球,「你同意嗎?」是討論裡
    正常的丟法。拿封閉式那一關去擋,重生成的指示會叫它「改成要對方講一段的
    問法」——那正是面試官的形狀。群面也沒有追問與主問題之分,講過的話不管
    哪一種都不該原樣再講一次,所以重複那一關一律檢查。
    """
    faults: list[tuple[str, str]] = []
    if clarify:
        return faults

    if group:
        repeat = is_verbatim_repeat(question, asked)
        if repeat:
            faults.append(
                (
                    f"與前面講過的幾乎同一句({repeat[:20]})",
                    _FIX_GROUP_REPEAT.format(repeat=repeat),
                )
            )
        return faults

    if is_follow_up:
        repeat = is_verbatim_repeat(question, asked)
        if repeat:
            faults.append(
                (
                    f"與問過的幾乎同一句({repeat[:20]})",
                    _FIX_REPEAT.format(repeat=repeat),
                )
            )

    if is_closed_question(question):
        faults.append(("只能用是/否答完", _FIX_CLOSED))

    return faults


async def _regenerate(
    llm: LLMCall, system: str, user: str, faults: list[tuple[str, str]]
) -> str:
    """再問一次。回傳新問題,失敗或沒產出回空字串。

    【為什麼要把毛病講出來】
    同樣的 system 加同樣的 user 會給出同樣的輸出,溫度 0.4 也只是換幾個字。
    不講它剛剛寫了什麼、哪裡不對,重生成就只是多花一次呼叫。

    【為什麼只重生成一次】
    面試進行中,每一次呼叫都是使用者在等。一次擋不住就照原樣送出,
    那比讓他多等一輪好。主問題撞重複那條路選的是備援池而不是重試,
    理由一樣;追問與封閉式沒有池可以換,所以這裡只能重試。
    """
    fixes = "\n\n".join(fix for _, fix in faults)
    try:
        raw = await asyncio.to_thread(
            llm, system, user + "\n\n" + REGENERATE.format(fixes=fixes)
        )
    except Exception:  # noqa: BLE001
        return ""
    return normalize_locale(str(parse_obj(raw).get("nextQuestion", "")).strip())


def dimension_speaker(mode: str, dimension: str, gi: int = 1, gs: int = 4) -> str:
    """這個維度的負責人在這一場的顯示名稱。沒有負責人或不在場就回空字串。

    一對一永遠回空字串——`display_name` 是空的,而且沒有別人可以派。
    """
    slot = DIMENSION_OWNER.get(dimension, "")
    if not slot:
        return ""
    for p in personas_for(mode, gi, gs):
        if p.id == slot and p.display_name:
            return p.display_name
    return ""


def _dimension_yields(name: str, counts: dict[str, int], names: Sequence[str]) -> str:
    """維度派發要不要讓位。回讓位的理由,不讓位回空字串。

    【為什麼這條規則要有兩份】
    `DISPATCH_ONLY` 那段 prompt 裡本來就寫了平衡規則,但那是寫給模型的,
    而維度派發不問模型。沒有這一份的話,確定性路徑會讓同一位連問四輪——
    使用者答得越一致,同一個維度就越可能連續是缺的那一個。

    兩份規則用同一組門檻(未發言過的優先、三倍以上讓位),所以文字那份
    改了這裡要跟著改。測試釘住的是行為不是數字,改門檻會紅在邊界那幾條。
    """
    mine = counts.get(name, 0)
    others = {n: counts.get(n, 0) for n in names if n != name}
    if not others:
        return ""
    silent = [n for n, c in others.items() if c == 0]
    if silent and mine >= 2:
        return f"{name}已發言 {mine} 次而{'、'.join(silent)}還沒開口"
    top = max(others.values())
    if top and mine >= 3 * top:
        return f"{name}的 {mine} 次是其他人的三倍以上"
    return ""


async def _dispatch(
    llm: LLMCall,
    mode: str,
    answer: str,
    counts: dict[str, int],
    gi: int,
    gs: int,
) -> tuple[str, str | None]:
    """先決定誰說話,再由生成那一步寫內容。

    【為什麼拆兩步】
    合併版要模型同時做三件事:決定誰說話、依那個人的人格寫問題、遵守追問規則。
    七位 persona 全部展開之後 prompt 超過 4200 字,實測 gpt-4o-mini 在那個長度下
    人格會塌陷——AI-邏輯 該質疑推論,寫出來卻是「你有沒有想過怎麼驗證」。

    拆開之後派發只有 542 字,而生成端只看一份人格。
    多一次呼叫的延遲換人格辨識度——群面的價值就在多種性格,
    性格塌陷等於這個模式沒有意義。

    失敗時回 ("", None),由呼叫端退回單步流程。
    """
    names = speaker_names(mode, gi, gs)
    if not names:
        return "", None
    user = f"【使用者這段回答】\n{answer}\n\n【目前為止的發言次數】\n" + "\n".join(
        f"  {n}:{counts.get(n, 0)} 次" for n in names
    )
    try:
        raw = await asyncio.to_thread(
            llm,
            compose_dispatch_prompt(mode, group_interviewers=gi, group_size=gs),
            user,
        )
    except Exception:  # noqa: BLE001
        return "", None
    return repair_speaker(raw, mode, gi, gs)


def salvage_question(raw: str) -> str | None:
    """模型沒包 JSON、直接回問題時,把它救回來。

    【為什麼要救】
    實測:模型回了「這個改進的過程是怎麼進行的?你是怎麼發現需要重寫查詢的?」
    ——那是一個好追問,卻因為沒包成 JSON 而被丟掉,改用備援池的罐頭問題。

    重試會增加面試中的延遲,而問題本身是好的,直接用就行。
    損失的只有 speaker / topic / reaction 這些後設欄位,那些可以另外補。

    【判準】
    不是 JSON、長度合理、而且看起來是在問問題。
    看起來像道歉或說明的一律不救——那種東西顯示出來比罐頭問題糟。
    """
    t = (raw or "").strip()
    if not t or t.startswith(("{", "[")):
        return None
    t = t.replace("\n", " ").strip()
    if not 4 <= len(t) <= MAX_SALVAGE_CHARS:
        return None
    if not any(mark in t for mark in ("?", "?", "嗎", "呢", "什麼", "如何", "怎麼")):
        return None
    if any(
        bad in t
        for bad in ("抱歉", "無法", "很遺憾", "作為一個", "我是 AI", "語言模型")
    ):
        return None
    return t


def required_speaker(
    mode: str,
    spoken_by: Sequence[str],
    turn_idx: int,
    answer: str = "",
    group_interviewers: int = 1,
    group_size: int = 4,
) -> str | None:
    """這一輪必須由誰開口。沒有硬性要求時回 None。

    回傳的名稱會同時寫進 prompt(讓問題的語氣對得上)與覆寫最終的 speaker
    (讓指派必然成立)。只做前者模型會不照做,只做後者問題的語氣會跟人格不符。
    """
    names = speaker_names(mode, group_interviewers, group_size)
    if not names or turn_idx < SILENT_FORCE_AFTER_TURNS:
        return None
    # 對方明確說不會時不強制指派——那一輪本來就該由主考官或 HR 主管接住。
    # 實測:AI-強勢 被強制指派到這種情境,講出「沒關係,換個問題」,
    # 那是主考官的工作,而且完全不符合它的人格。
    if any(k in answer for k in _ADMITS_UNKNOWN):
        return None
    # 主持人不列入強制指派。他的角色是條件性的——討論卡住或使用者說不會時
    # 才開口,不是輪流制。把他算進來會佔掉補位名額:
    # 實測群面五輪,第 3 輪強制給了主考官、第 5 輪給 AI-強勢,
    # AI-親切 因為排在宣告順序最後而永遠輪不到。
    fillable = [
        p.display_name
        for p in personas_for(mode, group_interviewers, group_size)
        if p.display_name and p.role != "moderator"
    ]
    counts = {n: list(spoken_by).count(n) for n in fillable}
    silent = [n for n, c in counts.items() if c == 0]
    return silent[0] if silent else None


def repair_speaker(
    raw: str, mode: str, group_interviewers: int = 1, group_size: int = 4
) -> tuple[str, str | None]:
    """把模型回的 speaker 修成合法值。回傳 (修好的名稱, 問題說明)。

    合約規定 speaker 是開放字串不是 Literal(CareerCategory 事故的教訓),
    但值域實際上封閉——前端顯示不出清單外的名稱。所以用機械修復維持。

    修復順序:完全相同 → 去空白後相同 → 被包含 → 都不是就取第一個。
    取第一個而不是留空,是因為 panel/group 模式沒有說話者的話畫面會壞掉。
    """
    names = speaker_names(mode, group_interviewers, group_size)
    if not names:  # single 模式不顯示說話者
        return "", None
    s = (raw or "").strip()
    if s in names:
        return s, None
    tight = s.replace(" ", "")
    for n in names:
        if tight == n.replace(" ", ""):
            return n, f"speaker「{s}」已正規化為「{n}」"
    for n in names:
        if n in s or s in n:
            return n, f"speaker「{s}」不精確,已對應到「{n}」"
    return names[0], f"speaker「{s}」不在清單中,已改為「{names[0]}」"


# ---------------------------------------------------------------------------
# 開場
# ---------------------------------------------------------------------------


async def _gen_opening(llm: LLMCall, mode: str, ctx_text: str, cfg: dict) -> dict:
    raw = await asyncio.to_thread(llm, compose_opening_prompt(mode, **cfg), ctx_text)
    return parse_obj(raw)


async def _gen_interrupts(llm: LLMCall, n: int) -> list[str]:
    raw = await asyncio.to_thread(llm, compose_interrupt_prompt(n), "產生搶話台詞。")
    return [
        normalize_locale(str(x).strip()) for x in parse_list(raw) if str(x).strip()
    ][:n]


def opening_question(ctx: InterviewContext) -> str:
    """開場的第一題:請使用者自我介紹。程式組的,不問模型。

    【為什麼開場固定成自我介紹】
    真實面試的順序是先自我介紹、再追履歷、最後問職缺。原本沒有這個順序,
    `OPENING_RULES` 反而明文寫「不是通用的『請自我介紹』」,所以一場四題
    全部是職缺題,而使用者連開口暖身的機會都沒有。

    三段的分法見 `probe_rules.interview_phase`。

    【為什麼不問模型】
    這一句的內容只隨職稱變動,其餘完全固定。問模型等於讓它在一件沒有
    變化空間的事情上產生變化,而那個變化只會是壞的——`DEFAULT_OPENING`
    本來就是這一句,它一直是生成失敗時的退路,現在扶正成正常路徑。

    省下來的不只是穩定性:開場那次呼叫現在只需要產備援追問。
    """
    role = (ctx.custom_role or "").strip()
    about = f"跟{role}這個職缺相關的經驗" if role else "跟這個職位相關的經驗"
    return f"先請你用一分鐘自我介紹,重點放在{about}。"


DEFAULT_OPENING = "先請你用一分鐘自我介紹,重點放在跟這個職位相關的經驗。"
"""`opening_question` 在沒有職稱時的輸出。測試與舊呼叫端還在用這個常數。"""

GROUP_OPENING_TOPIC = "小組討論"
"""群面開場題的領域標籤,模型沒標的時候用。"""


def group_default_opening(ctx: InterviewContext) -> str:
    """群面拿不到模型出的討論題時用的題目。

    【為什麼群面不用自我介紹】
    三段式那一版把開場固定成自我介紹,三種模式共用。群面用了之後,題目變成
    使用者一個人的經歷,AI 應徵者沒有可以表態的東西,只剩追問他這一條路——
    group-1(10/29)三位競爭者每一輪都是「你提到…能分享一下…」。

    群面的題目要讓每個人都有話說,而且說法不同。這一句是退路:不需要 JD,
    只要一個要大家做決定的情境。
    """
    role, firm = (ctx.custom_role or "").strip(), (ctx.custom_company or "").strip()
    if role and firm:
        team = f"{firm}的{role}團隊"
    elif role or firm:
        team = f"{role or firm}團隊"
    else:
        team = "一個新團隊"
    return (
        f"今天請大家一起討論一個情況:假設你們剛加入{team},主管只給一個月,"
        "要你們決定第一件該先做的事。請每個人先說你的選擇和理由,再互相討論。"
    )


DEFAULT_GROUP_PROBES = (
    "目前有幾種不同的做法,大家比較一下,你們會選哪一個?",
    "還沒表態的人,你怎麼看?",
    "時間差不多了,請大家收斂成一個結論。",
)
"""群面的備援句。主持人推進討論用,不是追問某個人的經歷。"""
DEFAULT_PROBES = (
    "這個判斷你當時是依據什麼下的?",
    "如果時間只剩一半,你會先砍掉哪一段?",
    "這件事你學到什麼,下次會怎麼做?",
)
DEFAULT_INTERRUPTS = (
    "我先說,這題我有現成的案子,等大家想完時間就沒了。",
    "(舉手)我插一個快的,你慢慢想,不衝突。",
    "我補一句,這件事的順序我覺得反了。",
)


@functools.lru_cache(maxsize=1)
def _adjacent_cache():
    """相鄰技能挑選的行程內快取,整個服務共用一份。

    【為什麼需要】
    實跑量到單次呼叫中位 3.91s。競爭者那段是兩次相依呼叫,所以群面開場
    從 3.9s 變成 7.8s。而挑技能的輸入只有職缺、溫度 0,同一個職缺答案本來就一樣。
    命中之後只剩寫人物那一次,跟開場、搶話並行,等待回到 3.9s。

    用 `lru_cache(1)` 當單例而不是模組層變數:模組層變數在 import 時就建好,
    而這個東西只有群面會用到。差別不大,但延後建立讓 import 這個模組的成本
    不含它——`tools/` 底下好幾支只為了一兩個常數 import 這裡。

    快取物件本身不是單例設計(見 `AdjacentCache` 的說明),是這個呼叫端
    選擇共用一份。測試與工具不傳就沒有快取。
    """
    from app.pipeline.peer_profile import AdjacentCache

    return AdjacentCache()


async def start_interview(
    *,
    mode: str,
    context: InterviewContext,
    llm: LLMCall,
    llm_exact: LLMCall | None = None,
    jd: JDInput | None = None,
    experiences: Sequence[ExperienceDTO] = (),
    session_id: str | None = None,
) -> StartInterviewResponse:
    """開場。群面另外並行生成搶話台詞與競爭者。

    任何一段失敗都退回預設值——面試按下開始卻噴 500,使用者連場都進不去。

    【llm_exact 是給需要可重現的那幾次呼叫用的】
    `llm` 是 temp 0.4 的生成用模型。競爭者的相鄰技能挑選是分類不是生成,
    用 0.4 會讓同一個職缺每次跑挑出不同的技能池,而那個變化會疊在
    seed 的變化上面——量測時分不出哪個變化來自哪裡。不給就兩邊共用。

    【競爭者那一段的延遲】
    它內部是兩次**相依**的呼叫(挑技能 → 排集合 → 寫人物),所以這一段
    是開場的關鍵路徑:原本是 max(開場, 搶話) ≈ 1 次呼叫的時間,
    現在是 max(1, 1, 2) = 2 次。換到的是競爭者的技能真的對著職缺,
    而那是群面這個模式的全部價值。
    """
    notices: list[str] = []
    # 群面的題目是給全組的,不貼著使用者一個人的履歷出。
    ctx_text = context_block(context, jd, () if mode == "group" else experiences)
    sid = session_id or f"itv_{uuid.uuid4().hex[:12]}"

    cfg = dict(
        group_interviewers=context.group_interviewers,
        group_size=context.group_size,
        difficulty=context.difficulty,
        group_role=context.group_role,
        # 有履歷的時候要換掉「不預設對方有什麼經歷」那一條,
        # 否則 prompt 一邊列出他的經歷、一邊叫模型不要假設他做過什麼。
        has_resume=bool(resume_block(experiences)),
    )
    tasks: dict[str, asyncio.Task] = {
        "opening": asyncio.ensure_future(_gen_opening(llm, mode, ctx_text, cfg))
    }
    if mode == "group":
        # ★ 在函式內 import。`peer_profile` 要這個模組的 LLMCall 與 parse_list,
        #   所以模組層 import 會循環——2c 那輪就是這個錯誤,症狀是
        #   ImportError 而不是功能失效,但它會讓整個服務起不來。
        from app.pipeline.peer_profile import build_peers

        # INTERRUPT_CAP 為 0 時不生成。帶著 0 去呼叫會白燒一次,
        # 而且那次呼叫的輸出會被下面的退回邏輯接住變成寫死的台詞。
        if INTERRUPT_CAP > 0:
            tasks["interrupts"] = asyncio.ensure_future(
                _gen_interrupts(llm, INTERRUPT_CAP)
            )
        tasks["peers"] = asyncio.ensure_future(
            build_peers(
                title=context.custom_role,
                required=list(jd.required_skills) if jd else [],
                llm=llm_exact or llm,
                write_llm=llm,
                group_size=context.group_size,
                group_interviewers=context.group_interviewers,
                seed=zlib.crc32(sid.encode("utf-8")),
                cache=_adjacent_cache(),
            )
        )

    results: dict[str, object] = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as exc:  # noqa: BLE001
            notices.append(f"開場:{name} 生成失敗({type(exc).__name__}),已使用預設")
            results[name] = {"opening": {}, "peers": None}.get(name, [])

    op = results.get("opening") or {}
    if mode == "group":
        # 群面的開場是全組的討論題,由模型依職缺出。見 group_default_opening。
        question = normalize_locale(str(op.get("openingQuestion", "")).strip())
        if not question:
            question = group_default_opening(context)
            notices.append("開場:未取得討論題目,已使用預設")
        topic = (
            normalize_locale(str(op.get("openingTopic", "")).strip())
            or GROUP_OPENING_TOPIC
        )
    else:
        # 開場固定是自我介紹,不採用模型回的 openingQuestion——
        # 那一句只隨職稱變動,沒有需要模型發揮的空間。見 opening_question。
        # 那次呼叫留著是因為備援追問仍然要生成。
        question = opening_question(context)
        topic = "自我介紹"
    probes = [
        normalize_locale(str(x).strip())
        for x in (op.get("fallbackProbes") or [])
        if str(x).strip()
    ]
    if not probes:
        probes = list(DEFAULT_GROUP_PROBES if mode == "group" else DEFAULT_PROBES)
        notices.append("開場:未取得備援追問,已使用預設")

    interrupts: list[str] = list(results.get("interrupts") or [])
    # 退回預設只在搶話開著的時候才做。關掉之後還退回的話,
    # 寫死的那三句會被裝回去——關等於沒關,而且 notices 看起來一切正常。
    if mode == "group" and INTERRUPT_CAP > 0 and not interrupts:
        interrupts = list(DEFAULT_INTERRUPTS)
        notices.append("開場:未取得搶話台詞,已使用預設")

    cast = results.get("peers")
    blurbs: dict[str, str] = {}
    if cast is not None:
        blurbs = {p.persona_id: p.background for p in cast.profiles}
        notices += [f"競爭者:{n}" for n in cast.notices]

    personas = _persona_dtos(
        mode, context.group_interviewers, context.group_size, blurbs
    )
    opening_speaker = personas[0].display_name if personas else ""

    return StartInterviewResponse(
        session_id=sid,
        mode=mode,
        # 程式組的,不問模型——面試官是誰是 personas_for 已經決定的事實,
        # 而模型猜錯的方式是編一個姓名出來。見 opening_intro 的註解。
        opening_intro=opening_intro(
            mode,
            role_title=context.custom_role,
            company=context.custom_company,
            round_name=context.round,
            group_interviewers=context.group_interviewers,
            group_size=context.group_size,
        ),
        opening_question=question,
        opening_speaker=opening_speaker,
        # 一對一與 panel 固定是自我介紹;群面是討論題,標籤由模型標。
        opening_topic=topic,
        personas=personas,
        interrupt_lines=interrupts,
        interrupt_cap=len(interrupts),
        fallback_probes=probes,
        notices=notices,
    )


# ---------------------------------------------------------------------------
# 每輪
# ---------------------------------------------------------------------------


def _fallback_turn(
    mode: str,
    fallback: Sequence[str],
    follow_up_idx: int,
    group_interviewers: int = 1,
    group_size: int = 4,
) -> TurnResponse:
    """LLM 不可用時的降級。用前端帶來的備援池,不要臨時編。

    【群面交給主持人】
    備援句是推進討論用的,由主持人講。原本照輪次挑說話者,會讓 AI 應徵者
    唸出一句面試官的追問。群面也沒有輪次上限,不標記推進。
    """
    if mode == "group":
        host = next(
            (
                p.display_name
                for p in personas_for(mode, group_interviewers, group_size)
                if p.role != "peer" and p.display_name
            ),
            "",
        )
        pool = list(fallback) or list(DEFAULT_GROUP_PROBES)
        return TurnResponse(
            speaker=host,
            next_question=pool[follow_up_idx % len(pool)],
            reaction="",
            is_follow_up=False,
            should_advance=False,
            topic="",
            notices=[f"本輪:生成失敗,{_fallback_used(mode)}"],
        )
    pool = list(fallback) or list(DEFAULT_PROBES)
    names = speaker_names(mode)
    return TurnResponse(
        speaker=names[follow_up_idx % len(names)] if names else "",
        next_question=pool[follow_up_idx % len(pool)],
        reaction="",
        is_follow_up=False,
        should_advance=True,
        topic="",
        notices=[f"本輪:生成失敗,{_fallback_used(mode)}"],
    )


def _fallback_used(mode: str) -> str:
    """降級時 notice 的說法。群面的備援是主持人的推進句,不是追問。"""
    return "已使用備援句" if mode == "group" else "已使用備援追問"


GROUP_HISTORY_UTTERANCES = 8
"""每輪帶給群面說話者的最近幾則發言。

整場可能二三十則,全部攤進 prompt 會把剛剛那一句淹掉,而回應的對象主要是
最近幾則。八則大約是兩輪:使用者一則、三位競爭者、主持人。
"""


def group_discussion_block(
    group_says: Sequence[UtteranceDTO], answer: str, opening: str = ""
) -> str:
    """群面的討論題目與最近幾則發言,給說話者回應用。沒有資料回空字串。

    題目先用 `opening`,也就是 `asked_questions` 的第一筆——合約寫明那一筆是
    開場題。沒帶才退回第一則非使用者的發言。退回那條靠的只是資料的形狀:
    前端照「新欄位那七題」的回覆把 `openingIntro` 顯示成獨立一則,如果也記進了
    groupTranscript,第一則就是「你好,我是主考官……」,題目跟著錯,
    而討論一長,真正的題目會被擠出最近那幾則,再也看不到。

    最後一則如果就是使用者剛剛講的那段,不再列一次,上面已經有了。
    使用者在前端的顯示名是「你」,這裡寫成「使用者」,跟 prompt 其他地方一致。
    """
    says = [u for u in group_says if (u.content or "").strip()]
    if not says:
        return ""
    topic = (opening or "").strip() or next(
        (u.content.strip() for u in says if not u.is_user), ""
    )
    if says[-1].is_user and says[-1].content.strip() == (answer or "").strip():
        says = says[:-1]
    recent = says[-GROUP_HISTORY_UTTERANCES:]
    lines = [f"【討論題目】{topic}"] if topic else []
    if recent:
        lines.append("【前面的討論(最近幾則,由舊到新)】")
        lines += [
            f"  {'使用者' if u.is_user else u.speaker}:{u.content.strip()}"
            for u in recent
        ]
    return "\n".join(lines)


async def next_turn(
    *,
    mode: str,
    answer: str,
    follow_up_idx: int,
    asked_questions: Sequence[str],
    asked_topics: Sequence[str] = (),
    spoken_by: Sequence[str] = (),
    question: str = "",
    prev_topic: str = "",
    ended_by: str = "unknown",
    fallback: Sequence[str],
    llm: LLMCall,
    context: InterviewContext | None = None,
    decision: "TurnDecision | None" = None,
    jd: JDInput | None = None,
    experiences: Sequence[ExperienceDTO] = (),
    force_speaker: str = "",
    group_says: Sequence[UtteranceDTO] = (),
) -> TurnResponse:
    """一輪接話。

    follow_up_idx 達到上限時強制推進,不問模型——
    「這是第幾次追問」是確定性事實,交給模型判斷會讓它有時多追一次。

    【asked_questions 與 asked_topics 分開傳,因為兩個用途不同】
    prompt 要的是「本場問過哪些」給模型參考,沒有問句原文時列領域標籤
    也比什麼都沒有好。機械查重要的是比對鍵,而標籤拿來比會**誤判**:
    `_overlapping` 的分母是較短那一邊,標籤只有四到十一個字,被一句問題
    整個包住的時候比例衝到 1.0。

        「那你在團隊合作上最不順的是哪一段?」vs 標籤「團隊合作」  1.000

    後果是模型產出的新主問題被判成重複,換成開場就生好的備援題——
    而備援題是通用的、一場只生一次,使用者看到的就是「問題差不多」。

    所以標籤只進 prompt,不進 `is_near_duplicate` 與 `is_verbatim_repeat`。
    前端沒帶 `askedQuestions` 的時候,那兩道是**關掉**而不是誤判,
    那才是 `routes._asked_questions` 的註解原本宣稱的行為。

    【decision】
    這一輪要做什麼(NEXT / FOLLOW_UP / CLARIFY),由 `turn_decision` 先決定好再傳進來。
    決定在 `routes.py` 那層算,不在這裡呼叫——這支函式只收資料不自己發起決策,
    跟 LLM 注入是同一個原則。

    傳 None 時維持舊行為:`is_follow_up` 取模型回的值,一對一沒回就預設 True。
    那是猜的,留著只是為了讓既有測試仍然測得到生成那一段。

    【force_speaker】
    指定這一輪由誰開口,跳過派發。走的是 `required_speaker` 既有的 forced
    路徑,所以下游行為跟「輪空強制指派」那條完全一樣,不是新的分支。

    生產路徑不傳這個值——誰開口由派發決定。它是給量測用的:要比較三位
    面試官對**同一段回答**會問出什麼,就得繞過派發,否則每一位拿到的
    回答都不同,量到的是回答的差異不是角色的差異。
    `tools/panel_role_eval.py` 是唯一的呼叫端。

    【group_says:群面要看得到別人講過什麼】
    沒有它,AI 應徵者只看得到使用者剛剛那一句,不可能附和、反駁或接著
    別人的論點講。群面同時不帶履歷:同場的應徵者看不到使用者的履歷,
    題目也是給全組的。追問判準與面試階段改用 GROUP_TURN_RULES。
    """
    cap = turn_cap_for(mode)
    must_advance = cap is not None and follow_up_idx >= cap
    if decision is not None and must_advance:
        # 輪次上限是確定性事實,比決定大。到上限時這一句就是最後一個主問題,
        # 所以語意上必然是 NEXT,也該過重複偵測。
        decision = None
        forced_next = True
    else:
        forced_next = False

    # 回答排在前面,而且明說要回應它。
    #
    # 早期版本把「上一個問題」排前面,實測模型會錨在前一題的主題上,
    # 把回答當成補充資料:使用者講協調排程,問的是 SQL;
    # 使用者說「我沒想過」,問的還是優化查詢——連續三輪不跟著回答走。
    # gpt-4o 也一樣,所以那不是模型能力問題,是輸入順序造成的錨定。
    group = mode == "group"
    if group:
        # 群面回應的是整場討論,不只是使用者這一句——見 group_discussion_block。
        user_parts = [
            f"【使用者剛剛說的】\n{answer}",
            "",
            "你要接的是這場討論:通常是上面這段,也可以接前面其他人講過的話。",
        ]
        discussion = group_discussion_block(
            group_says, answer, asked_questions[0] if asked_questions else ""
        )
        if discussion:
            user_parts.append(discussion)
        else:
            user_parts.append(f"【前一則(脈絡)】{question or '(未提供)'}")
    else:
        user_parts = [
            f"【他剛剛說的】\n{answer}",
            "",
            "你要回應的是上面這段話。下面的前一題只是脈絡,不是這次的主題。",
            f"【前一題(脈絡)】{question or '(未提供)'}",
        ]
    if context:
        # jd 與 experiences 原本都沒傳,所以每輪的 context_block 會寫
        # 「JD:使用者沒有提供」而實際上 `_turn_jd` 早就抽出來了,
        # 履歷也同樣看不到。職缺那一段跟履歷那一段都靠這兩份材料,
        # 不帶的話階段的指示會落在空的脈絡上。
        #
        # 群面不帶履歷:同場的應徵者看不到使用者的履歷。
        user_parts.append(
            "【職位脈絡】\n" + context_block(context, jd, () if group else experiences)
        )
    gi = context.group_interviewers if context else 1
    gs = context.group_size if context else 4
    names = speaker_names(mode, gi, gs)
    # force_speaker 走的是既有的 forced 路徑,不另開分支——
    # 派發那次呼叫一樣被跳過,下游一樣只看到一個已經決定好的說話者。
    forced = force_speaker or required_speaker(
        mode, spoken_by, follow_up_idx, answer, gi, gs
    )
    counts = {n: list(spoken_by).count(n) for n in names}

    # 第一步:決定誰說話。強制指派時跳過這次呼叫——答案已經確定了。
    focus = forced or ""
    dispatch_note = f"本輪:指派給未發言過的「{forced}」" if forced else ""

    # 缺的維度有負責人就直接給他,不問模型。語意派發是在沒有判準時用的,
    # 有判準的時候問模型只是多一次呼叫加一次不確定。
    #
    # 【只有 panel 走這條】
    # 群面場上有三位主管加三位 AI 應徵者,而 `derive_specs` 不含 peer——
    # 維度派發永遠選不到競爭者。那會把群面的前兩輪固定給主管,
    # 而群面的價值就在應徵者之間的互動(見 GROUP_PEER_STANCE 那一整套)。
    #
    # 更根本的是這在群面是範疇錯誤:群面的「誰接話」問的是討論動態
    # (有人被淹沒、討論變成兩人對話),不是「哪個能力維度還沒問到」。
    # 那兩件事用的是兩支不同的派發 prompt。
    #
    # 一對一不必擋:`dimension_speaker` 回空字串,因為一對一的
    # display_name 是空的——前端只有一位面試官,不顯示說話者。
    if (
        mode == "panel"
        and names
        and not focus
        and decision is not None
        and decision.dimension
    ):
        owner = dimension_speaker(mode, decision.dimension, gi, gs)
        if owner in names:
            yielding = _dimension_yields(owner, counts, names)
            if yielding:
                dispatch_note = (
                    f"本輪:缺「{decision.dimension}」本該給{owner},但{yielding},"
                    "改回語意派發"
                )
            else:
                focus = owner
                dispatch_note = f"本輪:缺「{decision.dimension}」,派給負責的{owner}"

    if names and not focus:
        picked, problem = await _dispatch(llm, mode, answer, counts, gi, gs)
        if picked:
            focus = picked
            if problem:
                dispatch_note = f"本輪:派發{problem}"
    if names and not focus:
        # 派發失敗:退回發言次數最少的一位,不要讓生成端自己猜。
        focus = min(names, key=lambda n: counts.get(n, 0))
        dispatch_note = "本輪:派發失敗,已改由發言最少的人開口"
    # 決定已經做完了,告訴生成端。不講的話決定說 NEXT 而模型生出追問,
    # is_follow_up 就會跟實際內容不符,重複偵測會對一句追問做比對。
    #
    # 【只傳動作,不傳 gap】
    # 原本會附上「缺的是:{gap}」。1d 實測發現 gap 常常指錯:A3 那筆答案講了結果
    # 只是沒有數字,20/20 次 gap 都寫「結果段」。照著生成出來的追問會去問一個
    # 他剛剛講過的東西,正好是「換個說法問同一件事」那個老問題的一個來源。
    # 生成端看得到完整的上一輪回答,不需要一個會指錯的標籤。
    decision_note = ""
    if decision is not None:
        what = ACTION_INSTRUCTION.get(decision.action, "")
        if what:
            user_parts.append(f"【這一輪要做什麼】{what}")
            decision_note = f"本輪:決定 {decision.action}({decision.source})"

    if must_advance:
        user_parts.append(
            f"【注意】整場已經問到第 {follow_up_idx + 1} 題,達到上限 "
            f"{cap} 題。請問最後一題,並把 shouldAdvance 設為 true。"
        )

    # system 與 user 留成變數,因為追問撞到重複時要用同一份再問一次。
    system = compose_turn_prompt(
        mode,
        list(asked_questions) or list(asked_topics),
        group_interviewers=gi,
        group_size=gs,
        focus_speaker=focus,
        truncated=ended_by == "timeout",
        engine=context.transcription_engine if context else "api",
        difficulty=context.difficulty if context else "中等",
        group_role=context.group_role if context else "一般應徵者",
        turn_index=follow_up_idx,
        has_resume=not group and bool(resume_block(experiences)),
    )
    user = "\n\n".join(user_parts)

    try:
        raw = await asyncio.to_thread(llm, system, user)
    except Exception as exc:  # noqa: BLE001
        r = _fallback_turn(mode, fallback, follow_up_idx, gi, gs)
        r.notices = [f"本輪:生成失敗({type(exc).__name__}),{_fallback_used(mode)}"]
        return r

    data = parse_obj(raw)
    nxt = normalize_locale(str(data.get("nextQuestion", "")).strip())
    salvage_note = ""
    if not nxt:
        rescued = salvage_question(raw)
        if rescued:
            nxt = normalize_locale(rescued)
            salvage_note = "本輪:模型沒有包成 JSON,已直接採用它產出的問題"
        else:
            r = _fallback_turn(mode, fallback, follow_up_idx, gi, gs)
            # 把原始輸出的開頭留下來。沒有它就查不出是格式問題、拒答、
            # 還是模型輸出了別的欄位——三種的修法完全不同。
            snippet = (raw or "").strip().replace("\n", " ")[:120] or "(空)"
            r.notices = [f"本輪:模型未產出問題,{_fallback_used(mode)}。原始輸出:{snippet}"]
            return r

    notices: list[str] = [x for x in (salvage_note, dispatch_note, decision_note) if x]
    # 說話者由派發那一步決定,不採用生成端回的值——
    # 生成端的 prompt 已經寫死「speaker 一律填 X」,回別的就是它沒照做。
    speaker, problem = repair_speaker(
        focus or str(data.get("speaker", "")), mode, gi, gs
    )
    if problem:
        notices.append(f"本輪:{problem}")
    if forced and speaker != forced:
        # 指派由程式保證。只寫進 prompt 的話模型會不照做——實測 group 五輪
        # 裡 AI-強勢 掛零,即使它的路由條件明確命中。
        notices.append(f"本輪:模型指派給「{speaker}」,已改為未發言過的「{forced}」")
        speaker = forced

    # isFollowUp:一對一的輸出格式拿掉了這一欄(欄位越少越可能被遵守),
    # 群面與 panel 仍然會回,有回就用它的。
    #
    # 【一對一怎麼判斷】
    # 早期版本用「topic 跟上一輪相同就是追問」,那是錯的:
    # 實測五輪全部在追問,但模型每輪都給了誠實的新 topic
    # (實習內容 → 查詢重寫的具體做法 → 協調過程),於是判成 0/5。
    #
    # 追問的定義是「接著剛剛那段回答問」,跟 topic 換不換沒有必然關係——
    # 追問也可以帶出新的領域名稱。把兩件事綁在一起是我的錯。
    #
    # 一對一沒有「新主問題」的概念:每一輪都是接著上一句問,
    # 直到整場輪次用完進入反問環節。所以預設為 True,
    # 只有明確換到不相干主題(topic 跟前一輪與上一題都無關)時才是 False。
    #
    # 【有 decision 的時候不用猜】
    # 決定本身就是答案,而且它是在生成之前做的,所以生成端也收到了同一個指示。
    if forced_next:
        is_follow_up = False
    elif decision is not None:
        is_follow_up = decision.is_follow_up
    elif "isFollowUp" in data:
        is_follow_up = bool(data["isFollowUp"])
    else:
        is_follow_up = True

    # 「整場該不該結束」由輪次決定,一律不採用模型的判斷。
    # 實測:模型因為使用者一句「我沒想過」就把 shouldAdvance 設成 true,
    # 提前結束了整場面試。那是確定性事實,不是判斷題。
    should_advance = must_advance
    if must_advance:
        notices.append(f"本輪:整場已達上限 {cap} 題,標記結束")
    elif bool(data.get("shouldAdvance", False)):
        if cap is None:
            notices.append("本輪:模型想結束討論,已忽略——群面由使用者自己決定何時結束")
        else:
            notices.append("本輪:模型想提前結束,已忽略——結束時機由輪次決定")

    # 新主問題跟問過的重複時,改用備援池裡沒用過的一題。
    # 重試會增加面試中的延遲,而備援池是開場時針對這份 JD 生成的,直接用就行。
    #
    # 追問不走這一關:它該留在同一個話題上,所以不能用「問同一件事」當判準,
    # 也不能換成備援題——備援題是另一個話題,那就不是追問了。
    # 追問的重複改在下面那一關擋,判準是「同一句話」而不是「同一件事」。
    #
    # 群面不走這一關:備援句是主持人的推進句,換進 AI 應徵者嘴裡就是
    # 一句面試官的話。群面的重複在下面那一關擋。
    if not is_follow_up and not group:
        dup = is_near_duplicate(nxt, asked_questions)
        if dup:
            unused = [p for p in fallback if not is_near_duplicate(p, asked_questions)]
            if unused:
                notices.append(f"本輪:新主問題與先前重複,改用備援題。原本:{nxt[:30]}")
                nxt = unused[0]
            else:
                notices.append(
                    f"本輪:新主問題與先前重複,但備援池已用盡。重複的是:{dup[:30]}"
                )
    # 送出前的形式檢查。這一關同時看兩件事(追問與問過的同一句、只能用是/否
    # 答完),共用一次重生成。擋到的機率不低——四次量測 144 題裡封閉式佔 25%,
    # 所以一場四題大約會多一次呼叫。
    clarify = decision is not None and decision.action == _CLARIFY
    # 群面比對的是前面所有非使用者的發言,不只問過的問題原文。
    said = list(asked_questions)
    if group:
        said += [u.content for u in group_says if not u.is_user and (u.content or "").strip()]
    faults = _question_faults(
        nxt, said, is_follow_up=is_follow_up, clarify=clarify, group=group
    )
    if faults:
        why = "、".join(label for label, _ in faults)
        again = await _regenerate(llm, system, user, faults)
        left = (
            _question_faults(
                again, said, is_follow_up=is_follow_up, clarify=clarify, group=group
            )
            if again
            else [("沒有產出", "")]
        )
        if left:
            # 重生成也擋不住就照原樣送出。留著一句不夠好的問題比沒有問題好,
            # 而且 notice 查得到。
            notices.append(
                f"本輪:問句{why},重生成沒有改善"
                f"({'、'.join(label for label, _ in left)}),已照原樣送出。"
            )
        else:
            notices.append(f"本輪:問句{why},已重生成。原本:{nxt[:30]}")
            nxt = again

    # topic 不可以留空。前端拿它累積 asked_topics,空的那一筆會讓序列斷掉,
    # 而斷掉的地方就是之後重複提問的破口。
    # 搶救路徑救回問題但沒有後設欄位時,沿用上一輪的領域——
    # 那通常是對的,因為搶救回來的多半是追問。
    topic = normalize_locale(str(data.get("topic", "")).strip())
    if not topic:
        topic = prev_topic or "延續上一題"
        notices.append(f"本輪:模型未標領域,已沿用「{topic}」")

    return TurnResponse(
        speaker=speaker,
        next_question=nxt,
        reaction=normalize_locale(str(data.get("reaction", "")).strip()),
        is_follow_up=is_follow_up,
        should_advance=should_advance,
        topic=topic,
        notices=notices,
    )
