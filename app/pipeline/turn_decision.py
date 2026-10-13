"""決定這一輪要做什麼,跟生成那一句分開。

【為什麼要把決定獨立出來】
現在沒有「決定」這個步驟——判準寫在 prompt 裡,模型直接吐出下一句問題,
所以系統不知道自己剛剛做了什麼。`TurnResponse.is_follow_up` 因此只能反推:
早期版本用 topic 相等推論,實測五題全是追問卻給 0/5,後來改成一對一一律
預設 True。那不是判斷,是猜。

決定變成資料之後拿到三件事:

    is_follow_up   是決定本身,不用猜
    追問率         可以數 → 難度第一次驗得出來
    why            每個決定都有理由可以逐項標注

【能用程式決定的不問模型】
`forced_decision()` 先跑一輪確定性判斷,命中就不呼叫 LLM。
這跟派發那邊 `required_speaker()` 先於 `_dispatch()` 是同一個做法:
有唯一正確答案的事交給程式,一次就 100% 解決。

【輸出一定要能壞】
Panfilova et al.(2026)實測 DeepSeek Chat V3.1 在三場面試裡累積
45 次 schema 驗證錯誤。所以解析失敗是常態不是例外,
這裡的每一條失敗路徑都有降級,而且降級會標在 source 上讓之後查得到。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.pipeline.interview_live import LLMCall, parse_obj
from app.pipeline.interviewer_spec import InterviewerSpec
from app.prompts.turn_decision import (
    DECISION_ACTIONS,
    GAP_VALUES,
    compose_decision_prompt,
)

# 明確表示答不出來。與 interview_live 共用同一組字面,改要一起改。
from app.pipeline.interview_live import _ADMITS_UNKNOWN

NEXT, FOLLOW_UP, CLARIFY = DECISION_ACTIONS

#: 模型實際會吐出來的各種寫法 → 正規值。
_ACTION_ALIASES: dict[str, str] = {
    "next": NEXT,
    "nextmain": NEXT,
    "next_main": NEXT,
    "advance": NEXT,
    "換題": NEXT,
    "推進": NEXT,
    "followup": FOLLOW_UP,
    "follow_up": FOLLOW_UP,
    "follow-up": FOLLOW_UP,
    "probe": FOLLOW_UP,
    "追問": FOLLOW_UP,
    "clarify": CLARIFY,
    "clarification": CLARIFY,
    "澄清": CLARIFY,
}

#: 整段都是道歉或免責時不採用,寧可降級。
#
#  只在**沒有結構化動作可用**的時候才比對(見 parse_decision)。
#  「無法」是中文最常用的詞之一,拿它去掃整段原文會掃到 why 欄位。
_REFUSAL_MARKS = ("抱歉", "無法", "很遺憾", "作為一個", "我是 AI", "語言模型")

MIN_ANSWER_CHARS = 2
"""短於這個長度視為沒有內容。空的 timeout 輪會走到這裡。"""

MAX_SUPERSEDED_CHARS = 300
"""記錄被取代的輸出時最多留幾個字。只為了事後查得出來,不需要全文。"""

DECISION_SOURCES: tuple[str, ...] = ("code", "llm", "fallback")
CODE, LLM, FALLBACK = DECISION_SOURCES
"""`TurnDecision.source` 的值域。

宣告出來是為了報表:工具自己列一份的話,這裡多一個值而它沒跟著加,
那一整類會被漏掉而且報表照樣印得出來。
"""


#: 模型實際會寫的講法 → GAP_VALUES。只在它沒照封閉集合填時當救援。
_GAP_ALIASES: dict[str, str] = {
    "情境": "缺情境",
    "背景": "缺情境",
    "任務": "缺任務",
    "目標": "缺任務",
    "行動": "缺行動",
    "做法": "缺行動",
    "作法": "缺行動",
    "量化": "結果沒量化",
    "數字": "結果沒量化",
    "數據": "結果沒量化",
    "指標": "結果沒量化",
    "取捨": "沒有取捨",
    "權衡": "沒有取捨",
    "放棄": "沒有取捨",
    "代價": "沒有取捨",
    "結果": "缺結果",
    "成果": "缺結果",
}


def normalize_gap(raw: str) -> str:
    """把 gap 修成封閉集合裡的值。認不得或指向多個就回空字串。

    【為什麼多個就放棄】
    「缺少結果段和量化數據」同時指向 `缺結果` 與 `結果沒量化`,而那兩個
    是互斥的事實——結果到底在不在,這句話答不出來。猜一個會把不確定
    變成確定,所以留空,讓報表看得出模型沒照封閉集合填。
    """
    text = (raw or "").strip().strip("\"'`。.、,,")
    if not text:
        return ""
    if text in GAP_VALUES:
        return text
    hit = {v for word, v in _GAP_ALIASES.items() if word in text}
    return hit.pop() if len(hit) == 1 else ""


@dataclass(frozen=True)
class TurnDecision:
    """這一輪的決定。"""

    action: str
    why: str = ""
    gap: str = ""
    """缺的是哪一項。只會是 GAP_VALUES 其中一個,或空字串。"""

    source: str = LLM
    """誰做的決定:code / llm / fallback。

    分開記是為了驗證時能把兩者拆開看——程式決定的那些不該拿去
    評模型的判斷品質,而降級的那些要看得到有多少。
    """

    superseded: str = ""
    """被這個決定取代掉的東西,截斷後留著。

    降級與(未來的)程式覆寫都會丟掉模型原本說了什麼,而那正是事後要查的。
    1e 那次就是:覆寫建了一個新的 TurnDecision,原本的 gap 沒帶過去,
    要查覆寫掉了什麼只能靠 why 裡剛好寫了推導過程——那是運氣不是設計。

    所以規則是:任何取代模型決定的程式路徑,都要把原本的東西放進這裡。
    """

    @property
    def is_follow_up(self) -> bool:
        """對應 TurnResponse.is_follow_up。CLARIFY 也算追問——沒有換題。"""
        return self.action in (FOLLOW_UP, CLARIFY)


def normalize_action(raw: str) -> str | None:
    """把模型回的動作修成合法值。認不得回 None。"""
    s = (raw or "").strip().strip("\"'`。.")
    if not s:
        return None
    if s.upper() in DECISION_ACTIONS:
        return s.upper()
    return _ACTION_ALIASES.get(s.lower().replace(" ", ""))


def parse_decision(raw: str) -> TurnDecision | None:
    """解析決策輸出。JSON 壞掉時從純文字救一次,再壞就回 None。

    【為什麼拒絕詞排在解析後面】
    原本在解析前就拿 `_REFUSAL_MARKS` 掃整段原文。但「無法」是中文最常用的
    詞之一,模型說明缺了哪一項時一定會寫「無法判斷」「無法確認」,
    於是一個合法的 FOLLOW_UP 會連著理由一起被丟掉。

    而且丟掉的方向是偏的:會寫「無法」的句子剛好都是在說少了什麼,
    也就是 FOLLOW_UP 那些。追問率因此被系統性壓低。

    1d 第一次實測量到降級 19/180,中等一個難度就佔 13 次,
    中等的追問率從期望的 0.500 掉到 0.283——比新手還低。
    這正是 `source` 那一欄存在的理由:降級後的決定長得跟
    「模型判斷講完了」完全一樣,只看追問率看不出來。

    所以順序是:有解析得出來的動作就收下,不管 why 寫了什麼;
    只有在完全沒有結構可用、要從純文字硬救的時候才擋拒絕詞。
    """
    text = (raw or "").strip()
    if not text:
        return None

    data = parse_obj(text)
    if data:
        action = normalize_action(str(data.get("action", "")))
        if action:
            return TurnDecision(
                action=action,
                why=str(data.get("why", "")).strip(),
                gap=normalize_gap(str(data.get("gap", ""))) if action == FOLLOW_UP else "",
                source=LLM,
            )

    # 到這裡表示沒有結構化的動作可用。整段是道歉的那種輸出不能硬救成動作名。
    if any(bad in text for bad in _REFUSAL_MARKS):
        return None

    # 沒包 JSON 但整段就是一個動作名——那個決定本身是好的,不要丟掉。
    action = normalize_action(text)
    if action:
        return TurnDecision(action=action, why="", gap="", source=LLM)
    return None


def forced_decision(
    *,
    spec: InterviewerSpec,
    answer: str,
    depth: int = 0,
    ended_by: str = "unknown",
) -> TurnDecision | None:
    """程式就能決定的情況。決定不了回 None,交給模型。

    順序是有意的,前面的優先:

    1. 沒有內容       沒東西可追。群面使用者沉默送的空輪會走到這裡。
    2. 被切斷         先讓他把話講完,其他判準這一輪不適用。
                      這條排在深度上限前面——被打斷不是他的問題,
                      不該因為「已經追三次了」就不讓他講完。
    3. 到深度上限     同一個缺口追到上限就換題,不管他補完整沒有。
                      追到第三次還講不出來,繼續追只是折磨,
                      而那本來就該誠實反映在報告裡。
    4. 明確說不會     換一個他答得出來的,不要問同一件事的變體。
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
    if depth >= spec.accept.max_depth:
        return TurnDecision(
            NEXT, f"同一個缺口已追問 {depth} 次,到這個難度的上限。", source=CODE
        )
    if any(k in text for k in _ADMITS_UNKNOWN):
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


async def decide_turn(
    *,
    spec: InterviewerSpec,
    answer: str,
    llm: LLMCall,
    question: str = "",
    depth: int = 0,
    ended_by: str = "unknown",
    transcription_engine: str = "api",
) -> TurnDecision:
    """這一輪要做什麼。永遠回一個決定,不拋例外。"""
    forced = forced_decision(spec=spec, answer=answer, depth=depth, ended_by=ended_by)
    if forced is not None:
        return forced

    user = "\n".join(
        [
            f"【他剛剛說的】\n{answer.strip()}",
            "",
            f"【上一題(脈絡)】{question or '(未提供)'}",
            "",
            f"【同一個缺口已經追問過 {depth} 次】",
        ]
    )
    system = compose_decision_prompt(spec, transcription_engine=transcription_engine)
    try:
        raw = await asyncio.to_thread(llm, system, user)
    except Exception as exc:  # noqa: BLE001
        return fallback_decision(f"決策呼叫失敗({type(exc).__name__}),改為推進。")

    decided = parse_decision(raw)
    if decided is None:
        return fallback_decision("決策輸出無法解析,改為推進。", raw)
    return decided
