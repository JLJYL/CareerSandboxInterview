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
from app.prompts.turn_decision import DECISION_ACTIONS, compose_decision_prompt

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

#: 看起來像道歉或免責的輸出一律不採用,寧可降級。
_REFUSAL_MARKS = ("抱歉", "無法", "很遺憾", "作為一個", "我是 AI", "語言模型")

MIN_ANSWER_CHARS = 2
"""短於這個長度視為沒有內容。空的 timeout 輪會走到這裡。"""


@dataclass(frozen=True)
class TurnDecision:
    """這一輪的決定。"""

    action: str
    why: str = ""
    gap: str = ""
    source: str = "llm"
    """誰做的決定:code / llm / fallback。

    分開記是為了驗證時能把兩者拆開看——程式決定的那些不該拿去
    評模型的判斷品質,而降級的那些要看得到有多少。
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
    """解析決策輸出。JSON 壞掉時從純文字救一次,再壞就回 None。"""
    text = (raw or "").strip()
    if not text or any(bad in text for bad in _REFUSAL_MARKS):
        return None

    data = parse_obj(text)
    if data:
        action = normalize_action(str(data.get("action", "")))
        if action:
            return TurnDecision(
                action=action,
                why=str(data.get("why", "")).strip(),
                gap=str(data.get("gap", "")).strip() if action == FOLLOW_UP else "",
                source="llm",
            )

    # 沒包 JSON 但整段就是一個動作名——那個決定本身是好的,不要丟掉。
    action = normalize_action(text)
    if action:
        return TurnDecision(action=action, why="", gap="", source="llm")
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
        return TurnDecision(NEXT, "這一輪沒有內容可以追問。", source="code")
    if ended_by == "timeout":
        return TurnDecision(
            FOLLOW_UP, "這段回答被語音辨識切斷,先請他把話接完。", "未講完的部分", "code"
        )
    if depth >= spec.accept.max_depth:
        return TurnDecision(
            NEXT, f"同一個缺口已追問 {depth} 次,到這個難度的上限。", source="code"
        )
    if any(k in text for k in _ADMITS_UNKNOWN):
        return TurnDecision(NEXT, "他明確表示答不出來,換一個他答得出來的。", source="code")
    return None


def fallback_decision(reason: str) -> TurnDecision:
    """決策完全失敗時的降級。

    選 NEXT 不選 FOLLOW_UP:一個解析不出來的決定不該把人釘在同一個點上,
    而且 max_depth 的保護本來就建立在決定可信之上。
    代價是這一輪的門檻被放寬了,所以 source 標成 fallback,驗證時看得到。
    """
    return TurnDecision(NEXT, reason, source="fallback")


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
        return fallback_decision("決策輸出無法解析,改為推進。")
    return decided
