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
import json
import re
import uuid
from collections.abc import Callable, Sequence

from app.contracts.interview_protocols import JDInput
from app.pipeline.jd_normalize import normalize_locale
from app.prompts.interview_live import (
    compose_interrupt_prompt,
    compose_opening_prompt,
    compose_turn_prompt,
)
from app.prompts.interview_personas import (
    INTERRUPT_CAP,
    personas_for,
    speaker_names,
)
from app.prompts.probe_rules import MAX_TURNS_PER_SESSION
from app.schemas.interview import (
    InterviewContext,
    Persona,
    StartInterviewResponse,
    TurnResponse,
)

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


def context_block(ctx: InterviewContext, jd: JDInput | None = None) -> str:
    """把設定頁的欄位攤成 prompt 材料。

    JD 是選填的。空的時候不要留一個空欄位讓模型腦補,直接寫明沒有——
    「JD:」後面接空白會讓它以為漏了什麼而去猜。
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
        lines.append("這份職缺需要的能力(依重要性排序):" + "、".join(jd.required_skills[:15]))
    elif ctx.custom_jd:
        lines.append(f"JD 原文:{ctx.custom_jd[:600]}")
    else:
        lines.append("JD:使用者沒有提供。只用上面的職位脈絡出題,不要假裝知道職缺內容。")
    return "\n".join(lines)


def _persona_dtos(mode: str) -> list[Persona]:
    return [
        Persona(id=p.id, display_name=p.display_name, role=p.role, blurb=p.blurb)
        for p in personas_for(mode)
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
    if any(bad in t for bad in ("抱歉", "無法", "很遺憾", "作為一個", "我是 AI", "語言模型")):
        return None
    return t


def required_speaker(
    mode: str, spoken_by: Sequence[str], turn_idx: int, answer: str = ""
) -> str | None:
    """這一輪必須由誰開口。沒有硬性要求時回 None。

    回傳的名稱會同時寫進 prompt(讓問題的語氣對得上)與覆寫最終的 speaker
    (讓指派必然成立)。只做前者模型會不照做,只做後者問題的語氣會跟人格不符。
    """
    names = speaker_names(mode)
    if not names or turn_idx < SILENT_FORCE_AFTER_TURNS:
        return None
    # 對方明確說不會時不強制指派——那一輪本來就該由主考官或 HR 主管接住。
    # 實測:AI-強勢 被強制指派到這種情境,講出「沒關係,換個問題」,
    # 那是主考官的工作,而且完全不符合它的人格。
    if any(k in answer for k in _ADMITS_UNKNOWN):
        return None
    counts = {n: list(spoken_by).count(n) for n in names}
    silent = [n for n, c in counts.items() if c == 0]
    return silent[0] if silent else None


def repair_speaker(raw: str, mode: str) -> tuple[str, str | None]:
    """把模型回的 speaker 修成合法值。回傳 (修好的名稱, 問題說明)。

    合約規定 speaker 是開放字串不是 Literal(CareerCategory 事故的教訓),
    但值域實際上封閉——前端顯示不出清單外的名稱。所以用機械修復維持。

    修復順序:完全相同 → 去空白後相同 → 被包含 → 都不是就取第一個。
    取第一個而不是留空,是因為 panel/group 模式沒有說話者的話畫面會壞掉。
    """
    names = speaker_names(mode)
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


async def _gen_opening(llm: LLMCall, mode: str, ctx_text: str) -> dict:
    raw = await asyncio.to_thread(llm, compose_opening_prompt(mode), ctx_text)
    return parse_obj(raw)


async def _gen_interrupts(llm: LLMCall, n: int) -> list[str]:
    raw = await asyncio.to_thread(llm, compose_interrupt_prompt(n), "產生搶話台詞。")
    return [normalize_locale(str(x).strip()) for x in parse_list(raw) if str(x).strip()][:n]


DEFAULT_OPENING = "先請你用一分鐘自我介紹,重點放在跟這個職位相關的經驗。"
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


async def start_interview(
    *,
    mode: str,
    context: InterviewContext,
    llm: LLMCall,
    jd: JDInput | None = None,
    session_id: str | None = None,
) -> StartInterviewResponse:
    """開場。群面另外並行生成搶話台詞。

    任何一段失敗都退回預設值——面試按下開始卻噴 500,使用者連場都進不去。
    """
    notices: list[str] = []
    ctx_text = context_block(context, jd)

    tasks: dict[str, asyncio.Task] = {"opening": asyncio.ensure_future(_gen_opening(llm, mode, ctx_text))}
    if mode == "group":
        tasks["interrupts"] = asyncio.ensure_future(_gen_interrupts(llm, INTERRUPT_CAP))

    results: dict[str, object] = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as exc:  # noqa: BLE001
            notices.append(f"開場:{name} 生成失敗({type(exc).__name__}),已使用預設")
            results[name] = {} if name == "opening" else []

    op = results.get("opening") or {}
    question = normalize_locale(str(op.get("openingQuestion", "")).strip()) or DEFAULT_OPENING
    if not op.get("openingQuestion"):
        notices.append("開場:未取得主問題,已使用預設")
    probes = [normalize_locale(str(x).strip()) for x in (op.get("fallbackProbes") or []) if str(x).strip()]
    if not probes:
        probes = list(DEFAULT_PROBES)
        notices.append("開場:未取得備援追問,已使用預設")

    interrupts: list[str] = list(results.get("interrupts") or [])
    if mode == "group" and not interrupts:
        interrupts = list(DEFAULT_INTERRUPTS)
        notices.append("開場:未取得搶話台詞,已使用預設")

    personas = _persona_dtos(mode)
    opening_speaker = personas[0].display_name if personas else ""

    return StartInterviewResponse(
        session_id=session_id or f"itv_{uuid.uuid4().hex[:12]}",
        mode=mode,
        opening_question=question,
        opening_speaker=opening_speaker,
        opening_topic=normalize_locale(str(op.get("openingTopic", "")).strip()) or "自我介紹",
        personas=personas,
        interrupt_lines=interrupts,
        interrupt_cap=len(interrupts),
        fallback_probes=probes,
        notices=notices,
    )


# ---------------------------------------------------------------------------
# 每輪
# ---------------------------------------------------------------------------


def _fallback_turn(mode: str, fallback: Sequence[str], follow_up_idx: int) -> TurnResponse:
    """LLM 不可用時的降級。用前端帶來的備援池,不要臨時編。"""
    pool = list(fallback) or list(DEFAULT_PROBES)
    names = speaker_names(mode)
    return TurnResponse(
        speaker=names[follow_up_idx % len(names)] if names else "",
        next_question=pool[follow_up_idx % len(pool)],
        reaction="",
        is_follow_up=False,
        should_advance=True,
        topic="",
        notices=["本輪:生成失敗,已使用備援追問"],
    )


async def next_turn(
    *,
    mode: str,
    answer: str,
    follow_up_idx: int,
    asked_questions: Sequence[str],
    spoken_by: Sequence[str] = (),
    question: str = "",
    fallback: Sequence[str],
    llm: LLMCall,
    context: InterviewContext | None = None,
) -> TurnResponse:
    """一輪接話。

    follow_up_idx 達到上限時強制推進,不問模型——
    「這是第幾次追問」是確定性事實,交給模型判斷會讓它有時多追一次。
    """
    must_advance = follow_up_idx >= MAX_TURNS_PER_SESSION

    user_parts = [f"【上一個問題】{question or '(未提供)'}", f"【他的回答】{answer}"]
    if context:
        user_parts.append("【職位脈絡】\n" + context_block(context))
    names = speaker_names(mode)
    forced = required_speaker(mode, spoken_by, follow_up_idx, answer)
    if names:
        counts = {n: list(spoken_by).count(n) for n in names}
        user_parts.append(
            "【目前為止的發言次數】\n"
            + "\n".join(f"  {n}:{c} 次" for n, c in counts.items())
        )
    if forced:
        user_parts.append(
            f"【這一輪必須由「{forced}」開口】\n"
            "他到目前為止一次都沒講過話。即使內容命中別人的領域,也由他來問——\n"
            "用他的人格與語氣寫這個問題,不要寫成別人會問的樣子。"
        )
    if must_advance:
        user_parts.append(
            f"【注意】整場已經問到第 {follow_up_idx + 1} 題,達到上限 "
            f"{MAX_TURNS_PER_SESSION} 題。請問最後一題,並把 shouldAdvance 設為 true。"
        )

    try:
        raw = await asyncio.to_thread(
            llm, compose_turn_prompt(mode, list(asked_questions)), "\n\n".join(user_parts)
        )
    except Exception as exc:  # noqa: BLE001
        r = _fallback_turn(mode, fallback, follow_up_idx)
        r.notices = [f"本輪:生成失敗({type(exc).__name__}),已使用備援追問"]
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
            r = _fallback_turn(mode, fallback, follow_up_idx)
            # 把原始輸出的開頭留下來。沒有它就查不出是格式問題、拒答、
            # 還是模型輸出了別的欄位——三種的修法完全不同。
            snippet = (raw or "").strip().replace("\n", " ")[:120] or "(空)"
            r.notices = [f"本輪:模型未產出問題,已使用備援追問。原始輸出:{snippet}"]
            return r

    notices: list[str] = [salvage_note] if salvage_note else []
    speaker, problem = repair_speaker(str(data.get("speaker", "")), mode)
    if problem:
        notices.append(f"本輪:{problem}")
    if forced and speaker != forced:
        # 指派由程式保證。只寫進 prompt 的話模型會不照做——實測 group 五輪
        # 裡 AI-強勢 掛零,即使它的路由條件明確命中。
        notices.append(f"本輪:模型指派給「{speaker}」,已改為未發言過的「{forced}」")
        speaker = forced

    is_follow_up = bool(data.get("isFollowUp", False))

    # 「整場該不該結束」由輪次決定,一律不採用模型的判斷。
    # 實測:模型因為使用者一句「我沒想過」就把 shouldAdvance 設成 true,
    # 提前結束了整場面試。那是確定性事實,不是判斷題。
    should_advance = must_advance
    if must_advance:
        notices.append(f"本輪:整場已達上限 {MAX_TURNS_PER_SESSION} 題,標記結束")
    elif bool(data.get("shouldAdvance", False)):
        notices.append("本輪:模型想提前結束,已忽略——結束時機由輪次決定")

    return TurnResponse(
        speaker=speaker,
        next_question=nxt,
        reaction=normalize_locale(str(data.get("reaction", "")).strip()),
        is_follow_up=is_follow_up,
        should_advance=should_advance,
        topic=normalize_locale(str(data.get("topic", "")).strip()),
        notices=notices,
    )
