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
from app.prompts.interview_live import (
    compose_dispatch_prompt,
    compose_interrupt_prompt,
    compose_opening_prompt,
    compose_turn_prompt,
)
from app.prompts.turn_decision import ACTION_INSTRUCTION
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


def resume_block(experiences: Sequence[ExperienceDTO]) -> str:
    """把經歷攤成 prompt 材料。開場與報告共用這一份。

    原本只有報告那條路有這個函式(`routes._resume_text`),而開場那邊
    根本沒用到履歷。兩邊各寫一份的話格式會分岔,而分岔的方向通常是
    「其中一邊忘了帶某個欄位」,然後沒有人發現。
    """
    lines: list[str] = []
    used = 0
    for e in experiences:
        bits = [e.title, e.role, e.action, e.result]
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


def is_near_duplicate(question: str, asked: Sequence[str]) -> str | None:
    """找出與先前問題高度重疊的一題。回傳那一題,沒有就 None。

    純字面重疊,不做語意。門檻訂得嚴,因為問句本來就會共用很多虛詞。

    【為什麼要機械檢查】
    已問問題原文已經傳給模型、規則也寫了不要重複,實測仍然會問近似題:
    群面第 5 輪主考官問「你過去如何處理資料清理的經驗」,
    而開場題是「你在資料分析方面的經驗,特別是如何處理數據或報表」。

    這跟派發平衡是同一類:資訊給了,模型不可靠地執行比對。
    """
    q = set(question) - _STOPWORDS
    if not q:
        return None
    for a in asked:
        av = set(a) - _STOPWORDS
        if not av:
            continue
        shared = len(q & av)
        if shared >= 4 and shared / min(len(q), len(av)) >= DUP_THRESHOLD:
            return a
    return None


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
    ctx_text = context_block(context, jd, experiences)
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

        tasks["interrupts"] = asyncio.ensure_future(_gen_interrupts(llm, INTERRUPT_CAP))
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
    question = (
        normalize_locale(str(op.get("openingQuestion", "")).strip()) or DEFAULT_OPENING
    )
    if not op.get("openingQuestion"):
        notices.append("開場:未取得主問題,已使用預設")
    probes = [
        normalize_locale(str(x).strip())
        for x in (op.get("fallbackProbes") or [])
        if str(x).strip()
    ]
    if not probes:
        probes = list(DEFAULT_PROBES)
        notices.append("開場:未取得備援追問,已使用預設")

    interrupts: list[str] = list(results.get("interrupts") or [])
    if mode == "group" and not interrupts:
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
        opening_topic=normalize_locale(str(op.get("openingTopic", "")).strip())
        or "自我介紹",
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
    mode: str, fallback: Sequence[str], follow_up_idx: int
) -> TurnResponse:
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
    prev_topic: str = "",
    ended_by: str = "unknown",
    fallback: Sequence[str],
    llm: LLMCall,
    context: InterviewContext | None = None,
    decision: "TurnDecision | None" = None,
) -> TurnResponse:
    """一輪接話。

    follow_up_idx 達到上限時強制推進,不問模型——
    「這是第幾次追問」是確定性事實,交給模型判斷會讓它有時多追一次。

    【decision】
    這一輪要做什麼(NEXT / FOLLOW_UP / CLARIFY),由 `turn_decision` 先決定好再傳進來。
    決定在 `routes.py` 那層算,不在這裡呼叫——這支函式只收資料不自己發起決策,
    跟 LLM 注入是同一個原則。

    傳 None 時維持舊行為:`is_follow_up` 取模型回的值,一對一沒回就預設 True。
    那是猜的,留著只是為了讓既有測試仍然測得到生成那一段。
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
    user_parts = [
        f"【他剛剛說的】\n{answer}",
        "",
        "你要回應的是上面這段話。下面的前一題只是脈絡,不是這次的主題。",
        f"【前一題(脈絡)】{question or '(未提供)'}",
    ]
    if context:
        user_parts.append("【職位脈絡】\n" + context_block(context))
    gi = context.group_interviewers if context else 1
    gs = context.group_size if context else 4
    names = speaker_names(mode, gi, gs)
    forced = required_speaker(mode, spoken_by, follow_up_idx, answer, gi, gs)
    counts = {n: list(spoken_by).count(n) for n in names}

    # 第一步:決定誰說話。強制指派時跳過這次呼叫——答案已經確定了。
    focus = forced or ""
    dispatch_note = f"本輪:指派給未發言過的「{forced}」" if forced else ""
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

    try:
        raw = await asyncio.to_thread(
            llm,
            compose_turn_prompt(
                mode,
                list(asked_questions),
                group_interviewers=gi,
                group_size=gs,
                focus_speaker=focus,
                truncated=ended_by == "timeout",
                engine=context.transcription_engine if context else "api",
                difficulty=context.difficulty if context else "中等",
                group_role=context.group_role if context else "一般應徵者",
            ),
            "\n\n".join(user_parts),
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
    # 追問不受此限——追問本來就該留在同一個話題上。
    if not is_follow_up:
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
