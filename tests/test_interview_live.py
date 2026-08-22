"""Live 引擎(A1/A2/A3)測試。不打真 LLM。

重點在**降級路徑**與**機械保證**:
面試進行中噴 500 等於整場中斷,所以每一條失敗路徑都要有測試。
"""

from __future__ import annotations

import json

import pytest

from app.contracts.interview_protocols import JDInput
from app.pipeline.interview_live import (
    context_block,
    next_turn,
    parse_obj,
    repair_speaker,
    start_interview,
)
from app.prompts.interview_live import compose_turn_prompt
from app.prompts.interview_personas import speaker_names
from app.prompts.probe_rules import MAX_TURNS_PER_SESSION
from app.schemas.interview import InterviewContext

CTX = InterviewContext(
    custom_role="資料分析實習生", custom_company="某電商", custom_industry="電子商務",
    custom_seniority="新鮮人", type="行為",
)


def llm_returning(payload) -> object:
    def call(system: str, user: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return call


def boom(system: str, user: str) -> str:
    raise RuntimeError("timeout")


OPENING = {
    "openingQuestion": "先講一個你用資料做出判斷的經驗。",
    "openingTopic": "資料分析",
    "fallbackProbes": ["那個判斷的依據是什麼?", "如果數字跟直覺打架你信哪個?"],
}
TURN = {
    "speaker": "技術主管", "nextQuestion": "那個數字的母數是多少?",
    "reaction": "嗯。", "isFollowUp": True, "topic": "資料分析", "shouldAdvance": False,
}


# ---------------------------------------------------------------------------
# 說話者的機械修復
# ---------------------------------------------------------------------------


def test_single_mode_has_no_speaker() -> None:
    """一對一前端不顯示說話者,回名稱反而會在畫面上多出東西。"""
    assert repair_speaker("面試官", "single") == ("", None)


def test_exact_speaker_passes() -> None:
    assert repair_speaker("技術主管", "panel") == ("技術主管", None)


@pytest.mark.parametrize("raw,want", [
    ("技術 主管", "技術主管"),
    ("技術主管:", "技術主管"),
    ("HR", "HR 主管"),
])
def test_near_miss_speaker_repaired(raw: str, want: str) -> None:
    got, problem = repair_speaker(raw, "panel")
    assert got == want
    assert problem


def test_unknown_speaker_falls_back_not_blank() -> None:
    """panel/group 沒有說話者的話畫面會壞掉,所以取第一個而不是留空。"""
    got, problem = repair_speaker("神秘嘉賓", "panel")
    assert got in speaker_names("panel")
    assert problem


def test_group_speakers_include_peers_and_moderator() -> None:
    names = speaker_names("group")
    assert "主考官" in names
    assert "AI-邏輯" in names and "AI-強勢" in names and "AI-親切" in names


# ---------------------------------------------------------------------------
# 開場
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_happy_path() -> None:
    r = await start_interview(mode="single", context=CTX, llm=llm_returning(OPENING))
    assert r.opening_question == OPENING["openingQuestion"]
    assert r.opening_topic == "資料分析"
    assert len(r.fallback_probes) == 2
    assert r.session_id.startswith("itv_")


@pytest.mark.asyncio
async def test_start_degrades_on_failure() -> None:
    """面試按下開始卻噴 500,使用者連場都進不去。"""
    r = await start_interview(mode="single", context=CTX, llm=boom)
    assert r.opening_question
    assert r.fallback_probes
    assert any("已使用預設" in n for n in r.notices)


@pytest.mark.asyncio
async def test_group_gets_interrupt_lines() -> None:
    """搶話在開場一次生成完——interruptLine(index) 是同步呼叫,
    那個時機發 HTTP 會卡住 UI。"""
    def call(system: str, user: str) -> str:
        if "搶節奏" in system:
            return json.dumps(["我先說。", "我插一個快的。", "我補一句。"], ensure_ascii=False)
        return json.dumps(OPENING, ensure_ascii=False)

    r = await start_interview(mode="group", context=CTX, llm=call)
    assert len(r.interrupt_lines) == 3
    assert r.interrupt_cap == 3


@pytest.mark.asyncio
async def test_non_group_has_no_interrupts() -> None:
    r = await start_interview(mode="panel", context=CTX, llm=llm_returning(OPENING))
    assert r.interrupt_lines == []
    assert r.interrupt_cap == 0


@pytest.mark.asyncio
async def test_personas_match_mode() -> None:
    single = await start_interview(mode="single", context=CTX, llm=llm_returning(OPENING))
    panel = await start_interview(mode="panel", context=CTX, llm=llm_returning(OPENING))
    assert single.personas == []          # 一對一不顯示名稱
    assert len(panel.personas) == 3
    assert {p.display_name for p in panel.personas} == {"HR 主管", "技術主管", "用人主管"}


# ---------------------------------------------------------------------------
# 每輪
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_happy_path() -> None:
    r = await next_turn(
        mode="panel", answer="我用 SQL 做了報表", follow_up_idx=0,
        asked_questions=["自我介紹"], question="請自我介紹", fallback=[],
        llm=llm_returning(TURN),
    )
    assert r.speaker == "技術主管"
    assert r.next_question == TURN["nextQuestion"]
    assert r.topic == "資料分析"


@pytest.mark.asyncio
async def test_turn_degrades_to_fallback_pool() -> None:
    """降級用前端帶來的備援池,不要臨時編。"""
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=["備援一", "備援二"], llm=boom,
    )
    assert r.next_question == "備援一"
    assert r.should_advance is True
    assert any("生成失敗" in n for n in r.notices)


@pytest.mark.asyncio
async def test_empty_question_falls_back() -> None:
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=["備援一"], llm=llm_returning({"nextQuestion": ""}),
    )
    assert r.next_question == "備援一"


@pytest.mark.asyncio
async def test_session_cap_is_enforced_in_code() -> None:
    """「整場問到第幾題」是確定性事實,交給模型判斷會讓它有時多問一題。"""
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=MAX_TURNS_PER_SESSION,
        asked_questions=[], question="q", fallback=[],
        llm=llm_returning({**TURN, "shouldAdvance": False}),
    )
    assert r.should_advance is True, "整場達上限時強制結束,不問模型"
    assert any("上限" in n for n in r.notices)


@pytest.mark.asyncio
async def test_locale_normalized_in_question() -> None:
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[], question="q",
        fallback=[], llm=llm_returning({**TURN, "nextQuestion": "你怎麼做用戶界面的交互設計?"}),
    )
    assert "使用者介面" in r.next_question
    assert "用戶" not in r.next_question


# ---------------------------------------------------------------------------
# prompt 組裝
# ---------------------------------------------------------------------------


def test_asked_questions_reach_the_prompt() -> None:
    """「不要問已經問過的東西」沒有清單就無法執行。"""
    t = compose_turn_prompt("single", ["請自我介紹", "你怎麼處理衝突?"])
    assert "請自我介紹" in t and "你怎麼處理衝突?" in t


def test_multi_mode_prompt_lists_legal_speakers() -> None:
    t = compose_turn_prompt("panel", [])
    for n in ("HR 主管", "技術主管", "用人主管"):
        assert n in t


def test_group_prompt_says_peers_are_competitors() -> None:
    """AI 同儕是競爭者不是評分者——這跟主管面試的目標相反。"""
    t = compose_turn_prompt("group", [])
    assert "競爭者" in t
    assert "不要引導使用者講得更完整" in t


def test_context_block_marks_missing_jd() -> None:
    """JD 空的時候要寫明沒有,留空欄位會讓模型以為漏了什麼而去猜。"""
    assert "沒有提供" in context_block(InterviewContext())


def test_context_block_prefers_extracted_skills() -> None:
    ctx = InterviewContext(custom_jd="一大段散文")
    jd = JDInput(required_skills=["SQL", "報表自動化"], source="extracted")
    text = context_block(ctx, jd)
    assert "SQL" in text
    assert "一大段散文" not in text, "有結構化技能時不需要再塞散文"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "抱歉", "[]", "null"])
def test_bad_output_returns_empty_dict(bad: str) -> None:
    assert parse_obj(bad) == {}


def test_strips_fence() -> None:
    assert parse_obj('```json\n{"a": 1}\n```') == {"a": 1}


# ---------------------------------------------------------------------------
# 派發平衡與結束時機:兩個確定性事實
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaker_counts_reach_the_prompt() -> None:
    """派發規則要求「某位沒開口就給他」,模型看不到發言紀錄就永遠不會生效。

    實測:未提供時,panel 五輪裡 HR 主管 4 次、用人主管 0 次。
    """
    seen = {}

    def spy(system: str, user: str) -> str:
        seen["user"] = user
        return json.dumps(TURN, ensure_ascii=False)

    await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        spoken_by=["HR 主管", "HR 主管", "技術主管"], question="q", fallback=[], llm=spy,
    )
    assert "HR 主管:2 次" in seen["user"]
    assert "用人主管:0 次" in seen["user"]


@pytest.mark.asyncio
async def test_single_mode_has_no_speaker_counts() -> None:
    """一對一只有一位面試官,沒有派發問題,不需要這段。"""
    seen = {}

    def spy(system: str, user: str) -> str:
        seen["user"] = user
        return json.dumps({"nextQuestion": "x"}, ensure_ascii=False)

    await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[], llm=spy,
    )
    assert "發言次數" not in seen["user"]


@pytest.mark.asyncio
async def test_model_cannot_end_session_early() -> None:
    """「整場該不該結束」由輪次決定,不是判斷題。

    實測:模型因為使用者一句「我沒想過」就把 shouldAdvance 設成 true,
    提前結束了整場面試。
    """
    r = await next_turn(
        mode="single", answer="我沒想過", follow_up_idx=1, asked_questions=[],
        question="q", fallback=[], llm=llm_returning({**TURN, "shouldAdvance": True}),
    )
    assert r.should_advance is False, "還沒到上限,模型不能自己結束"
    assert any("已忽略" in n for n in r.notices)


@pytest.mark.asyncio
async def test_cap_ends_session_even_if_model_says_no() -> None:
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=MAX_TURNS_PER_SESSION,
        asked_questions=[], question="q", fallback=[],
        llm=llm_returning({**TURN, "shouldAdvance": False}),
    )
    assert r.should_advance is True


@pytest.mark.asyncio
async def test_silent_persona_is_forced_by_code() -> None:
    """模型可以依內容判斷派給誰,但不能可靠地執行「誰還沒講過」這種計數。

    實測:發言次數已傳、規則已寫,group 五輪裡 AI-強勢 仍然掛零,
    而第 5 輪的回答正是它的路由觸發條件。
    """
    r = await next_turn(
        mode="group", answer="先做一個版本再修比較快", follow_up_idx=3,
        asked_questions=[], spoken_by=["主考官", "AI-邏輯", "AI-親切"],
        question="q", fallback=[],
        llm=llm_returning({**TURN, "speaker": "主考官"}),
    )
    assert r.speaker == "AI-強勢"
    assert any("未發言過" in n for n in r.notices)


@pytest.mark.asyncio
async def test_no_force_before_threshold() -> None:
    """前幾輪讓派發跟著內容走,不要一開始就強制輪流。"""
    r = await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        spoken_by=[], question="q", fallback=[], llm=llm_returning(TURN),
    )
    assert r.speaker == "技術主管", "模型的判斷應該被保留"


@pytest.mark.asyncio
async def test_failure_notice_keeps_raw_output() -> None:
    """沒有原始輸出就查不出是格式問題、拒答、還是輸出了別的欄位。"""
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=["備援"], llm=llm_returning("我沒辦法回答這個"),
    )
    assert "我沒辦法回答這個" in r.notices[0]


# ---------------------------------------------------------------------------
# 純文字搶救:不要因為格式問題丟掉好答案
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_text_question_is_salvaged() -> None:
    """實測:模型回了一個好追問,卻因為沒包成 JSON 而被換成罐頭問題。

    重試會增加面試中的延遲,而問題本身是好的,直接用就行。
    """
    good = "這個改進的過程是怎麼進行的?你是怎麼發現需要重寫查詢的?"
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=["罐頭問題"], llm=llm_returning(good),
    )
    assert r.next_question == good
    assert any("沒有包成 JSON" in n for n in r.notices)


@pytest.mark.parametrize("junk", [
    "抱歉,我無法回答這個問題",
    "作為一個語言模型,我沒辦法",
    "好的。",
    "這是一段沒有問句的敘述文字內容",
])
def test_junk_is_not_salvaged(junk: str) -> None:
    """看起來像道歉或說明的一律不救——顯示出來比罐頭問題糟。"""
    from app.pipeline.interview_live import salvage_question

    assert salvage_question(junk) is None


@pytest.mark.asyncio
async def test_honest_admission_turn_is_not_force_assigned() -> None:
    """對方說不會的那一輪該由主考官接住,不是拿來補發言次數。

    實測:AI-強勢 被強制指派到這種情境,講出「沒關係,換個問題」——
    那是主考官的工作,而且完全不符合它的人格。
    """
    from app.pipeline.interview_live import required_speaker

    assert required_speaker("group", ["主考官", "AI-邏輯"], 3, "這個我沒想過") is None
    assert required_speaker("group", ["主考官", "AI-邏輯"], 3, "我用 SQL 做報表") == "AI-強勢"
