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


def llm_returning(payload, dispatch: str | None = None) -> object:
    """dispatch 給定時,派發那一步回那個名字;其餘呼叫回 payload。

    兩步流程之後,群面與 panel 每輪會先呼叫一次派發(只回名字)。
    """
    def call(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            if dispatch is not None:
                return dispatch
            # 預設沿用 payload 裡的 speaker,讓舊測試的語意不變
            return payload.get("speaker", "") if isinstance(payload, dict) else ""
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
        llm=llm_returning(TURN, dispatch="技術主管"),
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
async def test_speaker_counts_reach_the_dispatch_prompt() -> None:
    """派發規則要求「某位沒開口就給他」,模型看不到發言紀錄就永遠不會生效。

    兩步流程之後,發言次數在派發那一步的 user message 裡。
    """
    seen = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            seen["dispatch"] = user
            return "技術主管"
        return json.dumps(TURN, ensure_ascii=False)

    await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        spoken_by=["HR 主管", "HR 主管", "技術主管"], question="q", fallback=[], llm=spy,
    )
    assert "HR 主管:2 次" in seen["dispatch"]
    assert "用人主管:0 次" in seen["dispatch"]


@pytest.mark.asyncio
async def test_dispatch_decides_speaker_not_generation() -> None:
    """說話者由派發那一步決定。生成端的 prompt 已寫死 speaker,
    回別的就是它沒照做,不採用。"""
    r = await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[],
        llm=llm_returning({**TURN, "speaker": "HR 主管"}, dispatch="用人主管"),
    )
    assert r.speaker == "用人主管"


@pytest.mark.asyncio
async def test_dispatch_failure_falls_back_to_least_spoken() -> None:
    """派發失敗時退回發言最少的一位,不要讓生成端自己猜。"""
    def call(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            raise RuntimeError("timeout")
        return json.dumps(TURN, ensure_ascii=False)

    r = await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        spoken_by=["HR 主管", "HR 主管", "技術主管"], question="q", fallback=[], llm=call,
    )
    assert r.speaker == "用人主管"
    assert any("派發失敗" in n for n in r.notices)


@pytest.mark.asyncio
async def test_focused_prompt_expands_only_one_persona() -> None:
    """七位 persona 全部展開 prompt 超過 4200 字,人格會塌陷。

    每一輪只有一位會說話,其餘六份完整設定是雜訊。
    """
    from app.prompts.interview_live import compose_turn_prompt

    full = compose_turn_prompt("group", [], group_interviewers=3, group_size=5)
    focused = compose_turn_prompt(
        "group", [], group_interviewers=3, group_size=5, focus_speaker="AI-邏輯"
    )
    assert len(focused) < len(full)
    assert "質疑數據與推論的跳躍" in focused
    assert "你是**同場競爭的應徵者**。你的風格是搶節奏" not in focused, \
        "其他人只列名字,不展開人格"
    assert "AI-強勢" in focused, "但名字要留著,使用者看得到他們在場"


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
        llm=llm_returning({**TURN, "speaker": "主考官"}, dispatch="主考官"),
    )
    assert r.speaker == "AI-強勢"
    assert any("未發言過" in n for n in r.notices)


@pytest.mark.asyncio
async def test_no_force_before_threshold() -> None:
    """前幾輪讓派發跟著內容走,不要一開始就強制輪流。"""
    r = await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        spoken_by=[], question="q", fallback=[],
        llm=llm_returning(TURN, dispatch="技術主管"),
    )
    assert r.speaker == "技術主管", "派發的判斷應該被保留,不該被強制指派蓋掉"


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


@pytest.mark.asyncio
async def test_missing_topic_falls_back_to_previous() -> None:
    """topic 不可以留空。

    前端拿它累積 asked_topics,空的那一筆會讓序列斷掉,
    而斷掉的地方就是之後重複提問的破口。
    """
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", prev_topic="資料分析", fallback=[],
        llm=llm_returning({"nextQuestion": "那你怎麼判斷的?"}),
    )
    assert r.topic == "資料分析"
    assert any("沿用" in n for n in r.notices)


@pytest.mark.asyncio
async def test_salvaged_question_still_has_topic() -> None:
    """搶救路徑救回問題但沒有後設欄位,topic 一樣不能空。"""
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", prev_topic="跨部門協調", fallback=[],
        llm=llm_returning("那你當時是怎麼協調的呢?"),
    )
    assert r.topic == "跨部門協調"


@pytest.mark.asyncio
async def test_topic_never_empty_even_without_prev() -> None:
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[], llm=llm_returning({"nextQuestion": "為什麼?"}),
    )
    assert r.topic


# ---------------------------------------------------------------------------
# 重複主問題的機械攔截
# ---------------------------------------------------------------------------


def test_near_duplicate_detection_calibrated() -> None:
    """實測校準:重複的落在 0.69 以上,不重複的是 0,門檻 0.6 兩邊都有餘裕。"""
    from app.pipeline.interview_live import is_near_duplicate

    opening = ["可以分享一次你在資料分析方面的經驗嗎?特別是如何處理數據或報表的部分。"]
    assert is_near_duplicate("那你能分享一下你過去如何處理資料清理的經驗嗎?", opening)
    assert is_near_duplicate("你能分享一下你在資料分析方面的具體經驗嗎?", opening)
    assert is_near_duplicate("你怎麼跟工程團隊協調排程?", opening) is None
    assert is_near_duplicate("你當時是怎麼發現需要重寫查詢的呢?", opening) is None


@pytest.mark.asyncio
async def test_duplicate_main_question_swapped_for_fallback() -> None:
    """已問問題原文已經傳給模型、規則也寫了,實測仍然會問近似題。

    重試會增加面試中的延遲,而備援池是開場時針對這份 JD 生成的,直接用就行。
    """
    opening = "可以分享一次你在資料分析方面的經驗嗎?特別是如何處理數據或報表的部分。"
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[opening],
        question=opening, fallback=["你怎麼跟工程團隊協調排程?"],
        llm=llm_returning({
            "nextQuestion": "你能分享一下你在資料分析方面的具體經驗嗎?",
            "isFollowUp": False, "topic": "資料分析",
        }),
    )
    assert r.next_question == "你怎麼跟工程團隊協調排程?"
    assert any("改用備援題" in n for n in r.notices)


@pytest.mark.asyncio
async def test_follow_up_may_stay_on_topic() -> None:
    """追問不受重複限制——追問本來就該留在同一個話題上。"""
    opening = "可以分享一次你在資料分析方面的經驗嗎?特別是如何處理數據或報表的部分。"
    same = "你能分享一下你在資料分析方面的具體經驗嗎?"
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[opening],
        question=opening, fallback=["別的題"],
        llm=llm_returning({"nextQuestion": same, "isFollowUp": True, "topic": "資料分析"}),
    )
    assert r.next_question == same


def test_group_prompt_requires_peers_to_assert() -> None:
    """面試官問問題,競爭者提主張。只問不說的那一句換成面試官講也成立。

    聚焦模式下這條判準要留著——那是人格的一部分,不是派發規則。
    """
    from app.prompts.interview_live import compose_turn_prompt

    focused = compose_turn_prompt("group", [], focus_speaker="AI-邏輯")
    assert "先表態再發問" in focused
    assert "換成面試官講也完全成立" in focused


@pytest.mark.asyncio
async def test_moderator_is_not_used_to_fill_quota() -> None:
    """主持人的角色是條件性的,不是輪流制。

    實測:把他算進補位名額,群面五輪裡第 3 輪給了主考官、第 5 輪給 AI-強勢,
    AI-親切 因為排在宣告順序最後而永遠輪不到。
    """
    from app.pipeline.interview_live import required_speaker

    assert required_speaker("group", ["AI-邏輯", "AI-邏輯"], 2) == "AI-強勢"
    assert required_speaker("group", ["AI-邏輯", "主考官", "AI-強勢"], 4) == "AI-親切"


@pytest.mark.asyncio
async def test_salvaged_question_defaults_to_follow_up() -> None:
    """搶救回來的問題是針對剛剛那段回答寫的,本質上就是追問。"""
    r = await next_turn(
        mode="single", answer="我用 SQL 重寫查詢", follow_up_idx=0,
        asked_questions=[], question="q", prev_topic="資料分析", fallback=[],
        llm=llm_returning("你當時是怎麼發現查詢效率有問題的呢?"),
    )
    assert r.is_follow_up is True


# ---------------------------------------------------------------------------
# 群面的兩種配置與小組人數
# ---------------------------------------------------------------------------


def test_group_solo_config_has_four_peers() -> None:
    """前端 baseRoster:主考官、你、AI-強勢、AI-邏輯、AI-親切、AI-沉默。"""
    from app.prompts.interview_personas import speaker_names

    assert speaker_names("group", 1, 5) == ("主考官", "AI-強勢", "AI-邏輯", "AI-親切", "AI-沉默")


def test_group_panel_config_replaces_moderator() -> None:
    """3 位面試官時主考官不出現,改由三位主管主持。

    前端:panel 模式時 speaker=="主考官" 會被換成 "用人主管"。
    """
    from app.prompts.interview_personas import speaker_names

    names = speaker_names("group", 3, 5)
    assert "主考官" not in names
    assert names[:3] == ("用人主管", "技術主管", "HR 主管")
    assert "AI-沉默" in names


@pytest.mark.parametrize("size,peers", [(3, 2), (4, 3), (5, 4)])
def test_group_size_controls_peer_count(size: int, peers: int) -> None:
    """小組人數含使用者本人,扣掉他就是 AI 應徵者的數量。"""
    from app.prompts.interview_personas import speaker_names

    names = speaker_names("group", 1, size)
    assert len([n for n in names if n.startswith("AI-")]) == peers


def test_peer_order_matches_frontend_roster() -> None:
    """順序錯的話會出場錯的人——選 3 人時前端顯示強勢與邏輯,
    後端卻回親切,畫面對不上。"""
    from app.prompts.interview_personas import speaker_names

    assert speaker_names("group", 1, 3) == ("主考官", "AI-強勢", "AI-邏輯")


def test_quiet_peer_has_a_distinct_trigger() -> None:
    """AI-沉默 的介入條件跟其他三位不同——不是「回答提到什麼」,
    是「討論的狀態」。沒有明確條件它會變成永遠不出現的裝飾。"""
    from app.prompts.interview_personas import GROUP_PEER_QUIET

    assert GROUP_PEER_QUIET.routes_when
    assert any("討論" in r for r in GROUP_PEER_QUIET.routes_when)


def test_difficulty_reaches_the_prompt() -> None:
    """難度傳了但沒有規則等於沒生效。"""
    from app.prompts.interview_live import compose_turn_prompt

    hard = compose_turn_prompt("single", [], difficulty="困難")
    assert "找到第一個就追" in hard, "難度改成換判準,不再是一段獨立的註記"


def test_group_role_only_in_group_mode() -> None:
    from app.prompts.interview_live import compose_turn_prompt

    assert "較資深應徵者" in compose_turn_prompt("group", [], group_role="較資深應徵者")
    assert "較資深應徵者" not in compose_turn_prompt("single", [], group_role="較資深應徵者")


@pytest.mark.asyncio
async def test_turn_uses_context_config() -> None:
    """設定要能逐輪生效,否則第二輪之後全部退回預設值。"""
    from app.schemas.interview import InterviewContext

    seen = {}

    def spy(system: str, user: str) -> str:
        seen["system"] = system
        return json.dumps({**TURN, "speaker": "AI-沉默"}, ensure_ascii=False)

    ctx = InterviewContext(group_interviewers=3, group_size=5, difficulty="困難")
    r = await next_turn(
        mode="group", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[], llm=spy, context=ctx,
    )
    assert "用人主管" in seen["system"]
    assert "找到第一個就追" in seen["system"]
    assert r.speaker == "AI-沉默"


# ---------------------------------------------------------------------------
# 輪次上限依模式而異
# ---------------------------------------------------------------------------


def test_group_has_no_turn_cap() -> None:
    """群面沒有上限——InterviewLiveGroupScreen 的 followUpIdx 只遞增,
    沒有任何上限檢查,也沒有進入反問環節的邏輯。

    早期版本把一對一的 4 題上限套過來,後果是選 5 人小組時只跑 5 輪,
    AI-親切 與 AI-沉默 永遠不會出場。
    """
    from app.prompts.probe_rules import turn_cap_for

    assert turn_cap_for("group") is None
    assert turn_cap_for("single") == 4
    assert turn_cap_for("panel") == 4


@pytest.mark.asyncio
async def test_group_does_not_end_at_four_turns() -> None:
    r = await next_turn(
        mode="group", answer="x", follow_up_idx=9, asked_questions=[],
        question="q", fallback=[], llm=llm_returning({**TURN, "speaker": "AI-邏輯"}),
    )
    assert r.should_advance is False, "群面不該因為輪次而結束"


@pytest.mark.asyncio
async def test_group_ignores_model_wanting_to_end() -> None:
    """群面由使用者自己決定何時結束,模型不能替他決定。"""
    r = await next_turn(
        mode="group", answer="x", follow_up_idx=9, asked_questions=[],
        question="q", fallback=[],
        llm=llm_returning({**TURN, "speaker": "AI-邏輯", "shouldAdvance": True}),
    )
    assert r.should_advance is False
    assert any("使用者自己決定" in n for n in r.notices)


@pytest.mark.asyncio
async def test_single_still_ends_at_cap() -> None:
    r = await next_turn(
        mode="single", answer="x", follow_up_idx=4, asked_questions=[],
        question="q", fallback=[], llm=llm_returning(TURN),
    )
    assert r.should_advance is True


# ---------------------------------------------------------------------------
# 輸入順序與 persona 定位
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_comes_before_previous_question() -> None:
    """回答排在前面,而且明說要回應它。

    實測:上一題排前面時模型會錨在前一題的主題上——
    使用者講協調排程,問的是 SQL;說「我沒想過」,問的還是優化查詢。
    連續三輪不跟著回答走,gpt-4o 也一樣,所以那是輸入順序造成的錨定。
    """
    seen = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return ""
        seen["user"] = user
        return json.dumps({"nextQuestion": "x", "topic": "t"}, ensure_ascii=False)

    await next_turn(
        mode="single", answer="我協調了兩邊的排程", follow_up_idx=0,
        asked_questions=[], question="你怎麼優化 SQL 查詢?", fallback=[], llm=spy,
    )
    u = seen["user"]
    assert u.index("我協調了兩邊的排程") < u.index("你怎麼優化 SQL 查詢?")
    assert "你要回應的是上面這段話" in u


def test_friendly_peer_is_an_integrator_not_a_helper() -> None:
    """「靠協作能力被看見」太抽象,模型只抓到「協作」而丟掉「被看見」。

    三種 prompt 方法都失敗、gpt-4o 也救不了,所以改的是定位本身:
    先指出分歧,再提出整合方案——綜合本身就是一種主張。
    """
    from app.prompts.interview_personas import GROUP_PERSONAS

    friendly = next(p for p in GROUP_PERSONAS if p.id == "peer_friendly")
    assert "整合" in friendly.blurb or "分歧" in friendly.blurb
    assert "分歧" in friendly.stance
    assert "整合本身就是一種主張" in friendly.stance
    assert any("分歧" in r or "兩派" in r for r in friendly.routes_when)


def test_stance_examples_use_unrelated_scenario() -> None:
    """示範的情境要跟面試無關,照抄就會明顯不合。

    實測:用「先做再修」「母數不到三十」當示範時,gpt-4o 與 gpt-4o-mini
    都一字不差照抄——那些句子剛好接得上使用者講的話,模型覺得直接用就好。
    那不是人格生效,是背答案。
    """
    from app.prompts.interview_live import GROUP_PEER_STANCE

    assert "一字不差照抄" in GROUP_PEER_STANCE
    assert "就是抄錯了" in GROUP_PEER_STANCE
    # 示範區塊必須用不相干的情境。舊句子只能出現在「為什麼」的說明裡,
    # 不能出現在示範區塊——那是會被照抄的位置。
    demo = GROUP_PEER_STANCE.split("【為什麼用不相干的情境當示範】")[0]
    assert "換供應商" in demo
    assert "先切一個最小可驗證的版本" not in demo
    assert "母數不到三十" not in demo


# ---------------------------------------------------------------------------
# 難度換判準,不是加註記
# ---------------------------------------------------------------------------


def test_difficulty_swaps_the_triggers_not_appends() -> None:
    """實測:難度寫成獨立段落放在 prompt 中段時,困難與新手的問句幾乎逐字相同。

    判準是模型真正會讀的部分(它要照著決定問什麼),
    所以難度要改的是判準本身。
    """
    from app.prompts.interview_live import compose_turn_prompt

    hard = compose_turn_prompt("single", [], difficulty="困難")
    easy = compose_turn_prompt("single", [], difficulty="新手")

    assert "找到第一個就追" in hard
    assert "找到第一個就追" not in easy
    assert "不要**追問數字" in easy or "**不要**追問數字" in easy
    assert hard.count("判斷追問什麼時") == 1, "兩份判準同時出現會互相打架"
    assert easy.count("判斷追問什麼時") == 1


def test_hard_mode_allows_repeated_probing() -> None:
    from app.prompts.interview_live import compose_turn_prompt

    assert "同一件事可以連續追問" in compose_turn_prompt("single", [], difficulty="困難")


def test_single_output_has_only_two_fields() -> None:
    """一對一實測 60–80% 的輪次沒包成 JSON,群面同樣長度卻幾乎不發生。

    差別在欄位數:speaker 恆為空、reaction 可有可無、
    isFollowUp 與 shouldAdvance 本來就由程式決定。
    """
    from app.prompts.interview_live import compose_turn_prompt

    t = compose_turn_prompt("single", [], difficulty="中等")
    # 只看 JSON 範例那一行,後面的說明會提到被拿掉的欄位名
    example = t.split("輸出格式")[1].split("\n")[2]
    assert "nextQuestion" in example and "topic" in example
    assert "shouldAdvance" not in example
    assert "speaker" not in example


@pytest.mark.asyncio
async def test_single_turns_default_to_follow_up() -> None:
    """一對一沒有 isFollowUp 欄位時預設為追問。

    早期版本用「topic 跟上一輪相同就是追問」,那是錯的:
    實測五輪全部在追問,但模型每輪都給了誠實的新 topic
    (實習內容 → 查詢重寫的具體做法 → 協調過程),於是判成 0/5。

    追問的定義是「接著剛剛那段回答問」,跟 topic 換不換沒有必然關係。
    """
    for topic in ("資料分析", "團隊衝突", ""):
        r = await next_turn(
            mode="single", answer="x", follow_up_idx=0, asked_questions=[],
            question="q", prev_topic="資料分析", fallback=[],
            llm=llm_returning({"nextQuestion": "那個數字怎麼算的?", "topic": topic}),
        )
        assert r.is_follow_up is True


@pytest.mark.asyncio
async def test_multi_mode_still_uses_the_field() -> None:
    """群面與 panel 的輸出格式仍有 isFollowUp,有回就用它的。"""
    r = await next_turn(
        mode="panel", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[],
        llm=llm_returning({**TURN, "isFollowUp": False}, dispatch="技術主管"),
    )
    assert r.is_follow_up is False
