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
    assert len(r.fallback_probes) == 2
    assert r.session_id.startswith("itv_")


@pytest.mark.asyncio
async def test_the_opening_always_asks_for_a_self_introduction() -> None:
    """開場固定是自我介紹,不採用模型回的 openingQuestion。

    三段式的第一段。原本 `OPENING_RULES` 明文寫「不是通用的『請自我介紹』」,
    所以一場四題全部是職缺題,使用者連開口暖身的機會都沒有。

    模型這裡回的是「先講一個你用資料做出判斷的經驗。」,不可以被採用。
    """
    r = await start_interview(mode="single", context=CTX, llm=llm_returning(OPENING))

    assert "自我介紹" in r.opening_question
    assert r.opening_question != OPENING["openingQuestion"]
    assert r.opening_topic == "自我介紹", "領域標籤跟著固定"
    # 職稱有填就帶進去,那是這一句唯一會變的地方
    assert CTX.custom_role in r.opening_question


@pytest.mark.asyncio
async def test_the_opening_still_reads_the_fallback_probes_from_the_model() -> None:
    """開場那次呼叫留著是因為備援追問仍然要生成。

    沒有這一條的話,「開場不問模型」可能被實作成整個呼叫都拿掉,
    而備援池會靜靜退回寫死的三句。
    """
    r = await start_interview(mode="single", context=CTX, llm=llm_returning(OPENING))
    assert r.fallback_probes == OPENING["fallbackProbes"]
    assert not any("備援追問" in n for n in r.notices)


@pytest.mark.asyncio
async def test_start_degrades_on_failure() -> None:
    """面試按下開始卻噴 500,使用者連場都進不去。"""
    r = await start_interview(mode="single", context=CTX, llm=boom)
    assert r.opening_question
    assert r.fallback_probes
    assert any("已使用預設" in n for n in r.notices)


@pytest.mark.asyncio
async def test_interrupts_are_off_and_nothing_puts_them_back() -> None:
    """AI 應徵者只在輪到自己時發言,不插話。產品決定。

    這一條擋的是**悄悄裝回去**:退回預設那段原本是
    「group 而且沒拿到台詞 → 用寫死的三句」,關掉之後它會照樣觸發,
    於是 interruptLines 裡又有三句,而 notices 看起來一切正常。

    模型這裡回了三句,也不可以被採用。
    """
    def call(system: str, user: str) -> str:
        if "搶節奏" in system:
            return json.dumps(["我先說。", "我插一個快的。", "我補一句。"], ensure_ascii=False)
        return json.dumps(OPENING, ensure_ascii=False)

    r = await start_interview(mode="group", context=CTX, llm=call)
    assert r.interrupt_lines == []
    assert r.interrupt_cap == 0
    assert not any("搶話台詞" in n for n in r.notices), "沒有生成不是降級,不要報"


def test_the_assertive_peer_no_longer_says_it_interrupts() -> None:
    """只關常數不夠。

    人格描述裡的「你會打斷,而且不覺得那是失禮」會讓模型在自己的回合
    寫出「我打斷一下」開頭的台詞——那是人格帶出來的,不是搶話機制帶出來的。
    """
    from app.prompts.interview_personas import INTERRUPT_PERSONA_ID, personas_for

    from app.prompts.interview_live import GROUP_PEER_STANCE

    p = next(x for x in personas_for("group") if x.id == INTERRUPT_PERSONA_ID)

    # 比對原本那句指示,不是「打斷」兩個字——禁令本身就含那兩個字,
    # 第一版就是這樣寫錯的。
    assert "你會打斷" not in p.stance
    assert "不覺得那是失禮" not in p.stance
    assert all(not line.startswith("我打斷") for line in p.sample_lines)
    # 其餘人格要留著——關掉的是插話,不是這個人的風格
    assert "搶節奏" in p.stance

    # 時機規則是**全體**的,放在共用區塊,不掛在單一 persona 上。
    # 掛在一位身上的話另外兩位沒有這條規則,而那很難從輸出看出來。
    assert "只在輪到你的時候" in GROUP_PEER_STANCE
    assert "不插話" in GROUP_PEER_STANCE


def test_each_peer_rebuts_differently() -> None:
    """反駁是允許的,但三個人不該用同一種方式。

    自己回合裡反駁本來就開著(`GROUP_PEER_STANCE` 寫了「可以反駁」),
    關掉的只有插話與輪次過後回頭補。缺的是**依個性區分**——
    三個人都寫「可以反駁」的話,輸出就會三個人都一樣。
    """
    from app.prompts.interview_live import GROUP_PEER_STANCE
    from app.prompts.interview_personas import personas_for

    peers = [p for p in personas_for("group") if p.role == "peer"]
    assert len(peers) >= 3

    described = {p.display_name: p.stance.split("反駁:", 1)[1]
                 for p in peers if "反駁:" in p.stance}
    assert len(described) == len(peers), "每一位都要寫明自己怎麼反駁"
    assert len(set(described.values())) == len(described), "三份不可以一樣"

    assert "最直接" in described["AI-強勢"]
    assert "不反駁立場" in described["AI-邏輯"]
    assert "最少正面反駁" in described["AI-親切"]

    # 共用區塊要說這件事因人而異,否則三份人格描述會被那句統一的
    # 「可以反駁」蓋過去
    assert "由你的人格決定" in GROUP_PEER_STANCE


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


# ---------------------------------------------------------------------------
# ended_by:被切斷跟自己講完是兩回事
# ---------------------------------------------------------------------------


def test_truncated_rule_only_when_timeout() -> None:
    """被引擎切斷時先把話接完,不要換新題。"""
    from app.prompts.probe_rules import compose_probe_rules

    cut = compose_probe_rules([], truncated=True)
    normal = compose_probe_rules([], truncated=False)
    assert "還沒講完的部分" in cut
    assert "還沒講完的部分" not in normal


def test_truncated_rule_forbids_blaming() -> None:
    """不要因為回答不完整而追問「為什麼沒有交代結果」——
    那是責備一個被打斷的人。"""
    from app.prompts.probe_rules import TRUNCATED_ANSWER

    assert "責備一個被打斷的人" in TRUNCATED_ANSWER
    assert "不會為麥克風道歉" in TRUNCATED_ANSWER


@pytest.mark.asyncio
async def test_ended_by_reaches_the_prompt() -> None:
    """欄位加了就要有人讀。沒人讀的欄位不該存在。"""
    seen = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return ""
        seen["system"] = system
        return json.dumps({"nextQuestion": "接著說", "topic": "t"}, ensure_ascii=False)

    await next_turn(
        mode="single", answer="我用 SQL 重寫查詢 然後", follow_up_idx=0,
        asked_questions=[], question="q", fallback=[], llm=spy, ended_by="timeout",
    )
    assert "還沒講完的部分" in seen["system"]


@pytest.mark.asyncio
async def test_normal_end_does_not_trigger_the_rule() -> None:
    seen = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return ""
        seen["system"] = system
        return json.dumps({"nextQuestion": "x", "topic": "t"}, ensure_ascii=False)

    await next_turn(
        mode="single", answer="x", follow_up_idx=0, asked_questions=[],
        question="q", fallback=[], llm=spy, ended_by="user",
    )
    assert "還沒講完的部分" not in seen["system"]


@pytest.mark.asyncio
async def test_force_speaker_pins_who_talks_and_skips_dispatch() -> None:
    """指定說話者時跳過派發,而且回的就是指定那一位。

    生產路徑不傳這個值——誰開口由派發決定。它存在是為了量測:
    要比較三位面試官對**同一段回答**會問什麼,就得繞過派發,
    否則每一位拿到的回答都不同,量到的是回答的差異不是角色的差異。

    第一版的 panel 量測就是栽在這裡:五場的派發順序一模一樣,
    於是「同角色」等於「同一段回答」,那份數字答不了原本的問題。
    """
    dispatched = []

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            dispatched.append(system)
            return "HR 主管"
        return json.dumps({**TURN, "speaker": "技術主管"}, ensure_ascii=False)

    r = await next_turn(
        mode="panel", answer="我用 SQL 重寫查詢。", follow_up_idx=0,
        asked_questions=[], question="請自我介紹", fallback=["備援"],
        llm=spy, context=CTX, force_speaker="用人主管",
    )

    assert r.speaker == "用人主管", "指定了誰就是誰,模型回別人也不算"
    assert not dispatched, "指定之後不該再問模型誰開口"


# ---------------------------------------------------------------------------
# HR 的守備範圍
# ---------------------------------------------------------------------------


def _hr() -> object:
    from app.prompts.interview_personas import PANEL_PERSONAS

    return next(p for p in PANEL_PERSONAS if p.id == "hr")


def test_hr_stance_carries_literal_question_shapes() -> None:
    """HR 的 stance 裡要有字面問句,不是描述題材也不是規則。

    三次量測的依據(`app/prompts/interview_personas.py` 上方那段註解):
    v3 給四個題材、v4 給一條主詞規則,兩次都沒移動角色效果,而且各被繞過
    一次。唯一問到人的那一題是照抄語氣示範來的,字元重疊 100%。

    一直穩的另外兩位,stance 裡本來就是字面問句:「那個結論怎麼來的」、
    「代價是什麼」、「為什麼選這個而不是那個」。這條把 HR 釘在同一個形狀。
    """
    import re

    stance = _hr().stance
    quoted = re.findall(r"「([^」]+)」", stance)
    assert len(quoted) >= 2, f"stance 裡只有 {len(quoted)} 個字面問句,至少要兩個"
    for shape in ("最難的是哪一段", "怎麼知道自己那次做得好"):
        assert any(shape in q for q in quoted), (
            f"「{shape}」要包在引號裡當字面問句,"
            f"描述成題材的版本量過沒用。目前引號裡是:{quoted}"
        )


def test_hr_stance_stays_in_the_same_size_class_as_the_two_that_work() -> None:
    """規則越加越多就是 v3 與 v4 的失敗路徑,這條擋住它。

    `_persona_block` 的註解記過 prompt 一長人格就塌。實際長度:
    技術主管 65 字、用人主管 47 字,兩位都穩;v4 的 HR 膨脹到三百多字,
    裡面有過濾規則、四個題材、一句禁令,一個都沒生效。

    HR 比另外兩位長是可以的——它多一個接住答不出來的人的職責,
    那是功能不是人格。但不該長成另一個數量級。
    """
    from app.prompts.interview_personas import PANEL_PERSONAS

    hr = _hr()
    others = [len(p.stance) for p in PANEL_PERSONAS if p.id != "hr"]
    assert len(hr.stance) <= max(others) * 2.5, (
        f"HR 的 stance {len(hr.stance)} 字,另外兩位 {others}。"
        "又在往裡面加規則了——先看註解裡那三次量測。"
    )


def test_hr_names_what_is_not_its_question() -> None:
    """界線要寫出來,但是一句話,不是一條規則。

    v4 的主詞規則被繞過了(「你當時是怎麼決定優先做那三個欄位的呢」——
    「你」是主詞,規則形式上過關)。所以這裡不再試圖讓模型自己檢查,
    只留一句話說那兩種問題是誰的,剩下交給上面那兩個字面問句去帶。
    """
    stance = _hr().stance
    assert "技術主管" in stance and "用人主管" in stance, "沒說那兩種問題是誰的"
    assert "主詞" not in stance, (
        "主詞規則量過會被繞過(十二題裡六題),不要再放回來"
    )


def test_hr_still_catches_the_candidate_who_cannot_answer() -> None:
    """接住答不出來的人是 HR 原本就有的功能,不能被改角度弄掉。

    這條跟評分那邊的 ADMITS_UNKNOWN 是一對:誠實說不會會拿最低錨點
    而不是不評分,所以場上也要有人把話接下去。
    """
    hr = _hr()
    assert "答不出來" in hr.stance
    assert "不是扣分項" in hr.stance
    assert any("答不出來" in r or "不確定" in r for r in hr.routes_when)


def test_hr_gets_routed_the_answers_nobody_else_can_use() -> None:
    """派發只讀 blurb 與 routes_when,不讀 stance。

    所以改 stance 而不動 routes_when 的話,生產路徑裡 HR 還是只會拿到
    講衝突的回答,新的切法永遠不會被用到——量測是用 force_speaker
    繞過派發的,它看不到這個漏洞。

    「沒有方法也沒有取捨可問」那一段就是泛泛而談,而泛泛而談正是技術與
    用人兩位一起漂掉的那一段。
    """
    routes = _hr().routes_when
    assert any("沒有方法" in r and "取捨" in r for r in routes), (
        "泛泛而談沒有歸屬,派發只能硬塞給技術或用人"
    )
    assert len(routes) > 2, "切法改了就要拓寬守備範圍,否則派發不會送過來"


def test_hr_blurb_stays_short_enough_for_the_frontend_roster() -> None:
    """blurb 會進開場回應的 roster,前端一行一個人顯示。

    `app/prompts/peer_profile.py` 量過既有 blurb 最長 14 個字。
    切法與過濾規則寫進 blurb 會爆掉那一行,所以細節留在 stance——
    stance 只進 prompt,不進回應。
    """
    hr = _hr()
    assert len(hr.blurb) <= 14, f"blurb {len(hr.blurb)} 字,roster 那一行放不下"
    assert "挫折" in hr.blurb, "派發讀 blurb,兩個切法要在裡面看得到"


@pytest.mark.asyncio
async def test_hr_angles_reach_the_prompt_the_model_actually_sees() -> None:
    """上面幾條驗的是常數,這一條驗那些字真的組進了 HR 開口那一輪的 prompt。

    `_persona_block` 只展開 focus 那一位,其餘只留名字與定位。
    改了常數卻沒進 prompt 的話,前面六條全綠而行為不變。
    """
    seen: dict[str, str] = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return "HR 主管"
        seen["system"] = system
        return json.dumps({**TURN, "speaker": "HR 主管"}, ensure_ascii=False)

    await next_turn(
        mode="panel", answer="我覺得做分析最難的是知道該問什麼問題。",
        follow_up_idx=0, asked_questions=[], question="請自我介紹",
        fallback=["備援"], llm=spy, context=CTX,
    )

    system = seen["system"]
    assert "最難的是哪一段" in system, "字面問句沒進 prompt"
    assert "怎麼知道自己那次做得好" in system, "字面問句沒進 prompt"
    assert "那一段裡最讓你卡住的是什麼?" in system, "語氣示範沒進 prompt"


def test_the_dispatcher_sees_hrs_new_territory() -> None:
    """派發 prompt 是另一次呼叫,組的是另一段文字。

    `compose_dispatch_prompt` 只放 display_name + blurb + routes_when。
    routes_when 改了但那個函式漏掉某一項的話,派發的判斷依據就跟
    persona 設定不一致,而那個錯是靜默的。
    """
    from app.prompts.interview_live import compose_dispatch_prompt

    prompt = compose_dispatch_prompt("panel")
    assert "挫折" in prompt, "blurb 沒進派發 prompt"
    for route in _hr().routes_when:
        assert route in prompt, f"守備範圍「{route}」沒進派發 prompt"


def test_route_conditions_stay_separable_when_they_contain_commas() -> None:
    """守備範圍含「、」,所以項與項之間不能也用「、」接。

    派發是獨立一次呼叫,判斷依據只有那一行。HR 拓寬到四項之後,
    用「、」接會變成讀不出邊界的一長串:

        提到團隊、合作、衝突、溝通、提到為什麼選這個、哪裡難…

    「溝通」與「提到為什麼選這個」之間那個「、」跟前面幾個長得一樣。
    """
    from app.prompts.interview_live import ROUTE_JOIN, compose_dispatch_prompt

    assert "、" not in ROUTE_JOIN, "分隔符跟守備範圍內部的標點撞了"

    hr = _hr()
    assert len(hr.routes_when) >= 4, "這條測的是多項的情況"
    line = next(
        ln for ln in compose_dispatch_prompt("panel").splitlines()
        if "什麼時候輪到他" in ln and hr.routes_when[0] in ln
    )
    assert line.count(ROUTE_JOIN) == len(hr.routes_when) - 1, (
        "項數與分隔數對不上,有一項被黏進別項裡了"
    )

def test_hrs_routing_hooks_do_not_land_in_another_interviewers_frame() -> None:
    """派發的鉤子不能用另外兩位的招牌問句。

    兩支 prompt 是兩次獨立呼叫:派發只讀 routes_when,生成只讀 stance。
    鉤子用了用人主管的招牌句時,HR 會被派到一段它接著不准照角度問的回答,
    然後只能借別人的框架——那正是 v3 十二題裡的七題。

    第二版實際發生的事:routes_when 寫「提到為什麼選這個」,而用人主管的
    stance 寫「為什麼選這個而不是那個才重要」。HR 在取捨那一段三次都問了
    優先順序怎麼評估。

    【這條只抓得到四個字以上的片段】
    招牌句是從另外兩位的 stance 比對出來的,所以「合作」「卡住」這種兩三
    個字的鉤子過不了這個篩子——它們本來也不會跟招牌句撞。
    """
    from app.prompts.interview_personas import PANEL_PERSONAS

    hr = _hr()
    others = " ".join(p.stance for p in PANEL_PERSONAS if p.id != "hr")
    checked = 0
    for route in hr.routes_when:
        for frag in route.replace("提到", "").split("、"):
            if len(frag) < 4:
                continue
            checked += 1
            assert frag not in others, (
                f"守備範圍的「{frag}」是另一位 stance 裡的招牌句:"
                "派發會送來這種回答,而 HR 照角度問不了"
            )
    assert checked, "沒有一個片段長到可以比對,這條測不到東西"


# ---------------------------------------------------------------------------
# 沒有履歷的前兩輪也要有換題的材料
# ---------------------------------------------------------------------------


def test_no_resume_early_turns_still_get_a_phase_block() -> None:
    """沒有履歷不等於那兩輪沒有階段。

    舊寫法是呼叫端判斷「履歷段而且沒履歷就整段不給」。不叫它編一筆經歷
    是對的,但那樣前兩輪完全沒有換題的材料。

    `data/panel_next_v1.json`(36 筆,決定全部指定成 NEXT):

                            通用的「有沒有遇到什麼挑戰」   引用回答
        沒有階段區塊(0-1)          12/18                1/18
        有 JD_PHASE  (2-3)           3/18                5/18

    同一句 NEXT 指令,有材料就照做,沒材料就退回通用問句。
    """
    from app.prompts.probe_rules import (
        JD_PHASE,
        NO_RESUME_PHASE,
        RESUME_PHASE,
        interview_phase,
    )

    assert interview_phase(0, has_resume=True) is RESUME_PHASE
    assert interview_phase(0, has_resume=False) is NO_RESUME_PHASE
    assert interview_phase(1, has_resume=False) is NO_RESUME_PHASE
    # 職缺那一段本來就不看履歷。
    assert interview_phase(2, has_resume=False) is JD_PHASE
    assert interview_phase(3, has_resume=True) is JD_PHASE


def test_the_no_resume_block_points_at_material_that_exists() -> None:
    """它指的東西必須真的在同一則訊息裡。

    `context_block` 一定會印【職位脈絡】,沒有 JD 時那裡仍然有職稱、產業、
    面試類型。所以指過去永遠有東西可讀——這跟指向一份不存在的履歷不同。
    """
    from app.prompts.probe_rules import NO_RESUME_PHASE

    assert "【職位脈絡】" in NO_RESUME_PHASE
    for pick_from_resume in ("從他履歷", "履歷上寫的", "履歷裡挑"):
        assert pick_from_resume not in NO_RESUME_PHASE, (
            f"還是叫它從履歷挑(「{pick_from_resume}」),而這一段的前提就是沒有履歷"
        )


def test_the_no_resume_block_still_forbids_inventing_experience() -> None:
    """補材料不能把原本那條保護弄掉。

    RESUME_PHASE 擋的是「問得像他做過某件事」。沒有履歷的時候那個風險更大,
    因為模型手上一筆經歷都沒有。
    """
    from app.prompts.probe_rules import NO_RESUME_PHASE

    assert "預設了沒有發生的事" in NO_RESUME_PHASE
    assert "只問他自己講出來的東西" in NO_RESUME_PHASE


@pytest.mark.asyncio
async def test_the_no_resume_block_reaches_the_prompt() -> None:
    """上面三條驗常數,這一條驗它真的組進了沒有履歷那一輪的 prompt。"""
    seen: dict[str, str] = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return "HR 主管"
        seen["system"] = system
        return json.dumps({**TURN, "speaker": "HR 主管"}, ensure_ascii=False)

    await next_turn(
        mode="panel", answer="我用 SQL 重寫了每週報表的查詢。", follow_up_idx=0,
        asked_questions=[], question="請自我介紹", fallback=["備援"],
        llm=spy, context=CTX,
    )

    system = seen["system"]
    assert "他沒有帶履歷" in system, "沒有履歷的階段區塊沒進 prompt"
    assert "從他履歷上寫的經歷" not in system, (
        "沒有履歷卻給了履歷段,模型會去編一筆經歷"
    )


def test_topic_labels_still_reach_the_prompt_when_no_questions_are_carried() -> None:
    """標籤只進 prompt,不進查重 —— 這一條驗「進 prompt」那半。

    前端沒帶 `askedQuestions` 的時候,列領域標籤比什麼都不列好:
    模型至少看得到問過哪些領域。查重那一側拿標籤去比會誤判
    (見 `next_turn` 的 asked_topics 那段),所以兩個用途分開傳。

    這一條存在是因為驗紅時發現:把 prompt 那一側的退路拿掉,
    整份測試照樣綠 —— 那表示保留退路的理由沒有人在守。
    """
    import asyncio

    seen: dict[str, str] = {}

    def spy(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            return "HR 主管"
        seen["system"] = system
        return json.dumps({**TURN, "speaker": "HR 主管"}, ensure_ascii=False)

    asyncio.run(
        next_turn(
            mode="panel", answer="我用 SQL 重寫查詢。", follow_up_idx=1,
            asked_questions=[], asked_topics=["團隊合作", "決策評估"],
            question="q", fallback=["備援"], llm=spy, context=CTX,
        )
    )
    assert "團隊合作" in seen["system"], "沒有問句原文時,標籤要列進 prompt"
    assert "決策評估" in seen["system"]
