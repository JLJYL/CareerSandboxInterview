"""群面的討論形狀:開場是全組的題目,AI 應徵者回應討論而不是追問使用者。

【為什麼有這一份】
group-1(前端 10/29 錄的那一場)三位 AI 應徵者每一輪都用「你提到…能分享一下…」
追問使用者,沒有人表態,彼此零互動。原因是三件事疊在一起:

    開場固定成自我介紹      題目變成使用者一個人的經歷,競爭者沒有東西可以表態
    面試官的規則一起送進去  STAR 追問的決定、追問判準、履歷/職缺階段
    看不到別人講過什麼      前端每輪送 groupSays,合約沒宣告,被丟掉了

這一份每一條都對應其中一件。不打真 LLM。
"""

from __future__ import annotations

import json

from app.pipeline.interview_live import (
    DEFAULT_GROUP_PROBES,
    group_default_opening,
    group_discussion_block,
    next_turn,
    opening_question,
    start_interview,
)
from app.prompts.interview_live import (
    GROUP_OPENING_RULES,
    GROUP_TURN_RULES,
    compose_opening_prompt,
    compose_turn_prompt,
)
from app.prompts.interview_personas import opening_intro
from app.schemas.interview import (
    ExperienceDTO,
    InterviewContext,
    TurnRequest,
    UtteranceDTO,
)
from tests.test_api import client  # noqa: F401  (fixture)

CTX = InterviewContext(
    custom_role="理財專員", custom_company="某銀行", custom_industry="金融",
    custom_seniority="新鮮人", type="行為",
)

TOPIC = "假設分行要在一個月內提升年輕客群的開戶數,各位會先做哪一件事?"
PEER_LINE = "我會先做社群活動,年輕人在那裡,預算也該先放那邊。"

SAYS = [
    UtteranceDTO(speaker="主考官", content=TOPIC, is_user=False),
    UtteranceDTO(speaker="你", content="我會先調查他們最常用的支付方式。", is_user=True),
    UtteranceDTO(speaker="AI-邏輯", content=PEER_LINE, is_user=False),
    UtteranceDTO(speaker="你", content="預算我會先放在線上開戶流程。", is_user=True),
]
ANSWER = "預算我會先放在線上開戶流程。"

RESUME = [ExperienceDTO(id="e1", title="股票分析競賽", description="用 Python 做風險控管模型")]


def spy(payload: dict, *, sink: list):
    """記下每次 (system, user),回固定 payload。"""
    def call(system: str, user: str) -> str:
        sink.append((system, user))
        return json.dumps(payload, ensure_ascii=False)
    return call


TURN = {
    "speaker": "AI-強勢", "nextQuestion": "線上流程誰都會做,我會先談跟學校合作。",
    "reaction": "", "isFollowUp": True, "topic": "開戶策略", "shouldAdvance": False,
}


# ---------------------------------------------------------------------------
# 開場
# ---------------------------------------------------------------------------


async def test_group_opening_is_the_models_discussion_question() -> None:
    def call(system: str, user: str) -> str:
        if "小組討論的題目" in system:
            return json.dumps(
                {"openingQuestion": TOPIC, "openingTopic": "開戶策略",
                 "fallbackProbes": ["還沒表態的人怎麼看?"]},
                ensure_ascii=False,
            )
        return "{}"

    r = await start_interview(mode="group", context=CTX, llm=call)
    assert r.opening_question == TOPIC
    assert r.opening_topic == "開戶策略"
    assert r.fallback_probes == ["還沒表態的人怎麼看?"]


async def test_group_opening_falls_back_to_a_discussion_not_a_self_intro() -> None:
    r = await start_interview(mode="group", context=CTX, llm=lambda s, u: "{}")
    assert r.opening_question == group_default_opening(CTX)
    assert "自我介紹" not in r.opening_question
    assert r.fallback_probes == list(DEFAULT_GROUP_PROBES)
    assert any("討論題目" in n for n in r.notices), "退回預設要講出來"


async def test_single_opening_is_still_the_self_intro() -> None:
    """三段式那一版的開場只對一對一與 panel 是對的,不要一起改掉。"""
    def call(system: str, user: str) -> str:
        return json.dumps({"openingQuestion": TOPIC, "fallbackProbes": ["x"]}, ensure_ascii=False)

    r = await start_interview(mode="single", context=CTX, llm=call)
    assert r.opening_question == opening_question(CTX)
    assert r.opening_topic == "自我介紹"


def test_group_opening_prompt_asks_for_a_discussion_question() -> None:
    group = compose_opening_prompt("group")
    single = compose_opening_prompt("single")
    assert GROUP_OPENING_RULES in group
    assert "開場主問題" not in group, "個人面試的開場判準不能一起送"
    assert GROUP_OPENING_RULES not in single


async def test_group_opening_does_not_see_the_resume() -> None:
    """題目是給全組的,不貼著使用者一個人的履歷出。"""
    seen: list = []
    await start_interview(
        mode="group", context=CTX, llm=spy({}, sink=seen), experiences=RESUME
    )
    opening_users = [u for s, u in seen if "小組討論的題目" in s]
    assert opening_users, "開場那次呼叫要有發出去"
    assert all("風險控管模型" not in u for u in opening_users)


def test_group_intro_announces_a_discussion() -> None:
    assert "討論" in opening_intro("group", role_title="理財專員")


# ---------------------------------------------------------------------------
# 每一輪
# ---------------------------------------------------------------------------


def test_group_turn_prompt_has_no_interview_rules() -> None:
    group = compose_turn_prompt("group", ["Q1"], focus_speaker="AI-邏輯", turn_index=0)
    single = compose_turn_prompt("single", ["Q1"], turn_index=0, has_resume=True)
    assert GROUP_TURN_RULES in group
    assert "追問時嚴禁" not in group, "追問判準是面試官的規則"
    assert "換新主問題" not in group, "履歷/職缺階段是面試官的規則"
    assert "追問時嚴禁" in single, "一對一的追問判準不能被拿掉"


def test_a_cut_off_answer_is_still_handled_in_group() -> None:
    assert "被語音辨識切斷" in compose_turn_prompt(
        "group", [], focus_speaker="AI-邏輯", truncated=True
    )


async def test_group_turn_sees_the_topic_and_the_other_speakers() -> None:
    seen: list = []
    await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[],
        fallback=[], llm=spy(TURN, sink=seen), context=CTX,
        force_speaker="AI-強勢", group_says=SAYS,
    )
    user = seen[-1][1]
    assert f"【討論題目】{TOPIC}" in user
    assert f"AI-邏輯:{PEER_LINE}" in user, "要看得到別的應徵者講過什麼"
    assert user.count(ANSWER) == 1, "使用者剛剛那段已經在最上面,不要再列一次"


def test_the_discussion_block_keeps_only_recent_lines() -> None:
    many = [UtteranceDTO(speaker="主考官", content=TOPIC, is_user=False)] + [
        UtteranceDTO(speaker="AI-邏輯", content=f"論點{i}", is_user=False) for i in range(20)
    ]
    block = group_discussion_block(many, "")
    assert f"【討論題目】{TOPIC}" in block, "題目永遠要在,不管討論多長"
    assert "論點19" in block and "論點5" not in block


INTRO = UtteranceDTO(
    speaker="主考官",
    content="你好,我是主考官。今天連你在內共四位應徵者。我會先出一個題目,請大家一起討論。",
    is_user=False,
)


async def test_the_topic_is_the_opening_question_even_if_the_intro_was_recorded() -> None:
    """題目先看 askedQuestions 第一筆。前端把 openingIntro 記成一則時,第一則是開場白。"""
    seen: list = []
    await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC, PEER_LINE],
        fallback=[], llm=spy(TURN, sink=seen), context=CTX,
        force_speaker="AI-強勢", group_says=[INTRO, *SAYS],
    )
    user = seen[-1][1]
    assert f"【討論題目】{TOPIC}" in user
    assert "【討論題目】你好" not in user


def test_without_asked_questions_the_first_host_line_is_the_topic() -> None:
    assert f"【討論題目】{TOPIC}" in group_discussion_block(SAYS, ANSWER)


async def test_group_turn_does_not_get_the_resume() -> None:
    """同場的應徵者看不到使用者的履歷。"""
    seen: list = []
    await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[],
        fallback=[], llm=spy(TURN, sink=seen), context=CTX,
        force_speaker="AI-強勢", group_says=SAYS, experiences=RESUME,
    )
    assert "風險控管模型" not in seen[-1][1]


async def test_a_repeated_group_line_is_regenerated_not_swapped_for_a_fallback() -> None:
    """備援句是主持人的推進句,換進 AI 應徵者嘴裡就是一句面試官的話。"""
    calls: list = []
    fresh = "社群活動太慢,我會直接跟學校簽約。"

    def call(system: str, user: str) -> str:
        calls.append(user)
        line = fresh if "重新問一句" in user else PEER_LINE
        return json.dumps({**TURN, "nextQuestion": line, "isFollowUp": False}, ensure_ascii=False)

    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[],
        fallback=["你們在過程中遇到最大的困難是什麼?"], llm=call, context=CTX,
        force_speaker="AI-強勢", group_says=SAYS,
    )
    assert r.next_question == fresh
    assert len(calls) == 2, "撞到重複要重生成一次"


async def test_a_group_line_that_repeats_an_asked_question_keeps_the_speakers_voice() -> None:
    """一對一那條「新主問題重複就換備援題」在群面不能走。

    前端在 askedQuestions 裡累積的是每一則 nextQuestion,群面就包含競爭者的發言。
    撞到的時候換成備援題,AI 應徵者就會唸出一句「你們遇到最大的困難是什麼?」。
    """
    fallback = "你們在過程中遇到最大的困難是什麼?"

    def call(system: str, user: str) -> str:
        line = "社群活動太慢,我會直接跟學校簽約。你覺得呢?" if "重新問一句" in user else PEER_LINE
        return json.dumps({**TURN, "nextQuestion": line, "isFollowUp": False}, ensure_ascii=False)

    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[PEER_LINE],
        fallback=[fallback], llm=call, context=CTX, force_speaker="AI-強勢",
    )
    assert r.next_question != fallback


# ---------------------------------------------------------------------------
# AI 應徵者不發問(10/30 定案):提問只有主考官和主管
# ---------------------------------------------------------------------------


def _scripted(*lines: str, sink: list | None = None, speaker: str = "AI-強勢"):
    """依序回這幾句,用完之後一直回最後一句。sink 記下每次的 user 訊息。"""
    it = iter(lines)

    def call(system: str, user: str) -> str:
        if sink is not None:
            sink.append(user)
        line = next(it, lines[-1])
        return json.dumps({**TURN, "speaker": speaker, "nextQuestion": line}, ensure_ascii=False)

    return call


async def test_a_peer_question_is_regenerated_once() -> None:
    calls: list = []
    clean = "社群活動太慢,我會直接跟學校簽約,開學季就看得到人數。"
    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC],
        fallback=[], llm=_scripted("我覺得先做社群,你們怎麼看?", clean, sink=calls),
        context=CTX, force_speaker="AI-強勢", group_says=SAYS,
    )
    assert r.next_question == clean
    assert len(calls) == 2
    assert "不發問" in calls[-1], "重生成要講清楚錯在哪"
    assert any("是應徵者在發問" in n for n in r.notices)


async def test_a_tag_question_left_after_regeneration_is_cut_off() -> None:
    """重生成還是丟球的話,只切掉逗號後面那一小段,主張留著。"""
    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC],
        fallback=[], llm=_scripted("我覺得先做客群分析比較穩,你同意嗎?"),
        context=CTX, force_speaker="AI-強勢", group_says=SAYS,
    )
    assert r.speaker == "AI-強勢"
    assert r.next_question == "我覺得先做客群分析比較穩。"
    assert any("問句已刪除" in n for n in r.notices)


async def test_a_peer_line_that_is_only_a_question_goes_to_the_host() -> None:
    """整句都是問句就沒有主張可留。讓應徵者唸一句問句,就是回到 group-1。"""
    host_line = "目前有兩種做法,大家比較一下?"
    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC],
        fallback=[host_line], llm=_scripted("那你們的預算是怎麼估的?"),
        context=CTX, force_speaker="AI-強勢", group_says=SAYS,
    )
    assert r.speaker == "主考官"
    assert r.next_question == host_line
    assert any("改由主持人推進" in n for n in r.notices)
    assert r.topic, "topic 不可以留空,前端拿它累積 askedTopics"


async def test_the_host_may_ask() -> None:
    calls: list = []
    q = "目前有兩種做法,大家比較一下,你們會選哪一個?"
    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC],
        fallback=[], llm=_scripted(q, sink=calls, speaker="主考官"), context=CTX,
        force_speaker="主考官", group_says=SAYS,
    )
    assert r.next_question == q
    assert len(calls) == 1, "主持人的問句不該重生成"


async def test_the_three_managers_may_ask_too() -> None:
    ctx = CTX.model_copy(update={"group_interviewers": 3})
    q = "這個做法要成立,前提是什麼?"
    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=2, asked_questions=[TOPIC],
        fallback=[], llm=_scripted(q, speaker="技術主管"), context=ctx,
        force_speaker="技術主管", group_says=SAYS,
    )
    assert (r.speaker, r.next_question) == ("技術主管", q)


async def test_interviewer_modes_still_ask() -> None:
    """一對一與 panel 的面試官本來就該發問,這條規則只管群面的應徵者。"""
    q = "那個數字的母數是多少?"
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=[],
        fallback=[], llm=_scripted(q, speaker="技術主管"), force_speaker="技術主管",
    )
    assert r.next_question == q


def test_the_peer_rules_no_longer_ask_for_a_question() -> None:
    from app.prompts.interview_live import GROUP_PEER_STANCE

    assert "先表態再發問" not in GROUP_PEER_STANCE
    assert "不發問" in GROUP_PEER_STANCE and "不發問" in GROUP_TURN_RULES
    demo = GROUP_PEER_STANCE.split("正確的形狀")[1].split("【為什麼用不相干")[0]
    assert "?" not in demo and "？" not in demo, "模型學的是示範的形狀,示範帶問句就白寫了"


def test_no_peer_persona_is_shown_asking() -> None:
    """語氣示範會被照著學。示範句帶問號,前面那條規則等於沒寫。"""
    from app.prompts.interview_personas import personas_for

    for gi in (1, 3):
        for p in personas_for("group", gi, 5):
            if p.role != "peer":
                continue
            assert not any("?" in s or "？" in s for s in p.sample_lines), p.id
            assert "就問" not in p.stance, p.id


async def test_a_group_failure_falls_back_to_the_moderator() -> None:
    def boom(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=1, asked_questions=[],
        fallback=[], llm=boom, context=CTX, force_speaker="AI-強勢", group_says=SAYS,
    )
    assert r.speaker == "主考官"
    assert r.next_question in DEFAULT_GROUP_PROBES
    assert r.should_advance is False, "群面沒有輪次上限,降級不能標記推進"


# ---------------------------------------------------------------------------
# 合約與路由
# ---------------------------------------------------------------------------


def test_turn_request_keeps_group_says() -> None:
    req = TurnRequest.model_validate(
        {"answer": "x", "groupSays": [{"speaker": "AI-邏輯", "content": "母數多少", "isUser": False}]}
    )
    assert [u.content for u in req.group_says] == ["母數多少"]


def test_group_turns_skip_the_star_decision(client, monkeypatch) -> None:  # noqa: F811
    """STAR 追問的決定是面試官評估一個人的方式,群面不做。"""
    import app.api.routes as routes

    called: list = []

    async def fake_decide(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(routes, "decide_turn", fake_decide)

    body = {"answer": ANSWER, "followUpIdx": 2, "context": {"customRole": "理財專員"}}
    client.post("/interviews/itv_x/turns", json={**body, "mode": "group"})
    assert called == [], "群面不該呼叫追問決策"

    client.post("/interviews/itv_x/turns", json={**body, "mode": "single"})
    assert called, "一對一照舊要做決定"


def test_group_says_reach_the_turn_prompt_through_the_api(client, monkeypatch) -> None:  # noqa: F811
    import app.api.routes as routes

    seen: list = []
    monkeypatch.setattr(routes, "get_llm_live", lambda: spy(TURN, sink=seen))
    client.post(
        "/interviews/itv_x/turns",
        json={
            "answer": ANSWER, "followUpIdx": 2, "mode": "group",
            "groupSays": [u.model_dump(by_alias=True) for u in SAYS],
        },
    )
    assert any(f"AI-邏輯:{PEER_LINE}" in u for _, u in seen)


def test_generated_peer_profiles_are_not_primed_to_ask() -> None:
    """背景與說話風格會顯示在小組名單上。範例寫「會反問」,就跟不發問的規則打架。"""
    from app.prompts.peer_profile import compose_profile_prompt

    prompt = compose_profile_prompt()
    assert "反問" not in prompt and "先問" not in prompt
