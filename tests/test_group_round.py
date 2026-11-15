"""群面一題一輪:每題每人答一次,AI 依反應速度先後開口,全部答完才換題。

規格(11/15 定案):
    主持方出一題,使用者與每位 AI 應徵者各答一次,全部答完主持方才出下一題
    沒有搶答按鈕:AI 有快有慢(強勢快、邏輯居中、親切慢、沉默最慢,每題加一點隨機),
    使用者在某位 AI 開口前開始回答,就排在他前面
    使用者答完之後,剩下的 AI 不再等,直接依序答完
    AI 都答完而使用者還沒答,就等他
    後答的人可以接前面的說法;誰先誰後不計分
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.group_round import (
    REACTION_BASE_MS,
    REACTION_JITTER_MS,
    REACTION_MIN_MS,
    next_step,
    peer_order,
    reaction_ms,
    round_state,
)
from app.pipeline.interview_live import (
    DEFAULT_GROUP_ROUND_QUESTIONS,
    DEFAULT_ROUND_PEER_LINES,
    USER_LABEL_NOTE,
    group_default_opening,
    next_turn,
    start_interview,
)
from app.prompts.interview_live import GROUP_ROUND_RULES
from app.prompts.interview_personas import opening_intro
from app.schemas.interview import InterviewContext, UtteranceDTO
from tests.test_api import client  # noqa: F401  (fixture)

CTX = InterviewContext(
    custom_role="理財專員", custom_company="某銀行", custom_industry="金融",
    custom_seniority="新鮮人", type="行為",
)
Q1 = "假設分行要在一個月內提升年輕客群的開戶數,各位會先做哪一件事?"
Q2 = "如果主管給的預算砍半,你會先砍哪一塊?"
PEERS = ("AI-強勢", "AI-邏輯", "AI-親切")
SEED = "itv_test"


def host(text: str, who: str = "主考官") -> UtteranceDTO:
    return UtteranceDTO(speaker=who, content=text, is_user=False)


def peer(name: str, text: str) -> UtteranceDTO:
    return UtteranceDTO(speaker=name, content=text, is_user=False)


def me(text: str) -> UtteranceDTO:
    return UtteranceDTO(speaker="你", content=text, is_user=True)


def order_of(question: str, seed: str = SEED) -> list[str]:
    return peer_order(round_state([host(question)]), seed)


def payload(speaker: str, line: str, topic: str = "預算取捨") -> str:
    return json.dumps(
        {"speaker": speaker, "nextQuestion": line, "isFollowUp": False,
         "shouldAdvance": False, "topic": topic},
        ensure_ascii=False,
    )


def spy(reply, sink: list):
    """記下每次 (system, user)。reply 是字串或依呼叫次數回字串的函式。"""
    def call(system: str, user: str) -> str:
        sink.append((system, user))
        return reply(len(sink)) if callable(reply) else reply
    return call


def never(system: str, user: str) -> str:
    raise AssertionError("這一輪不該呼叫模型")


async def round_call(says, llm, **kw):
    return await next_turn(
        mode="group", answer=kw.pop("answer", ""), follow_up_idx=kw.pop("follow_up_idx", 0),
        asked_questions=kw.pop("asked_questions", [Q1]), fallback=kw.pop("fallback", []),
        llm=llm, context=kw.pop("context", CTX), group_says=says,
        group_flow="round", session_id=SEED, **kw,
    )


# ---------------------------------------------------------------------------
# 反應時間
# ---------------------------------------------------------------------------


def test_reaction_time_is_stable_for_the_same_question() -> None:
    """同一題算幾次都一樣,否則 App 兩次呼叫之間順序會跳。"""
    for pid in REACTION_BASE_MS:
        assert reaction_ms(pid, Q1, SEED) == reaction_ms(pid, Q1, SEED)


def test_reaction_time_stays_within_the_jitter_band() -> None:
    for pid, base in REACTION_BASE_MS.items():
        for i in range(50):
            ms = reaction_ms(pid, f"題目{i}", SEED)
            assert max(REACTION_MIN_MS, base - REACTION_JITTER_MS) <= ms <= base + REACTION_JITTER_MS


def test_assertive_always_beats_quiet() -> None:
    """相鄰兩位可以換順序,但強勢不會慢過沉默。"""
    for i in range(100):
        q = f"題目{i}"
        assert reaction_ms("peer_assertive", q, SEED) < reaction_ms("peer_quiet", q, SEED)


def test_order_varies_between_questions() -> None:
    """每題加一點隨機,所以不會每一題都是同一個順序。"""
    orders = {tuple(order_of(f"第{i}題:你會怎麼做")) for i in range(30)}
    assert len(orders) > 1


def test_order_is_fastest_first() -> None:
    ids = {"AI-強勢": "peer_assertive", "AI-邏輯": "peer_logic", "AI-親切": "peer_friendly"}
    order = order_of(Q1)
    times = [reaction_ms(ids[n], Q1, SEED) for n in order]
    assert times == sorted(times)
    assert set(order) == set(PEERS)


# ---------------------------------------------------------------------------
# 這一題的現況
# ---------------------------------------------------------------------------


def test_state_reads_the_current_question_and_its_answers() -> None:
    says = [host(Q1), peer("AI-強勢", "先做社群。"), me("我會先調查支付習慣。"),
            peer("AI-邏輯", "先看數據。"), peer("AI-親切", "同意前面。"),
            host(Q2), peer("AI-邏輯", "砍廣告。")]
    st = round_state(says)
    assert st.question == Q2
    assert st.index == 2
    assert st.asked == (Q1, Q2)
    assert st.answered == ("AI-邏輯",)
    assert st.user_answered is False
    assert [u.content for u in st.answers] == ["砍廣告。"]


def test_state_without_a_recorded_opening_uses_the_fallback_question() -> None:
    """舊版 App 沒把開場題記進 groupSays 時,所有發言都算第一題的回答。"""
    st = round_state([me("我先講。"), peer("AI-強勢", "我也有想法。")], fallback_question=Q1)
    assert st.question == Q1
    assert st.index == 1
    assert st.user_answered is True
    assert st.answered == ("AI-強勢",)


# ---------------------------------------------------------------------------
# 下一步
# ---------------------------------------------------------------------------


def test_fresh_question_starts_with_the_fastest_peer_and_waits() -> None:
    step = next_step(round_state([host(Q1)]), SEED)
    assert step.kind == "peer"
    assert step.speaker == order_of(Q1)[0]
    assert step.delay_ms >= REACTION_MIN_MS
    assert step.position == 1


def test_after_the_user_answers_nobody_waits() -> None:
    first = order_of(Q1)[0]
    step = next_step(round_state([host(Q1), peer(first, "我先說。"), me("我的想法是……")]), SEED)
    assert step.kind == "peer"
    assert step.speaker == order_of(Q1)[1]
    assert step.delay_ms == 0
    assert step.position == 3


def test_user_jumping_in_does_not_change_the_peer_order() -> None:
    """使用者搶在最快那位前面答,AI 的順序照舊,只是不再等。"""
    step = next_step(round_state([host(Q1), me("我先講。")]), SEED)
    assert step.speaker == order_of(Q1)[0]
    assert step.delay_ms == 0


def test_everyone_answers_once_before_the_host_moves_on() -> None:
    says = [host(Q1)] + [peer(n, f"{n} 的回答。") for n in PEERS]
    assert next_step(round_state(says), SEED).kind == "await_user"
    says.append(me("我最後一個答。"))
    step = next_step(round_state(says), SEED)
    assert step.kind == "host"
    assert step.speaker == "主考官"
    assert step.delay_ms == 0


def test_a_peer_never_answers_twice_in_one_question() -> None:
    first = order_of(Q1)[0]
    says = [host(Q1), peer(first, "我先說。"), peer(first, "我再補一句。")]
    st = round_state(says)
    assert st.answered == (first,)
    assert next_step(st, SEED).speaker != first


def test_three_hosts_take_turns_asking() -> None:
    says = [host(Q1, "用人主管")] + [peer(n, "答。") for n in PEERS] + [me("答。")]
    st = round_state(says, group_interviewers=3)
    step = next_step(st, SEED)
    assert step.kind == "host"
    assert step.speaker == "技術主管"


def test_five_person_group_includes_the_quiet_peer_last_most_of_the_time() -> None:
    st = round_state([host(Q1)], group_size=5)
    assert "AI-沉默" in st.peers
    assert peer_order(st, SEED)[-1] == "AI-沉默"


# ---------------------------------------------------------------------------
# 經過 next_turn
# ---------------------------------------------------------------------------


async def test_peer_turn_answers_the_question_first() -> None:
    seen: list = []
    first = order_of(Q1)[0]
    r = await round_call([host(Q1)], spy(payload(first, "我會先跟學校社團合作。"), seen))
    system, user = seen[0]
    assert GROUP_ROUND_RULES in system
    assert "這是小組討論" not in system
    assert user.startswith(f"【主考官問的這一題】{Q1}")
    assert "你是這一題第一個回答的人" in user
    assert r.speaker == first
    assert r.next_question == "我會先跟學校社團合作。"
    assert r.delay_ms >= REACTION_MIN_MS
    assert r.await_user is False
    assert r.topic == "", "應徵者的話不是新題目,不累積進 askedTopics"


async def test_later_peer_sees_earlier_answers_and_does_not_wait() -> None:
    seen: list = []
    first, second = order_of(Q1)[:2]
    says = [host(Q1), peer(first, "先做社群。"), me("我會先調查支付習慣。")]
    r = await round_call(says, spy(payload(second, "我接著前面,先看開戶流程卡在哪。"), seen))
    user = seen[0][1]
    assert f"  {first}:先做社群。" in user
    assert "  使用者:我會先調查支付習慣。" in user
    assert USER_LABEL_NOTE in user
    assert "你是第 3 個" in user
    assert r.speaker == second
    assert r.delay_ms == 0


async def test_waiting_for_the_user_calls_no_model() -> None:
    says = [host(Q1)] + [peer(n, f"{n} 的回答。") for n in PEERS]
    r = await round_call(says, never)
    assert r.await_user is True
    assert r.speaker == ""
    assert r.next_question == ""


async def test_host_asks_the_next_question_without_the_peer_stance() -> None:
    seen: list = []
    says = [host(Q1)] + [peer(n, f"{n} 的回答。") for n in PEERS] + [me("我的回答。")]
    r = await round_call(says, spy(payload("主考官", "謝謝大家。" + Q2), seen))
    system, user = seen[0]
    assert "你是同場競爭的應徵者" not in system
    assert "【本場已經問過的題目】" in user and Q1 in user
    assert r.speaker == "主考官"
    assert Q2 in r.next_question
    assert r.topic == "預算取捨"
    assert r.delay_ms == 0


async def test_host_repeating_an_old_question_falls_back_to_an_unused_one() -> None:
    seen: list = []
    says = [host(Q1)] + [peer(n, "答。") for n in PEERS] + [me("答。")]
    r = await round_call(
        says, spy(payload("主考官", Q1), seen), fallback=[Q1, "如果要你帶新人,你第一週會做什麼?"]
    )
    assert len(seen) == 2, "先重生成一次"
    assert r.next_question == "如果要你帶新人,你第一週會做什麼?"
    assert any("改用備援題" in n for n in r.notices)


async def test_peer_line_that_is_all_questions_stays_with_the_peer() -> None:
    """整句都是問句時換備援句,不交給主持方——大家還沒答完,主持方不能出題。"""
    first = order_of(Q1)[0]
    r = await round_call([host(Q1)], spy(payload(first, "大家覺得呢?"), []))
    assert r.speaker == first
    assert r.next_question in DEFAULT_ROUND_PEER_LINES


async def test_generation_failure_keeps_the_speaker_and_the_delay() -> None:
    def down(system: str, user: str) -> str:
        raise RuntimeError("llm down")

    first = order_of(Q1)[0]
    r = await round_call([host(Q1)], down)
    assert r.speaker == first
    assert r.next_question in DEFAULT_ROUND_PEER_LINES
    assert r.delay_ms >= REACTION_MIN_MS


async def test_host_generation_failure_uses_a_backup_question() -> None:
    def down(system: str, user: str) -> str:
        raise RuntimeError("llm down")

    says = [host(Q1)] + [peer(n, "答。") for n in PEERS] + [me("答。")]
    r = await round_call(says, down)
    assert r.speaker == "主考官"
    assert r.next_question in DEFAULT_GROUP_ROUND_QUESTIONS


async def test_model_naming_another_speaker_is_overruled() -> None:
    first = order_of(Q1)[0]
    other = next(n for n in PEERS if n != first)
    r = await round_call([host(Q1)], spy(payload(other, "我先說,先做社群。"), []))
    assert r.speaker == first
    assert any("已改回" in n for n in r.notices)


async def test_direct_callers_still_get_the_discussion_flow() -> None:
    """不帶 group_flow 的呼叫端(測試、量測工具)維持自由討論。"""
    seen: list = []
    await next_turn(
        mode="group", answer="我會先調查支付習慣。", follow_up_idx=2,
        asked_questions=[Q1], fallback=[], llm=spy(payload("AI-強勢", "先做社群。"), seen),
        context=CTX, group_says=[host(Q1), me("我會先調查支付習慣。")],
        force_speaker="AI-強勢",
    )
    assert "這是小組討論" in seen[0][0]


# ---------------------------------------------------------------------------
# 開場
# ---------------------------------------------------------------------------


async def test_round_opening_asks_for_a_question_everyone_answers() -> None:
    seen: list = []
    r = await start_interview(mode="group", context=CTX, llm=spy("{}", seen))
    opening_system = next(s for s, _ in seen if "開場問題" in s)
    assert "每位應徵者輪流回答" in opening_system
    assert r.opening_question == group_default_opening(CTX, "round")
    assert r.opening_question.endswith("請每個人輪流說你的選擇和理由。")
    assert r.fallback_probes == list(DEFAULT_GROUP_ROUND_QUESTIONS)
    assert "不用照座位順序" in r.opening_intro


def test_discussion_intro_is_unchanged() -> None:
    assert opening_intro("group").endswith("請大家一起討論。")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _says_json(says) -> list[dict]:
    return [u.model_dump(by_alias=True) for u in says]


def test_api_defaults_to_round_and_reports_await_user(client) -> None:  # noqa: F811
    says = [host(Q1)] + [peer(n, f"{n} 的回答。") for n in PEERS]
    d = client.post(
        "/interviews/itv_x/turns",
        json={"answer": "", "mode": "group", "askedQuestions": [Q1],
              "groupSays": _says_json(says)},
    ).json()
    assert d["awaitUser"] is True
    assert d["speaker"] == ""
    assert d["delayMs"] == 0


def test_api_peer_turn_carries_a_delay(client) -> None:  # noqa: F811
    d = client.post(
        "/interviews/itv_x/turns",
        json={"answer": "", "mode": "group", "askedQuestions": [Q1],
              "groupSays": _says_json([host(Q1)])},
    ).json()
    assert d["speaker"] in PEERS
    assert d["delayMs"] >= REACTION_MIN_MS
    assert d["awaitUser"] is False


@pytest.mark.parametrize("flow", ["discussion"])
def test_api_discussion_flow_is_still_selectable(client, flow) -> None:  # noqa: F811
    d = client.post(
        "/interviews/itv_x/turns",
        json={"answer": "我會先調查支付習慣。", "mode": "group", "askedQuestions": [Q1],
              "context": {"groupFlow": flow},
              "groupSays": _says_json([host(Q1), me("我會先調查支付習慣。")])},
    ).json()
    assert d["awaitUser"] is False
    assert d["delayMs"] == 0
