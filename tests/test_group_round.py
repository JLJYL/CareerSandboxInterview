"""群面一題一輪:主持方點名,每題每人答一次,全部答完才換題。

規格(11/15 定案):
    主持方出一題,使用者與每位 AI 應徵者各答一次,全部答完主持方才出下一題
    後答的人可以接前面的說法;誰先誰後不計分

11/18 早上(11/17 實測:答案都一樣、一則接一則搶不到):
    等待 = 上一則的講話時間(一字 0.22 秒)+ 停頓
    每題每位 AI 分到不同的切入角度,固定一位唱反調
    跟這一題前面的回答(含使用者)太像就重生成一次

11/18 下午(使用者選的):
    發言順序由主持方點名,每題隨機,使用者也排在裡面
    出題那句最後是「請 X 先說。」,之後每位答完,主持方說一句點名句
    主持方不講第幾題
    搶答保留:使用者在下一則出現前開口,就排到前面,主持方接著點被他搶先的那位
    使用者答完之後,後面照樣等讀完的時間

11/18 下午實測太慢(第一位 AI 等二十幾秒、兩位 AI 之間十幾秒):
    等待改成閱讀速度(一字 0.12 秒,最多 8 秒),被點到後停 0.5 到 2 秒
"""

from __future__ import annotations

import json

import pytest

from app.contracts.interview_protocols import FakeCollabObserver
from app.pipeline.group_round import (
    ANGLES,
    DISSENT,
    DISSENT_FIRST,
    DISSENT_LATER,
    HOST_PAUSE_MS,
    PERSONA_ANGLES,
    REACTION_BASE_MS,
    REACTION_JITTER_MS,
    REACTION_MIN_MS,
    SPEAK_MAX_MS,
    SPEAK_MS_PER_CHAR,
    USER_SLOT,
    clean_question,
    cue_line,
    designation,
    is_cue_line,
    next_step,
    reaction_ms,
    round_angles,
    round_order,
    round_state,
    speak_ms,
    strip_question_number,
    too_similar,
    with_designation,
)
from app.pipeline.interview_live import (
    DEFAULT_GROUP_ROUND_QUESTIONS,
    DEFAULT_ROUND_PEER_LINES,
    USER_LABEL_NOTE,
    group_default_opening,
    next_turn,
    round_peer_prompt,
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
IDS = {"AI-強勢": "peer_assertive", "AI-邏輯": "peer_logic",
       "AI-親切": "peer_friendly", "AI-沉默": "peer_quiet"}
FORTY = "我會先找三個老客戶試用,看他們願不願意付錢,再決定功能要做到多完整,兩週內給結論。"


def peers_of(size: int = 4, gi: int = 1) -> tuple[str, ...]:
    return round_state([], group_interviewers=gi, group_size=size).peers


PEERS = peers_of(4)


def _seed_where(pred, index: int = 1, size: int = 4) -> str:
    return next(
        s for s in (f"itv_{i}" for i in range(1000))
        if pred(round_order(peers_of(size), index, s))
    )


SEED = _seed_where(lambda o: USER_SLOT not in o[:2])
"""第一題前兩位都是 AI 的場次。多數測試要先看 AI 怎麼答。"""
SEED_USER_FIRST = _seed_where(lambda o: o[0] == USER_SLOT)
ORDER = round_order(PEERS, 1, SEED)
FIRST, SECOND = ORDER[0], ORDER[1]


def host(text: str, who: str = "主考官") -> UtteranceDTO:
    return UtteranceDTO(speaker=who, content=text, is_user=False)


def peer(name: str, text: str) -> UtteranceDTO:
    return UtteranceDTO(speaker=name, content=text, is_user=False)


def me(text: str) -> UtteranceDTO:
    return UtteranceDTO(speaker="你", content=text, is_user=True)


def qline(q: str = Q1, seed: str = SEED, index: int = 1, size: int = 4,
          who: str = "主考官", gi: int = 1) -> UtteranceDTO:
    """出題那一句,App 記進 groupSays 的樣子:題目加上「請 X 先說。」"""
    return host(with_designation(q, round_order(peers_of(size, gi), index, seed)[0]), who)


def cue(target: str, turn: int, last: bool = False, who: str = "主考官") -> UtteranceDTO:
    return host(cue_line(target, last=last, turn=turn), who)


def play(says, *, seed: str = SEED, size: int = 4, gi: int = 1):
    """照 next_step 把這一題走完,走到主持方要出下一題為止。回 (says, steps)。

    點到使用者就替他答一句;AI 的回答用一句固定的話代替。
    """
    says, steps = list(says), []
    for _ in range(30):
        st = round_state(says, group_interviewers=gi, group_size=size)
        step = next_step(st, seed)
        steps.append(step)
        if step.kind == "host":
            break
        if step.kind == "await_user":
            says.append(me("我會先問客戶最在意什麼,再決定。"))
        elif step.kind == "cue":
            says.append(cue(step.target, len(st.answers), step.last, step.speaker))
        else:
            says.append(peer(step.speaker, f"{step.speaker}的回答,講一個具體的做法。"))
    return says, steps


def answer_order(says) -> list[str]:
    hosts = {"主考官", "用人主管", "技術主管", "HR 主管"}
    return [USER_SLOT if u.is_user else u.speaker for u in says if u.is_user or u.speaker not in hosts]


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
        group_flow="round", session_id=kw.pop("seed", SEED), **kw,
    )


# ---------------------------------------------------------------------------
# 停頓與講話時間
# ---------------------------------------------------------------------------


def test_reaction_time_is_stable_for_the_same_question() -> None:
    for pid in REACTION_BASE_MS:
        assert reaction_ms(pid, Q1, SEED) == reaction_ms(pid, Q1, SEED)


def test_reaction_time_stays_within_the_jitter_band() -> None:
    for pid, base in REACTION_BASE_MS.items():
        for i in range(50):
            ms = reaction_ms(pid, f"題目{i}", SEED)
            assert max(REACTION_MIN_MS, base - REACTION_JITTER_MS) <= ms <= base + REACTION_JITTER_MS


def test_assertive_starts_sooner_than_quiet_when_called() -> None:
    for i in range(50):
        q = f"題目{i}"
        assert reaction_ms("peer_assertive", q, SEED) < reaction_ms("peer_quiet", q, SEED)


def test_speaking_time_counts_characters_not_punctuation() -> None:
    assert speak_ms("我會先做。") == 4 * SPEAK_MS_PER_CHAR
    assert speak_ms("  ,。?!「」") == 0
    assert speak_ms("") == 0
    assert speak_ms("字" * 500) == SPEAK_MAX_MS


def test_waits_leave_time_to_read_but_keep_the_round_moving() -> None:
    """11/18 下午實測:第一位 AI 等二十幾秒、兩位 AI 之間十幾秒,太慢。

    四十字的回答之後,主持方要等到讀得完才點名,但不超過 6.5 秒;
    點名之後 AI 開口不超過 3.5 秒;題目出來到第一位 AI 開口不超過 8 秒。
    """
    called = next_step(round_state([qline(), peer(FIRST, FORTY)]), SEED)
    assert called.kind == "cue"
    assert speak_ms(FORTY) <= called.delay_ms <= 6500
    answer = next_step(round_state([qline(), peer(FIRST, FORTY), cue(SECOND, 1)]), SEED)
    assert answer.kind == "peer"
    assert answer.delay_ms <= 3500
    assert next_step(round_state([qline()]), SEED).delay_ms <= 8000


# ---------------------------------------------------------------------------
# 點名順序
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 4, 5])
def test_order_has_everyone_once(size: int) -> None:
    order = round_order(peers_of(size), 1, SEED)
    assert sorted(order) == sorted(peers_of(size) + (USER_SLOT,))


def test_order_is_stable_within_a_question_and_changes_across_questions() -> None:
    assert round_order(PEERS, 1, SEED) == round_order(PEERS, 1, SEED)
    assert len({round_order(PEERS, i, SEED) for i in range(1, 21)}) > 5


def test_users_position_is_random_across_questions() -> None:
    """使用者選的:每題隨機。四個位置都要常出現,不是固定在某一格。"""
    counts = [0, 0, 0, 0]
    for i in range(1, 401):
        counts[round_order(PEERS, i, SEED).index(USER_SLOT)] += 1
    assert all(c >= 70 for c in counts), counts


# ---------------------------------------------------------------------------
# 點名句與題目
# ---------------------------------------------------------------------------


def test_every_cue_line_is_recognised() -> None:
    for who in peers_of(5) + (USER_SLOT,):
        for last in (False, True):
            for turn in (0, 1, 2):
                assert is_cue_line(cue_line(who, last=last, turn=turn)), (who, last, turn)


@pytest.mark.parametrize("text", [
    "謝謝。下一位請 AI-親切,他剛剛說的我同意。",
    "好,接下來請大家想一想預算怎麼分。",
    "謝謝主考官。我會先做社群。",
    Q1,
    with_designation(Q1, "AI-親切"),
])
def test_answers_and_questions_are_not_cue_lines(text: str) -> None:
    assert not is_cue_line(text)


def test_cue_wording_marks_the_last_and_addresses_the_user() -> None:
    assert cue_line(USER_SLOT) == "謝謝。接下來請你。"
    assert cue_line(USER_SLOT, last=True) == "謝謝。最後請你。"
    assert cue_line("AI-親切", last=True) == "謝謝。最後請 AI-親切。"
    assert cue_line("AI-親切", turn=0) != cue_line("AI-親切", turn=1)


@pytest.mark.parametrize("raw,clean", [
    ("第一題:假設預算只剩一半,你會先做什麼?", "假設預算只剩一半,你會先做什麼?"),
    ("下一題:如果主管給的時程不夠?", "如果主管給的時程不夠?"),
    ("謝謝大家的分享,我們來看下一題:如果預算砍半?", "謝謝大家的分享,如果預算砍半?"),
    ("接下來是第三題,如果團隊有人不配合?", "如果團隊有人不配合?"),
    ("謝謝。我們進入第二題。團隊有人不配合怎麼辦?", "謝謝。團隊有人不配合怎麼辦?"),
    ("第 2 題:預算砍半怎麼辦?", "預算砍半怎麼辦?"),
])
def test_question_numbers_are_stripped(raw: str, clean: str) -> None:
    """使用者要主考官直接講問題,不講第幾題。"""
    assert strip_question_number(raw) == clean


def test_a_mid_sentence_reference_to_another_question_is_kept() -> None:
    text = "跟第一題比,這題多了預算限制,你會怎麼調整?"
    assert strip_question_number(text) == text


def test_designation_is_added_once() -> None:
    q = with_designation(Q1, "AI-親切")
    assert q == Q1 + "請 AI-親切 先說。"
    assert with_designation(q, USER_SLOT) == Q1 + "請你先說。"
    assert clean_question(q) == Q1
    assert with_designation("你會怎麼做", "AI-邏輯") == "你會怎麼做。請 AI-邏輯 先說。"


# ---------------------------------------------------------------------------
# 這一題的現況
# ---------------------------------------------------------------------------


def test_state_reads_the_current_question_and_its_answers() -> None:
    a, b, c = PEERS
    says = [host(with_designation(Q1, a)), peer(a, "先做社群。"), cue(USER_SLOT, 1), me("我先調查。"),
            cue(b, 2), peer(b, "先看數據。"), cue(c, 3, last=True), peer(c, "先問客戶。"),
            host(with_designation("第二題:" + Q2, b)), peer(b, "砍廣告。")]
    st = round_state(says)
    assert st.question == Q2
    assert st.index == 2
    assert st.asked == (Q1, Q2)
    assert st.answered == (b,)
    assert st.user_answered is False
    assert [u.content for u in st.answers] == ["砍廣告。"]
    assert st.last_kind == "peer"


def test_cue_lines_do_not_start_a_new_question() -> None:
    st = round_state([qline(), peer(FIRST, "先做社群。"), cue(SECOND, 1)])
    assert st.index == 1
    assert st.question == Q1
    assert st.answered == (FIRST,)
    assert st.last_kind == "cue"


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


def test_the_question_names_who_answers_first() -> None:
    line = qline()
    assert line.content.endswith(designation(FIRST))
    step = next_step(round_state([line]), SEED)
    assert step.kind == "peer"
    assert step.speaker == FIRST
    assert step.delay_ms == speak_ms(line.content) + reaction_ms(IDS[FIRST], Q1, SEED)
    assert step.position == 1


def test_when_the_user_is_named_first_the_app_waits_for_him() -> None:
    line = qline(seed=SEED_USER_FIRST)
    assert line.content.endswith("請你先說。")
    assert next_step(round_state([line]), SEED_USER_FIRST).kind == "await_user"


def test_after_each_answer_the_host_calls_the_next_in_order() -> None:
    says, steps = play([qline()])
    assert answer_order(says) == list(ORDER)
    cues = [s for s in steps if s.kind == "cue"]
    assert [s.target for s in cues] == list(ORDER[1:])
    assert [s.last for s in cues] == [False] * (len(cues) - 1) + [True]
    assert all(s.speaker == "主考官" for s in cues)
    assert steps[-1].kind == "host"
    assert steps[-1].target == round_order(PEERS, 2, SEED)[0]


def test_cue_waits_for_the_answer_to_be_read_out() -> None:
    step = next_step(round_state([qline(), peer(FIRST, FORTY)]), SEED)
    assert step.kind == "cue"
    assert step.target == SECOND
    assert step.delay_ms == speak_ms(FORTY) + HOST_PAUSE_MS


def test_called_peer_waits_for_the_cue_then_its_own_pause() -> None:
    called = cue(SECOND, 1)
    step = next_step(round_state([qline(), peer(FIRST, FORTY), called]), SEED)
    assert step.kind == "peer"
    assert step.speaker == SECOND
    assert step.delay_ms == speak_ms(called.content) + reaction_ms(IDS[SECOND], Q1, SEED)


def test_host_only_pauses_after_the_users_own_answer() -> None:
    step = next_step(round_state([qline(), me(FORTY)]), SEED)
    assert step.kind == "cue"
    assert step.delay_ms == HOST_PAUSE_MS


def test_user_jumping_in_is_followed_by_the_peer_he_skipped() -> None:
    """主持方點了 FIRST,使用者搶先答完,主持方接著點的還是 FIRST。"""
    step = next_step(round_state([qline(), me("我先講。")]), SEED)
    assert step.kind == "cue"
    assert step.target == FIRST


def test_user_jumping_in_before_a_cue_keeps_the_rest_of_the_order() -> None:
    step = next_step(round_state([qline(), peer(FIRST, "先做社群。"), me("我先講。")]), SEED)
    rest = [s for s in ORDER if s not in (FIRST, USER_SLOT)]
    assert step.kind == "cue"
    assert step.target == rest[0]


def test_user_jumping_in_still_ends_with_everyone_answering_once() -> None:
    says, steps = play([qline(), me("我先講。")])
    order = answer_order(says)
    assert order[0] == USER_SLOT
    assert sorted(order) == sorted(ORDER)
    assert steps[-1].kind == "host"


def test_a_peer_never_answers_twice_in_one_question() -> None:
    st = round_state([qline(), peer(FIRST, "我先說。"), peer(FIRST, "我再補一句。")])
    assert st.answered == (FIRST,)
    step = next_step(st, SEED)
    assert step.kind == "cue" and step.target == SECOND


def test_three_hosts_take_turns_asking_and_the_asker_calls_names() -> None:
    says, steps = play([qline(who="用人主管", gi=3)], gi=3)
    assert all(s.speaker == "用人主管" for s in steps if s.kind == "cue")
    assert steps[-1].kind == "host"
    assert steps[-1].speaker == "技術主管"


def test_five_person_group_calls_the_quiet_peer_too() -> None:
    says, _ = play([qline(size=5)], size=5)
    assert answer_order(says).count("AI-沉默") == 1
    assert len(answer_order(says)) == 5


# ---------------------------------------------------------------------------
# 切入角度
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 4, 5])
def test_every_peer_gets_a_different_angle_and_one_dissents(size: int) -> None:
    for i in range(40):
        st = round_state([host(f"第{i}題:你會怎麼做?")], group_size=size)
        angles = round_angles(st, SEED)
        assert set(angles) == set(st.peers)
        assert len(set(angles.values())) == len(angles), angles
        assert list(angles.values()).count(DISSENT) == 1


def test_angles_are_stable_within_a_question_and_change_across_questions() -> None:
    st = round_state([host(Q1)], group_size=5)
    assert round_angles(st, SEED) == round_angles(st, SEED)
    seen: dict[str, set[str]] = {}
    for i in range(30):
        angles = round_angles(round_state([host(f"你會怎麼做,情境{i}?")], group_size=5), SEED)
        for name, angle in angles.items():
            seen.setdefault(name, set()).add(angle)
    assert all(len(v) >= 3 for v in seen.values()), seen


def test_angles_come_from_the_persona_list_or_dissent() -> None:
    for i in range(40):
        st = round_state([host(f"你會怎麼做,情境{i}?")], group_size=5)
        for name, angle in round_angles(st, SEED).items():
            assert angle == DISSENT or angle in PERSONA_ANGLES[IDS[name]], (name, angle)
            assert angle == DISSENT or angle in ANGLES


def test_dissent_wording_depends_on_whether_anyone_answered() -> None:
    a, b = PEERS[:2]
    alone = round_peer_prompt(round_state([host(Q1)]), a, 1, DISSENT)
    later = round_peer_prompt(round_state([host(Q1), peer(a, "先做社群。")]), b, 2, DISSENT)
    assert DISSENT_FIRST in alone and DISSENT_LATER not in alone
    assert DISSENT_LATER in later and DISSENT_FIRST not in later


# ---------------------------------------------------------------------------
# 跟前面的回答像不像
# ---------------------------------------------------------------------------

Q_BUDGET = "新產品下個月要上線,但預算只剩一半,你會先做什麼?"
Q_URGENT = "同時有客戶投訴和報表要交,你會先處理哪一件?"
Q_SHIP = "產品要先上線再修,還是修好再上線?請說你的選擇和理由。"
Q_BOSS = "如果主管給的時程明顯不夠,你會怎麼跟主管談?"
BUDGET_A = "我會先確認預算夠不夠,不夠的話先砍掉次要功能,確保核心功能能準時上線。"

PARAPHRASES = [
    (Q_BUDGET, BUDGET_A, "我覺得要先看預算,如果預算不足就先把次要的功能拿掉,讓核心功能準時上線。"),
    (Q_URGENT, "我會先處理客戶投訴那件,因為客戶流失的影響比內部報表延遲大。",
     "我會優先處理客戶的投訴,客戶流失造成的影響比報表晚交還要大。"),
    (Q_SHIP, "結論先講:先上線再修,兩週內拿到使用者回饋比較實際。",
     "我也覺得應該先上線,之後再慢慢修,這樣兩週內就能拿到使用者的回饋。"),
    (Q_BOSS, "我會直接跟主管說時程不夠,然後提出砍掉哪些功能可以準時交。",
     "我會跟主管坦白時程不夠,同時提出要砍哪些功能才能準時完成。"),
]
DIFFERENT = [
    (Q_BUDGET, BUDGET_A, "我會先找三個老客戶試用,看他們願不願意付錢,再決定功能要做到多完整。"),
    (Q_URGENT, "我會先處理客戶投訴那件,因為客戶流失的影響比內部報表延遲大。",
     "這題預設兩件都要自己做,其實報表可以先請同事幫忙,我專心處理投訴。"),
    (Q_SHIP, "結論先講:先上線再修,兩週內拿到使用者回饋比較實際。",
     "我反而會晚兩週上線,先把付款流程測完,上線後出錯的成本比延後高。"),
    (Q_BOSS, "我會直接跟主管說時程不夠,然後提出砍掉哪些功能可以準時交。",
     "時程不夠通常是需求沒對齊,我會先拉產品和工程一起把範圍講清楚。"),
]


@pytest.mark.parametrize("question,earlier,line", PARAPHRASES)
def test_paraphrase_of_an_earlier_answer_is_caught(question, earlier, line) -> None:
    hit = too_similar(line, [peer("AI-強勢", earlier)], question)
    assert hit is not None and hit.content == earlier


@pytest.mark.parametrize("question,earlier,line", DIFFERENT)
def test_a_different_angle_on_the_same_question_is_not_flagged(question, earlier, line) -> None:
    assert too_similar(line, [peer("AI-強勢", earlier)], question) is None


def test_words_taken_from_the_question_do_not_count() -> None:
    """題目的字大家都會用,不拿掉的話兩個不同的答案也會被當成換句話說。"""
    a = "新產品下個月要上線,預算只剩一半的話,我會先砍廣告。"
    b = "新產品下個月要上線又只剩一半預算,我會先把外包的設計費停掉。"
    assert too_similar(b, [peer("AI-強勢", a)], "") is not None, "沒拿掉題目的字:誤判"
    assert too_similar(b, [peer("AI-強勢", a)], Q_BUDGET) is None


def test_the_users_answer_counts_too() -> None:
    hit = too_similar(PARAPHRASES[0][2], [me(BUDGET_A)], Q_BUDGET)
    assert hit is not None and hit.is_user


# ---------------------------------------------------------------------------
# 經過 next_turn
# ---------------------------------------------------------------------------


async def test_peer_turn_answers_the_question_without_the_calling_line() -> None:
    seen: list = []
    line = qline()
    r = await round_call([line], spy(payload(FIRST, "我會先跟學校社團合作。"), seen))
    system, user = seen[0]
    assert GROUP_ROUND_RULES in system
    assert "這是小組討論" not in system
    assert user.startswith(f"【主考官問的這一題】{Q1}\n")
    assert "先說。" not in user.split("\n")[0], "點名不是題目的一部分"
    assert "你是這一題第一個回答的人" in user
    assert r.speaker == FIRST
    assert r.next_question == "我會先跟學校社團合作。"
    assert r.delay_ms == speak_ms(line.content) + reaction_ms(IDS[FIRST], Q1, SEED)
    assert r.cue is False and r.await_user is False
    assert r.topic == "", "應徵者的話不是新題目,不累積進 askedTopics"


async def test_cue_turn_needs_no_model() -> None:
    r = await round_call([qline(), peer(FIRST, FORTY)], never)
    assert r.cue is True
    assert r.speaker == "主考官"
    assert r.next_question == cue_line(SECOND, turn=1)
    assert r.delay_ms == speak_ms(FORTY) + HOST_PAUSE_MS
    assert r.await_user is False


async def test_waiting_for_the_user_calls_no_model() -> None:
    r = await round_call([qline(seed=SEED_USER_FIRST)], never, seed=SEED_USER_FIRST)
    assert r.await_user is True
    assert r.speaker == ""
    assert r.next_question == ""
    assert r.cue is False


async def test_later_peer_sees_earlier_answers() -> None:
    seen: list = []
    says = [qline(), peer(FIRST, "先做社群。"), me("我會先調查支付習慣。"), cue(SECOND, 2)]
    r = await round_call(says, spy(payload(SECOND, "我先看開戶流程卡在哪。"), seen))
    user = seen[0][1]
    assert f"  {FIRST}:先做社群。" in user
    assert "  使用者:我會先調查支付習慣。" in user
    assert USER_LABEL_NOTE in user
    assert "你是第 3 個" in user
    assert r.speaker == SECOND
    assert r.delay_ms == speak_ms(says[-1].content) + reaction_ms(IDS[SECOND], Q1, SEED)


async def test_host_asks_the_next_question_and_names_who_starts() -> None:
    seen: list = []
    says, _ = play([qline()])
    r = await round_call(says, spy(payload("主考官", "謝謝大家。" + Q2), seen))
    system, user = seen[0]
    assert "你是同場競爭的應徵者" not in system
    assert "【本場已經問過的題目】" in user and Q1 in user
    assert "不要說「第幾題」" in user
    nxt = round_order(PEERS, 2, SEED)[0]
    assert r.speaker == "主考官"
    assert r.next_question == "謝謝大家。" + Q2 + designation(nxt)
    assert r.cue is False
    assert r.topic == "預算取捨"


async def test_host_saying_the_question_number_has_it_removed() -> None:
    says, _ = play([qline()])
    r = await round_call(says, spy(payload("主考官", "第二題:" + Q2), []))
    assert r.next_question.startswith(Q2)
    assert "第二題" not in r.next_question
    assert any("已拿掉" in n for n in r.notices)


async def test_host_repeating_an_old_question_falls_back_to_an_unused_one() -> None:
    seen: list = []
    says, _ = play([qline()])
    backup = "如果要你帶新人,你第一週會做什麼?"
    r = await round_call(says, spy(payload("主考官", Q1), seen), fallback=[Q1, backup])
    assert len(seen) == 2, "先重生成一次"
    assert r.next_question == backup + designation(round_order(PEERS, 2, SEED)[0])
    assert any("改用備援題" in n for n in r.notices)


async def test_peer_line_that_is_all_questions_stays_with_the_peer() -> None:
    """整句都是問句時換備援句,不交給主持方——大家還沒答完,主持方不能出題。"""
    r = await round_call([qline()], spy(payload(FIRST, "大家覺得呢?"), []))
    assert r.speaker == FIRST
    assert r.next_question in DEFAULT_ROUND_PEER_LINES


async def test_generation_failure_keeps_the_speaker_and_the_delay() -> None:
    def down(system: str, user: str) -> str:
        raise RuntimeError("llm down")

    line = qline()
    r = await round_call([line], down)
    assert r.speaker == FIRST
    assert r.next_question in DEFAULT_ROUND_PEER_LINES
    assert r.delay_ms == speak_ms(line.content) + reaction_ms(IDS[FIRST], Q1, SEED)


async def test_host_generation_failure_uses_a_backup_question() -> None:
    def down(system: str, user: str) -> str:
        raise RuntimeError("llm down")

    says, _ = play([qline()])
    r = await round_call(says, down)
    assert r.speaker == "主考官"
    assert clean_question(r.next_question) in DEFAULT_GROUP_ROUND_QUESTIONS
    assert r.next_question.endswith(designation(round_order(PEERS, 2, SEED)[0]))


async def test_model_naming_another_speaker_is_overruled() -> None:
    other = next(n for n in PEERS if n != FIRST)
    r = await round_call([qline()], spy(payload(other, "我先說,先做社群。"), []))
    assert r.speaker == FIRST
    assert any("已改回" in n for n in r.notices)


async def test_peer_prompt_carries_the_assigned_angle() -> None:
    seen: list = []
    angle = round_angles(round_state([qline()]), SEED)[FIRST]
    r = await round_call([qline()], spy(payload(FIRST, "我會先跟學校社團合作。"), seen))
    user = seen[0][1]
    assert f"【你這一題的切入角度】{angle}:" in user
    assert (DISSENT_FIRST if angle == DISSENT else ANGLES[angle]) in user
    assert f"切入角度:{angle}" in r.notices[0]


async def test_later_peer_is_told_that_paraphrasing_counts_as_repeating() -> None:
    seen: list = []
    says = [qline(), peer(FIRST, "先做社群。"), cue(SECOND, 1)]
    await round_call(says, spy(payload(SECOND, "我先看開戶流程。"), seen))
    user = seen[0][1]
    assert "換句話說也算重複" in user
    assert "不要用「我同意」" in user
    assert "原樣" not in user, "11/15 的「不要原樣再講一次」模型只守字面"


def _budget_says(first_answer: UtteranceDTO, next_up: str) -> list[UtteranceDTO]:
    return [qline(Q_BUDGET), first_answer, cue(next_up, 1)]


async def test_peer_paraphrasing_an_earlier_answer_is_regenerated() -> None:
    seen: list = []
    paraphrase, different = PARAPHRASES[0][2], DIFFERENT[0][2]
    r = await round_call(
        _budget_says(peer(FIRST, BUDGET_A), SECOND),
        spy(lambda n: payload(SECOND, paraphrase if n == 1 else different), seen),
        asked_questions=[Q_BUDGET],
    )
    assert len(seen) == 2
    assert f"那一句跟{FIRST}的回答太像" in seen[1][1]
    assert r.speaker == SECOND
    assert r.next_question == different
    assert any("已重生成" in n for n in r.notices)


async def test_paraphrasing_the_user_is_regenerated_too() -> None:
    seen: list = []
    paraphrase, different = PARAPHRASES[0][2], DIFFERENT[0][2]
    r = await round_call(
        _budget_says(me(BUDGET_A), FIRST),
        spy(lambda n: payload(FIRST, paraphrase if n == 1 else different), seen),
        asked_questions=[Q_BUDGET],
    )
    assert "那一句跟使用者的回答太像" in seen[1][1]
    assert r.next_question == different


async def test_retry_that_is_less_similar_is_used() -> None:
    """重生成還是偏像,但比原本那句不像:用新的。"""
    near_copy = "我也會先確認預算夠不夠,不夠就先砍掉次要功能,確保核心功能準時上線。"
    closer_to_new = "預算不夠就先砍次要功能,但我會先問客戶哪些功能他們最在意。"
    r = await round_call(
        _budget_says(peer(FIRST, BUDGET_A), SECOND),
        spy(lambda n: payload(SECOND, near_copy if n == 1 else closer_to_new), []),
        asked_questions=[Q_BUDGET],
    )
    assert r.next_question == closer_to_new
    assert any("比原本好" in n for n in r.notices)


async def test_retry_that_is_just_as_similar_keeps_the_original() -> None:
    line = "預算不夠就先砍次要功能,但我會先問客戶哪些功能他們最在意。"
    r = await round_call(
        _budget_says(peer(FIRST, BUDGET_A), SECOND), spy(payload(SECOND, line), []),
        asked_questions=[Q_BUDGET],
    )
    assert r.next_question == line
    assert any("照原樣送出" in n for n in r.notices)


async def test_host_questions_are_not_checked_against_the_answers() -> None:
    """主持方的下一題本來就可能用到大家回答裡的字,只查跟問過的題目重不重複。"""
    seen: list = []
    says, _ = play([qline(Q_BUDGET)])
    nxt = "剛剛有人提到先砍次要功能、確保核心功能準時上線。如果團隊有人一直不配合,你會怎麼處理?"
    r = await round_call(says, spy(payload("主考官", nxt), seen), asked_questions=[Q_BUDGET])
    assert len(seen) == 1
    assert clean_question(r.next_question) == nxt


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


def _opening_llm(question: str, probes: list[str]):
    body = json.dumps(
        {"openingQuestion": question, "openingTopic": "開戶", "fallbackProbes": probes},
        ensure_ascii=False,
    )
    return lambda system, user: body if "開場問題" in system else "{}"


def _first_of(start) -> str:
    peers = [p.display_name for p in start.personas if p.role == "peer"]
    return round_order(peers, 1, start.session_id)[0]


async def test_round_opening_names_who_answers_first() -> None:
    seen: list = []
    r = await start_interview(mode="group", context=CTX, llm=spy("{}", seen))
    opening_system = next(s for s, _ in seen if "開場問題" in s)
    assert "不說「第一題」" in opening_system
    assert r.opening_question == with_designation(group_default_opening(CTX, "round"), _first_of(r))
    assert "第一題" not in r.opening_question
    assert r.fallback_probes == list(DEFAULT_GROUP_ROUND_QUESTIONS)
    assert "由我點名" in r.opening_intro
    assert "也可以直接開口" in r.opening_intro


async def test_model_opening_loses_its_number_and_gets_the_designation() -> None:
    r = await start_interview(
        mode="group", context=CTX, llm=_opening_llm("第一題:" + Q1, ["第二題:" + Q2])
    )
    assert r.opening_question == with_designation(Q1, _first_of(r))
    assert r.fallback_probes == [Q2]


async def test_first_turn_follows_the_opening_designation() -> None:
    """開場和每一輪各自算順序,兩邊要點到同一個人。"""
    start = await start_interview(mode="group", context=CTX, llm=_opening_llm(Q1, [Q2]))
    first = _first_of(start)
    r = await next_turn(
        mode="group", answer="", follow_up_idx=0, asked_questions=[start.opening_question],
        fallback=start.fallback_probes, llm=spy(payload(first, "我先答。"), []), context=CTX,
        group_says=[host(start.opening_question, start.opening_speaker)],
        group_flow="round", session_id=start.session_id,
    )
    if first == USER_SLOT:
        assert r.await_user is True
    else:
        assert r.speaker == first


def test_discussion_intro_is_unchanged() -> None:
    assert opening_intro("group").endswith("請大家一起討論。")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _says_json(says) -> list[dict]:
    return [u.model_dump(by_alias=True) for u in says]


def test_api_defaults_to_round_and_reports_await_user(client) -> None:  # noqa: F811
    d = client.post(
        f"/interviews/{SEED_USER_FIRST}/turns",
        json={"answer": "", "mode": "group", "askedQuestions": [Q1],
              "groupSays": _says_json([qline(seed=SEED_USER_FIRST)])},
    ).json()
    assert d["awaitUser"] is True
    assert d["speaker"] == ""
    assert d["delayMs"] == 0
    assert d["cue"] is False


def test_api_cue_turn_is_flagged(client) -> None:  # noqa: F811
    d = client.post(
        f"/interviews/{SEED}/turns",
        json={"answer": "", "mode": "group", "askedQuestions": [Q1],
              "groupSays": _says_json([qline(), peer(FIRST, FORTY)])},
    ).json()
    assert d["cue"] is True
    assert d["speaker"] == "主考官"
    assert d["nextQuestion"] == cue_line(SECOND, turn=1)
    assert d["delayMs"] == speak_ms(FORTY) + HOST_PAUSE_MS


def test_api_peer_turn_carries_a_delay(client) -> None:  # noqa: F811
    d = client.post(
        f"/interviews/{SEED}/turns",
        json={"answer": "", "mode": "group", "askedQuestions": [Q1],
              "groupSays": _says_json([qline()])},
    ).json()
    assert d["speaker"] == FIRST
    assert d["delayMs"] >= speak_ms(qline().content) + REACTION_MIN_MS
    assert d["awaitUser"] is False
    assert d["cue"] is False


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
    assert d["cue"] is False


def test_report_leaves_cue_lines_out_of_the_collab_input(client, monkeypatch) -> None:  # noqa: F811
    """點名句是主考官的流程話,不是論點,不送進協作評分。"""
    import app.api.routes as routes

    seen: list = []

    class Spy(FakeCollabObserver):
        def observe(self, utterances):
            seen.extend(utterances)
            return super().observe(utterances)

    monkeypatch.setattr(routes, "get_collab", lambda: Spy())
    says = [qline(), peer(FIRST, "先做社群。"), cue(USER_SLOT, 1), me("我會先調查支付習慣。")]
    client.post(
        f"/interviews/{SEED}/report",
        json={"mode": "group", "turns": [{"question": Q1, "answer": "我會先調查支付習慣。"}],
              "groupSays": _says_json(says)},
    )
    texts = [u.text for u in seen]
    assert says[0].content in texts, "出題那句留著"
    assert cue_line(USER_SLOT) not in texts
    assert "我會先調查支付習慣。" in texts
