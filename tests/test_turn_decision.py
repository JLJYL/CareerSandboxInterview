"""決策步驟測試。不打真 LLM。

重點在**兩件事**:

    程式能決定的不呼叫模型   forced_decision 命中時 llm 不該被碰到
    每一條失敗路徑都有降級   Panfilova 實測 DeepSeek 三場累積 45 次
                             schema 錯誤,解析失敗是常態不是例外
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.contracts.interview_protocols import JDInput
from app.pipeline.interviewer_spec import ACCEPT_RULES, derive_specs
from app.pipeline.star_read import AnswerFacts
from app.pipeline.transcript import segment_text
from app.prompts.star_read import FACT_QUESTION
from app.pipeline.turn_decision import (
    CLARIFY,
    DECISION_SOURCES,
    decide_from_facts,
    FOLLOW_UP,
    MAX_SUPERSEDED_CHARS,
    NEXT,
    TurnDecision,
    decide_turn,
    fallback_decision,
    forced_decision,
    normalize_action,
    normalize_gap,
    parse_decision,
)
from app.prompts.turn_decision import GAP_VALUES
from app.prompts.turn_decision import compose_decision_prompt, render_accept_rule

JD = JDInput(required_skills=["Python", "資料分析"], description="…")
GOOD_ANSWER = "那個專案我負責報表自動化,把每週三小時的人工整理縮到二十分鐘。"

_LADDER_RAW = json.loads(
    Path("fixtures/golden/decision/answers_v1.json").read_text(encoding="utf-8")
)["answers"]

LADDER: tuple[dict, ...] = tuple(a["label"] for a in _LADDER_RAW)
"""階梯七筆的標註。拿真的那組當事實,不要在測試裡自己編一組。"""

RICH_ANSWER: str = next(a["text"] for a in _LADDER_RAW if a["answer_id"] == "A5")
"""四段齊全的那一筆。多句,所以句子分派表達得出四個標籤。"""


def spec_of(difficulty: str = "中等", jd: JDInput | None = JD):
    return derive_specs("single", jd=jd, difficulty=difficulty)[0]


class Recorder:
    """記錄有沒有被呼叫,以及拿到什麼 prompt。"""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


# ---------------------------------------------------------------------------
# 程式能決定的不問模型
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"answer": ""}, NEXT),
        ({"answer": "  "}, NEXT),
        ({"answer": GOOD_ANSWER, "ended_by": "timeout"}, FOLLOW_UP),
        ({"answer": GOOD_ANSWER, "depth": 2}, NEXT),
        ({"answer": "這個我沒想過耶"}, NEXT),
    ],
)
def test_forced_cases(kwargs, expected):
    d = forced_decision(spec=spec_of(), **kwargs)
    assert d is not None
    assert d.action == expected
    assert d.source == "code"
    assert d.why


def test_normal_answer_is_not_forced():
    assert forced_decision(spec=spec_of(), answer=GOOD_ANSWER) is None


async def test_forced_decision_does_not_call_the_model():
    llm = Recorder()
    d = await decide_turn(spec=spec_of(), answer="", llm=llm)
    assert d.source == "code"
    assert llm.calls == []


async def test_normal_answer_does_call_the_model():
    """prompt 路徑:一次呼叫,動作由模型挑。這是 1d 量到的那條路。"""
    llm = Recorder(json.dumps({"why": "沒有數字", "action": "FOLLOW_UP", "gap": "結果沒量化"}))
    d = await decide_turn(
        spec=spec_of(), answer=GOOD_ANSWER, llm=llm, structure_from="prompt"
    )
    assert len(llm.calls) == 1
    assert d.action == FOLLOW_UP
    assert d.gap == "結果沒量化"
    assert d.source == "llm"


def test_truncated_beats_depth_cap():
    """被打斷不是他的問題,不該因為追過三次就不讓他講完。"""
    d = forced_decision(spec=spec_of("困難"), answer=GOOD_ANSWER, depth=9, ended_by="timeout")
    assert d is not None and d.action == FOLLOW_UP


def test_depth_cap_follows_difficulty():
    """新手 1 次就收,困難可以追到 3 次。"""
    assert forced_decision(spec=spec_of("新手"), answer=GOOD_ANSWER, depth=1).action == NEXT
    assert forced_decision(spec=spec_of("困難"), answer=GOOD_ANSWER, depth=1) is None


# ---------------------------------------------------------------------------
# 解析與降級
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NEXT", NEXT),
        ("next", NEXT),
        ("follow-up", FOLLOW_UP),
        ("FOLLOW UP", FOLLOW_UP),
        ("followup", FOLLOW_UP),
        ("追問", FOLLOW_UP),
        ("clarify", CLARIFY),
        ("澄清", CLARIFY),
        ("隨便", None),
        ("", None),
    ],
)
def test_normalize_action(raw, expected):
    assert normalize_action(raw) == expected


def test_parse_accepts_fenced_json():
    raw = '```json\n{"why":"完整","action":"NEXT","gap":""}\n```'
    d = parse_decision(raw)
    assert d is not None and d.action == NEXT


def test_parse_salvages_bare_action():
    """沒包 JSON 但整段就是一個動作名——決定本身是好的,不要丟掉。"""
    d = parse_decision("FOLLOW_UP")
    assert d is not None and d.action == FOLLOW_UP and d.source == "llm"


@pytest.mark.parametrize(
    "why",
    [
        "他講了結果但沒有可查證的數字,無法判斷成果規模。",
        "缺少量化,無法確認改善幅度。",
        "沒有交代取捨,無法看出判斷依據。",
        "很遺憾他完全沒提到結果段。",
    ],
)
def test_reasoning_may_contain_refusal_words(why):
    """「無法」是中文常用詞,不是拒絕。

    1d 第一次實測:拒絕詞原本掃解析前的整段原文,中等難度 60 次決策有 13 次
    因此降級,追問率從期望的 0.500 掉到 0.283,比新手還低。

    而且方向是偏的——會寫「無法」的句子剛好都是在說少了哪一項,
    所以被破壞的一律是 FOLLOW_UP。
    """
    raw = json.dumps({"why": why, "action": "FOLLOW_UP", "gap": "量化"}, ensure_ascii=False)
    d = parse_decision(raw)
    assert d is not None
    assert d.action == FOLLOW_UP
    assert d.source == "llm"
    assert d.why == why


def test_parse_rejects_apology():
    """整段是道歉、沒有可用動作時才擋。"""
    assert parse_decision("抱歉,我無法判斷這個回答。") is None
    assert parse_decision("作為一個語言模型,我不便評論這段回答。") is None


def test_apology_is_not_salvaged_into_an_action():
    """純文字救援不能把夾在道歉裡的動作名撿回來。"""
    assert parse_decision("抱歉,FOLLOW_UP") is None


def test_parse_rejects_unknown_action():
    assert parse_decision(json.dumps({"action": "PROBE_HARDER"})) is None


def test_gap_only_kept_for_follow_up():
    raw = json.dumps({"action": "NEXT", "gap": "結果段", "why": "x"})
    d = parse_decision(raw)
    assert d is not None and d.gap == ""


async def test_unparseable_output_degrades_to_next():
    d = await decide_turn(
        spec=spec_of(), answer=GOOD_ANSWER, llm=Recorder("??? ###"), structure_from="prompt"
    )
    assert d.action == NEXT
    assert d.source == "fallback"
    assert d.why


async def test_llm_exception_degrades_to_next():
    def boom(system: str, user: str) -> str:
        raise RuntimeError("boom")

    d = await decide_turn(
        spec=spec_of(), answer=GOOD_ANSWER, llm=boom, structure_from="prompt"
    )
    assert d.action == NEXT
    assert d.source == "fallback"
    assert "RuntimeError" in d.why


# ---------------------------------------------------------------------------
# gap 是封閉集合
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", GAP_VALUES)
def test_every_closed_value_survives_parsing(value):
    raw = json.dumps({"why": "x", "action": "FOLLOW_UP", "gap": value}, ensure_ascii=False)
    d = parse_decision(raw)
    assert d is not None and d.gap == value


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("結果段", "缺結果"),
        ("成果", "缺結果"),
        ("量化", "結果沒量化"),
        ("量化數據", "結果沒量化"),
        ("取捨", "沒有取捨"),
        ("放棄了什麼", "沒有取捨"),
        ("情境段", "缺情境"),
        ("「缺任務」", "缺任務"),
    ],
)
def test_normalize_gap_maps_what_the_model_actually_writes(raw, expected):
    assert normalize_gap(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "缺少結果段和量化數據",
        "結果段、量化、取捨",
        "任務段和結果段",
        "他的角色不清楚",
        "",
        "   ",
    ],
)
def test_ambiguous_or_unreadable_gap_becomes_empty(raw):
    """指向兩個以上互斥的事實就放棄,不要猜一個。

    「缺少結果段和量化數據」同時指 `缺結果` 與 `結果沒量化`,而結果到底在不在
    這句話答不出來。猜一個會把不確定變成確定,而 1d 實測就是這個欄位指錯
    20/20 次。留空讓報表看得出模型沒照封閉集合填。
    """
    assert normalize_gap(raw) == ""


def test_missing_result_and_unquantified_result_are_different_values():
    """這兩個混在一起,下一句就會去問他已經講過的東西。"""
    assert normalize_gap("缺結果") != normalize_gap("結果沒量化")
    assert "缺結果" in GAP_VALUES and "結果沒量化" in GAP_VALUES


def test_prompt_lists_exactly_the_closed_set():
    """選項文字從 GAP_VALUES 生出來,改了清單 prompt 跟著動。"""
    p = compose_decision_prompt(spec_of())
    for v in GAP_VALUES:
        assert v in p
    assert "不要列兩個以上" in p


# ---------------------------------------------------------------------------
# 被取代的輸出要留著
# ---------------------------------------------------------------------------


async def test_fallback_keeps_what_it_replaced():
    """光看 source=fallback 只知道失敗了,不知道失敗長什麼樣。"""
    broken = "我覺得這個回答還可以再補充一點細節。"
    d = await decide_turn(
        spec=spec_of(), answer=GOOD_ANSWER, llm=Recorder(broken), structure_from="prompt"
    )
    assert d.source == "fallback"
    assert d.superseded == broken


async def test_superseded_is_truncated():
    d = await decide_turn(
        spec=spec_of(), answer=GOOD_ANSWER, llm=Recorder("壞" * 900), structure_from="prompt"
    )
    assert d.source == "fallback"
    assert 0 < len(d.superseded) <= MAX_SUPERSEDED_CHARS


def test_every_gap_a_decision_can_carry_is_declared():
    """gap 的值域只有一個來源。集合外的值跑出去,下游沒辦法照清單處理。

    `forced_decision` 的 timeout 那條原本填「未講完的部分」,不在集合裡。
    """
    allowed = set(GAP_VALUES) | {""}
    seen = {
        forced_decision(spec=spec_of(), answer="").gap,
        forced_decision(spec=spec_of(), answer=GOOD_ANSWER, ended_by="timeout").gap,
        forced_decision(spec=spec_of("新手"), answer=GOOD_ANSWER, depth=1).gap,
        forced_decision(spec=spec_of(), answer="這個我沒想過耶").gap,
        fallback_decision("x").gap,
    }
    seen |= {
        decide_from_facts(ACCEPT_RULES[d], AnswerFacts.from_label(lab)).gap
        for d in ACCEPT_RULES
        for lab in LADDER
    }
    assert seen <= allowed, sorted(seen - allowed)


def test_every_source_is_declared():
    seen = {
        TurnDecision(NEXT).source,
        forced_decision(spec=spec_of(), answer="").source,
        fallback_decision("x").source,
        decide_from_facts(ACCEPT_RULES["困難"], AnswerFacts.from_label(LADDER[0])).source,
        decide_from_facts(ACCEPT_RULES["新手"], AnswerFacts()).source,
    }
    assert seen <= set(DECISION_SOURCES)
    assert len(seen) >= 4, "四種來源應該都走得到,少一種表示有一條路不會發生"


def test_a_clean_decision_supersedes_nothing():
    raw = json.dumps({"why": "x", "action": "NEXT", "gap": ""}, ensure_ascii=False)
    d = parse_decision(raw)
    assert d is not None and d.superseded == ""


# ---------------------------------------------------------------------------
# 結構判斷交給程式
# ---------------------------------------------------------------------------


def facts_llm(label: dict, answer: str = RICH_ANSWER):
    """照標註回答事實判讀的三次呼叫。分派與單題都要認得。

    答案一定要是多句的:分派一句只掛一個標籤,所以單句的答案最多只讀出一段。
    `GOOD_ANSWER` 只有一句,拿它當這裡的輸入會量到「S 以外都沒有」
    然後誤以為是程式錯了——我第一次就是這樣寫的。
    """

    def call(system: str, user: str) -> str:
        if "把每一句分派到一個標籤" in system:
            sentences, _ = segment_text(answer, level="sentence")
            order = [f for f in ("S", "T", "A", "R") if f in label["star_parts"]]
            assert len(sentences) >= len(order), "句數不夠,分派表達不出這組標註"
            return json.dumps(
                {
                    str(i + 1): (order[i] if i < len(order) else "其他")
                    for i in range(len(sentences))
                },
                ensure_ascii=False,
            )
        for fact in ("quantified", "tradeoff"):
            if FACT_QUESTION[fact] in system:
                has = bool(label[fact])
                return json.dumps(
                    {"present": has, "quote": answer[:12] if has else ""},
                    ensure_ascii=False,
                )
        raise AssertionError(f"假模型認不得這個 prompt:{system[:40]}")

    return call


async def test_facts_path_asks_only_about_facts():
    """三次呼叫,全部是在問「這段話裡有什麼」,沒有一次在問「要不要追問」。

    這是 2c 的重點:放行條件不再進 prompt。1d 實測它進了也沒生效。
    """
    llm = Recorder()
    llm_calls: list[str] = []

    def spy(system: str, user: str) -> str:
        llm_calls.append(system)
        return facts_llm(LADDER[4])(system, user)

    d = await decide_turn(spec=spec_of(), answer=RICH_ANSWER, llm=spy)
    assert len(llm_calls) == 3
    assert not any("什麼時候算講完了" in s for s in llm_calls)
    assert not any(a in s for s in llm_calls for a in ("NEXT", "FOLLOW_UP"))
    assert d.source == "facts"
    assert llm.calls == []


async def test_facts_path_lets_difficulty_decide():
    """同一段回答,三種難度要給出不同的決定。

    1d 量到的失敗就是這一條不成立:三種難度的追問率都是 0.500。
    A4 是中等 / 困難的分界——三段加結果加數字,中等全滿足,困難還要四段跟取捨。
    """
    a4 = {"star_parts": ["S", "A", "R"], "quantified": True, "tradeoff": False}
    got = {}
    for diff in ("新手", "中等", "困難"):
        d = await decide_turn(spec=spec_of(diff), answer=RICH_ANSWER, llm=facts_llm(a4))
        got[diff] = d.action
    assert got["新手"] == NEXT
    assert got["中等"] == NEXT
    assert got["困難"] == FOLLOW_UP


async def test_the_gap_is_computed_not_asked():
    """A3:講了結果只是沒數字。模型在 1d 裡 20/20 次寫「缺結果」。

    算出來的是「結果沒量化」,而且這一輪模型根本沒被問到缺口是什麼。
    """
    a3 = {"star_parts": ["S", "A", "R"], "quantified": False, "tradeoff": False}
    d = await decide_turn(spec=spec_of("中等"), answer=RICH_ANSWER, llm=facts_llm(a3))
    assert d.action == FOLLOW_UP
    assert d.gap == "結果沒量化"
    assert d.source == "facts"


async def test_an_incomplete_read_degrades_and_keeps_what_it_read():
    """判讀沒讀完 → 不拿不完整的事實下判斷。

    「四段都沒讀到」與「讀出來四段都沒有」在 accepts() 眼裡一模一樣,
    所以這條路徑一定要標成 fallback,而且把讀到的部分留在 superseded。
    """

    def half(system: str, user: str) -> str:
        if "把每一句分派到一個標籤" in system:
            raise RuntimeError("timeout")
        return json.dumps({"present": True, "quote": RICH_ANSWER[:12]}, ensure_ascii=False)

    d = await decide_turn(spec=spec_of("困難"), answer=RICH_ANSWER, llm=half)
    assert d.action == NEXT
    assert d.source == "fallback"
    assert d.superseded


def test_decide_from_facts_is_pure():
    """同樣的事實加同樣的條件,永遠同一個決定。不碰模型。"""
    facts = AnswerFacts.from_label(LADDER[3])
    rule = ACCEPT_RULES["困難"]
    first = decide_from_facts(rule, facts)
    assert decide_from_facts(rule, facts) == first


def test_why_says_what_it_read_and_what_is_missing():
    """決定變成程式算之後,why 是事實摘要加缺口,不是模型的說法。

    報告逐題講評與人工標注看的都是這一句,空的或含糊的話那兩件事就做不了。
    """
    d = decide_from_facts(ACCEPT_RULES["困難"], AnswerFacts.from_label(LADDER[5]))
    assert d.action == FOLLOW_UP
    assert "取捨" in d.why
    assert d.gap in d.why


def test_the_facts_path_never_says_clarify():
    """facts 路徑沒有 CLARIFY——那需要問模型,而那次呼叫還沒做。

    這是刻意留的缺口(見 STRUCTURE_SOURCES)。釘住它是為了之後補上來的時候
    有一條測試會紅,而不是靠人記得。
    """
    seen = {
        decide_from_facts(ACCEPT_RULES[d], AnswerFacts.from_label(lab)).action
        for d in ACCEPT_RULES
        for lab in LADDER
    }
    assert CLARIFY not in seen
    assert seen <= {NEXT, FOLLOW_UP}


async def test_an_unknown_structure_source_is_refused():
    """認不得的值要炸,不要默默走其中一條——那會讓量測以為量的是另一條。"""
    with pytest.raises(ValueError):
        await decide_turn(
            spec=spec_of(), answer=GOOD_ANSWER, llm=Recorder(), structure_from="code"
        )


# ---------------------------------------------------------------------------
# is_follow_up
# ---------------------------------------------------------------------------


def test_is_follow_up_covers_clarify():
    """CLARIFY 沒有換題,對前端來說一樣是追問。"""
    assert TurnDecision(FOLLOW_UP).is_follow_up
    assert TurnDecision(CLARIFY).is_follow_up
    assert not TurnDecision(NEXT).is_follow_up


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def test_accept_rule_text_comes_from_the_dataclass():
    """條件只有一個來源。改了 dataclass,prompt 文字跟著動。"""
    easy = render_accept_rule(spec_of("新手").accept)
    hard = render_accept_rule(spec_of("困難").accept)
    assert "至少要交代 2 段" in easy
    assert "至少要交代 4 段" in hard
    assert "取捨" not in easy
    assert "取捨" in hard


def test_prompt_states_when_there_is_no_jd():
    """沒有 JD 時要明說,不然模型會自行想像這個職位需要什麼。"""
    p = compose_decision_prompt(spec_of(jd=None))
    assert "沒有職缺需求可用" in p
    assert "不要自行想像" in p


def test_prompt_adapts_to_transcription_engine():
    """兩種引擎的失真方式相反,CLARIFY 的判準要跟著反過來。"""
    api = compose_decision_prompt(spec_of(), transcription_engine="api")
    dev = compose_decision_prompt(spec_of(), transcription_engine="device")
    assert "不漏字" in api
    assert "會漏字" in dev
    assert api != dev


def test_prompt_has_no_persona():
    """決策不帶 persona——prompt 一長人格就塌,派發拆兩步就是為了這件事。"""
    p = compose_decision_prompt(spec_of())
    for name in ("AI-強勢", "AI-邏輯", "HR 主管", "技術主管"):
        assert name not in p
