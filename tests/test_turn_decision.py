"""決策步驟測試。不打真 LLM。

重點在**三件事**:

    程式能決定的不呼叫模型   forced_decision 命中時 llm 不該被碰到
    結構判斷不問模型         三次呼叫只問「這段話裡有什麼」,不問要不要追問
    每一條失敗路徑都有降級   Panfilova 實測 DeepSeek 三場累積 45 次
                             schema 錯誤,判讀失敗是常態不是例外

2d 之後模型挑動作那條路刪掉了,所以這個檔案裡也沒有解析動作、
修正 gap、檢查 prompt 文字那幾節——那些驗的行為已經不存在。
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
from app.pipeline.interviewer_spec import GAP_VALUES
from app.pipeline.turn_decision import (
    CLARIFY,
    DECISION_SOURCES,
    FOLLOW_UP,
    MAX_SUPERSEDED_CHARS,
    NEXT,
    TurnDecision,
    decide_from_facts,
    decide_turn,
    fallback_decision,
    forced_decision,
)

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


def test_truncated_beats_depth_cap():
    """被打斷不是他的問題,不該因為追過三次就不讓他講完。"""
    d = forced_decision(spec=spec_of("困難"), answer=GOOD_ANSWER, depth=9, ended_by="timeout")
    assert d is not None and d.action == FOLLOW_UP


def test_depth_cap_follows_difficulty():
    """新手 1 次就收,困難可以追到 3 次。"""
    assert forced_decision(spec=spec_of("新手"), answer=GOOD_ANSWER, depth=1).action == NEXT
    assert forced_decision(spec=spec_of("困難"), answer=GOOD_ANSWER, depth=1) is None


# ---------------------------------------------------------------------------
# 被取代的輸出要留著
# ---------------------------------------------------------------------------


def test_fallback_keeps_what_it_replaced():
    """光看 source=fallback 只知道失敗了,不知道失敗長什麼樣。"""
    d = fallback_decision("判讀沒完成。", "讀到的是這些")
    assert d.action == NEXT
    assert d.source == "fallback"
    assert d.superseded == "讀到的是這些"


def test_superseded_is_truncated():
    d = fallback_decision("x", "壞" * 900)
    assert 0 < len(d.superseded) <= MAX_SUPERSEDED_CHARS


def test_a_clean_decision_supersedes_nothing():
    """沒有取代任何東西的決定,superseded 要是空的——有值表示有東西被丟掉。"""
    facts = AnswerFacts.from_label(LADDER[4])
    assert decide_from_facts(ACCEPT_RULES["困難"], facts).superseded == ""


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
    assert seen == set(DECISION_SOURCES), (
        "宣告出來的每一種來源都要有生產者。多了表示那條路不存在卻列在值域裡"
        "(報表會印一個永遠是 0 的欄位);少了表示有一條路沒被宣告,報表會漏掉它。"
    )


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

    這是刻意留的缺口(理由寫在 DECISION_SOURCES 的 docstring)。
    釘住它是為了之後補上來的時候有一條測試會紅,而不是靠人記得。
    """
    seen = {
        decide_from_facts(ACCEPT_RULES[d], AnswerFacts.from_label(lab)).action
        for d in ACCEPT_RULES
        for lab in LADDER
    }
    assert CLARIFY not in seen
    assert seen <= {NEXT, FOLLOW_UP}


# ---------------------------------------------------------------------------
# is_follow_up
# ---------------------------------------------------------------------------


def test_is_follow_up_covers_clarify():
    """CLARIFY 沒有換題,對前端來說一樣是追問。"""
    assert TurnDecision(FOLLOW_UP).is_follow_up
    assert TurnDecision(CLARIFY).is_follow_up
    assert not TurnDecision(NEXT).is_follow_up
