"""分數區分度驗收(階段一第五步)。

材料是前端組的「個人面試評分範例集」,四題 × 四個品質等級。
他們量到的問題是**順序不對**:

    離題 65 > 亂答 64 > 拒答 59,全距只有 18

這一支驗順序有沒有修好,以及已知還沒修好的是哪幾格。

【只驗公式那兩項】
六項裡四項是 LLM 評的,測試不打 LLM。所以這裡驗的是內容深度與表達流暢度,
加上可評性閘門。LLM 那四項的區分度要實跑才量得到。

【範例的限制寫在 fixture 裡】
十六筆只有兩筆是實測,其餘是寫出來補齊等級的。寫的有標點、沒有填充詞,
所以表達那一面這一份驗不了——那要等 Whisper 錄音。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.contracts.interview_protocols import TextStats
from app.pipeline.interview_report import score_depth, score_fluency
from app.pipeline.scorability import LOWEST_ANCHOR, assess

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scoring" / "answer_levels.json"


@pytest.fixture(scope="module")
def levels() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _stats(answer: dict) -> TextStats:
    """讀 fixture 裡烤好的統計,不在這裡重算。

    【為什麼烤進 fixture】
    第一版在這裡用正則重寫了一份量化詞計數,題目 2 的「好」算出 8 處而
    `text_stats` 算的是 3 處。測試量的跟上線的不是同一件事,而那個差距
    讓分數看起來高了一整個帶。計數只能有一個來源。

    不直接呼叫 `text_stats` 是因為 `TranscriptAnalyzer` 會載 bge-m3(2.3GB)。
    規則改了就跑 `tools/refresh_answer_levels.py` 重烤。

    填充詞恆為 0:這一份範例全是寫出來的,本來就沒有。
    """
    return TextStats(
        char_count=answer["char_count"],
        filler_count=0,
        quantifier_count=answer["quantifier_count"],
        filler_reliability="measured",
    )


def _depth(answer: dict) -> int | None:
    """一筆範例的內容深度。閘門擋下來時回 None。"""
    gate = assess(answer["text"].split("\n"))
    if not gate.scored:
        return None
    if gate.anchored:
        return LOWEST_ANCHOR
    return score_depth(_stats(answer), answer["star"])[0]


def _by_level(levels: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {lv: [] for lv in levels["levels"]}
    for case in levels["cases"]:
        for lv in levels["levels"]:
            got = _depth(case["answers"][lv])
            if got is not None:
                out[lv].append(got)
    return out


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# 他們回報的問題
# ---------------------------------------------------------------------------


def test_the_levels_come_out_in_the_right_order(levels) -> None:
    """前端量到的是離題 65 > 拒答 59。順序反了。

    這一條是那份範例集存在的理由,排在最前面。
    """
    m = {lv: _mean(v) for lv, v in _by_level(levels).items()}

    assert m["good"] > m["fair"] > m["off_topic"] >= m["refusal"], (
        f"順序不對:{ {k: round(v, 1) for k, v in m.items()} }"
    )


def test_good_and_fair_are_far_apart(levels) -> None:
    """改之前最差的回答也有 50–60 分,好跟普通擠在一起。"""
    m = {lv: _mean(v) for lv, v in _by_level(levels).items()}
    assert m["good"] - m["fair"] >= 25


def test_good_answers_land_in_the_expected_band(levels) -> None:
    """前端給的預期:好 80–90。四題裡至少三題要進去。"""
    lo, hi = levels["expected_overall"]["good"]
    got = _by_level(levels)["good"]
    inside = [v for v in got if lo <= v <= hi]
    assert len(inside) >= 3, f"好的四題:{got},預期 {lo}–{hi}"


# ---------------------------------------------------------------------------
# 已知還沒修好的,釘住
# ---------------------------------------------------------------------------


def test_off_topic_is_still_scored_too_high(levels) -> None:
    """離題壓不下來,因為公式看得到結構、看不到內容在講什麼。

    題目 1 的離題範例有情境有行動(前端記錄模型把 S 與 A 標成有),
    只是講的是旅行。內容深度拿到跟「普通」一樣的 56。

    要分開得先知道切不切題,而離題偵測卡在逐字稿沒有標點——
    句子分派只收真句末標點切出來的句子。見 PLAN 第三節 J。

    **這一條在離題偵測做完之後會變紅。** 那時候把它刪掉。
    """
    lo, hi = levels["expected_overall"]["off_topic"]
    got = _by_level(levels)["off_topic"]
    assert max(got) > hi, "如果這條綠了代表離題偵測做好了,刪掉這條"


def test_a_refusal_can_outscore_an_off_topic_answer(levels) -> None:
    """明確說不知道拿最低錨點 45,而離題落在地板 40。

    所以題目 3 的拒答(45)比離題(40)高。

    這不完全是壞事——「我不知道」比硬掰一段不相干的誠實,
    而最低錨點的語意是「評過了,幾乎沒表現」。但前端的預期是
    拒答 < 離題,兩邊要先談好哪一個才對,不要默默各做各的。
    """
    per_case = []
    for case in levels["cases"]:
        r = _depth(case["answers"]["refusal"])
        o = _depth(case["answers"]["off_topic"])
        if r is not None and o is not None:
            per_case.append((r, o))
    assert any(r > o for r, o in per_case), "如果這條綠了代表刻度改過了"


def test_star_depth_under_scores_non_behavioural_questions(levels) -> None:
    """自我介紹與動機題沒有 STAR 結構,所以內容深度被壓低。

    題目 3(自我介紹)的「好」只拿 75,題目 4(動機)的「普通」掉到地板 40
    跟離題同分——不是因為答得差,是因為那兩題本來就不是行為題。

    這是 STAR 當主軸的已知代價,寫在 `STAR_BANDS` 的註解裡。
    要修的話得讓內容深度知道這一題是不是行為題,而題型目前只在
    `InterviewContext.type` 上(整場一個值),不是逐題的。
    """
    behavioural = [_depth(levels["cases"][i]["answers"]["good"]) for i in (0, 1)]
    other = [_depth(levels["cases"][i]["answers"]["good"]) for i in (2, 3)]
    assert min(behavioural) > min(other), "如果這條綠了代表逐題題型做好了"


# ---------------------------------------------------------------------------
# 範例本身的限制
# ---------------------------------------------------------------------------


def test_written_examples_cannot_test_the_delivery_dimension(levels) -> None:
    """寫出來的範例沒有填充詞,所以表達流暢度量不到真實的口語習慣。

    這一條不是在驗分數,是在釘住 fixture 的限制——免得有人拿這一份
    去宣稱表達那一面也驗過了。
    """
    got = [
        score_fluency(_stats(case["answers"][lv]), True)[0]
        for case in levels["cases"]
        for lv in levels["levels"]
    ]
    scored = [v for v in got if v is not None]

    # 夠長的那幾筆會拿到分數,但全部是同一個分數——填充詞恆為 0,
    # 所以每一筆都落在最高帶。好的回答跟亂答在這個維度上分不開。
    assert scored, "全部未評分的話這條就沒驗到東西"
    assert len(set(scored)) == 1, (
        f"寫出來的範例沒有填充詞,分數應該全部相同,實際拿到 {sorted(set(scored))}"
    )


def test_the_star_labels_are_marked_as_unreviewed(levels) -> None:
    """STAR 段數直接決定內容深度,標錯會直接變成測試通過或不通過。

    黃金集的慣例是 A-draft + B-review 兩個人標。這一份只有一半,
    所以 provenance 要寫明,不要讓之後的人以為它複核過了。
    """
    p = levels["_provenance"]
    assert p["star_double_labeled"] is False
    assert "B-draft" in p["star_labeled_by"]
