"""可評性閘門測試(問題 A / B / C)。

這一支的重點是**分界**,不是門檻值本身。門檻會因為拿到校準資料而改,
但下面這幾條分界改不了:

    薄而真的回答  要被評,拿它應得的低分
    廢話         流暢度不可以拿高分
    答不出來      最低錨點,不是未評分
    沒講話        未評分,不是低分

第一版的門檻訂在 30 字,`test_a_thin_but_real_answer_is_still_content`
就是當時打掉它的那一條。
"""

from __future__ import annotations

import pytest

from app.contracts.interview_protocols import TextStats
from app.pipeline.interview_report import (
    MIN_FLUENCY_CHARS,
    overall,
    score_fluency,
)
from app.pipeline.scorability import (
    ADMITS_UNKNOWN,
    LOWEST_ANCHOR,
    SCORABLE,
    TOO_SHORT,
    assess,
    classify,
)
from app.schemas.interview import SubScoreDTO


# ---------------------------------------------------------------------------
# 逐輪判定
# ---------------------------------------------------------------------------


def test_admitting_you_dont_know_is_scorable_not_unscorable() -> None:
    """「不知道」有回應,那是關於這項能力的觀察證據。

    判成不可評的話,全場答不知道的人會拿到一份六項全空的報告——
    他什麼都得不到。最低錨點加上說明才有資訊。
    """
    v = classify("不知道。")
    assert v.label == ADMITS_UNKNOWN
    assert v.scorable, "答不出來要進評分,只是分數固定在最低錨點"


def test_admitting_beats_too_short_when_both_match() -> None:
    """「不知道。」同時夠短也命中字面。判定順序決定他拿到低分還是空白。"""
    assert classify("不知道。").label == ADMITS_UNKNOWN
    assert classify("嗯。").label == TOO_SHORT


def test_a_thin_but_real_answer_is_still_content() -> None:
    """薄的回答要被評成薄,不可以被判成沒有內容。

    這一條打掉了第一版 30 字的門檻。那一版想用長度判斷「這是不是認真的
    回答」,但長度分不開下面這兩句,而且方向還相反:

        「我用 SQL 重寫查詢」        11 字,是真的回答
        「喔好啊這樣啊對啊我覺得還好」 14 字,是廢話

    真回答被判成不可評的代價特別高:使用者的報告少一塊,而他不會知道為什麼。
    """
    assert classify("我用 SQL 重寫查詢").label == SCORABLE


def test_a_hedge_inside_a_real_answer_is_not_an_admission() -> None:
    """字面集合是子字串比對,長回答裡的保留語會誤觸。

    這一句命中「不確定」,但它有具體數字、有對象,是個好回答。
    沒有長度上限的話它會被打到 45,而且錯得很難發現——
    分數低看起來只是「這題答得不好」。
    """
    good = "我不確定當時的確切數字,但那一季的回訪率從 18% 拉到 26%,主要是靠推播的分眾。"
    assert classify(good).label == SCORABLE


# ---------------------------------------------------------------------------
# 整場彙整
# ---------------------------------------------------------------------------


def test_a_session_with_one_bad_turn_is_still_scorable() -> None:
    """四題裡有一題亂答,整場照評。那不是 A/B/C 要解決的。"""
    sc = assess(["嗯。", "我在系學會負責行銷,把粉專追蹤數從零做到一千二。"])
    assert sc.scored
    assert not sc.anchored
    assert sc.reason() == "", "有可評的題就不需要解釋"


def test_a_session_of_nothing_but_admissions_is_anchored() -> None:
    sc = assess(["不知道。", "這個我沒想過。"])
    assert sc.scored, "答不出來是可評的"
    assert sc.anchored, "但分數固定在最低錨點"
    assert "答不出來" in sc.reason()


def test_a_session_with_no_content_is_not_scored() -> None:
    sc = assess(["嗯。", "喔。"])
    assert not sc.scored
    assert not sc.anchored
    assert "太短" in sc.reason()


def test_an_empty_session_is_not_scored() -> None:
    assert not assess([]).scored
    assert not assess(["", "   "]).scored


# ---------------------------------------------------------------------------
# A:亂答不可以贏過認真回答
# ---------------------------------------------------------------------------


def _stats(chars: int, fillers: int) -> TextStats:
    return TextStats(
        char_count=chars, filler_count=fillers, filler_reliability="measured"
    )


def test_gibberish_no_longer_outscores_a_real_answer() -> None:
    """A 的頭條。改之前實測:

        「喔好啊這樣啊對啊我覺得還好。」 14 字,0 個填充詞 → **90**
        認真回答,每百字 3.0                                →   82

    亂答高 8 分。原因是密度的分母太小——14 個字裡觀察到 0 個填充詞,
    對一個每百字 3.0 的正常講者來說完全是預期結果。
    """
    junk, _ = score_fluency(_stats(14, 0), measurable=True)
    real, _ = score_fluency(_stats(1000, 30), measurable=True)

    assert junk is None, "樣本量不足就不要給分"
    assert real is not None and real > 0
    # 原本的失敗長這樣,擋住它
    assert not (junk is not None and junk > real)


def test_the_formula_itself_was_never_the_problem() -> None:
    """刻意講亂拿 60、認真回答拿 82,公式對填充詞率的方向是對的。

    壞的是在沒有樣本的時候還硬算。所以修的是樣本量,不是 band。
    """
    normal, _ = score_fluency(_stats(1000, 30), measurable=True)
    messy, _ = score_fluency(_stats(1000, 98), measurable=True)
    assert messy < normal


def test_the_sample_floor_does_not_touch_any_real_recording() -> None:
    """黃金集十份整場 254–690 字,門檻不可以動到它們。"""
    assert MIN_FLUENCY_CHARS < 254
    shortest_real, _ = score_fluency(_stats(254, 8), measurable=True)
    assert shortest_real is not None


# ---------------------------------------------------------------------------
# 總分(G / H)
# ---------------------------------------------------------------------------


def test_the_total_only_averages_what_was_actually_scored() -> None:
    """未評分的項不進平均,也不當成 0。"""
    subs = [
        SubScoreDTO(name="內容深度", score=80),
        SubScoreDTO(name="邏輯清晰度", score=70),
        SubScoreDTO(name="表達流暢度", score=None),
    ]
    score, basis = overall(subs)
    assert basis == 2
    assert score == pytest.approx(75.0)


def test_an_unscored_item_does_not_drag_the_total_down() -> None:
    """F 的回歸測試:補 0 的時候同一組會算出 50。"""
    subs = [
        SubScoreDTO(name="內容深度", score=80),
        SubScoreDTO(name="邏輯清晰度", score=70),
        SubScoreDTO(name="表達流暢度", score=None),
    ]
    score, _ = overall(subs)
    as_if_zero = (80 + 70 + 0) / 3
    assert score > as_if_zero


def test_the_total_keeps_its_decimals() -> None:
    """G:前端取整數截斷,64.67 變 64。後端不可以把小數先丟掉。"""
    subs = [SubScoreDTO(name=n, score=v) for n, v in
            zip(("內容深度", "邏輯清晰度", "表達流暢度"), (64, 65, 65))]
    score, _ = overall(subs)
    assert score != int(score), "小數要留給前端決定顯示幾位"
    assert score == pytest.approx(64.666, abs=0.01)


def test_no_scores_means_no_total_not_a_default_number() -> None:
    """H:前端在陣列為空時填 74。後端要回 null 讓它沒得填。"""
    score, basis = overall([SubScoreDTO(name="內容深度", score=None)])
    assert score is None
    assert basis == 0


def test_the_lowest_anchor_comes_from_the_collab_scale() -> None:
    """不要自己寫一個 45。兩邊是同一個刻度,寫兩次就會分岔。"""
    from app.prompts.collab_rubric import LEVEL_TO_SCORE

    assert LOWEST_ANCHOR == LEVEL_TO_SCORE[1]
