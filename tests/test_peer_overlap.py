"""競爭者重疊度測試。純函式。

重點在**一件事**:指標選錯的話旋鈕會轉但數字不動。

`test_overlap_coefficient_cannot_tell_a_subset_apart` 就是選 Jaccard 的依據。
它不是裝飾——拿掉它之後,有人改回 overlap coefficient 不會有任何東西變紅。
"""

from __future__ import annotations

from app.pipeline.peer_overlap import (
    coverage,
    jaccard,
    mean_coverage,
    mean_pairwise_jaccard,
    overlap_coefficient,
)

# 成員 A 在 234 vocab 上手捏的三種情境。數字是示範不是門檻。
SIMILAR = [
    {"a", "b", "c", "d"},
    {"a", "b", "c", "e"},
    {"a", "b", "d", "e"},
    {"a", "b", "c", "d"},
]
DISTINCT = [
    {"a", "b"},
    {"c", "d"},
    {"e", "f"},
    {"g", "h"},
]
SUBSET = [
    {"a", "b"},
    {"a", "b", "c", "d"},
    {"a", "b", "c", "d", "e", "f"},
    {"a", "b", "c", "d", "e", "f", "g", "h"},
]


def test_jaccard_separates_the_three_shapes():
    """三種情境要給出分得開的數字,不然這個旋鈕沒用。"""
    sim = mean_pairwise_jaccard(SIMILAR)
    dis = mean_pairwise_jaccard(DISTINCT)
    sub = mean_pairwise_jaccard(SUBSET)
    assert dis == 0.0, "完全不重疊要是 0"
    assert 0.0 < sub < sim, f"子集要落在中間,拿到 子集={sub:.3f} 相近={sim:.3f}"


def test_overlap_coefficient_cannot_tell_a_subset_apart():
    """子集陷阱:A ⊆ B 時 overlap coef 回 1.0,而那是「各有強項」的一種。

    這條測試是選 Jaccard 的依據。overlap coef 把
    「B 完全包含 A 而且多了四個 A 沒有的強項」判成完全相近,
    於是生成器往「各有強項」調的時候,這個數字不會動。
    """
    a, b = {"a", "b"}, {"a", "b", "c", "d", "e", "f"}
    assert overlap_coefficient(a, b) == 1.0
    assert jaccard(a, b) < 0.4

    assert mean_pairwise_jaccard(SUBSET) < 1.0
    sub_overlap = (
        overlap_coefficient(SUBSET[0], SUBSET[1])
        + overlap_coefficient(SUBSET[0], SUBSET[2])
        + overlap_coefficient(SUBSET[0], SUBSET[3])
    ) / 3
    assert sub_overlap == 1.0, "三個都包含第一個,overlap coef 全部是 1.0"


def test_jaccard_is_symmetric_and_bounded():
    a, b = {"x", "y"}, {"y", "z", "w"}
    assert jaccard(a, b) == jaccard(b, a)
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert jaccard(a, a) == 1.0


def test_two_empty_sets_are_not_called_identical():
    """0/0 在這裡要回 0。

    回 1 的話「生成失敗,四個人都沒技能」會長得像「同質度最高」,
    而那是最糟的誤讀方向:它會讓一次失敗看起來像一次成功。
    """
    assert jaccard(set(), set()) == 0.0
    assert mean_pairwise_jaccard([set(), set(), set(), set()]) == 0.0


def test_fewer_than_two_peers_has_no_homogeneity():
    assert mean_pairwise_jaccard([]) == 0.0
    assert mean_pairwise_jaccard([{"a", "b"}]) == 0.0


def test_four_peers_means_six_pairs():
    """四個人兩兩是六對。算錯對數會讓平均值系統性偏掉。"""
    sets = [{"a"}, {"a"}, {"b"}, {"b"}]
    # 同組兩對是 1.0,跨組四對是 0.0 → 2/6
    assert abs(mean_pairwise_jaccard(sets) - 2 / 6) < 1e-9


# ---------------------------------------------------------------------------
# 壓迫度
# ---------------------------------------------------------------------------


def test_coverage_is_against_the_jd_not_the_peer():
    """分母是職缺要求,不是競爭者的技能數。

    用競爭者當分母的話,一個只寫兩個技能但都命中的人會拿到 1.0,
    而那不是「他對這個職缺很強」,是「他的履歷很短」。
    """
    required = {"a", "b", "c", "d"}
    assert coverage({"a", "b"}, required) == 0.5
    assert coverage({"a", "b", "x", "y", "z"}, required) == 0.5


def test_coverage_with_no_requirements_is_zero():
    """JD 抽取失敗時回 0,不是 1。

    回 1 會讓「沒有職缺需求」長得像「完全命中」,
    而 JD 抽取的 recall 目前是 0.375,這條路會走到。
    """
    assert coverage({"a", "b"}, set()) == 0.0
    assert mean_coverage([{"a"}, {"b"}], set()) == 0.0


def test_the_two_metrics_move_independently():
    """同質度與壓迫度要能獨立調,不然就是同一個旋鈕換個名字。

    這一組:四個人彼此完全不重疊(同質度 0),但每個人都踩滿職缺要求
    (壓迫度 1.0)。混成一個數字的話這種組合表達不出來,
    而它正是「每個人都很強但各有強項」——最難應付的那種群面。
    """
    required = {"a", "b", "c", "d", "e", "f", "g", "h"}
    peers = [{"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}]
    assert mean_pairwise_jaccard(peers) == 0.0
    assert mean_coverage(peers, required) == 0.25
    # 反過來:彼此一樣,但都只踩到一小部分
    same = [{"a"}, {"a"}, {"a"}, {"a"}]
    assert mean_pairwise_jaccard(same) == 1.0
    assert mean_coverage(same, required) == 0.125


def test_no_threshold_constants_are_declared():
    """這個模組不該有門檻常數。

    成員 A 給的 0.548 / 0.000 / 0.333 是手捏技能的示範,只證明指標會分離。
    真正的門檻要拿生成器實際輸出在 234 vocab 上重跑才能定。
    寫死一個數字等於把示範當驗收標準——1d 那輪就是這樣,期望值要從資料算。
    """
    import app.pipeline.peer_overlap as mod

    numeric = {
        k: v
        for k, v in vars(mod).items()
        if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert numeric == {}, f"出現門檻常數 {numeric},那個數字要附一次量測"
