"""競爭者彼此有多像,以及他們對職缺的壓迫度。

【為什麼用 Jaccard 不用 overlap coefficient】
overlap coefficient 是 |A∩B| / min(|A|,|B|),它有子集陷阱:
只要 A ⊆ B 就等於 1.0,不管 B 多了幾個 A 沒有的強項。

而「各有強項」正是這個旋鈕要做出來的那一端。成員 A 在 234 版上用真技能跑的對照:

                            條件相近   各有強項   子集 A⊆B
    mean pairwise Jaccard      0.548     0.000     0.333   ✓
    mean pairwise overlap      0.700     0.000     1.000   ✗

子集那一欄就是結論。`A ⊆ B` 是「各有強項」的一種(B 多了一堆),
overlap coef 把它判成完全相近。選錯指標的話旋鈕會轉但數字不動。

【這兩個指標問的是不同的問題】
    同質度    競爭者彼此有多像        mean pairwise Jaccard
    壓迫度    競爭者對這個職缺多強    coverage = |C ∩ R| / |R|

同質度高 → 四個人看起來都差不多,使用者感受到的是「大家都一樣」。
壓迫度高 → 每個人都踩在職缺要的能力上,使用者感受到的是「他們都很強」。
兩者可以獨立調,混成一個數字就調不動。

【門檻要拿真輸出重跑】
上面那組數字是成員 A 手捏競爭者技能的示範,只證明指標會分離,
不是我們的生成器會落在哪。真正的 homogeneous / heterogeneous 門檻
要拿生成器實際輸出在 234 vocab 上重跑 `mean_pairwise_jaccard` 才能定。

這跟 1d 的處境一樣:期望值要從資料算,不能寫死。所以這個檔**沒有門檻常數**——
有人要加 `HOMOGENEOUS_MIN = 0.5` 的時候,那個數字必須附一次量測。
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """|A∩B| / |A∪B|。兩邊都空的時候回 0.0。

    兩邊都空回 0 而不是 1:數學上 0/0 未定義,而在這裡「兩個都沒有技能」
    不該被報成「完全相同」——那會讓生成失敗看起來像同質度最高。
    """
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def mean_pairwise_jaccard(sets: Sequence[Iterable[str]]) -> float:
    """同質度指數:兩兩 Jaccard 再平均。四個人是六對。

    少於兩個人回 0.0——一個人沒有「彼此多像」可言。
    """
    pairs = list(combinations([set(s) for s in sets], 2))
    if not pairs:
        return 0.0
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def coverage(peer: Iterable[str], required: Iterable[str]) -> float:
    """壓迫度:這個競爭者踩到職缺要求的幾成。|C ∩ R| / |R|。

    職缺沒有要求任何技能時回 0.0:沒有 R 就沒有「踩到幾成」,
    而回 1.0 會讓「JD 抽取失敗」長得像「完全命中」。
    """
    r = set(required)
    return len(set(peer) & r) / len(r) if r else 0.0


def mean_coverage(peers: Sequence[Iterable[str]], required: Iterable[str]) -> float:
    """一組競爭者對職缺的平均壓迫度。"""
    if not peers:
        return 0.0
    return sum(coverage(p, required) for p in peers) / len(peers)


def overlap_coefficient(a: Iterable[str], b: Iterable[str]) -> float:
    """|A∩B| / min(|A|,|B|)。**不要拿它當同質度指標。**

    留在這裡只為了讓那個子集陷阱可以被測試證明,不是給生成端用的。
    `test_peer_overlap.py` 用它跟 Jaccard 對照,證明 A ⊆ B 時它回 1.0
    而 Jaccard 不會——那條測試是這個選擇的依據,不是裝飾。
    """
    sa, sb = set(a), set(b)
    smaller = min(len(sa), len(sb))
    return len(sa & sb) / smaller if smaller else 0.0
