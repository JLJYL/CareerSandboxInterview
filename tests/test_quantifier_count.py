"""Quantifier counting: articles and ordinals are not numbers.

`count_quantifiers` is a module-level function so these tests never build a
TranscriptAnalyzer (which loads bge-m3). The boundary table comes from
member B's request; the rows marked "decided" are the cases it left open.
"""

from __future__ import annotations

import glob
import json

import pytest

from app.pipeline.transcript import count_quantifiers


@pytest.mark.parametrize(
    "text, expected",
    [
        # article uses: not a quantity
        ("我做了一個專案", 0),
        ("有一次我負責", 0),
        ("一個人負責", 0),
        ("每一個人都要報告", 0),
        # durations and real counts
        ("花了一個月", 1),
        ("花了一個小時", 1),
        ("花了一個學期", 1),
        ("花了三個月", 1),
        ("做了一年", 1),
        ("一週一次", 1),
        ("十一個人", 1),
        ("一百個", 1),
        ("兩個國中生", 1),
        ("一成", 1),
        ("提升 30%", 1),
        # decided: ordinals are order, not quantity
        ("第一次做專案", 0),
        ("第二次面試", 0),
        # decided: Arabic 1 follows the same rule as 一
        ("做了1個專案", 0),
        ("做了11個專案", 1),
        ("做了1.5個月", 1),
        # decided: approximations still count
        ("做了一兩個專題", 1),
        # not a counter at all
        ("我有一點點經驗", 0),
        ("", 0),
    ],
)
def test_boundary_table(text: str, expected: int) -> None:
    assert count_quantifiers(text) == expected


def test_each_counted_phrase_counts_once() -> None:
    assert count_quantifiers("做了一個專案,花了三個月,提升 30%") == 2


def _recordings() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(glob.glob("fixtures/golden/interview/ivw-*.json")):
        case = json.load(open(path, encoding="utf-8"))
        segs = case.get("transcript_segments") or []
        out.setdefault(
            case["meta"]["recording_id"], "\n".join(s for s in segs if s.strip())
        )
    return out


def test_golden_recordings_counts_are_pinned() -> None:
    """Before the fix these were T1 2, T2 2, T3 12, T4 4, T5 3, T6 3.

    T3's remaining 7 are two real figures (四個人, 60個小時) plus five
    "兩人" that come from an STT error ("兩人位" heard as "兩人味"), a
    separate problem. Every dropped count was a "一個" or "一次" used as an
    article.
    """
    got = {rid: count_quantifiers(t) for rid, t in _recordings().items()}
    assert got == {"T1": 0, "T2": 0, "T3": 7, "T4": 0, "T5": 2, "T6": 2}
