"""多方對話切片的黃金案例回歸(成員 A,W3)。

案例放在 `fixtures/golden/collab/*.json`,每格記錄一段多方對話與
四個維度應該切出來的形狀。新增案例只要丟一個 JSON 進去,不用改這支。

## 這支測什麼、不測什麼

測**切片形狀**:幾則、標記什麼、配對配給誰、原文有沒有被動過。
這部分完全由規則決定,可以斷言。

不測**BARS 等第**:那是語意判斷,由 B 的 LLM 做,驗證它需要真實對話
加人工標記。案例裡的 `bars_labels` 目前一律 null,真實資料到位再填。

★ 合成資料只能鎖形狀。拿合成對話去驗 LLM 評分會得到一個看起來
  有在測、實際上在測自己編的預期的東西——那比沒有測試糟。

## case-05 是刻意記錄「目前行為」而非「正確行為」

主考官與 AI 同儕在 `UtteranceDTO` 上無法區分(只有 speaker 顯示名稱,
而合約明文警告不可用它判斷身分)。那一格的 `known_issue` 寫明了影響,
`expected` 記的是現行規則下的結果。欄位補上後預期值要改,
而這支測試會讓那件事非被處理不可,不會被默默忘掉。
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from app.contracts.interview_protocols import Utterance
from app.pipeline.collab import CollabObserver
from app.schemas.interview import COLLAB_DIM_NAMES

CASES_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "collab"

_QUOTE = re.compile(r"「(.+?)」")


def _load_cases() -> list[dict]:
    if not CASES_DIR.is_dir():
        return []
    out = []
    for path in sorted(CASES_DIR.glob("case-*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


CASES = _load_cases()
CASE_IDS = [c["case_id"] for c in CASES]


def _to_utterances(raw: list[dict]) -> list[Utterance]:
    return [
        Utterance(speaker_id=u["speaker"], text=u["text"], is_user=u["is_user"])
        for u in raw
    ]


def _by_name(slices, name):
    return next(s for s in slices if s.name == name)


@pytest.fixture(scope="module")
def observer():
    return CollabObserver()


def test_cases_directory_is_not_empty():
    """案例目錄空掉的話下面全部會靜默跳過,那等於沒有測試。"""
    assert CASES, f"沒有讀到任何案例,檢查 {CASES_DIR}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_slice_counts_match_expected(case, observer):
    """四個維度各自的切片則數要跟案例記錄的一致。"""
    slices = observer.observe(_to_utterances(case["utterances"]))
    for dim in COLLAB_DIM_NAMES:
        exp = case["expected"].get(dim)
        if exp is None or "count" not in exp:
            continue
        actual = len(_by_name(slices, dim).excerpts)
        assert actual == exp["count"], (
            f"{case['case_id']} 的「{dim}」預期 {exp['count']} 則,實得 {actual}"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_participation_tags_match_expected(case, observer):
    """參與主動性每則的接續/起頭標記要對。"""
    exp = case["expected"].get(COLLAB_DIM_NAMES[0], {})
    tags = exp.get("tags")
    if not tags:
        return
    excerpts = _by_name(observer.observe(_to_utterances(case["utterances"])),
                        COLLAB_DIM_NAMES[0]).excerpts
    assert len(excerpts) == len(tags)
    for i, (excerpt, want) in enumerate(zip(excerpts, tags)):
        assert f"[{want}]" in excerpt, (
            f"{case['case_id']} 第 {i + 1} 則預期標記「{want}」,實得:{excerpt!r}"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_responsiveness_pairs_with_expected_speaker(case, observer):
    """傾聽與回應每一對配到的他人,要是案例記錄的那一位。"""
    exp = case["expected"].get(COLLAB_DIM_NAMES[1], {})
    want_speakers = exp.get("paired_with")
    if not want_speakers:
        return
    excerpts = _by_name(observer.observe(_to_utterances(case["utterances"])),
                        COLLAB_DIM_NAMES[1]).excerpts
    assert len(excerpts) == len(want_speakers)
    for i, (excerpt, want) in enumerate(zip(excerpts, want_speakers)):
        assert f"（{want}）" in excerpt, (
            f"{case['case_id']} 第 {i + 1} 對預期配對「{want}」,實得:{excerpt!r}"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_empty_dims_carry_a_reason(case, observer):
    """切片為空時必須說明原因,不能只回一個空清單。

    空切片會讓 LLM 回 level 0,報告顯示「這次沒有可觀察的內容」。
    沒有 note 的話,沒有人知道是資訊不在場還是切錯了。
    """
    slices = observer.observe(_to_utterances(case["utterances"]))
    for dim in COLLAB_DIM_NAMES:
        exp = case["expected"].get(dim, {})
        if not exp.get("note_required"):
            continue
        sl = _by_name(slices, dim)
        assert not sl.excerpts
        assert sl.note.strip(), f"{case['case_id']} 的「{dim}」切片為空但沒有說明原因"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_user_text_preserved_verbatim(case, observer):
    """使用者發言的原文不可被切片層改動。

    B 的 verify_evidence 要求 evidence 是逐字稿的子字串。切片層只要
    動了一個字(修正錯字、補標點、清空白),LLM 引用的就會對不上原文,
    誠實的評分會被判成編造。
    """
    utts = _to_utterances(case["utterances"])
    user_texts = {u.text.strip() for u in utts if u.is_user and u.text.strip()}
    if not user_texts:
        return
    excerpts = _by_name(observer.observe(utts), COLLAB_DIM_NAMES[2]).excerpts
    assert set(excerpts) == user_texts, (
        f"{case['case_id']} 的論點建構切片跟原文不一致——切片層動了文字"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_all_tagged_excerpts_wrap_text_in_quotes(case, observer):
    """帶標籤的切片,真正的發言文字必須被「」包住。

    標籤(接續/起頭、講者名稱)跟原文黏在一起沒有邊界的話,
    照 prompt 指示引用原文的 LLM 可能連標籤一起抄進 evidence,
    而標籤不存在於逐字稿,會被 verify_evidence 判成編造。
    """
    slices = observer.observe(_to_utterances(case["utterances"]))
    for dim in (COLLAB_DIM_NAMES[0], COLLAB_DIM_NAMES[3]):
        for excerpt in _by_name(slices, dim).excerpts:
            assert _QUOTE.search(excerpt), (
                f"{case['case_id']} 的「{dim}」切片沒有引號邊界:{excerpt!r}"
            )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_bars_labels_are_null_for_synthetic_cases(case):
    """合成案例不可以帶 BARS 等第標記。

    合成對話的「正確等第」是編的,拿它當 ground truth 會讓評分測試
    測到自己編的預期。真實錄音 + 人工標記的案例才可以填 bars_labels,
    而那種案例的 meta.source 要標 real。
    """
    if case["meta"].get("source") == "synthetic":
        assert case.get("bars_labels") is None, (
            f"{case['case_id']} 是合成案例,不該有 BARS 標記"
        )


def test_known_issue_cases_document_the_fix_condition():
    """記錄已知限制的案例,必須寫明修好之後預期值要怎麼改。

    不然欄位補上時沒有人知道這一格的預期值是「刻意記錄的錯誤行為」,
    會以為它是對的而照著保留。
    """
    for case in CASES:
        issue = case.get("known_issue")
        if not issue:
            continue
        for key in ("summary", "root_cause", "impact", "when_fixed"):
            assert issue.get(key), f"{case['case_id']} 的 known_issue 缺少 {key}"
