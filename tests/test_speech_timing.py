"""段落時間特徵測試。純函式,不打 LLM。

重點在**兩件事**:

    算得出來的要精確   停頓是 starts[i+1] - ends[i],不是估計
    算不出來的要說算不出來   資料不足或不自洽一律回 unavailable,不回估計值

第二件比第一件重要。「平均每段 76 字」被當成精確句長用過一次,
而它其實是「她停頓前講了多長」——過度宣稱比不宣稱糟。
"""

from __future__ import annotations

from app.pipeline.speech_timing import (
    MAX_REASONABLE_PAUSE_MS,
    MIN_SEGMENT_MS,
    TIMESTAMPS,
    TIMING_SOURCES,
    UNAVAILABLE,
    SpeechTiming,
    describe_pauses,
    measure,
)

SEGS = ["那個專案原本都是人工整理", "我用 SQL 重寫查詢", "每週三小時降到二十分鐘"]


def voice(starts, ends, segments=None):
    return measure(
        starts_ms=starts,
        ends_ms=ends,
        segments=segments if segments is not None else SEGS[: len(ends)],
        input_mode="voice",
    )


# ---------------------------------------------------------------------------
# 算得出來的
# ---------------------------------------------------------------------------


def test_pause_is_the_gap_between_one_end_and_the_next_start():
    t = voice([0, 5200, 12700], [4000, 8200, 15000])
    assert t.source == TIMESTAMPS
    assert t.pauses_ms == (1200, 4500)
    assert t.longest_pause_ms == 4500
    assert t.total_pause_ms == 5700


def test_speaking_time_excludes_the_pauses():
    """語速的分母是講話時間,不是總時長。

    含停頓的話算出來的數字同時反映講多快與停多久,兩個訊號混在一起之後
    哪一個都讀不出來——而這兩個訊號在報告裡的意思完全不同。
    """
    t = voice([0, 5200], [4000, 8200])
    assert t.speaking_ms == 4000 + 3000
    elapsed = 8200 - 0
    assert t.speaking_ms < elapsed


def test_rate_is_chars_over_speaking_seconds():
    t = voice([0], [4000], ["十二個字的一句話啊"])
    assert t.chars == 9
    assert abs(t.chars_per_second - 9 / 4.0) < 1e-9


def test_a_single_segment_has_no_pauses_but_is_still_usable():
    """只有一段表示他一口氣講完,那是資訊不是缺資料。"""
    t = voice([0], [4000])
    assert t.usable
    assert t.pauses_ms == ()
    assert t.longest_pause_ms == 0


def test_an_extra_start_is_tolerated():
    """合約寫明 starts 可能比段落多 1:按下結束時系統正要開始聽新的一段。

    那不是錯誤,所以不該讓整輪變成不可用。
    """
    t = voice([0, 5200, 20000], [4000, 8200], SEGS[:2])
    assert t.source == TIMESTAMPS
    assert t.pauses_ms == (1200,)


# ---------------------------------------------------------------------------
# 算不出來的
# ---------------------------------------------------------------------------


def test_typed_answers_have_no_speaking_time():
    """打字沒有「講話時長」。硬算會變成打字速度,而那跟猶豫無關。"""
    t = measure(starts_ms=[0], ends_ms=[4000], segments=["x" * 20], input_mode="typed")
    assert t.source == UNAVAILABLE
    assert not t.usable


def test_unknown_input_mode_is_not_treated_as_voice():
    t = measure(starts_ms=[0], ends_ms=[4000], segments=["abc"], input_mode="unknown")
    assert not t.usable


def test_no_end_times_means_unavailable():
    """前端還沒送 segment_ends_ms 的時候,整塊安靜退回今天的行為。"""
    assert not voice([0, 5200], [], SEGS[:2]).usable


def test_overlapping_segments_are_rejected():
    """重疊表示兩邊的時點不在同一個時間軸上,相減出來的停頓沒有意義。"""
    assert not voice([0, 3000], [4000, 8000], SEGS[:2]).usable


def test_an_impossibly_short_segment_is_rejected():
    assert not voice([0], [MIN_SEGMENT_MS - 1], ["字"]).usable


def test_an_absurd_pause_is_rejected_not_clamped():
    """一分鐘沒出聲不是在想,是中斷(接電話、切背景、引擎掛掉)。

    夾到上限等於宣稱「他停了 60 秒」,那是我們編的不是量到的。
    """
    t = voice([0, MAX_REASONABLE_PAUSE_MS + 5000], [4000, MAX_REASONABLE_PAUSE_MS + 9000], SEGS[:2])
    assert t.source == UNAVAILABLE


def test_empty_input_is_unavailable():
    assert not voice([], []).usable
    assert not measure(starts_ms=[0], ends_ms=[4000], segments=[], input_mode="voice").usable


# ---------------------------------------------------------------------------
# 值域與 prompt 錨點
# ---------------------------------------------------------------------------


def test_sources_are_declared():
    assert set(TIMING_SOURCES) == {TIMESTAMPS, UNAVAILABLE}
    assert SpeechTiming().source == UNAVAILABLE, "預設要是不可用,不是可用但全為 0"


def test_the_anchor_is_empty_when_there_is_nothing_to_say():
    """沒有資料就整行不出現。印「停頓:無資料」會被讀成「沒有停頓」。"""
    assert describe_pauses(SpeechTiming()) == ""
    assert describe_pauses(voice([0], [4000])) == ""


def test_the_anchor_states_the_measured_numbers():
    note = describe_pauses(voice([0, 5200, 12700], [4000, 8200, 15000]))
    assert "2 次" in note
    assert "4.5 秒" in note
    assert "實測值" in note, "沒有標成實測值的話,下游分不出它是量到的還是推論的"


def test_the_anchor_does_not_mention_rate():
    """文件那張判讀依據表有「長停頓」,沒有語速。

    prompt 多列一條會改變行為(gap 的選項說明把追問率從 0.500 推到 0.567),
    所以設計沒有在用的訊號不進 prompt。語速算出來備用。
    """
    t = voice([0, 5200], [4000, 8200])
    assert t.chars_per_second > 0
    assert "字/秒" not in describe_pauses(t)
    assert "語速" not in describe_pauses(t)
