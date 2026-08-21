"""CollabObserver 驗收測試（成員 A，W2）。

守三件事：契約符合性、禁用指標不被拿來計量、資訊不在場時回空而不近似。
"""

from __future__ import annotations

from app.contracts.interview_protocols import (
    CollabObserver as CollabObserverProto,
    Utterance,
)
from app.pipeline.collab import CollabObserver
from app.schemas.interview import COLLAB_DIM_NAMES

ONLY_USER = [
    Utterance("user", "我覺得應該先做市場調查因為不知道客群就沒辦法定價"),
    Utterance("user", "對啊我同意"),
]

WITH_OTHERS = [
    Utterance("AI-邏輯", "我覺得應該先看預算"),
    Utterance("AI-強勢", "預算不是重點吧"),
    Utterance("user", "我建議先釐清目標客群，因為不知道客群就沒辦法決定預算"),
    Utterance("AI-親切", "有道理"),
    Utterance("AI-邏輯", "那要怎麼做"),
    Utterance("user", "可以分成兩個方向，第一是問卷，第二是訪談"),
]


# ---------------------------------------------------------------- 契約符合性


def test_satisfies_protocol():
    assert isinstance(CollabObserver(), CollabObserverProto)


def test_returns_four_signals_aligned_to_dim_names():
    """回傳四筆、順序對齊 COLLAB_DIM_NAMES。

    ★ 名字直接 import 常數而非在實作裡再寫一份字串。
      合約只寫「name 必須是 COLLAB_DIM_NAMES 四個之一」但沒定義它,
      定義在 app/schemas/interview.py。兩份字串各自維護就是
      D1 那個「skill_id vs 顯示字串」的同款坑——對不上不會報錯,只會安靜失效。
    """
    got = CollabObserver().observe(WITH_OTHERS)
    assert [s.name for s in got] == list(COLLAB_DIM_NAMES)


def test_level_is_always_none():
    """level 由 B 的 LLM 對照 BARS 指派，A 一律不填。"""
    for utts in (ONLY_USER, WITH_OTHERS, []):
        assert all(s.level is None for s in CollabObserver().observe(utts))


# ---------------------------------------------------------------- 禁用指標


def test_utterance_count_is_marked_as_background_only():
    """★ 發言次數只能當背景資訊，不能是計量訊號。

    合約 COLLAB_PROHIBITED_INDICATORS 依 babble 假說禁用發言量指標，
    但允許放進 signals 當背景。本實作用 bg_ 前綴讓那道界線在**資料裡**
    看得見——只寫在 prompt 或註解裡的「不要依賴這個」是很弱的保護，
    它是那堆數字裡最直觀的一個，LLM 會錨定上去。
    """
    sig = CollabObserver().observe(WITH_OTHERS)[0]
    counts = [k for k in sig.signals if "count" in k and "connector" not in k]
    assert counts, "發言次數應該仍被帶出來當背景"
    assert all(k.startswith("bg_") for k in counts), \
        f"發言量指標必須加 bg_ 前綴標示不參與評分：{counts}"


def test_participation_does_not_rank_by_volume():
    """講得多不代表參與主動性訊號比較好。

    兩份逐字稿發言次數差三倍，但首次發言位置相同——
    非背景訊號應該一致。
    """
    o = CollabObserver()
    few = [Utterance("AI", "先看預算"), Utterance("user", "我建議先釐清客群")]
    many = few + [Utterance("user", f"補充第{i}點") for i in range(5)]
    a = {k: v for k, v in o.observe(few)[0].signals.items() if not k.startswith("bg_")}
    b = {k: v for k, v in o.observe(many)[0].signals.items() if not k.startswith("bg_")}
    assert a["first_speak_position"] == b["first_speak_position"]
    assert a["framing_in_first"] == b["framing_in_first"]


def test_no_interruption_metric():
    """打斷次數合約明確禁用：介面在 AI 發言時擋住輸入，結構上恆為 0。"""
    for sig in CollabObserver().observe(WITH_OTHERS):
        assert not any("interrupt" in k for k in sig.signals)


# ---------------------------------------------------------------- 資訊不在場


def test_two_dims_empty_without_speaker_info():
    """★ 沒有他人發言時，後兩項回空而不做近似。

    可以想到的替代品（數「剛剛那位」「我同意」這類詞）測的是
    「有沒有做出回應的姿態」，不是「有沒有接住論點」——
    那會產出一個看起來合理、實際上量錯東西的數字，比沒有數字糟。
    """
    got = {s.name: s for s in CollabObserver().observe(ONLY_USER)}
    assert got[COLLAB_DIM_NAMES[1]].signals == {}
    assert got[COLLAB_DIM_NAMES[3]].signals == {}
    assert "不可用" in got[COLLAB_DIM_NAMES[1]].evidence
    assert "不可用" in got[COLLAB_DIM_NAMES[3]].evidence


def test_computable_dims_work_without_speaker_info():
    """相對地，前兩項不依賴他人發言，沒有發言者資料照樣算得出來。"""
    got = {s.name: s for s in CollabObserver().observe(ONLY_USER)}
    assert got[COLLAB_DIM_NAMES[0]].signals
    assert got[COLLAB_DIM_NAMES[2]].signals


def test_empty_input_does_not_raise():
    """沒有任何發言是正常情況，不拋例外。"""
    got = CollabObserver().observe([])
    assert len(got) == 4
    assert all(s.signals == {} for s in got)


# ---------------------------------------------------------------- 論點建構


def test_causal_rate_is_per_hundred_chars_not_per_utterance():
    """密度用每百字而非每則發言——後者會讓話多的人分數高，那又繞回發言量。"""
    o = CollabObserver()
    short = [Utterance("user", "因為時間不夠所以先做問卷")]
    padded = [Utterance("user", "因為時間不夠所以先做問卷" + "然後我們就開始執行了" * 5)]
    a = o.observe(short)[2].signals["causal_connector_rate"]
    b = o.observe(padded)[2].signals["causal_connector_rate"]
    assert a > b, "同樣的因果連接詞被大量無關文字稀釋後，密度應下降"
