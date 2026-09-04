"""CollabObserver 的測試(成員 A,W3)。

之前的驗證都是手動腳本,沒有進到 CI 會跑的測試,任何人改動
`collab.py` 都沒有防護網。這支補上,尤其涵蓋兩類手動腳本才發現的情況：

    浮動掃描邏輯    使用者連續發言時,參與主動性/傾聽與回應該不該
                    因為斷句方式改變判斷(見 test_participation_floating_*)
    引號邊界        標籤跟真正的發言文字之間要有「」當邊界,不然會讓
                    B 的 verify_evidence 把老實的 LLM 判成編造引用
                    (見 test_*_wraps_text_in_quotes)

第二類是串到 `collab_score.score_collab` 才發現的問題,單獨測切片
格式測不出來——這支測試把兩種都涵蓋,以後回歸不會被漏掉。
"""

from __future__ import annotations

import re

import pytest

from app.contracts.interview_protocols import (
    CollabObserver as CollabObserverProto,
)
from app.contracts.interview_protocols import Utterance
from app.pipeline.collab import CollabObserver, FakeCollabObserver
from app.schemas.interview import COLLAB_DIM_NAMES

PARTICIPATION, RESPONSIVENESS, ARGUMENT, COLLABORATION = COLLAB_DIM_NAMES

_QUOTE = re.compile(r"「(.+?)」")


def _by_name(slices, name):
    return next(s for s in slices if s.name == name)


# --------------------------------------------------------------------- 基本形狀

def test_protocol_compliance():
    """CollabObserver 必須滿足合約的 Protocol,否則 deps.py 不會選用它。"""
    assert isinstance(CollabObserver(), CollabObserverProto)


def test_observe_returns_four_slices_in_dim_order():
    obs = CollabObserver()
    slices = obs.observe([Utterance("你", "測試", is_user=True)])
    assert len(slices) == 4
    assert [s.name for s in slices] == list(COLLAB_DIM_NAMES)


def test_empty_utterances_all_dims_empty():
    obs = CollabObserver()
    for s in obs.observe([]):
        assert s.excerpts == []


def test_fake_observer_shape_matches_real():
    """Fake 的形狀要跟真實作一致,測試才能在兩者間自由替換。"""
    fake_slices = FakeCollabObserver().observe([])
    real_slices = CollabObserver().observe([])
    assert [s.name for s in fake_slices] == [s.name for s in real_slices]


# --------------------------------------------------------------- 只有使用者發言

class TestOnlyUserSpeaks:
    """沒有他人發言時,四維度該有的降級行為。"""

    @pytest.fixture
    def slices(self):
        utts = [
            Utterance("你", "我覺得應該先做市場調查", is_user=True),
            Utterance("你", "因為不知道客群就沒辦法定價", is_user=True),
        ]
        return CollabObserver().observe(utts)

    def test_participation_still_works(self, slices):
        """參與主動性不需要他人在場,純看使用者自己的發言。"""
        s = _by_name(slices, PARTICIPATION)
        assert len(s.excerpts) == 2

    def test_argument_still_works(self, slices):
        """論點建構是唯一完全不需要他人發言就能算的維度。"""
        s = _by_name(slices, ARGUMENT)
        assert s.excerpts == ["我覺得應該先做市場調查", "因為不知道客群就沒辦法定價"]

    def test_responsiveness_empty_with_reason(self, slices):
        """沒有他人發言可比對——資訊不在場,不是難,不做近似。"""
        s = _by_name(slices, RESPONSIVENESS)
        assert s.excerpts == []
        assert s.note  # 必須說明原因,不能是空字串

    def test_collaboration_gets_user_only_transcript(self, slices):
        """協作姿態一律拿到整段(這裡只有使用者),不是空的。"""
        s = _by_name(slices, COLLABORATION)
        assert len(s.excerpts) == 2


# ------------------------------------------------------------- 浮動掃描邏輯

class TestFloatingNearestOther:
    """使用者把回應拆成連續多則時,不該因為斷句方式改變判斷。

    這是跟 B 的 FakeCollabObserver(整場只判一次)刻意不同的地方,
    也是這支測試要鎖住的行為,回歸了要能立刻抓到。
    """

    @pytest.fixture
    def utts(self):
        return [
            Utterance("AI-邏輯", "我覺得應該先看預算", is_user=False),
            Utterance("你", "我建議先看客群", is_user=True),
            Utterance("你", "因為客群決定預算怎麼配", is_user=True),  # 連續第二則
            Utterance("AI-親切", "有道理", is_user=False),
            Utterance("你", "可以分兩個方向做", is_user=True),
        ]

    def test_second_consecutive_utterance_still_tagged_as_following(self, utts):
        """第二則使用者發言前面是自己講的,但往前找最近他人仍然存在，
        該標「接續他人之後」，不是「自己起頭」。"""
        s = _by_name(CollabObserver().observe(utts), PARTICIPATION)
        assert "[接續他人之後]" in s.excerpts[0]
        assert "[接續他人之後]" in s.excerpts[1], (
            "連續第二則不該因為斷句方式被標成自己起頭"
        )

    def test_both_consecutive_utterances_pair_with_same_other(self, utts):
        """連續兩則使用者發言都該配對同一則他人發言，直到下一位他人出現。"""
        s = _by_name(CollabObserver().observe(utts), RESPONSIVENESS)
        assert len(s.excerpts) == 3
        assert "我覺得應該先看預算" in s.excerpts[0]
        assert "我覺得應該先看預算" in s.excerpts[1], (
            "連續第二則應配對同一則他人發言，不是空手"
        )
        assert "有道理" in s.excerpts[2]

    def test_first_utterance_with_no_prior_other_is_starting(self, utts):
        """整場第一則使用者發言前面沒有任何他人，該標自己起頭。"""
        only_user_first = [Utterance("你", "開場我先講", is_user=True)]
        s = _by_name(CollabObserver().observe(only_user_first), PARTICIPATION)
        assert "[自己起頭]" in s.excerpts[0]


# ----------------------------------------------------------- 引號邊界(evidence 保護)

class TestQuoteBoundaryForEvidenceSafety:
    """標籤跟真正的話之間要有「」邊界。

    B 的 verify_evidence 要求 LLM 回的 evidence 是逐字稿裡真的出現過的
    子字串;prompt 又要求 LLM「引用逐字稿裡的話」。如果標籤跟話黏在一起
    沒有邊界,誠實的 LLM 也可能把標籤一起引用進去,導致被誤判成編造。
    這裡直接檢查:每一則帶標籤的切片,真正的發言文字都必須被「」包住,
    且用正規表達式抓出來的內容要跟原始文字完全一致。
    """

    @pytest.fixture
    def utts(self):
        return [
            Utterance("AI-邏輯", "我覺得應該先看預算", is_user=False),
            Utterance("你", "我建議先看客群", is_user=True),
        ]

    def test_participation_excerpt_wraps_text_in_quotes(self, utts):
        s = _by_name(CollabObserver().observe(utts), PARTICIPATION)
        match = _QUOTE.search(s.excerpts[0])
        assert match is not None, "參與主動性的發言文字必須用「」包住"
        assert match.group(1) == "我建議先看客群"

    def test_collaboration_excerpt_wraps_text_in_quotes(self, utts):
        s = _by_name(CollabObserver().observe(utts), COLLABORATION)
        for excerpt in s.excerpts:
            match = _QUOTE.search(excerpt)
            assert match is not None, f"協作姿態的每則發言都必須用「」包住:{excerpt!r}"

    def test_responsiveness_excerpt_wraps_both_sides_in_quotes(self, utts):
        s = _by_name(CollabObserver().observe(utts), RESPONSIVENESS)
        quotes = _QUOTE.findall(s.excerpts[0])
        assert len(quotes) == 2, "他人那句、使用者那句都要各自被「」包住"
        assert quotes == ["我覺得應該先看預算", "我建議先看客群"]


# --------------------------------------------------------------------- is_user

def test_uses_is_user_not_speaker_id_string_match():
    """前端送的 speaker_id 是「你」不是 "user"——絕對不能用字串比對。

    這是曾經讓四個維度全部切錯、且不報錯的靈點失敗風險,鎖住不能回歸。
    """
    utts = [
        Utterance("AI-邏輯", "我覺得應該先看預算", is_user=False),
        Utterance("你", "我建議先看客群", is_user=True),  # speaker_id 是「你」
    ]
    slices = CollabObserver().observe(utts)
    assert _by_name(slices, ARGUMENT).excerpts == ["我建議先看客群"], (
        "若誤用 speaker_id == 'user' 判斷,這裡會是空的"
    )


# --------------------------------------------------------------------- 論點建構

def test_argument_excerpts_are_independent_no_cross_reference():
    """論點建構每則獨立,不應該把他人發言混進去。"""
    utts = [
        Utterance("AI-邏輯", "我覺得應該先看預算", is_user=False),
        Utterance("你", "我建議先看客群", is_user=True),
        Utterance("你", "因為客群決定預算", is_user=True),
    ]
    s = _by_name(CollabObserver().observe(utts), ARGUMENT)
    assert s.excerpts == ["我建議先看客群", "因為客群決定預算"]
    assert not any("預算" in e and "我建議" in e for e in s.excerpts if e != s.excerpts[1])


# --------------------------------------------------------------------- 空白文字

def test_blank_text_utterances_are_skipped():
    """空字串或純空白的發言不該產出空切片。"""
    utts = [
        Utterance("你", "   ", is_user=True),
        Utterance("你", "", is_user=True),
        Utterance("你", "這則才算數", is_user=True),
    ]
    s = _by_name(CollabObserver().observe(utts), ARGUMENT)
    assert s.excerpts == ["這則才算數"]
