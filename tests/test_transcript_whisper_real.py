"""Whisper 真實輸出的斷句行為(成員 A,W3)。

## 這支鎖住什麼

怡君 2026-09-08 提供的實測樣本:同一支手機、同一個 App、刻意講得平穩、
逐字稿標了停頓位置。這是唯一一份「可對照原稿」的長樣本,價值在於
它能證偽假設,不只是提供資料。

它證偽了一個我差點做進去的設計:**拿中文字之間的空格當句界**。

    …決定要做 / 真正 / 開始動手之後…      「真正」後面的空格是講到一半停頓
    …比我們想象 / 原本的 / 多很多          口誤自我修正,原稿是「比我們原本想像的」

18 個空格裡至少 4 個在句子中間。照那個設計會把平均句長算成真實值的一半,
而且會回報 punctuation／stt_segment 這種「實測值」等級的模式——過度宣稱。

## 為什麼不需要改 _segment

現行四態拿這段真實資料跑,兩種情況都已經誠實:

    送了 segments   stt_segment      邊界確實來自引擎,觸發「不得講成句長」
    沒送 segments   discourse_marker 數字難看,但那一態的定義就是估算值

要做的是讓前端把 answerSegments 送上來,不是改這裡。
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from app.pipeline.transcript import TranscriptAnalyzer

VOCAB_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "vocab" / "skills_v1.json"
)

#: 怡君 2026-09-08 的實測輸出。零改寫,原樣保留。
#: 講者刻意在兩處停頓約三秒（第 1、7 段之後），並標了原稿供對照。
WHISPER_REAL = (
    "我想聊一下我觉得做这个专案最有挑战性的一个部分 "
    "其实一开始我们对于要不要做语音输入这件事内部讨论了很久 "
    "因为纯文字输入明显简单很多 风险也比较低 但后来考量到真实面试环境 "
    "如果只能用打字练习效果会差很多 所以还是决定要做 真正 开始动手之后才发现 "
    "语音这块要处理的细节 比我们想象 原本的 多很多 包括录音格式 "
    "上传流程还有转录结果要怎么跟后面的评分逻辑串接 每个环节都需要花时间去确认清楚 "
    "整体来说 这是这次专案里我觉得学到最多但也最花时间的一个部分"
)


@pytest.fixture(scope="module")
def analyzer():
    raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab = raw.get("skills", raw) if isinstance(raw, dict) else raw
    return TranscriptAnalyzer(vocab, engine_filler_policy="partial")


class TestRealWhisperOutput:
    """真實樣本的既定事實。這些變了代表 STT 行為變了,要重新看。"""

    def test_no_punctuation_at_all(self):
        """App 錄音路徑零標點。斷句邏輯不可依賴標點。"""
        assert not re.search(r"[。！？；，、.!?;,]", WHISPER_REAL)

    def test_output_is_simplified(self):
        """Whisper 對繁簡沒有一致保證,而且跟錄音來源無關。

        先前推測是電腦錄音 vs App 錄音的差別,這段是 App 錄的,照樣簡體。
        """
        assert any(c in WHISPER_REAL for c in "这觉专挑战们语")


class TestNotOverClaiming:
    """★ 核心:不可宣稱超過實際擁有的精確度。"""

    def test_bare_text_is_estimated_not_measured(self, analyzer):
        """沒送 segments 時只能估算,不可回 punctuation 或 stt_segment。

        回那兩態會讓 B 的 prompt 把平均句長講成量出來的值。
        """
        _, mode = analyzer._segment(WHISPER_REAL)
        assert mode in ("discourse_marker", "unavailable"), (
            f"整段一塊時宣稱了 {mode},那是過度宣稱"
        )

    def test_spaces_are_not_treated_as_sentence_boundaries(self, analyzer):
        """空格是停頓不是句界,不可拿來切。

        「真正」後面、「想象/原本的」前後那幾個空格都在句子中間。
        照空格切會切出 18 段,平均句長變成真實值的一半。
        """
        parts, _ = analyzer._segment(WHISPER_REAL)
        assert len(parts) < 10, (
            f"切出 {len(parts)} 段,接近空格數(18)——空格被當成句界了"
        )

    def test_segments_path_reports_stt_segment(self, analyzer):
        """送了 segments 時回 stt_segment:邊界確實來自引擎,是實測值。

        這一態同樣會觸發合約裡「不得把平均句長講得像量出來的」。
        """
        joined = "\n".join(WHISPER_REAL.split(" "))
        parts, mode = analyzer._segment(joined)
        assert mode == "stt_segment"
        assert len(parts) == 18


class TestSpaceIsPauseNotBoundary:
    """把「空格 = 停頓」這個事實記成可執行的斷言。

    這幾格是從原稿對照出來的——沒有原稿就分不出哪個空格是句界。
    """

    @pytest.mark.parametrize("fragment", [
        "所以还是决定要做 真正 开始动手之后才发现",   # 「真正」後停頓，非句界
        "比我们想象 原本的 多很多",                    # 口誤自我修正，非句界
    ])
    def test_mid_sentence_spaces_exist_in_real_data(self, fragment):
        """這些片段確實出現在真實輸出裡,不是我編的。"""
        assert fragment in WHISPER_REAL
