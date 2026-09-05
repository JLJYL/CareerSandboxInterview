"""繁簡輸出不一致的正規化測試(成員 A,W3)。

## 為什麼需要

Whisper 對中文繁簡輸出**沒有一致保證**。怡君 2026-09-05 的實測報告:
同一位講者、同一支後端、同樣的呼叫方式,五段 App 錄音輸出繁體,
一段電腦錄音輸出簡體。原因未定(懷疑是收音鏈路差異),
但不影響結論——不能假設逐字稿一定是繁體。

## 不修的後果

詞彙表是繁體。逐字稿回簡體時中文技能全部比對不到,而
gap = 履歷∩JD − 提及,提及變小,交集裡的技能就變成假指控。

拿黃金集實測(只把逐字稿轉簡體、其餘完全不動):

    提及集 recall    0.488 → 0.366
    漏講 precision   0.533 → 0.444
    假指控             7 → 10 筆

而且不報錯,只是告訴使用者「你什麼都沒講到」。

## 這支鎖住什麼

    繁簡一致      同一個技能,繁體與簡體輸入要得到同一個 skill_id
    等長          fold_chars 與 normalize_for_scan 都不可改變字元數,
                  gap.py 用位移切原文當證據,長度一變證據就錯位
    冪等          已經是簡體的輸入再過一次不可被改壞
    無回歸        繁體(現有黃金集的形狀)的行為完全不變
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.pipeline.transcript import (
    T2S_CHARS_PATH,
    TranscriptAnalyzer,
    _T2S_CHARS,
    fold_chars,
    normalize_for_scan,
)

VOCAB_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "vocab" / "skills_v1.json"
)

#: 詞彙表裡真實存在的技能,繁簡對照。
#: 修正前實測:右欄全部比對不到。
SKILL_PAIRS = [
    ("專案管理", "专案管理"),
    ("軟體程式設計", "软体程式设计"),
    ("報表製作", "报表制作"),
    ("溝通協調", "沟通协调"),
    ("文書處理", "文书处理"),
    ("時間管理", "时间管理"),
    ("簡報表達", "简报表达"),
]


@pytest.fixture(scope="module")
def analyzer():
    raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab = raw.get("skills", raw) if isinstance(raw, dict) else raw
    return TranscriptAnalyzer(vocab)


class TestTableIntegrity:
    """對照表本身的性質。"""

    def test_table_is_loaded(self):
        assert len(_T2S_CHARS) > 3000, (
            f"只載入 {len(_T2S_CHARS)} 筆,檢查 {T2S_CHARS_PATH} 是否存在"
        )

    def test_every_mapping_is_one_to_one(self):
        """一律 1 字元對 1 字元,這是等長的來源。"""
        for trad, simp in _T2S_CHARS.items():
            assert len(trad) == 1 and len(simp) == 1, (
                f"非 1:1 對照會破壞等長:{trad!r} → {simp!r}"
            )

    def test_no_identity_mappings(self):
        """自己映射到自己是無效條目,徒增查表成本。"""
        same = [k for k, v in _T2S_CHARS.items() if k == v]
        assert not same, f"有 {len(same)} 筆自我映射:{same[:5]}"


class TestLengthPreserved:
    """fold_chars 與 normalize_for_scan 都不可改變字元數。

    gap.py 用 scan() 回傳的位移去切**原始**文字當證據。
    長度一變,證據就會切到錯的位置,而且不會報錯——
    使用者會看到一段對不上的引用。
    """

    @pytest.mark.parametrize("text", [
        "專案管理與軟體程式設計",
        "頭髮長得很快乾了",
        "我在這裡面試",
        "React 跟 Angular 的差異",
        "幹部乾淨",
        "我做過專案管理的工作",
    ])
    def test_fold_chars_preserves_length(self, text):
        assert len(fold_chars(text)) == len(text)

    @pytest.mark.parametrize("text", [
        "我做過專案管理的工作",
        "我做过专案管理的工作",
        "我們用 Python 寫的",
    ])
    def test_normalize_for_scan_offset_map_aligns(self, text):
        """位移對照表長度要跟輸出字串一致,否則證據會指錯位置。"""
        out, idx = normalize_for_scan(text)
        assert len(out) == len(idx)


class TestTraditionalSimplifiedMatch:
    """同一個技能,繁體與簡體輸入要落到同一個 skill_id。"""

    @pytest.mark.parametrize("trad,simp", SKILL_PAIRS)
    def test_fold_chars_converges(self, trad, simp):
        assert fold_chars(trad) == fold_chars(simp)

    @pytest.mark.parametrize("trad,simp", SKILL_PAIRS)
    def test_mentioned_skills_same_for_both(self, analyzer, trad, simp):
        """實際跑 mentioned_skills,兩種寫法要抓到同一組技能。

        修正前:簡體那邊回空集合,而空集合會讓 gap 變大,
        履歷∩JD 裡的技能全部變成假指控。
        """
        got_trad = analyzer.mentioned_skills(f"我做過{trad}的工作")
        got_simp = analyzer.mentioned_skills(f"我做过{simp}的工作")
        assert got_trad, f"繁體「{trad}」本來就該抓到,測試前提不成立"
        assert got_simp == got_trad, (
            f"簡體「{simp}」抓到 {got_simp},繁體抓到 {got_trad}"
        )


class TestIdempotent:
    """已經是簡體的輸入,再過一次不可被改壞。"""

    @pytest.mark.parametrize("text", [
        "专案管理", "软体程式设计", "沟通协调", "报表制作",
    ])
    def test_simplified_input_unchanged(self, text):
        assert fold_chars(text) == fold_chars(fold_chars(text))


class TestNoRegressionOnTraditional:
    """繁體(現有黃金集的形狀)的行為完全不變。

    現有 10 個黃金案例全部是繁體。這裡壞掉會讓 W2 校準結果失效。
    """

    def test_traditional_skills_still_detected(self, analyzer):
        for trad, _ in SKILL_PAIRS:
            assert analyzer.mentioned_skills(f"我做過{trad}的工作"), (
                f"繁體「{trad}」抓不到了,這是回歸"
            )

    def test_english_terms_unaffected(self, analyzer):
        """英文技能名不含中文字,繁簡轉換不該碰到它們。"""
        got = analyzer.mentioned_skills("我用 Python 跟 React 寫的")
        assert got, "英文技能名被繁簡轉換影響了"

    def test_golden_transcripts_fold_without_error(self):
        """10 份真實逐字稿都要能正常處理且等長。"""
        cases = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "interview"
        n = 0
        for p in sorted(cases.glob("ivw-*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            segs = d.get("transcript_segments") or []
            text = "".join(segs) if segs else d.get("transcript", "")
            if not text:
                continue
            n += 1
            assert len(fold_chars(text)) == len(text), f"{p.name} 折疊後長度變了"
        assert n >= 5, f"只讀到 {n} 份逐字稿,測試涵蓋不足"
