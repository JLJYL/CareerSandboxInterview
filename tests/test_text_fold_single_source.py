"""折疊只能有一份實作(成員 A,W3)。

## 為什麼需要

`fold_chars` 原本住在 `transcript.py`,B 側 verbatim 比對需要同一套繁簡
折疊時自建了 `text_fold.py`。兩邊各自跑了一陣子——**兩份實作分岔會很難查**:
症狀是「這裡比對得到、那裡比對不到」,而兩邊的程式碼各自看起來都對,
沒有任何一行是錯的。

合併之後這支確保它不會再長出第二份,以及那張表的性質不被破壞。
"""

from __future__ import annotations

import pathlib

import pytest

from app.pipeline import text_fold, transcript


def test_transcript_reexports_the_same_object():
    """`transcript.fold_chars` 必須就是 `text_fold.fold_chars` 本人。

    不是「行為一樣的另一個函式」——是同一個物件。複製一份過去行為也會
    一樣,而那正是要防的情況。
    """
    assert transcript.fold_chars is text_fold.fold_chars
    assert transcript._T2S_CHARS is text_fold._T2S


def test_no_duplicate_definition_in_transcript():
    src = pathlib.Path(transcript.__file__).read_text(encoding="utf-8")
    assert "def fold_chars" not in src, "transcript.py 又長出一份定義,兩份會分岔"


def test_both_functions_share_one_table():
    """fold_variants 與 fold_chars 用同一張表。

    分成兩張的話,verbatim 比對得到的東西索引可能比對不到,反之亦然。
    """
    for trad, simp in [("髮", "发"), ("報", "报"), ("軟", "软")]:
        assert text_fold.fold_variants(trad) == simp
        assert text_fold.fold_chars(trad) == simp


class TestTableIsUsable:
    """表的性質——壞了會讓上面兩支還是綠的,但行為錯掉。"""

    def test_table_is_loaded(self):
        assert len(text_fold._T2S) > 3000, (
            f"只載入 {len(text_fold._T2S)} 筆,檢查 {text_fold._T2S_PATH}"
        )

    def test_every_entry_is_one_to_one(self):
        """非 1:1 會破壞 fold_chars 的等長,gap.py 的證據就會錯位。"""
        for k, v in text_fold._T2S.items():
            assert len(k) == 1 == len(v), f"非 1:1:{k!r} → {v!r}"
        for k, v in text_fold._LOOKALIKE_MAP.items():
            assert len(k) == 1 == len(v), f"非 1:1:{k!r} → {v!r}"

    @pytest.mark.parametrize("text", [
        "專案成本╱品質╱風險管理",
        "設計導入製造執行系統（MES）",
        "頭髮長得很快乾了",
        "React 跟 Angular 的差異",
        "a\u200bb",
    ])
    def test_fold_chars_preserves_length(self, text):
        assert len(text_fold.fold_chars(text)) == len(text)


class TestCoverageThatUsedToFail:
    """手寫 157 筆時對不上的那幾個,現在要通過。

    這兩個是黃金測試集裡真實存在的技能——涵蓋不足時
    verify_evidence 會把正確的引用判成編造。
    """

    @pytest.mark.parametrize("trad,simp", [
        ("軟體程式設計", "软体程式设计"),
        ("報表製作", "报表制作"),
        ("專案管理", "专案管理"),
        ("溝通協調", "沟通协调"),
    ])
    def test_matches_after_fold(self, trad, simp):
        assert text_fold.same_text(trad, simp)
        assert text_fold.fold_chars(trad) == text_fold.fold_chars(simp)
