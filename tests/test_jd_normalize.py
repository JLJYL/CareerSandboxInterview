"""JD 抽取結果的機械正規化測試。

這一層存在的理由是 prompt 修過兩次都有殘留(見 jd_normalize.py 檔頭)。
測試的重點有兩個:該改的要改,**不該動的一個都不能動**。
誤傷比殘留嚴重——把「Google雲端試算表」砍成「Google」會讓那條永遠抓不到。
"""

from __future__ import annotations

import pytest

from app.pipeline.jd_normalize import (
    normalize_locale,
    normalize_skill,
    normalize_skills,
    strip_proper_noun_suffix,
)


# ---------------------------------------------------------------------------
# 用語:實測出現過的失敗案例
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,want",
    [
        ("代碼測試", "程式碼測試"),
        ("用戶界面", "使用者介面"),
        ("用戶體驗優化", "使用者體驗優化"),
        ("交互功能", "互動功能"),
        ("數據可視化", "資料視覺化"),
        ("數據庫設計", "資料庫設計"),
        ("軟件開發", "軟體開發"),
    ],
)
def test_locale_fixes_real_failures(src: str, want: str) -> None:
    assert normalize_locale(src) == want


@pytest.mark.parametrize("term", ["數據分析", "前端開發", "品質管理", "效能優化", "專案管理"])
def test_locale_leaves_taiwan_terms_alone(term: str) -> None:
    """台灣本來就這樣講的不可以動。數據分析在台灣通用,不可換成資料分析。"""
    assert normalize_locale(term) == term


# ---------------------------------------------------------------------------
# 專有名詞去後綴
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,want",
    [
        ("C#程式開發", "C#"),
        ("Excel操作", "Excel"),
        ("TypeScript開發", "TypeScript"),
        ("SQL撰寫", "SQL"),
        ("Python程式設計", "Python"),
    ],
)
def test_strips_generic_suffix(src: str, want: str) -> None:
    assert strip_proper_noun_suffix(src) == want


@pytest.mark.parametrize(
    "term",
    [
        "RESTful API",  # 後面不是中文
        "Node.js",
        "C#",
        "前端開發",  # 開頭不是拉丁字母
        "Google雲端試算表",  # 後綴不在通用清單,是產品名的一部分
        "鼎新Smart系統",
        "Google Sheets",
    ],
)
def test_does_not_touch_valid_terms(term: str) -> None:
    """誤傷比殘留嚴重。砍錯會讓那條技能永遠對不上詞彙表。"""
    assert strip_proper_noun_suffix(term) == term


# ---------------------------------------------------------------------------
# 整批
# ---------------------------------------------------------------------------


def test_order_is_preserved() -> None:
    """順序即重要性順序,JD_POSITION_DECAY 依賴它。"""
    out, _ = normalize_skills(["代碼測試", "React", "C#程式開發", "HTML"])
    assert out == ["程式碼測試", "React", "C#", "HTML"]


def test_dedup_after_normalization() -> None:
    """正規化會製造重複:用戶界面與使用者介面正規化後同名。"""
    out, notices = normalize_skills(["用戶界面", "使用者介面", "React"])
    assert out == ["使用者介面", "React"]
    assert any("去重" in n for n in notices)


def test_changes_are_recorded_in_notices() -> None:
    """改了什麼要看得見,否則之後查不出數字為什麼變。"""
    _, notices = normalize_skills(["C#程式開發"])
    assert any("C#程式開發→C#" in n for n in notices)


def test_no_change_means_no_notice() -> None:
    _, notices = normalize_skills(["React", "HTML", "專案管理"])
    assert notices == []


def test_empty_input() -> None:
    assert normalize_skills([]) == ([], [])


def test_full_pipeline_on_observed_output() -> None:
    """用 ivw-002 實際抽出的那批跑一次。"""
    raw = ["TypeScript", "前端開發", "React", "RESTful API", "前端狀態管理",
           "Redux", "代碼測試", "問題修復", "數據可視化", "用戶體驗優化"]
    out, _ = normalize_skills(raw)
    assert "程式碼測試" in out
    assert "資料視覺化" in out
    assert "使用者體驗優化" in out
    assert "代碼測試" not in out
    for kept in ("TypeScript", "React", "RESTful API", "Redux", "前端開發"):
        assert kept in out
