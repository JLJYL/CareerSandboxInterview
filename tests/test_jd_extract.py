"""JD 抽取器測試。不打真 LLM,llm 參數傳假的。"""

from __future__ import annotations

import pytest

from app.pipeline.jd_extract import (
    MIN_DESCRIPTION_CHARS,
    clean_skills,
    extract_jd,
    from_catalog,
    parse_skills,
)
from app.prompts.jd_extract import MAX_SKILLS, compose_jd_prompt

LONG_JD = "協助建立銷售報表與資料清理,需具備 SQL 撰寫能力,能建立自動化報表流程。" * 2


def fake_llm(payload: str):
    return lambda system, user: payload


def boom_llm(system, user):
    raise RuntimeError("模型服務逾時")


# ---------------------------------------------------------------------------
# 解析:壞輸出不可拋例外
# ---------------------------------------------------------------------------


def test_parses_clean_json() -> None:
    items, notices = parse_skills('["SQL", "報表自動化"]')
    assert items == ["SQL", "報表自動化"]
    assert notices == []


def test_strips_markdown_fence() -> None:
    items, _ = parse_skills('```json\n["SQL"]\n```')
    assert items == ["SQL"]


def test_extracts_array_from_chatty_output() -> None:
    """模型加了前言時擷取其中的陣列,並記錄下來。"""
    items, notices = parse_skills('好的,以下是結果:\n["SQL", "Python"]\n希望有幫助')
    assert items == ["SQL", "Python"]
    assert any("夾雜" in n for n in notices)


@pytest.mark.parametrize("bad", ["", "完全不是 JSON", '{"skills": ["SQL"]}', "[不合法]"])
def test_bad_output_returns_empty_not_exception(bad: str) -> None:
    items, notices = parse_skills(bad)
    assert items == []
    assert notices


# ---------------------------------------------------------------------------
# 清洗:順序必須保留
# ---------------------------------------------------------------------------


def test_order_is_preserved() -> None:
    """順序即重要性順序,A 的 JD_POSITION_DECAY 依賴它,不可重排。"""
    src = ["報表自動化", "SQL", "跨部門溝通"]
    out, _ = clean_skills(src, [])
    assert out == src


def test_dedup_keeps_first_occurrence() -> None:
    out, notices = clean_skills(["SQL", "sql", " SQL ", "Python"], [])
    assert out == ["SQL", "Python"]
    assert any("去重" in n for n in notices)


def test_drops_non_string_items() -> None:
    out, notices = clean_skills(["SQL", 42, None, {"x": 1}, "Python"], [])
    assert out == ["SQL", "Python"]
    assert any("非字串" in n for n in notices)


def test_truncates_at_cap() -> None:
    out, notices = clean_skills([f"技能{i}" for i in range(40)], [])
    assert len(out) == MAX_SKILLS
    assert out[0] == "技能0", "截斷要砍後段,前面是比較重要的"
    assert any("截斷" in n for n in notices)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def test_happy_path() -> None:
    r = extract_jd(LONG_JD, fake_llm('["SQL", "報表自動化"]'))
    assert r.jd.required_skills == ["SQL", "報表自動化"]
    assert r.jd.source == "extracted"
    assert r.jd.description == LONG_JD.strip()


def test_empty_jd_skips_the_call() -> None:
    """customJd 是選填,使用者跳過是常態,不該浪費一次呼叫。"""
    called = []

    def spy(system, user):
        called.append(1)
        return "[]"

    r = extract_jd("", spy)
    assert r.jd.required_skills == []
    assert called == []
    assert any("過短或未提供" in n for n in r.notices)


def test_short_jd_skips_the_call() -> None:
    r = extract_jd("找前端", fake_llm('["React"]'))
    assert r.jd.required_skills == []
    assert len("找前端") < MIN_DESCRIPTION_CHARS


def test_llm_failure_degrades_not_raises() -> None:
    """抽取失敗不可讓整份報告掛掉。"""
    r = extract_jd(LONG_JD, boom_llm)
    assert r.jd.required_skills == []
    assert any("呼叫失敗" in n for n in r.notices)
    assert r.jd.description == LONG_JD.strip()


def test_empty_result_is_a_legal_answer() -> None:
    """散文太空泛抽不出東西是正常的,不要為了避免空值而捏造。"""
    r = extract_jd(LONG_JD, fake_llm("[]"))
    assert r.jd.required_skills == []
    assert any("抽不出" in n for n in r.notices)


# ---------------------------------------------------------------------------
# 型錄路徑
# ---------------------------------------------------------------------------


def test_catalog_path_marks_source() -> None:
    """兩條路交給 A 的形狀相同,靠 source 區分校準分佈與正式分佈。"""
    jd = from_catalog(["SQL", "Python"], "描述")
    assert jd.source == "catalog"
    assert jd.required_skills == ["SQL", "Python"]


def test_catalog_preserves_order() -> None:
    src = ["專案時間╱進度控管", "專案成本╱品質╱風險管理", "SQL"]
    assert from_catalog(src).required_skills == src


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def test_prompt_carries_the_inclusion_test() -> None:
    """納入判準要與黃金集規則四一致,否則抽取器會產出黃金集判定不算需求的項目。"""
    p = compose_jd_prompt()
    assert "少一項" in p


def test_prompt_marks_context_as_non_source() -> None:
    """職稱叫「資料分析師」不代表 JD 要求「資料分析」。"""
    p = compose_jd_prompt(title="資料分析實習生", company="某電商")
    assert "資料分析實習生" in p
    assert "不可" in p and "抽取來源" in p


def test_prompt_forbids_fabrication() -> None:
    p = compose_jd_prompt()
    assert "捏造" in p or "沒提就是沒提" in p


# ---------------------------------------------------------------------------
# 實測回饋後補的規則
# ---------------------------------------------------------------------------


def test_prompt_maps_mainland_terms_explicitly() -> None:
    """原則性說明不夠——第二版只把「實現」拿掉,詞根沒改。要給對照表。"""
    from app.prompts.jd_extract import JD_LOCALE

    for pair in ("用戶", "使用者", "代碼", "程式碼", "交互", "互動"):
        assert pair in JD_LOCALE
    assert "這不是選項" in JD_LOCALE


def test_prompt_protects_proper_nouns_from_suffixes() -> None:
    """去修飾語是拿掉前面的形容詞,不是在後面補說明。

    C# 被寫成「C#程式開發」就對不上詞彙表的「C#」條目。
    """
    from app.prompts.jd_extract import JD_PROHIBITIONS

    assert "C#程式開發" in JD_PROHIBITIONS
    assert "不可加後綴" in JD_PROHIBITIONS


def test_prompt_has_two_layer_test() -> None:
    """第一層問是不是需求,第二層問是不是能力。兩層都過才輸出。"""
    from app.prompts.jd_extract import compose_jd_prompt

    t = compose_jd_prompt()
    assert "別家公司" in t
    assert "兩層都通過" in t
