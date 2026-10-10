"""面試官規格的推導測試。純函式,不打 LLM。

這一版只宣告規格,還沒接進 next_turn,所以沒有行為可測。
這裡驗的是三件事:

    推導是可重現的        同樣的 JD 加同樣的難度,每次都一樣
    難度真的逐級變嚴      放行條件單調,不是三段讀起來不同的文字
    槽位 id 對齊 persona  前端的頭像以 id 查表,錯了圖就對不上
"""

from __future__ import annotations

import pytest

from app.contracts.interview_protocols import JDInput
from app.pipeline.interviewer_spec import (
    ACCEPT_RULES,
    DEFAULT_DIFFICULTY,
    AcceptRule,
    InterviewerSpec,
    accept_rule_for,
    derive_specs,
)
from app.prompts.interview_personas import personas_for

JD = JDInput(required_skills=["Python", "資料分析", "跨部門溝通"], description="…")


# ---------------------------------------------------------------------------
# 可重現
# ---------------------------------------------------------------------------


def test_same_input_gives_same_specs():
    """推導不是生成:同樣的輸入每次都該一樣。"""
    a = derive_specs("panel", jd=JD, difficulty="困難")
    b = derive_specs("panel", jd=JD, difficulty="困難")
    assert a == b


def test_cares_about_keeps_jd_order():
    """JD 的順序即優先序,不可重排。"""
    spec = derive_specs("single", jd=JD)[0]
    assert spec.cares_about == ("Python", "資料分析", "跨部門溝通")


def test_no_jd_is_not_an_error():
    """JD 抽取失敗或使用者沒填時照常推導,只是沒有 cares_about。"""
    spec = derive_specs("single", jd=None)[0]
    assert spec.cares_about == ()
    assert not spec.has_jd()
    assert spec.accept is accept_rule_for(DEFAULT_DIFFICULTY)


# ---------------------------------------------------------------------------
# 難度逐級變嚴
# ---------------------------------------------------------------------------


def test_accept_rules_are_monotonic():
    """新手 → 中等 → 困難,每一欄都只會變嚴,不會鬆回去。

    這條是「追問率應該是 新手 < 中等 < 困難」的前提。
    條件不單調的話,那個預期就不成立,實跑數出來的數字也沒有意義。
    """
    easy, mid, hard = (ACCEPT_RULES[k] for k in ("新手", "中等", "困難"))
    assert easy.min_star_parts < mid.min_star_parts < hard.min_star_parts
    assert easy.max_depth < mid.max_depth < hard.max_depth
    for field in ("require_result", "require_quantification", "require_tradeoff"):
        seq = [getattr(r, field) for r in (easy, mid, hard)]
        assert seq == sorted(seq), f"{field} 不單調:{seq}"


def test_unknown_difficulty_falls_back_to_medium():
    assert accept_rule_for("") is ACCEPT_RULES["中等"]
    assert accept_rule_for("地獄") is ACCEPT_RULES["中等"]


def test_accept_rule_fields_are_decidable():
    """每一欄都是可判定的型別,不是形容詞。

    「更嚴格」沒辦法照著做,int 與 bool 可以。
    """
    rule = ACCEPT_RULES["困難"]
    assert isinstance(rule.min_star_parts, int)
    assert isinstance(rule.max_depth, int)
    for field in ("require_result", "require_quantification", "require_tradeoff"):
        assert isinstance(getattr(rule, field), bool)


def test_max_depth_has_an_upper_bound():
    """追問深度不能無上限——Grok 4 一場 45.7 次被評為 over-questioning。"""
    assert max(r.max_depth for r in ACCEPT_RULES.values()) <= 3


# ---------------------------------------------------------------------------
# 槽位
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,gi,expected",
    [
        ("single", 1, 1),
        ("panel", 1, 3),
        ("group", 1, 1),  # 一位主考官
        ("group", 3, 3),  # 三位主管,主考官不出現
    ],
)
def test_slot_count_matches_personas(mode: str, gi: int, expected: int):
    assert len(derive_specs(mode, jd=JD, group_interviewers=gi)) == expected


def test_slot_ids_align_with_personas():
    """前端的頭像與色票以 persona id 查表,對不上圖就不會顯示。"""
    for mode, gi in (("single", 1), ("panel", 1), ("group", 1), ("group", 3)):
        specs = derive_specs(mode, jd=JD, group_interviewers=gi)
        ids = {p.id for p in personas_for(mode, gi, 4) if p.role != "peer"}
        assert {s.slot_id for s in specs} == ids


def test_competitors_are_excluded():
    """AI 競爭者不追問,動作空間完全不同,不該有面試官規格。"""
    specs = derive_specs("group", jd=JD, group_size=5)
    assert all(s.role != "peer" for s in specs)
    assert not any(s.slot_id.startswith("peer_") for s in specs)


def test_every_slot_shares_the_same_accept_rule():
    """「回答有沒有缺口」跟誰在問無關,同一場的放行條件只有一套。"""
    specs = derive_specs("panel", jd=JD, difficulty="新手")
    assert len({s.accept for s in specs}) == 1


def test_spec_is_frozen():
    import dataclasses

    spec = derive_specs("single", jd=JD)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.accept = AcceptRule(1, False, False, False, 1)  # type: ignore[misc]
    assert isinstance(spec, InterviewerSpec)
