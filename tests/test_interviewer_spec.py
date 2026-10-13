"""面試官規格的推導測試。純函式,不打 LLM。

這裡驗的是四件事:

    推導是可重現的        同樣的 JD 加同樣的難度,每次都一樣
    難度真的逐級變嚴      放行條件單調,不是三段讀起來不同的文字
    放行與缺口算得出來    accepts() 與 first_gap() 窮舉所有事實組合都自洽
    槽位 id 對齊 persona  前端的頭像以 id 查表,錯了圖就對不上
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.contracts.interview_protocols import JDInput
from app.pipeline.interviewer_spec import (
    ACCEPT_RULES,
    DEFAULT_DIFFICULTY,
    GAP_QUANTIFICATION,
    GAP_TRADEOFF,
    GAP_VALUES,
    STAR_PART_GAP,
    STAR_PART_KEYS,
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
# 放行與缺口都是算出來的
# ---------------------------------------------------------------------------


def all_facts():
    """16 種段落組合 × 量化 × 取捨。窮舉,因為手挑樣本會漏。

    這條檢查原本寫在 interviewer_spec 的 import 時,我手挑了六種組合,
    結果誤判「缺情境」算不出來——它要的是 (T, A, R) 在困難難度,
    剛好不在我挑的那六種裡。窮舉才不會有這種事。
    """
    for n in range(len(STAR_PART_KEYS) + 1):
        for parts in itertools.combinations(STAR_PART_KEYS, n):
            for quantified in (False, True):
                for tradeoff in (False, True):
                    yield parts, quantified, tradeoff


def test_every_gap_is_reachable():
    """每一個宣告出來的缺口都要有組合算得出來。

    算不出來的值是死的值域,而死的值域會被當成活的拿去寫報表。
    """
    produced = {
        rule.first_gap(star_parts=parts, quantified=q, tradeoff=t)
        for parts, q, t in all_facts()
        for rule in ACCEPT_RULES.values()
    }
    assert not set(GAP_VALUES) - produced


def test_not_accepting_always_names_a_gap():
    """不放行就一定講得出缺哪一項。

    反過來也要成立:放行的時候不能還掛著一個缺口,那會讓報告說
    「這題過了,但缺結果」。
    """
    for parts, q, t in all_facts():
        for diff, rule in ACCEPT_RULES.items():
            ok = rule.accepts(star_parts=parts, quantified=q, tradeoff=t)
            gap = rule.first_gap(star_parts=parts, quantified=q, tradeoff=t)
            assert bool(gap) == (not ok), (diff, parts, q, t, ok, gap)
            assert gap in ("",) + tuple(GAP_VALUES)


def test_gap_only_names_something_actually_missing():
    """算出來的缺口必須真的缺。指錯比不指更糟——下一句會去問他講過的東西。"""
    for parts, q, t in all_facts():
        for rule in ACCEPT_RULES.values():
            gap = rule.first_gap(star_parts=parts, quantified=q, tradeoff=t)
            if not gap:
                continue
            if gap == GAP_QUANTIFICATION:
                assert not q
            elif gap == GAP_TRADEOFF:
                assert not t
            else:
                missing = {p for p in STAR_PART_KEYS if p not in parts}
                assert gap in {STAR_PART_GAP[p] for p in missing}


def test_a_result_that_lacks_a_number_is_not_called_missing():
    """1d 最明確的一次指錯:A3 講了結果只是沒數字,20/20 次都寫「缺結果」。

    算出來的版本在同一筆上必須是「結果沒量化」。
    `缺結果` 與 `結果沒量化` 分成兩個值就是為了這件事,
    而現在挑哪一個不是模型挑的。
    """
    gap = ACCEPT_RULES["中等"].first_gap(
        star_parts=("S", "A", "R"), quantified=False, tradeoff=False
    )
    assert gap == GAP_QUANTIFICATION


def test_missing_result_outranks_a_short_count():
    """缺結果排第一——它是唯一能單獨決定放不放行的一段。"""
    gap = ACCEPT_RULES["中等"].first_gap(
        star_parts=("S", "A"), quantified=False, tradeoff=False
    )
    assert gap == STAR_PART_GAP["R"]


def test_a_context_only_answer_is_asked_what_he_did():
    """只講情境的時候要問行動,不要問任務。

    PROBE_ORDER 的第一順位是行動:行為面試裡沒有行動就沒有東西可評。
    新手不要求結果,所以這一筆會落在段數不足那條。
    """
    gap = ACCEPT_RULES["新手"].first_gap(
        star_parts=("S",), quantified=False, tradeoff=False
    )
    assert gap == STAR_PART_GAP["A"]


def test_undeclared_parts_do_not_count():
    """認不得的段落不能墊高段數——墊高的方向是放行。"""
    rule = ACCEPT_RULES["困難"]
    assert not rule.accepts(
        star_parts=("S", "T", "A", "X", "Y"), quantified=True, tradeoff=True
    )
    assert rule.accepts(
        star_parts=("S", "T", "A", "R", "X"), quantified=True, tradeoff=True
    )


def test_repeated_parts_do_not_count_twice():
    assert not ACCEPT_RULES["中等"].accepts(
        star_parts=("S", "S", "S", "S"), quantified=True, tradeoff=True
    )


def test_the_golden_ladder_still_climbs():
    """階梯的放行判斷必須與 fixture 的排法一致:同一筆越嚴的難度不會變成更容易過。"""
    data = json.loads(
        Path("fixtures/golden/decision/answers_v1.json").read_text(encoding="utf-8")
    )
    for answer in data["answers"]:
        lab = answer["label"]
        seq = [
            ACCEPT_RULES[d].accepts(
                star_parts=lab["star_parts"],
                quantified=lab["quantified"],
                tradeoff=lab["tradeoff"],
            )
            for d in ("新手", "中等", "困難")
        ]
        assert seq == sorted(seq, reverse=True), (answer["answer_id"], seq)


def test_gap_values_are_the_only_source():
    """prompts 那邊轉出去的 GAP_VALUES 必須是同一個物件,不是另一份一樣的清單。"""
    from app.prompts.turn_decision import GAP_VALUES as PROMPT_GAP_VALUES

    assert PROMPT_GAP_VALUES is GAP_VALUES


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
