"""技能池測試。純函式,讀真的 234 筆詞彙表。

重點在**一件事**:篩的方向對不對。

成員 A 的結論是「像履歷看 layer,別用 prefix」,而 prefix 跟 zhang_matched
在這份資料上是一對一的,所以用 prefix 篩等於用背書篩,而背書那批
包含一堆 O*NET 抽象職能。方向會剛好反過來。

這個檔用真資料把那件事釘住,不是用手捏的例子。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.skill_pool import (
    MAX_NAME_CHARS,
    NOT_A_SOURCE_LAYERS,
    OPTIONAL_LAYERS,
    PROVENANCE_PATH,
    RESUME_LIKE_LAYERS,
    audit,
    load_pool,
)

ROOT = Path(__file__).resolve().parents[1]
PROV = json.loads((ROOT / PROVENANCE_PATH).read_text(encoding="utf-8"))
LAYERS = {sid: set(p.get("layers") or ()) for sid, p in PROV.items()}


def pool(**kw):
    return load_pool(
        vocab_path=ROOT / "fixtures/vocab/skills_v1.json",
        provenance_path=ROOT / PROVENANCE_PATH,
        **kw,
    )


# ---------------------------------------------------------------------------
# 大小對得上成員 A 算的
# ---------------------------------------------------------------------------


def test_default_pool_is_the_number_a_computed():
    """L0 ∪ L3 = 152。成員 A 說「直接用」,所以這個數字要對得上。

    對不上的話不是四捨五入問題,是篩選規則跟他算的不是同一件事。
    """
    assert len(pool()) == 152


def test_adding_l2_matches_too():
    assert len(pool(layers=RESUME_LIKE_LAYERS + OPTIONAL_LAYERS)) == 171


def test_pure_onet_entries_are_all_excluded():
    """純 L1_onet 那 34 筆一筆都不該在池子裡。

    它們是「主動聆聽」「判斷與決策」這類抽象職能,
    拿去當競爭者履歷技能會很假——那是這個池存在的理由。
    """
    only_onet = {s for s, ls in LAYERS.items() if ls == {"L1_onet"}}
    assert len(only_onet) == 34
    assert not {s.skill_id for s in pool()} & only_onet


# ---------------------------------------------------------------------------
# 方向:prefix 不能當篩子
# ---------------------------------------------------------------------------


def test_prefix_is_just_zhang_matched_in_disguise():
    """`sk:` 與 `zhang_matched=True` 在這份資料上一對一。

    所以「用 prefix 篩」跟「用背書篩」是同一件事,而後者方向是反的。
    這條在資料上證明那個等價,不是引用別人的話。
    """
    for sid, p in PROV.items():
        assert sid.startswith("sk:") == bool(p.get("zhang_matched")), sid


def test_filtering_by_prefix_would_invert_the_selection():
    """用 prefix 篩會砍掉最像履歷的、留下最抽象的。

    這是不照 A 的建議做會發生的事,用真資料算出來。
    """
    ids = {s.skill_id for s in pool()}
    by_prefix = {s for s in PROV if s.startswith("sk:")}
    only_onet = {s for s, ls in LAYERS.items() if ls == {"L1_onet"}}

    assert only_onet <= by_prefix, "最抽象的那 34 筆全部會被 prefix 篩收進來"
    lost = len({s for s in ids if s.startswith("skm:")})
    assert lost > 100, f"用 prefix 篩會丟掉 {lost} 筆最像履歷的工具名"


def test_the_most_resume_like_tool_names_survive():
    """CSS、HTML、Illustrator、Adobe Photoshop 要在池子裡。

    這四筆同時屬於 L3_market 與 L4_merge_alias。第一版把 L4 寫成硬排除,
    結果砍掉的正好是整份詞彙表裡最像履歷的那幾個——
    L4 那個標籤講的是「這一筆吸收了別名」,跟品質正交。
    """
    names = {s.display for s in pool()}
    assert {"CSS", "HTML", "Illustrator", "Adobe Photoshop"} <= names


def test_an_entry_whose_only_layer_is_not_a_source_is_refused_even_via_also():
    """`also` 不能繞過規則。

    「我特別指定的」不等於「可以繞過」,不然那條規則遲早被繞光。
    """
    sole = [s for s, ls in LAYERS.items() if ls and ls <= set(NOT_A_SOURCE_LAYERS)]
    assert sole, "資料裡應該有只屬於非來源層的條目,不然這條測試沒在測東西"
    got = {s.skill_id for s in pool(also=sole)}
    assert not got & set(sole)


# ---------------------------------------------------------------------------
# 形狀
# ---------------------------------------------------------------------------


def test_the_pool_is_sorted_and_reproducible():
    """生成要可重現,而 dict 的順序不是合約。"""
    a, b = pool(), pool()
    assert [s.skill_id for s in a] == [s.skill_id for s in b]
    assert [s.skill_id for s in a] == sorted(s.skill_id for s in a)


def test_every_skill_has_a_display_name():
    """沒有名字的技能寫不進履歷。"""
    assert all(s.display for s in pool())


def test_names_are_within_the_length_limit():
    assert all(len(s.display) <= MAX_NAME_CHARS for s in pool())


def test_the_audit_shows_how_lopsided_the_pool_is():
    """L3_market 佔絕大多數這件事要看得見。

    它決定了這個池的品質上限,而那一層是爬蟲頻次來的、沒有人逐筆看過。
    不印出來的話,生成出奇怪的技能名時會先懷疑生成器。
    """
    a = audit(pool())
    assert a["_total"] == 152
    assert a["L3_market"] > a["L0_curated"] * 3, "L3 佔壓倒性多數,這是已知風險"
    assert a["_longest_name"] <= MAX_NAME_CHARS


def test_interview_coverage_layer_is_not_taken_wholesale():
    """L5 預設不收。

    它的措辭取自真實逐字稿,裡面有「電話接聽與人員接待事項」這種整句。
    整層收會讓競爭者履歷出現一整句話當技能名。
    """
    l5_only = {s for s, ls in LAYERS.items() if ls == {"L5_interview_coverage"}}
    assert l5_only, "L5 應該有單層成員"
    assert not {s.skill_id for s in pool()} & l5_only
