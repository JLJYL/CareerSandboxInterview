"""競爭者技能生成測試。純函式,讀真的 234 筆詞彙表與十個黃金案例。

重點在**三件事**:

    人數       `group_size` 含使用者,所以 AI 競爭者是三位不是四位
    精確性     同質度是封閉解,挑完要精確等於階梯上的值,不是接近
    可行域     低同質度 × 高單人壓迫度 互斥,而且要被回報出來

第三件最容易被當成 bug 修掉。它是算術:分母只有 d 個,三個人要各自踩很多個
又不能重疊,分不出來。所以 `test_low_homogeneity_and_high_pressure_collide`
存在的理由是防止有人把那條 notice 刪掉改成默默達標。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from app.pipeline.peer_generate import (
    JD_PREFIX,
    SKILLS_PER_PEER,
    JDCore,
    generate,
    homogeneity_ladder,
    names_for,
    peer_count,
    plan_skills,
    resolve_jd,
    verify,
)
from app.pipeline.skill_pool import load_pool
from app.prompts.interview_personas import personas_for

ROOT = Path(__file__).resolve().parents[1]
POOL = load_pool(
    vocab_path=ROOT / "fixtures/vocab/skills_v1.json",
    provenance_path=ROOT / "fixtures/vocab/skills_v1.provenance.json",
)
VOCAB = ROOT / "fixtures/vocab/skills_v1.json"
K = SKILLS_PER_PEER


def jd_of(case: str) -> list[str]:
    d = json.loads((ROOT / f"fixtures/golden/interview/{case}.json").read_text("utf-8"))
    return list((d.get("jd") or {}).get("requiredSkills") or [])


def core_of(case: str) -> JDCore:
    return resolve_jd(jd_of(case), POOL, vocab_path=VOCAB)


def plan(core: JDCore, **kw):
    kw.setdefault("peers", peer_count())
    return plan_skills(core, POOL, **kw)


# ---------------------------------------------------------------------------
# 人數
# ---------------------------------------------------------------------------


def test_the_group_of_four_has_three_ai_peers():
    """`group_size=4` 含使用者,所以 AI 競爭者是三位。

    第一版寫成四位。四位不會報錯,它會讓可行域整個往下移
    (c=1 的單人壓迫度上限從 0.375 掉到 0.250),也就是指標系統性偏低
    而且看起來像生成器不夠強。這條把那個數字釘在 persona 層上。
    """
    assert peer_count(4) == 3
    assert peer_count(5) == 4
    assert peer_count(3) == 2


def test_the_peer_count_is_whatever_the_persona_layer_says():
    """不是重算 `group_size - 1`,是數 `personas_for` 真的給了幾個 peer。

    那條規則的所有者是 persona 層(它還夾在 1–4 之間、還要扣掉主持人)。
    抄一份過來的話,兩份分岔時是指標偏掉而不是報錯。
    """
    for size in (3, 4, 5, 6):
        for hosts in (1, 3):
            expect = sum(
                1 for p in personas_for("group", hosts, size) if p.role == "peer"
            )
            assert peer_count(size, hosts) == expect


def test_plan_skills_refuses_to_guess_the_peer_count():
    """`peers` 是必填。有預設值的話人數錯了也不報錯。"""
    with pytest.raises(TypeError):
        plan_skills(core_of("ivw-004"), POOL)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 同質度:封閉解要精確
# ---------------------------------------------------------------------------


def test_the_ladder_is_the_closed_form():
    """階梯就是 c/(2k-c),而它的反函式是 c = 2kJ/(1+J)。

    兩個方向都驗,因為檔頭用封閉解解釋階梯為什麼長這樣,
    而程式用搜尋挑 c。兩者算的要是同一件事。
    """
    ladder = homogeneity_ladder(K)
    assert ladder[0] == 0.0
    assert ladder[K] == 1.0
    for c, j in enumerate(ladder):
        assert abs(j - c / (2 * K - c)) < 1e-12
        if j < 1.0:
            assert abs(2 * K * j / (1 + j) - c) < 1e-9


def test_every_reachable_homogeneity_is_hit_exactly():
    """目標落在階梯上的時候,實際值要**精確**相等,不是接近。

    構造法保證任兩人交集恰好是 c,所以每一對的 Jaccard 都一樣,
    平均值沒有誤差。拿到近似值就是構造走偏了。
    """
    core = core_of("ivw-008")
    for target in homogeneity_ladder(K):
        got = plan(core, homogeneity=target, pressure=0.0, seed=3)
        assert got.homogeneity == pytest.approx(target, abs=1e-12)


def test_every_pair_shares_exactly_the_same_count():
    """封閉解的前提:任兩人的交集大小相同。

    不成立的話平均 Jaccard 還是算得出來,但它不再等於 c/(2k-c),
    於是「目標 0.5 拿到 0.5」變成巧合而不是保證。
    """
    core = core_of("ivw-008")
    for target in homogeneity_ladder(K):
        got = plan(core, homogeneity=target, pressure=0.0, seed=5)
        sizes = {
            len(set(got.peers[a]) & set(got.peers[b]))
            for a in range(len(got.peers))
            for b in range(a + 1, len(got.peers))
        }
        assert len(sizes) == 1, f"目標 {target} 的交集大小不一致:{sizes}"


def test_a_target_between_two_rungs_is_reported_not_hidden():
    """0.6 在 k=6 的階梯上不存在,所以要說一聲。

    默默給 0.5 的話,呼叫端會以為旋鈕壞了而去調別的地方。
    """
    got = plan(core_of("ivw-008"), homogeneity=0.6, pressure=0.0)
    assert got.homogeneity == pytest.approx(0.5)
    assert any("不在 k=6 的階梯上" in n for n in got.notices)


def test_peers_each_get_exactly_k_distinct_skills():
    core = core_of("ivw-008")
    for target in homogeneity_ladder(K):
        got = plan(core, homogeneity=target, pressure=0.0, seed=11)
        for p in got.peers:
            assert len(set(p)) == K


# ---------------------------------------------------------------------------
# 可行域:這是算術不是 bug
# ---------------------------------------------------------------------------


def test_low_homogeneity_and_high_pressure_collide():
    """同質度 0 的時候單人壓迫度上不去,而且要被回報。

    分母 8、三個人、各自要踩 4 個不重疊的職缺技能 = 要 12 個,只有 8 個。
    所以上限是 2/8 = 0.25。這條不是「調參調得不夠好」,是分不出來。
    """
    got = plan(core_of("ivw-008"), homogeneity=0.0, pressure=0.5)
    assert got.homogeneity == 0.0
    assert got.mean_pressure == pytest.approx(0.25)
    assert any("做不出來" in n and "可行上限" in n for n in got.notices)


def test_the_union_is_what_rises_when_homogeneity_falls():
    """單人壓迫度固定不動,同質度降 → **聯集壓迫度升**。

    這是三個數字要一起看的理由,也是最強的那個說法:
    同質度這個旋鈕可以在單人數字完全不變的情況下把聯集從 0.25 推到 0.75。
    使用者感受到的「被包圍」是聯集買的,不是每人幾個技能買的。

    第一版這條寫成「兩個目標都推到上限再比聯集」,而那樣兩邊都是 6/8:
    c=6 是三個人共用六個職缺技能,c=0 是三個人各拿兩個不重疊的,
    用掉的職缺技能數剛好都是 6。推到上限比不出東西,要固定一個變另一個。
    """
    core = core_of("ivw-008")
    tight = plan(core, homogeneity=1.0, pressure=0.25, seed=2)
    loose = plan(core, homogeneity=0.0, pressure=0.25, seed=2)
    assert tight.mean_pressure == loose.mean_pressure == pytest.approx(0.25)
    assert tight.union_pressure == pytest.approx(0.25)
    assert loose.union_pressure == pytest.approx(0.75)


def test_the_reported_numbers_are_recomputed_not_the_target():
    """回報的是實際值。拿一個做不到的目標進去,出來的數字要不等於它。

    回報目標值等於把願望寫成事實。這條是「過度宣稱比不宣稱糟」的機械版。
    """
    got = plan(core_of("ivw-008"), homogeneity=0.0, pressure=1.0)
    assert got.mean_pressure != pytest.approx(1.0)
    assert verify(got.peers, got.denominator) == (
        got.homogeneity,
        got.mean_pressure,
        got.union_pressure,
    )


def test_pressure_ceiling_is_k_over_the_denominator():
    """一個人只有 k 個技能,踩不到比 k 更多的要求。"""
    small = core_of("ivw-003")  # 可覆蓋 5 條
    big = core_of("ivw-006")  # 可覆蓋 23 條,分母取前 8
    assert small.pressure_ceiling(K) == pytest.approx(1.0)  # min(6,5)/5
    assert big.pressure_ceiling(K) == pytest.approx(6 / 8)


# ---------------------------------------------------------------------------
# 隨機性的方向
# ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_and_another_seed_does_not():
    """候選人要多樣,但量測要可重現。所以可重現**給定 seed**。

    `interviewer_spec` 記的那條:面試官相反,同一份履歷每次該問一樣的問題。
    """
    req = jd_of("ivw-004")
    a = generate(req, seed=1, pool=POOL, vocab_path=VOCAB)
    b = generate(req, seed=1, pool=POOL, vocab_path=VOCAB)
    c = generate(req, seed=2, pool=POOL, vocab_path=VOCAB)
    assert a.peers == b.peers
    assert a.peers != c.peers


def test_different_seeds_keep_the_same_metrics():
    """換 seed 換人,但三個數字不該跟著跳。

    會跳的話「同質度 0.5」就不是一個設定而是一次抽獎。
    """
    req = jd_of("ivw-004")
    runs = [generate(req, seed=s, pool=POOL, vocab_path=VOCAB) for s in range(6)]
    assert len({r.homogeneity for r in runs}) == 1
    assert len({r.mean_pressure for r in runs}) == 1
    assert len({r.union_pressure for r in runs}) == 1


# ---------------------------------------------------------------------------
# 職缺解析:用真資料
# ---------------------------------------------------------------------------


def test_every_golden_jd_entry_resolves_to_the_vocab():
    """118 條職缺要求,一條都不該對不到。

    這個數字是選「沿用 SurfaceIndex 而不另寫精確字典」的依據:
    精確字典是 117/118,差的那條是全形括號的「設計導入製造執行系統（MES）」。
    """
    total = unresolved = partial = in_pool = 0
    for path in sorted(glob.glob(str(ROOT / "fixtures/golden/interview/ivw-*.json"))):
        d = json.loads(Path(path).read_text("utf-8"))
        req = list((d.get("jd") or {}).get("requiredSkills") or [])
        core = resolve_jd(req, POOL, vocab_path=VOCAB)
        total += core.requested
        unresolved += len(core.unresolved)
        partial += len(core.partial)
        in_pool += len(core.coverable)
    assert total == 118
    assert unresolved == 0, f"對不到詞彙表的有 {unresolved} 條"
    assert partial == 0, f"只命中一部分的有 {partial} 條"
    assert in_pool == 95, f"在技能池裡的應該是 95,拿到 {in_pool}"


def test_requirements_outside_the_pool_stay_out_of_the_denominator():
    """對到詞彙表但不在技能池裡的要求不算進分母。

    那些幾乎都是純 L1_onet 的抽象職能,技能池裡不存在,
    所以**沒有任何競爭者可能踩到**。算進分母的話壓迫度有一個跟 k 無關的
    天花板,而讀數字的人會以為是生成器不夠強。
    """
    core = core_of("ivw-001")  # 11 條,可覆蓋 7,池子外 4
    assert core.requested == 11
    assert len(core.coverable) == 7
    assert len(core.out_of_pool) == 4
    assert not core.lossless
    assert set(core.denominator()) <= set(core.coverable)

    got = plan(core)
    assert any("不在技能池裡" in n for n in got.notices)


def test_a_short_jd_shrinks_the_denominator_and_says_so():
    core = core_of("ivw-003")  # 可覆蓋 5 條,少於 JD_PREFIX
    got = plan(core)
    assert len(got.denominator) == 5 < JD_PREFIX
    assert any("分母用 5 不是 8" in n for n in got.notices)


def test_a_failed_jd_extraction_still_produces_peers():
    """職缺抽取失敗(recall 目前 0.375,這條路會走到)不該讓生成爆掉。

    回的是完整的三位競爭者、壓迫度 0、而且 notice 說清楚壓迫度沒有分母。
    壓迫度回 0 而不是 1 跟 `coverage` 同一個方向:
    「沒有分母」不該長得像「完全命中」。
    """
    core = resolve_jd([], POOL, vocab_path=VOCAB)
    assert core.coverable == ()
    got = plan(core)
    assert len(got.peers) == peer_count()
    assert all(len(set(p)) == K for p in got.peers)
    assert got.mean_pressure == 0.0
    assert got.union_pressure == 0.0
    assert got.homogeneity == pytest.approx(0.5)
    assert any("沒有分母" in n for n in got.notices)


def test_the_jd_order_is_the_priority_order():
    """分母取前 N 個,而「前」是職缺原本的順序。

    合約明訂 `required_skills` 順序即優先序,`gap.JD_POSITION_DECAY` 依賴它。
    這裡重排的話,壓迫度就不是「踩到最關鍵的幾成」。
    """
    req = jd_of("ivw-006")  # 31 條,可覆蓋 23
    core = resolve_jd(req, POOL, vocab_path=VOCAB)
    first_eight = core.denominator(8)
    assert first_eight == core.coverable[:8]
    assert len(first_eight) == 8


# ---------------------------------------------------------------------------
# 分母外那幾格
# ---------------------------------------------------------------------------


def test_the_default_filler_admits_it_is_random():
    """均勻亂挑的那條路要自己說一聲。

    拿 ivw-004(前端)實跑,亂挑給出「Vmware、Canva、社群媒體經營管理、
    ERP 系統」——三個數字全部達標而三份履歷都不可信。
    池子裡沒有領域訊號(ucan_code 與 onet_code 實測 0/152),
    所以程式挑不好,這條路只適合測試與量測。
    """
    got = plan(core_of("ivw-004"))
    assert any("均勻亂挑" in n for n in got.notices)


def test_a_supplied_deck_does_not_move_the_metrics():
    """分母外那幾格換成別的技能,三個數字不變。

    這是「交給模型填」成立的前提:兩個壓迫度只跟分母內的技能有關,
    同質度只需要那幾格互不重複。所以模型可以挑領域貼合的,指標仍然精確。
    """
    core = core_of("ivw-004")
    base = plan(core, seed=7)
    deck = [s.skill_id for s in POOL if s.skill_id not in set(core.coverable)][:40]
    curated = plan(core, seed=7, off_jd_deck=deck)
    assert (curated.homogeneity, curated.mean_pressure, curated.union_pressure) == (
        base.homogeneity,
        base.mean_pressure,
        base.union_pressure,
    )
    assert any(set(a) != set(b) for a, b in zip(curated.peers, base.peers))


def test_a_deck_from_another_pool_is_refused():
    """混用兩個池子會讓指標對不上,所以拋錯而不是默默跳過。"""
    with pytest.raises(ValueError, match="不在技能池裡"):
        plan(core_of("ivw-004"), off_jd_deck=["sk:deadbeef"] * 30)


def test_a_deck_containing_a_jd_requirement_is_refused():
    """分母內的技能不能從「分母外」那條路進來。

    進來的話某個人會多踩一個職缺要求,而回報的壓迫度是算出來的,
    於是數字對了但不是這個構造算出來的 —— 旋鈕與結果脫鉤。
    """
    core = core_of("ivw-004")
    with pytest.raises(ValueError, match="是職缺要求"):
        plan(core, off_jd_deck=list(core.coverable))


def test_a_deck_too_small_raises_instead_of_short_peers():
    """技能不夠就拋錯,不給短少的集合。

    一個只有四個技能的「六技能競爭者」會讓三個指標全部算錯,
    而錯的方向是壓迫度偏低——看起來像競爭者不夠強,不像程式壞了。
    """
    core = core_of("ivw-004")
    deck = [s.skill_id for s in POOL if s.skill_id not in set(core.coverable)][:2]
    with pytest.raises(ValueError, match="分母外的技能不夠"):
        plan(core, off_jd_deck=deck)


def test_off_jd_lists_what_is_not_in_the_denominator():
    """`off_jd` 是「他還會別的」。

    給模型寫背景敘述要用:一份只寫職缺要求的履歷讀起來像照職缺抄的。
    """
    got = plan(core_of("ivw-004"), seed=7)
    d = set(got.denominator)
    for own, rest in zip(got.peers, got.off_jd):
        assert set(rest) == set(own) - d
        assert rest


# ---------------------------------------------------------------------------
# 名字
# ---------------------------------------------------------------------------


def test_names_come_out_in_the_order_given():
    got = plan(core_of("ivw-004"), seed=7)
    ids = got.denominator
    names = names_for(ids, POOL)
    assert len(names) == len(ids)
    assert names[0] in {s.display for s in POOL}


def test_an_unknown_id_is_skipped_not_placeheld():
    """對不到的 id 直接跳過。

    放佔位字串的話,呼叫端混用兩個池子的錯誤會變成競爭者履歷上的一行怪字,
    而那比少一個技能難追。
    """
    real = POOL[0].skill_id
    assert names_for([real, "sk:nope"], POOL) == (POOL[0].display,)
