"""模型那一側的測試:兩次呼叫、兩次驗證。讀真的 234 筆詞彙表。

重點在**三件事**:

    分派       兩個 prompt 要分得開,假 LLM 依子字串分派已經誤導過三次
    對人       人物用 id 對,不是用順序
    誇大       背景宣稱了它沒有的職缺技能,程式要抓得到

第三件是這一支唯一有意思的驗證。背景寫得好不好程式不知道,
但「它說自己做過 React 而技能清單裡沒有 React」查得出來,
而那種誇大剛好會讓體感壓迫度高過三個指標說的數字。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.pipeline.peer_generate import (
    SKILLS_PER_PEER,
    names_for,
    off_jd_needed,
    peer_count,
    plan_skills,
    resolve_jd,
)
from app.pipeline.peer_profile import (
    FALLBACK,
    FIRST_PERSON,
    AdjacentCache,
    LLM,
    PROFILE_FALLBACK,
    PROFILE_LLM,
    TOPPED_UP,
    build_peers,
    find_overclaims,
    pick_adjacent,
    read_profiles,
)
from app.pipeline.skill_pool import load_pool
from app.prompts.interview_personas import personas_for
from app.prompts.peer_profile import (
    ADJACENT_MARK,
    ADJACENT_TARGET,
    SCHOOLS,
    NAME_MAX_CHARS,
    PROFILE_MARK,
    compose_adjacent_prompt,
    compose_profile_prompt,
    render_pool,
)

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "fixtures/vocab/skills_v1.json"
POOL = load_pool(
    vocab_path=VOCAB, provenance_path=ROOT / "fixtures/vocab/skills_v1.provenance.json"
)
BY_NAME = {s.display: i for i, s in enumerate(POOL, 1)}
PEERS = peer_count()
SPECS = [p for p in personas_for("group", 1, 4) if p.role == "peer"]

FRONTEND = (
    "Git",
    "TypeScript",
    "Figma",
    "Node.js",
    "SQL",
    "Python",
    "Java",
    "RESTful API",
    "軟體程式設計",
    "使用者體驗設計",
    "網頁設計",
)


def jd_of(case: str = "ivw-004") -> list[str]:
    d = json.loads((ROOT / f"fixtures/golden/interview/{case}.json").read_text("utf-8"))
    return list((d.get("jd") or {}).get("requiredSkills") or [])


def core_of(case: str = "ivw-004"):
    return resolve_jd(jd_of(case), POOL, vocab_path=VOCAB)


def adjacent_reply(names=FRONTEND) -> str:
    return json.dumps([BY_NAME[n] for n in names if n in BY_NAME])


def profile_reply(user: str, **over) -> str:
    ids = [
        line.split("應徵者 ")[1].rstrip("】")
        for line in user.splitlines()
        if "【應徵者" in line
    ]
    rows = []
    for i, pid in enumerate(ids, 1):
        row = {
            "id": pid,
            "name": f"林{'一二三四'[i - 1]}哲",
            "school": "政治大學",
            "background": "政治大學資訊管理畢業,在電商做過半年的前端開發。",
            "style": "講話快,先講結論",
        }
        row.update(over.get(pid, {}))
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False)


def llm_of(*, adjacent=None, profile=None, seen=None):
    """依 MARK 分派的假 LLM。沒有分支吃的 prompt 直接炸,不要靜默。"""

    def call(system: str, user: str) -> str:
        if seen is not None:
            seen.append((system, user))
        if ADJACENT_MARK in system:
            if adjacent is None:
                raise RuntimeError("adjacent down")
            return adjacent(user) if callable(adjacent) else adjacent
        if PROFILE_MARK in system:
            if profile is None:
                raise RuntimeError("profile down")
            return profile(user) if callable(profile) else profile
        raise AssertionError(f"沒有分支吃這個 prompt:{system[:40]!r}")

    return call


def cast_of(*, adjacent=None, profile=None, seen=None, **kw):
    kw.setdefault("title", "前端網頁工程師")
    kw.setdefault("required", jd_of())
    kw.setdefault("seed", 7)
    kw.setdefault("pool", POOL)
    kw.setdefault("vocab_path", VOCAB)
    return asyncio.run(
        build_peers(llm=llm_of(adjacent=adjacent, profile=profile, seen=seen), **kw)
    )


# ---------------------------------------------------------------------------
# 分派
# ---------------------------------------------------------------------------


def test_the_two_prompts_cannot_be_confused():
    """兩個 MARK 各只出現在自己那個 prompt 裡。

    假 LLM 依 system prompt 的子字串分派,已經誤導過三次
    (1c 的決策分支、2c 的事實分支、情緒批次),症狀每次都一樣:
    某一支收到另一支的回覆,而測試全綠。
    """
    adj, prof = compose_adjacent_prompt(), compose_profile_prompt()
    assert ADJACENT_MARK in adj and ADJACENT_MARK not in prof
    assert PROFILE_MARK in prof and PROFILE_MARK not in adj


def test_a_prompt_nobody_dispatches_on_blows_up():
    """假 LLM 碰到沒有分支的 prompt 要炸,不要回空字串。

    回空字串的話,prompt 改字之後分派失效會表現成「模型沒答話」,
    而那條路有 fallback,所以測試還是綠的。
    """
    call = llm_of(adjacent="[]", profile="[]")
    try:
        call("完全不相干的 system prompt", "")
    except AssertionError:
        return
    raise AssertionError("應該炸")


def test_the_pool_is_numbered_from_one():
    """編號從 1 開始,而程式用 n-1 索引。差一就全部對到隔壁技能。"""
    text = render_pool(["甲", "乙", "丙"])
    assert text.splitlines() == ["1. 甲", "2. 乙", "3. 丙"]


def test_the_prompt_asks_for_more_than_the_minimum():
    """prompt 要的數量要高過構造的最低需求。

    剛好等於需求的話,每個 seed 只是把同一批技能重排,換 seed 換不到人,
    而「同一個職缺每次跑該產出不一樣的競爭者」是這個模組的設計要求。
    """
    assert ADJACENT_TARGET > off_jd_needed(core_of(), peers=PEERS)
    assert ADJACENT_TARGET >= PEERS * SKILLS_PER_PEER


# ---------------------------------------------------------------------------
# 第一次呼叫的驗證
# ---------------------------------------------------------------------------


def test_a_number_outside_the_list_is_dropped_and_counted():
    pick = asyncio.run(
        pick_adjacent(
            title="前端",
            core=core_of(),
            pool=POOL,
            llm=llm_of(adjacent=json.dumps([1, 2, 9999, 0, -5])),
            need=2,
        )
    )
    assert pick.out_of_range == 3
    assert pick.kept == 2


def test_a_non_integer_entry_is_dropped():
    """模型回技能名而不是編號的時候要丟掉,不要去猜。

    猜的話就等於在這裡又寫了一份名稱解析,而兩份解析遲早分岔。
    """
    pick = asyncio.run(
        pick_adjacent(
            title="前端",
            core=core_of(),
            pool=POOL,
            llm=llm_of(adjacent=json.dumps(["Git", None, {"a": 1}, 3])),
            need=1,
        )
    )
    assert pick.kept == 1


def test_a_pick_that_lands_on_a_jd_requirement_is_dropped():
    """職缺要求不能從「分母外」那條路進來,而且要數出來。

    `on_jd` 是 prompt 規則 4 失效的直接指標。長期偏高的話那條規則
    該換做法,不是寫得更長——1d 那輪就是這樣,prompt 改兩次都沒生效。
    """
    core = core_of()
    on_jd = [i for i, s in enumerate(POOL, 1) if s.skill_id in set(core.coverable)]
    assert on_jd, "ivw-004 的職缺要求應該有在池子裡的"
    pick = asyncio.run(
        pick_adjacent(
            title="前端",
            core=core,
            pool=POOL,
            llm=llm_of(adjacent=json.dumps(on_jd + [BY_NAME["Git"]])),
            need=1,
        )
    )
    assert pick.on_jd == len(on_jd)
    assert set(pick.ids).isdisjoint(set(core.coverable))


def test_a_failed_call_returns_nothing_rather_than_a_fake_deck():
    """呼叫失敗回空清單與 fallback,不偽造一份 deck。

    偽造的話「模型沒答話」會長得像「模型挑的」,而兩者的輸出品質差很多。
    """
    pick = asyncio.run(
        pick_adjacent(title="前端", core=core_of(), pool=POOL, llm=llm_of(), need=6)
    )
    assert pick.source == FALLBACK
    assert pick.ids == ()


def test_a_short_pick_is_topped_up_not_discarded():
    """挑太少就補,不要整批丟掉。部分訊號比沒有訊號好。"""
    pick = asyncio.run(
        pick_adjacent(
            title="前端",
            core=core_of(),
            pool=POOL,
            llm=llm_of(adjacent=json.dumps([BY_NAME["Git"]])),
            need=6,
        )
    )
    assert pick.source == TOPPED_UP
    assert pick.kept == 6
    assert POOL[BY_NAME["Git"] - 1].skill_id in pick.ids


def test_the_requirement_is_the_real_one_not_the_worst_case():
    """最低需求要照設定算,不是 `peers × k`。

    用最壞情況的話,預設設定(同質度 0.5,只要 6 個)碰到模型挑 10 個
    也會被判不足,然後補 8 個亂挑的進去稀釋。實跑看過一次:
    前端職缺的競爭者因此拿到「規劃與管理防火牆」。
    """
    core = core_of()
    assert off_jd_needed(core, peers=PEERS) == 6  # 預設 0.5 / 0.5
    assert off_jd_needed(core, peers=PEERS, homogeneity=0.0) == 12
    # 真正的最壞情況是兩個旋鈕都歸零:沒有共用、也沒有人踩職缺要求
    assert (
        off_jd_needed(core, peers=PEERS, homogeneity=0.0, pressure=0.0)
        == PEERS * SKILLS_PER_PEER
    )

    cast = cast_of(adjacent=adjacent_reply(FRONTEND[:10]), profile=profile_reply)
    assert cast.adjacent.source == LLM, "十個夠用,不該被判定不足"


def test_the_denominator_is_printed_as_resolved_names():
    """user 訊息裡的職缺要求用解析後的名稱。

    印原文的話「C#程式開發」會讓模型以為「C#」是清單外的東西而去挑它,
    然後那一筆被當成撞 JD 丟掉——等於白跑一次。
    """
    seen: list = []
    cast_of(adjacent=adjacent_reply(), profile=profile_reply, seen=seen)
    user = next(u for s, u in seen if ADJACENT_MARK in s)
    assert "HTML" in user and "JavaScript" in user


# ---------------------------------------------------------------------------
# 第二次呼叫的驗證
# ---------------------------------------------------------------------------


def plan_of(**kw):
    kw.setdefault("peers", PEERS)
    kw.setdefault("seed", 7)
    return plan_skills(core_of(), POOL, **kw)


def profiles_of(reply, **kw):
    return asyncio.run(
        read_profiles(
            title="前端",
            plan=plan_of(),
            pool=POOL,
            llm=llm_of(profile=reply, adjacent="[]"),
            vocab_path=VOCAB,
            **kw,
        )
    )


def test_profiles_are_matched_by_id_not_by_order():
    """模型把陣列順序倒過來,每個人還是要拿到自己的那一份。

    靠順序對人的話,模型重排就會把「強勢」的背景配到「親切」身上,
    而那不會報錯——它會變成一個人格與背景不符的競爭者。
    """

    def reply(user: str) -> str:
        rows = json.loads(profile_reply(user))
        rows.reverse()
        for row in rows:
            row["background"] = f"政治大學畢業,{row['id']} 的背景"
        return json.dumps(rows, ensure_ascii=False)

    out = profiles_of(reply)
    assert [p.persona_id for p in out] == [s.id for s in SPECS[:PEERS]]
    for p in out:
        assert p.background == f"政治大學畢業,{p.persona_id} 的背景"


def test_the_order_is_the_roster_order():
    """回傳順序照 `personas_for` 的 peer 順序,也就是前端 roster 的順序。"""
    out = profiles_of(profile_reply)
    assert [p.persona_id for p in out] == [
        "peer_assertive",
        "peer_logic",
        "peer_friendly",
    ]


def test_a_missing_peer_falls_back_to_the_fixed_persona():
    """模型漏掉一位,那一位用固定人設,其餘照用模型的。

    整批退回的話一個欄位格式錯就毀掉三個人物。
    """

    def reply(user: str) -> str:
        rows = json.loads(profile_reply(user))
        return json.dumps(rows[1:], ensure_ascii=False)

    out = profiles_of(reply)
    assert out[0].source == PROFILE_FALLBACK
    assert out[0].name == SPECS[0].display_name
    assert out[0].background == SPECS[0].blurb
    assert all(p.source == PROFILE_LLM for p in out[1:])


def test_an_overlong_name_is_replaced_not_truncated():
    """過長的欄位退回預設,不截斷。

    截一半的句子是壞資料不是短資料,而它會直接出現在使用者看到的 roster 上。
    """
    long_name = "資深前端工程師小林同學"
    assert len(long_name) > NAME_MAX_CHARS
    out = profiles_of(lambda u: profile_reply(u, **{SPECS[0].id: {"name": long_name}}))
    assert out[0].name == SPECS[0].display_name
    assert "name" in out[0].repaired
    assert out[0].source == PROFILE_LLM, "欄位不合格不等於呼叫失敗"


def test_a_name_colliding_with_a_display_name_is_refused():
    """生成的名字撞到現有顯示名的話不收。

    收了的話 roster 上會有兩個「AI-邏輯」,而說話者修復是靠顯示名比對的。
    """
    out = profiles_of(
        lambda u: profile_reply(u, **{SPECS[0].id: {"name": SPECS[1].display_name}})
    )
    assert "name" in out[0].repaired


def test_a_blank_style_is_recorded_but_not_substituted():
    """風格空的就空的,不拿 blurb 去填。

    那一句是給 live 那一端接話用的,塞一句語意不同的東西進去
    比留空更難追——而留空的後果只是 live 端少一個提示。
    """
    out = profiles_of(lambda u: profile_reply(u, **{SPECS[0].id: {"style": ""}}))
    assert out[0].style == ""
    assert "style" in out[0].repaired


def test_the_stance_comes_from_the_persona_layer():
    """人格指示直接用 `PersonaSpec.stance`,不另寫一份。

    抄一份過來的話調 persona 會調不到這裡,而症狀是
    「我改了人格但競爭者的背景沒變」。
    """
    seen: list = []
    cast_of(adjacent=adjacent_reply(), profile=profile_reply, seen=seen)
    user = next(u for s, u in seen if PROFILE_MARK in s)
    for spec in SPECS[:PEERS]:
        assert spec.stance[:20] in user


def test_the_skills_in_the_message_are_names_not_ids():
    """給模型的是技能名稱,不是 skill_id。

    餵 id 的話它會把 `skm:1b24149e61` 寫進背景敘述。
    """
    seen: list = []
    cast_of(adjacent=adjacent_reply(), profile=profile_reply, seen=seen)
    user = next(u for s, u in seen if PROFILE_MARK in s)
    assert "skm:" not in user and "sk:" not in user
    assert "HTML" in user


# ---------------------------------------------------------------------------
# 誇大
# ---------------------------------------------------------------------------


def test_a_background_claiming_a_skill_it_lacks_is_recorded():
    """背景宣稱了它沒有的職缺技能要被抓到。

    那種誇大會讓使用者感受到的壓迫度高過三個指標說的數字,
    也就是指標與體感脫鉤——那比文案寫得差嚴重。
    """
    plan = plan_of()
    lacking = [s for s in plan.denominator if s not in set(plan.peers[0])]
    assert lacking, "第一位應該有幾個分母技能沒拿到"
    name = names_for(lacking[:1], POOL)[0]
    got = find_overclaims(
        f"台科大畢業,主力是 {name} 的大型專案。",
        plan.peers[0],
        plan.denominator,
        vocab_path=VOCAB,
    )
    assert got == tuple(lacking[:1])


def test_a_background_claiming_a_skill_it_has_is_clean():
    plan = plan_of()
    mine = [s for s in plan.denominator if s in set(plan.peers[0])]
    assert mine
    name = names_for(mine[:1], POOL)[0]
    assert (
        find_overclaims(
            f"寫了三年 {name}。", plan.peers[0], plan.denominator, vocab_path=VOCAB
        )
        == ()
    )


def test_overclaims_only_cover_the_denominator():
    """分母外的技能被提到不算誇大。

    背景提到清單外的東西大多是無害的顏色(「在社團辦過活動」)。
    全部抓的話這個訊號會被雜訊淹掉,然後沒人看它。
    """
    plan = plan_of()
    outside = next(
        s.display
        for s in POOL
        if s.skill_id not in set(plan.denominator)
        and s.skill_id not in set(plan.peers[0])
    )
    assert (
        find_overclaims(
            f"也做過 {outside}。", plan.peers[0], plan.denominator, vocab_path=VOCAB
        )
        == ()
    )


def test_an_empty_background_has_nothing_to_overclaim():
    plan = plan_of()
    assert find_overclaims("", plan.peers[0], plan.denominator, vocab_path=VOCAB) == ()


def test_an_overclaim_shows_up_in_the_notices():
    plan = plan_of()
    lacking = [s for s in plan.denominator if s not in set(plan.peers[0])]
    name = names_for(lacking[:1], POOL)[0]
    cast = cast_of(
        adjacent=adjacent_reply(),
        profile=lambda u: profile_reply(
            u, **{SPECS[0].id: {"background": f"政治大學畢業,主力是 {name}。"}}
        ),
    )
    assert any("宣稱了它沒有的職缺技能" in n for n in cast.notices)


# ---------------------------------------------------------------------------
# 串起來
# ---------------------------------------------------------------------------


def test_both_calls_failing_still_produces_a_full_cast():
    """兩次呼叫都掛掉也要回得出完整一組。

    不好看,但群面開得起來,而且 notices 說清楚發生了什麼。
    """
    cast = cast_of()
    assert len(cast.profiles) == PEERS
    assert all(len(p.skills) == SKILLS_PER_PEER for p in cast.profiles)
    assert all(p.source == PROFILE_FALLBACK for p in cast.profiles)
    assert cast.adjacent.source == FALLBACK
    assert any("退回亂挑" in n for n in cast.notices)
    assert any("退回固定人設" in n for n in cast.notices)


def test_the_model_side_does_not_move_the_metrics():
    """模型挑的 deck 換成亂挑的,三個數字不變。

    這是把分母外那幾格交給模型的前提:兩個壓迫度只跟分母內的技能有關,
    同質度只需要那幾格互不重複。不成立的話旋鈕與結果就脫鉤了。
    """
    with_model = cast_of(adjacent=adjacent_reply(), profile=profile_reply)
    without = plan_of()
    assert (
        with_model.plan.homogeneity,
        with_model.plan.mean_pressure,
        with_model.plan.union_pressure,
    ) == (without.homogeneity, without.mean_pressure, without.union_pressure)


def test_the_same_seed_reproduces_the_whole_cast():
    kw = dict(adjacent=adjacent_reply(), profile=profile_reply)
    a, b = cast_of(**kw), cast_of(**kw)
    assert a.plan.peers == b.plan.peers
    assert [p.name for p in a.profiles] == [p.name for p in b.profiles]


def test_a_failed_jd_extraction_still_produces_a_cast():
    """職缺抽取失敗(recall 目前 0.375)不該讓群面開不起來。"""
    cast = cast_of(required=[], adjacent=adjacent_reply(), profile=profile_reply)
    assert len(cast.profiles) == PEERS
    assert cast.plan.mean_pressure == 0.0
    assert any("沒有分母" in n for n in cast.notices)


def test_the_plan_notices_are_carried_through():
    """`plan_skills` 的 notice 不能在這一層被吃掉。

    分母不足 8、有要求不在池子裡,那些是讀三個數字的前提。
    """
    cast = cast_of(adjacent=adjacent_reply(), profile=profile_reply)
    assert any("分母用 7 不是 8" in n for n in cast.notices)


# ---------------------------------------------------------------------------
# 快取
# ---------------------------------------------------------------------------


def adjacent_calls(seen: list) -> int:
    return sum(1 for system, _user in seen if ADJACENT_MARK in system)


def test_the_same_jd_only_asks_the_model_once():
    """同一個職缺跑兩次,挑技能那次呼叫只該發生一次。

    實跑量到單次呼叫中位 3.91s,而競爭者那段是兩次相依呼叫,所以群面開場
    從 3.9s 變成 7.8s。挑技能的輸入只有職缺、溫度 0,同一個職缺答案本來就一樣,
    所以那一次是可以省的——省掉之後只剩寫人物,跟開場、搶話並行。
    """
    cache = AdjacentCache()
    seen: list = []
    for _ in range(2):
        cast_of(
            adjacent=adjacent_reply(), profile=profile_reply, seen=seen, cache=cache
        )
    assert adjacent_calls(seen) == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_the_same_jd_still_gets_fresh_people():
    """快取只省挑技能那一次,寫人物每次都要重問。

    快到連人都一樣的話,同一個職缺重跑會看到同一組競爭者,
    而「同一個職缺每次跑該產出不一樣的競爭者」是這個模組的設計要求。
    """
    cache = AdjacentCache()
    seen: list = []
    for _ in range(3):
        cast_of(
            adjacent=adjacent_reply(), profile=profile_reply, seen=seen, cache=cache
        )
    assert sum(1 for system, _u in seen if PROFILE_MARK in system) == 3


def test_a_different_jd_is_not_a_hit():
    cache = AdjacentCache()
    seen: list = []
    cast_of(adjacent=adjacent_reply(), profile=profile_reply, seen=seen, cache=cache)
    cast_of(
        required=jd_of("ivw-008"),
        adjacent=adjacent_reply(),
        profile=profile_reply,
        seen=seen,
        cache=cache,
    )
    assert adjacent_calls(seen) == 2


def test_a_degraded_pick_does_not_stick():
    """降級結果不進快取。

    存起來的話一次失敗會變成那個職缺的永久答案,而那比當次失敗嚴重得多——
    它會在之後每一場群面重現,而且看起來不像故障。
    """
    cache = AdjacentCache()
    dead = cast_of(profile=profile_reply, cache=cache)
    assert dead.adjacent.source == FALLBACK
    assert len(cache) == 0

    ok = cast_of(adjacent=adjacent_reply(), profile=profile_reply, cache=cache)
    assert ok.adjacent.source == LLM
    assert len(cache) == 1


def test_a_cached_pick_too_small_for_the_need_is_a_miss():
    """存的那筆不夠新設定用的時候要重問。

    同質度 0.5 只要 6 個分母外技能,兩個旋鈕歸零要 18 個。
    不對 need 的話後者會拿到一筆不夠的 deck,然後被補亂挑的——
    而那看起來像模型挑得不好,不像快取的問題。
    """
    cache = AdjacentCache()
    small = adjacent_reply(FRONTEND[:7])
    seen: list = []
    a = cast_of(adjacent=small, profile=profile_reply, seen=seen, cache=cache)
    assert a.adjacent.source == LLM and a.adjacent.kept == 7

    b = cast_of(
        adjacent=small,
        profile=profile_reply,
        seen=seen,
        cache=cache,
        homogeneity=0.0,
        pressure=0.0,
    )
    assert adjacent_calls(seen) == 2, "need 變成 18,7 個不夠,要重問"
    assert b.adjacent.source == TOPPED_UP


def test_without_a_cache_nothing_is_shared():
    """預設不快取。測試與工具共用一份的話,同一個職缺餵不同假 LLM
    會拿到第一次的結果,而那種錯誤表現成「測試互相影響」。"""
    seen: list = []
    for _ in range(2):
        cast_of(adjacent=adjacent_reply(), profile=profile_reply, seen=seen)
    assert adjacent_calls(seen) == 2


def test_the_cache_is_bounded():
    """無上限的 dict 在長命的行程裡只會長大。"""
    cache = AdjacentCache(max_entries=2)
    for case in ("ivw-001", "ivw-004", "ivw-008"):
        cast_of(
            required=jd_of(case),
            adjacent=adjacent_reply(),
            profile=profile_reply,
            cache=cache,
        )
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# 校系
# ---------------------------------------------------------------------------


def test_a_school_outside_the_list_is_refused():
    out = profiles_of(
        lambda u: profile_reply(u, **{SPECS[0].id: {"school": "某某大學"}})
    )
    assert "school" in out[0].repaired
    assert out[0].background == SPECS[0].blurb
    assert out[0].school == ""


def test_a_fabricated_school_in_the_background_is_refused():
    """這是實跑抓到的那個 bug:「國防醫學院會計系畢業」。

    十個職缺的三十份背景裡「國防醫學院」出現 6 次,全部落在 peer_assertive,
    而且掛著不存在的科系。國防醫學院是軍醫學校,沒有資訊工程也沒有會計系。

    只驗 `school` 欄位擋不住——那一句是寫在背景敘述裡的。
    """
    out = profiles_of(
        lambda u: profile_reply(
            u,
            **{
                SPECS[0].id: {
                    "school": "政治大學",
                    "background": "國防醫學院會計系畢業,曾在事務所實習。",
                }
            },
        )
    )
    assert "background_school" in out[0].repaired
    assert "國防醫學院" not in out[0].background


def test_an_abbreviation_instead_of_the_listed_name_is_refused():
    """「政大資工」配 school=政治大學 不算過。

    prompt 規則 3 明說要用清單裡的寫法。放寬成「簡寫也算」的話,
    這個檢查就要去維護一張簡寫對照表,而那張表是下一個會跟清單分岔的東西。
    """
    out = profiles_of(
        lambda u: profile_reply(
            u,
            **{
                SPECS[0].id: {
                    "school": "政治大學",
                    "background": "政大資工畢業,在電商做過半年。",
                }
            },
        )
    )
    assert "background_school" in out[0].repaired


def test_an_internal_college_is_not_a_fake_school():
    """「臺灣大學商學院畢業」要過。

    第一版用正則掃帶「學院」後綴的校名,結果實跑輸出裡的
    「台大商學院畢業」「台大社會科學院畢業」被判成假校名——
    商學院是學校內部的學院,完全合法。這條把那個誤殺釘住。
    """
    out = profiles_of(
        lambda u: profile_reply(
            u,
            **{
                SPECS[0].id: {
                    "school": "臺灣大學",
                    "background": "臺灣大學商學院畢業,曾在非營利組織做行政。",
                }
            },
        )
    )
    assert out[0].repaired == ()
    assert out[0].school == "臺灣大學"


def test_the_school_list_holds_only_broad_universities():
    """清單裡每一所都以「大學」結尾。

    程式驗得了校名、驗不了「這個學校有沒有這個系」。專門學院
    (醫學院、藝術學院)掛錯科系的機率最高,所以靠清單把那個風險擋在外面。
    加一所「○○醫學院」進來就等於把驗不到的風險放回去,這條會變紅。
    """
    assert all(name.endswith("大學") for name in SCHOOLS), [
        n for n in SCHOOLS if not n.endswith("大學")
    ]
    assert len(set(SCHOOLS)) == len(SCHOOLS)


def test_the_prompt_carries_the_school_list():
    """清單要真的印在 prompt 裡,不然程式驗的是模型沒看過的規則。"""
    text = compose_profile_prompt()
    for name in SCHOOLS:
        assert name in text


# ---------------------------------------------------------------------------
# 人稱
# ---------------------------------------------------------------------------


def test_a_first_person_background_is_refused():
    """背景敘述出現「我」就退回預設。

    實跑三十份裡有三份混了第一人稱(「習慣直接提出我的見解」),其餘是第三人稱。
    這段顯示在小組成員名單上,是別人在介紹他,混用讀起來像兩個人寫的。
    """
    out = profiles_of(
        lambda u: profile_reply(
            u, **{SPECS[0].id: {"background": "政治大學畢業,習慣直接提出我的見解。"}}
        )
    )
    assert "background_voice" in out[0].repaired
    assert out[0].background == SPECS[0].blurb


def test_a_third_person_background_passes():
    out = profiles_of(profile_reply)
    assert all(p.repaired == () for p in out)
    assert all(FIRST_PERSON not in p.background for p in out)


# ---------------------------------------------------------------------------
# 名字
# ---------------------------------------------------------------------------


def test_two_peers_cannot_share_a_name():
    """同一批不能有兩個同名的。

    小組名單上出現兩個「林一哲」,使用者會以為畫面壞了。
    跨職缺重複(實跑裡李明軒出現四次)擋不掉,那只能靠 prompt。
    """
    out = profiles_of(
        lambda u: profile_reply(
            u, **{spec.id: {"name": "林一哲"} for spec in SPECS[:PEERS]}
        )
    )
    names = [p.name for p in out]
    assert len(set(names)) == len(names)
    assert any("name" in p.repaired for p in out[1:])
