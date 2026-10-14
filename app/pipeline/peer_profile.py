"""模型那一側:挑相鄰技能、寫人物,兩次呼叫都由程式驗證。

【分工的邊界在哪】
`peer_generate` 挑技能 id,因為同質度與壓迫度可算。這一支做的是
那兩件算不出正確答案的事,外加一件程式做不到的事:

    挑相鄰技能   程式做不到——池子裡沒有領域訊號(ucan_code 與 onet_code 實測 0/152)
    寫人物       沒有可算的正確答案

兩者都是模型提案、程式驗證。驗的方式不同:

    相鄰技能   提案是編號,程式對照清單換回 id,換不到的丟掉。完全可驗。
    人物       提案是散文,只能驗**可以falsify的部分**——
               長度、id 對不對、有沒有宣稱它沒有的職缺技能。

第三項是這一支唯一有意思的驗證。背景敘述寫得好不好程式不知道,
但「它說自己做過 React 而技能清單裡沒有 React」是查得出來的,
而那種誇大剛好會讓使用者感受到的壓迫度高過三個指標說的數字。

【為什麼不驗背景敘述寫得好不好】
那要人看。這一支把可驗的驗完、把不可驗的原樣交出去,
而不是寫一條「背景敘述要有具體名目」的規則然後假裝驗過了——
1d 那輪的教訓是 prompt 寫兩次沒生效就不要寫第三次,
這裡的對應是:驗不到的東西不要假裝有驗。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.pipeline.interview_live import LLMCall, parse_list
from app.pipeline.peer_generate import (
    JDCore,
    PeerPlan,
    SKILLS_PER_PEER,
    load_index,
    names_for,
    off_jd_needed,
    peer_count,
    plan_skills,
    resolve_jd,
)
from app.pipeline.skill_pool import VOCAB_PATH, Skill, default_pool
from app.prompts.interview_personas import personas_for
from app.prompts.peer_profile import (
    BACKGROUND_MAX_CHARS,
    NAME_MAX_CHARS,
    STYLE_MAX_CHARS,
    adjacent_user_message,
    compose_adjacent_prompt,
    compose_profile_prompt,
    profile_user_message,
)

ADJACENT_SOURCES: tuple[str, ...] = ("llm", "topped_up", "fallback")
LLM, TOPPED_UP, FALLBACK = ADJACENT_SOURCES
"""`AdjacentPick.source` 的值域。

三個值而不是兩個,因為「模型挑的不夠多所以補了亂挑的」跟
「模型整個失敗所以全部亂挑」是不同的事實,而它們的輸出品質差很多。
合成一個值的話,看到 fallback 會以為呼叫壞了,而實際上可能只是少挑了五個。
"""

PROFILE_SOURCES: tuple[str, ...] = ("llm", "fallback")
PROFILE_LLM, PROFILE_FALLBACK = PROFILE_SOURCES
"""`PeerProfile.source` 的值域。"""


@dataclass(frozen=True)
class AdjacentPick:
    """模型挑的相鄰技能,以及挑的過程掉了什麼。

    四個計數都要留。它們之間掉東西的理由不一樣,而混在一起看不出是哪一種:

        returned      模型回了幾個編號
        out_of_range  編號不在 1..len(pool) 裡——模型自己編號碼
        on_jd         挑到職缺要求上——規則 4 沒生效
        kept          最後留下幾個

    `on_jd` 特別值得盯:它是 prompt 規則失效的直接指標。如果它長期偏高,
    那條規則就該從 prompt 搬到別的地方,而不是把規則寫得更長。
    """

    ids: tuple[str, ...]
    source: str
    returned: int = 0
    out_of_range: int = 0
    on_jd: int = 0

    @property
    def kept(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class PeerProfile:
    """一位競爭者的人物設定。技能是程式給的,其餘是模型寫的。

    `source` 是「這一位有沒有拿到可用的模型回覆」。
    `repaired` 是「有拿到,但某幾個欄位不合格所以退回預設」——
    兩者分開,因為「呼叫失敗」跟「名字寫了十五個字」要分得出來。
    """

    persona_id: str
    name: str
    background: str
    style: str
    skills: tuple[str, ...]
    source: str
    repaired: tuple[str, ...] = ()
    overclaims: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeerCast:
    """一場群面的競爭者:技能集合、三個指標、人物設定。

    `plan` 裡的三個數字是實際值,`notices` 是整條流程掉過的東西。
    呼叫端把 `profiles` 的 `background` 寫進 `Persona.blurb` 就接完了。
    """

    plan: PeerPlan
    profiles: tuple[PeerProfile, ...]
    adjacent: AdjacentPick
    notices: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 第一次呼叫
# ---------------------------------------------------------------------------


async def pick_adjacent(
    *,
    title: str,
    core: JDCore,
    pool: Sequence[Skill],
    llm: LLMCall,
    need: int,
    rng_seed: int = 0,
) -> AdjacentPick:
    """問模型哪些技能跟這個職缺相鄰。回傳的 id 保證在池子裡、不在職缺要求裡。

    `need` 是構造最少要幾個(`peers × k` 的最壞情況)。模型挑不到那麼多就補,
    補的是亂挑的,而 `source` 會記成 `topped_up`。

    呼叫失敗回 `fallback` 與空清單,呼叫端據此把 `off_jd_deck` 留空,
    讓 `plan_skills` 走它自己的亂挑路徑並發它自己的 notice。
    不在這裡偽造一份 deck:那會讓「模型沒答話」長得像「模型挑的」。
    """
    excluded = set(core.coverable) | set(core.out_of_pool)
    candidates = [s for s in pool if s.skill_id not in excluded]
    if not candidates:
        return AdjacentPick((), FALLBACK)

    try:
        raw = await asyncio.to_thread(
            llm,
            compose_adjacent_prompt(),
            adjacent_user_message(
                title, core_names(core, pool), [s.display for s in pool]
            ),
        )
    except Exception:  # noqa: BLE001
        return AdjacentPick((), FALLBACK)

    picked: list[str] = []
    out_of_range = on_jd = returned = 0
    seen: set[str] = set()
    for item in parse_list(raw):
        try:
            n = int(item)
        except (TypeError, ValueError):
            out_of_range += 1
            continue
        returned += 1
        if not 1 <= n <= len(pool):
            out_of_range += 1
            continue
        sid = pool[n - 1].skill_id
        if sid in excluded:
            on_jd += 1
            continue
        if sid in seen:
            continue
        seen.add(sid)
        picked.append(sid)

    if not picked:
        return AdjacentPick((), FALLBACK, returned, out_of_range, on_jd)

    source = LLM
    if len(picked) < need:
        source = TOPPED_UP
        rest = [s.skill_id for s in candidates if s.skill_id not in seen]
        random.Random(rng_seed).shuffle(rest)
        picked += rest[: need - len(picked)]

    return AdjacentPick(tuple(picked), source, returned, out_of_range, on_jd)


def core_names(core: JDCore, pool: Sequence[Skill]) -> tuple[str, ...]:
    """職缺要求的顯示名稱,用來印在 user 訊息裡。

    用解析後的名稱而不是職缺原文:原文可能是「C#程式開發」而池子裡是「C#」,
    印原文會讓模型以為那是清單外的東西而去挑「C#」。
    """
    return names_for(core.coverable, pool)


# ---------------------------------------------------------------------------
# 第二次呼叫
# ---------------------------------------------------------------------------


def _clean(value: object, limit: int) -> str:
    """取一個欄位,過長或空的回空字串。

    **不截斷。** 截一半的句子是壞資料而不是短資料,而它會直接出現在
    使用者看到的 roster 上。退回預設至少是通順的。
    """
    text = str(value or "").strip().replace("\n", " ")
    return text if 0 < len(text) <= limit else ""


def find_overclaims(
    background: str,
    own: Sequence[str],
    denominator: Sequence[str],
    *,
    vocab_path: Path | str = VOCAB_PATH,
) -> tuple[str, ...]:
    """背景敘述裡宣稱了哪些它沒有的職缺技能。

    【為什麼只掃分母內的】
    背景提到清單外的技能大多是無害的顏色(「在社團辦過活動」)。
    但宣稱一個**職缺要求的**技能而技能清單裡沒有,會讓使用者感受到的
    壓迫度高過三個指標說的數字——那是指標與體感脫鉤,不是文案問題。

    掃不到就是乾淨。掃得到也不改寫:記下來給呼叫端決定,
    因為「把那一句刪掉」需要重新生成,而重新生成的成本不該由這一層決定。
    """
    if not background:
        return ()
    index = load_index(vocab_path)
    hits = {sid for _s, _e, _surface, sid in index.scan(background)}
    return tuple(sid for sid in denominator if sid in hits and sid not in set(own))


async def read_profiles(
    *,
    title: str,
    plan: PeerPlan,
    pool: Sequence[Skill],
    llm: LLMCall,
    group_size: int = 4,
    group_interviewers: int = 1,
    vocab_path: Path | str = VOCAB_PATH,
) -> tuple[PeerProfile, ...]:
    """問模型寫人物。每一位都回得出東西,失敗的那幾位退回固定人設。

    人格指示直接用 `PersonaSpec.stance`,不另寫一份——人格的所有者是
    `interview_personas`,抄一份過來的話調 persona 會調不到這裡。

    回傳順序照 `personas_for` 的 peer 順序,也就是前端 roster 的順序。
    """
    specs = [
        p
        for p in personas_for("group", group_interviewers, group_size)
        if p.role == "peer"
    ]
    specs = specs[: len(plan.peers)]
    rows = [
        (spec.id, spec.stance, names_for(skills, pool))
        for spec, skills in zip(specs, plan.peers)
    ]

    data: list = []
    try:
        raw = await asyncio.to_thread(
            llm, compose_profile_prompt(), profile_user_message(title, rows)
        )
        data = parse_list(raw)
    except Exception:  # noqa: BLE001
        data = []

    by_id: dict[str, dict] = {}
    for item in data:
        if isinstance(item, dict):
            pid = str(item.get("id") or "").strip()
            if pid and pid not in by_id:
                by_id[pid] = item

    taken = {spec.display_name for spec in specs}
    out: list[PeerProfile] = []
    for spec, skills in zip(specs, plan.peers):
        entry = by_id.get(spec.id)
        if entry is None:
            out.append(
                PeerProfile(
                    persona_id=spec.id,
                    name=spec.display_name,
                    background=spec.blurb,
                    style="",
                    skills=tuple(skills),
                    source=PROFILE_FALLBACK,
                )
            )
            continue

        repaired: list[str] = []
        name = _clean(entry.get("name"), NAME_MAX_CHARS)
        if not name or name in taken:
            name = spec.display_name
            repaired.append("name")
        background = _clean(entry.get("background"), BACKGROUND_MAX_CHARS)
        if not background:
            background = spec.blurb
            repaired.append("background")
        style = _clean(entry.get("style"), STYLE_MAX_CHARS)
        if not style:
            repaired.append("style")

        out.append(
            PeerProfile(
                persona_id=spec.id,
                name=name,
                background=background,
                style=style,
                skills=tuple(skills),
                source=PROFILE_LLM,
                repaired=tuple(repaired),
                overclaims=find_overclaims(
                    background, skills, plan.denominator, vocab_path=vocab_path
                ),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# 串起來
# ---------------------------------------------------------------------------


async def build_peers(
    *,
    title: str,
    required: Sequence[str],
    llm: LLMCall,
    write_llm: LLMCall | None = None,
    group_size: int = 4,
    group_interviewers: int = 1,
    homogeneity: float = 0.5,
    pressure: float = 0.5,
    seed: int = 0,
    k: int = SKILLS_PER_PEER,
    pool: Sequence[Skill] | None = None,
    vocab_path: Path | str = VOCAB_PATH,
) -> PeerCast:
    """職缺 → 一組完整的競爭者。兩次呼叫,順序是硬的。

    第一次挑相鄰技能、程式排技能集合、第二次寫人物。不能併成一次:
    技能集合要先定下來,模型才拿得到「已經定好的技能」而不是「請你發揮」。
    也不能並行:第二次呼叫的輸入是第一次的結果經過程式處理之後的東西。

    `llm` 跑挑技能那一次,`write_llm` 跑寫人物那一次。分開的理由是溫度:
    挑技能是分類,要可重現(temp 0),不然同一個職缺每次跑的 deck 都不一樣,
    那會疊在 seed 的變化上面,量測就分不出哪個變化來自哪裡。
    寫人物要有變化(temp 0.4)。不給 `write_llm` 就兩邊用同一支。

    兩次呼叫都失敗也回得出完整的一組:亂挑的技能 + 固定人設。
    那一組不好看,但它不會讓群面開不起來,而且 `notices` 會說清楚發生了什麼。
    """
    pool = list(default_pool()) if pool is None else pool
    peers = peer_count(group_size, group_interviewers)
    core = resolve_jd(required, pool, vocab_path=vocab_path)

    need = off_jd_needed(
        core, peers=peers, homogeneity=homogeneity, pressure=pressure, k=k
    )
    pick = await pick_adjacent(
        title=title, core=core, pool=pool, llm=llm, need=need, rng_seed=seed
    )
    notices: list[str] = []
    if pick.source == FALLBACK:
        notices.append(
            f"相鄰技能那一次呼叫沒拿到可用結果(回了 {pick.returned} 個編號),"
            "分母外的技能退回亂挑"
        )
    elif pick.source == TOPPED_UP:
        notices.append(
            f"模型只挑到 {pick.returned - pick.out_of_range - pick.on_jd} 個相鄰技能,"
            f"不足構造需要的 {need} 個,其餘補亂挑的"
        )
    if pick.on_jd:
        notices.append(f"模型挑了 {pick.on_jd} 個職缺已經要求的技能,已丟掉")
    if pick.out_of_range:
        notices.append(f"模型回了 {pick.out_of_range} 個清單外的編號,已丟掉")

    plan = plan_skills(
        core,
        pool,
        peers=peers,
        homogeneity=homogeneity,
        pressure=pressure,
        k=k,
        seed=seed,
        off_jd_deck=pick.ids or None,
    )

    profiles = await read_profiles(
        title=title,
        plan=plan,
        pool=pool,
        llm=write_llm or llm,
        group_size=group_size,
        group_interviewers=group_interviewers,
        vocab_path=vocab_path,
    )

    missing = [p.persona_id for p in profiles if p.source == PROFILE_FALLBACK]
    if missing:
        notices.append(f"{len(missing)} 位競爭者退回固定人設:{'、'.join(missing)}")
    for p in profiles:
        if p.repaired:
            notices.append(
                f"{p.persona_id} 的 {'、'.join(p.repaired)} 不合格,已退回預設"
            )
        if p.overclaims:
            names = "、".join(names_for(p.overclaims, pool))
            notices.append(f"{p.persona_id} 的背景宣稱了它沒有的職缺技能:{names}")

    return PeerCast(
        plan=plan,
        profiles=profiles,
        adjacent=pick,
        notices=tuple(list(plan.notices) + notices),
    )
