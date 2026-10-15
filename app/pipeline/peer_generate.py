"""群面競爭者的技能集合由程式挑,不由模型挑。

【為什麼這一段不給模型做】
「同質度」與「壓迫度」是兩個可算的量(見 `peer_overlap`)。叫模型「生成四個
條件相近的競爭者」,出來的東西沒有辦法驗——你只能看著四份履歷說「嗯好像有點像」。
而這兩個數字一算就知道,所以這裡沿用專案的老規則:能用程式算的不要問 LLM。

分工:

    程式   從 152 筆技能池裡挑出每人一組 skill_id,並回報挑出來的實際指標
    模型   名字、學校、背景敘述、說話風格 —— 沒有可算的正確答案的那部分

模型拿到的是**已經定好的技能集合**,不是「請你發揮」。

【構造法:同質度有封閉解】
讓每個人共用 c 個技能、各自再拿 k-c 個互不重複的技能。任兩人的交集恰好是 c、
聯集恰好是 2k-c,所以

    J = c / (2k - c)        反過來   c = 2kJ / (1+J)

每一對都一樣,所以 mean pairwise Jaccard 精確等於 J,不是近似。
c 是整數,所以同質度只能落在一個階梯上。k=6 的階梯:

    c      0      1      2      3      4      5      6
    J  0.000  0.091  0.200  0.333  0.500  0.714  1.000

k 越大階梯越密(k=4 最大跳 0.400、k=5 是 0.333、k=6 是 0.286),
但 k 也是單人壓迫度的上限(一個人只有 k 個技能,踩不滿 k 個以上的職缺要求)。
k=6 是這兩件事的折衷,量測記錄在 `SKILLS_PER_PEER`。

【壓迫度沒有封閉解,而且跟同質度搶同一批技能】
單人壓迫度 = 這個人踩到職缺要求的幾成,分母是職缺要求(的前 N 個)。
要讓每個人踩到 j 個職缺要求,而且各自那塊不能重複,那麼:

    用掉的職缺技能數 = min(c,j) + peers × (j - min(c,j))   ≤  d

j 超過 c 的部分每個人都要一個不同的職缺技能,所以乘人數。
於是(d=8, k=6,預設的三位 AI 競爭者):

    同質度 0.000   單人壓迫度最高 0.250
    同質度 0.200   單人壓迫度最高 0.500
    同質度 0.500   單人壓迫度最高 0.625
    同質度 1.000   單人壓迫度最高 0.750

**低同質度與高單人壓迫度是互斥的**,而這是算術,不是調參調得不夠好。
只有 8 個職缺要求可分,三個人要各自踩很多個又不能重疊,分不出來。

重點是:這時候**聯集壓迫度反而最高**。c=0、j=2 的那組,單人只有 0.250,
聯集是 6/8 = 0.750 —— 三個人加起來蓋掉的遠多於任何一個人,
而且 c=0、k=6 的時候聯集吃滿 min(peers×k, d)/d。
使用者在群面裡感受到的「被包圍」就是這個,不是單人數字。

所以 `plan_skills` 回報三個數字,而且沒有一個可以單獨讀。

【分母外的那幾個技能,程式挑不好,而這是資料的問題】
每個人 k=6 個技能裡只有 j 個來自職缺要求,剩下的是「他還會別的」。
第一版從池子裡均勻亂挑,拿 ivw-004(前端網頁工程師)實跑:

    競爭者1  Vmware、Canva、CSS、jQuery、HTML、JavaScript
    競爭者2  社群媒體經營管理、市場調查資料分析與報告撰寫、CSS、jQuery、HTML、JavaScript
    競爭者3  軟體程式設計、ERP 系統、CSS、jQuery、HTML、JavaScript

三個數字全部精確達標,而三份履歷都不可信 —— 應徵前端的人不會把
「媒體關係經營」「ERP 系統」寫在六個技能裡。使用者會先懷疑這整個功能。

想過用領域相近度挑,但**池子裡沒有領域訊號**:152 筆的 `ucan_code` 與
`onet_code` 全部是空字串(實測 0/152),也沒有分類欄位。L3_market 是爬蟲
頻次來的平鋪名單,所以程式手上只有名字,挑不出「跟前端相近」的技能。

所以分母外那幾格交給模型,而程式驗證:它們必須彼此不重複、且不在分母裡。
這兩條一驗,三個指標就仍然精確 —— `mean_pressure` 與 `union_pressure`
只跟分母內的技能有關,`homogeneity` 只需要分母外那幾格互不重複。
`off_jd_deck` 就是這條路:呼叫端把模型提的(已解析成 id 的)技能餵進來,
不傳就退回均勻亂挑,而那條路只適合測試與量測,不適合給使用者看。

【人數不是四】
`group_size` 是**含使用者**的小組人數(預設 4),所以 AI 競爭者預設是三位
(`personas_for` 裡 `n_peers = group_size - 1`)。這一版第一次寫成四位,
而四位會讓可行域整個往下移(c=1 的單人上限從 0.375 掉到 0.250),
也就是指標會系統性偏低而程式不報錯。

所以這個模組**不宣告人數常數**,人數一律從 `personas_for` 數出來
(`peer_count`),`plan_skills` 的 `peers` 是必填。宣告第二個人數來源
就是埋下一次這種偏差。

【為什麼回報的是實際值不是目標值】
目標值是呼叫端給的願望,實際值是挑出來的東西。兩者在整數階梯上必然有差,
在可行域邊界上差更多。回報目標值等於把願望寫成事實 —— 專案裡
「過度宣稱比不宣稱糟」那條規則講的就是這個。所以 `PeerPlan` 上的
三個指標一律是**拿挑好的集合重算**出來的,不是從目標抄的。

【隨機性的方向跟面試官相反】
`interviewer_spec` 記了一條:面試官要穩定,同一份履歷每次跑該問一樣的問題;
候選人要多樣,同一個職缺每次跑該產出不一樣的競爭者。所以這裡吃 `seed`:
給同一個 seed 完全可重現(測試與重跑量測需要),換 seed 就換一組人。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from app.pipeline.peer_overlap import (
    mean_coverage,
    mean_pairwise_jaccard,
    union_coverage,
)
from app.pipeline.skill_pool import VOCAB_PATH, Skill, load_pool
from app.pipeline.transcript import SurfaceIndex, normalize_for_scan
from app.prompts.interview_personas import personas_for

SKILLS_PER_PEER = 6
"""每個競爭者幾個技能(k)。

k 同時決定兩件事,而它們要的方向相反:

    同質度階梯的密度   k 越大越密   k=4 最大跳 0.400、k=5 0.333、k=6 0.286
    單人壓迫度的上限   k 越大越高   上限 = min(k, d) / d

黃金集的 d(可覆蓋的職缺要求數)中位數是 10、範圍 5–23,所以 k=6 時
單人壓迫度上限大約 0.60,夠表達「他很強」而不必假裝一個新人踩滿職缺。
再往上加 k 的邊際收益(階梯 0.286 → 0.250)不值得讓每個競爭者
背七八個技能 —— 那會開始像履歷造假而不像競爭者。
"""

JD_PREFIX = 8
"""壓迫度的分母取職缺要求的前幾個(N)。

合約明訂 `required_skills` 「順序即優先序」(`gap.JD_POSITION_DECAY` 依賴這條),
所以前 N 個就是最關鍵的 N 個。取前 N 而不是全取的理由是分母穩定:
黃金集裡 d 從 5 到 23,全取的話同一組競爭者在 d=23 的案子上
壓迫度自動掉到 d=10 案子的四成多,而競爭者並沒有變弱。

N=8 落在 d 的中位數 10 以下,所以多數案子的分母真的是 8。
d 不足 8 的案子(黃金集有兩個是 5)分母就是 d 本身,見 `JDCore.denominator`。
"""


def peer_count(group_size: int = 4, group_interviewers: int = 1) -> int:
    """這一場有幾位 AI 競爭者。數 `personas_for` 真的給出幾個 peer。

    不自己算 `group_size - 1`:那條規則的所有者是 `personas_for`
    (它還夾在 1–4 之間、還要扣掉主持人),抄一份過來就是同一條規則寫兩次,
    而兩份分岔的時候是指標偏掉而不是報錯。
    """
    return sum(
        1
        for p in personas_for("group", group_interviewers, group_size)
        if p.role == "peer"
    )


# ---------------------------------------------------------------------------
# 職缺要求 → skill_id
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JDCore:
    """一份職缺的要求解析成技能池裡的 id,順序即優先序。

    【為什麼四個欄位都要留】
    `requested` 到 `coverable` 之間會掉東西,而掉的理由不只一種:

        unresolved   對不到詞彙表                 黃金集 118 條裡 0 條
        out_of_pool  對到詞彙表但不在技能池裡       118 條裡 23 條
        partial      只命中條目的一部分(可能誤解析) 118 條裡 0 條

    `out_of_pool` 那 23 條幾乎都是純 L1_onet 的抽象職能(主動聆聽、判斷與決策)。
    它們在技能池裡不存在,所以**沒有任何競爭者可能踩到**。把它們算進壓迫度的
    分母,壓迫度就有一個跟 k 無關的天花板,而讀數字的人會以為是生成器不夠強。

    `partial` 目前是 0,但它不是死欄位:表面形掃描是子字串掃描,
    「設計導入製造執行系統（MES）」整條命中只是因為詞彙表剛好有整條。
    哪天職缺寫「設計導入 MES 系統並導入 SPC」,掃描會命中其中一段而回報成功。
    那是誤解析不是命中,所以要分開數。
    """

    coverable: tuple[str, ...]
    requested: int
    unresolved: tuple[str, ...] = ()
    out_of_pool: tuple[str, ...] = ()
    partial: tuple[str, ...] = ()

    def denominator(self, n: int = JD_PREFIX) -> tuple[str, ...]:
        """壓迫度的分母:可覆蓋的前 n 個。不足 n 就是全部。"""
        return self.coverable[:n]

    def pressure_ceiling(self, k: int = SKILLS_PER_PEER, n: int = JD_PREFIX) -> float:
        """單人壓迫度的上限。一個人只有 k 個技能,踩不到比 k 更多的要求。

        職缺要求全部解析失敗時回 0.0 —— 跟 `coverage` 同一個方向,
        「沒有分母」不該長得像「完全命中」。
        """
        d = len(self.denominator(n))
        return min(k, d) / d if d else 0.0

    @property
    def lossless(self) -> bool:
        """職缺要求有沒有整份進得了技能池。

        不是給流程判斷用的,是給量測工具分層用的:掉了東西的案子,
        壓迫度要分開看,不能跟沒掉的混在一起算平均。
        """
        return len(self.coverable) == self.requested


@lru_cache(maxsize=4)
def load_index(vocab_path: Path | str = VOCAB_PATH) -> SurfaceIndex:
    """詞彙表的表面形索引,按路徑快取。

    快取的理由是呼叫路徑:`create_interview` 每一場都要解析職缺,而建索引
    要讀 234 筆 JSON 再分桶排序。那是每次請求都重做一次同樣的工作。

    `with_heads=False`:詞頭索引是為了掃逐字稿補洞的,職缺要求與背景敘述
    都是短句,補洞只會製造誤命中。
    """
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return SurfaceIndex.from_vocab(vocab, with_heads=False)


def resolve_jd(
    required: Sequence[str],
    pool: Sequence[Skill],
    *,
    vocab_path: Path | str = VOCAB_PATH,
) -> JDCore:
    """職缺要求字串 → `JDCore`。用既有的 `SurfaceIndex`,不另寫解析器。

    【為什麼不自己寫一個精確字典】
    第一版想過:拿 name_zh / name_en / aliases 建一張精確表。在黃金集上
    那張表命中 117/118,而 `SurfaceIndex` 是 118/118。差的那一條是
    「設計導入製造執行系統（MES）」—— 全形括號。`SurfaceIndex` 的掃描前會
    做全形轉半形、簡繁折疊、英文轉小寫,精確表沒有。

    再寫一份就是同一件事寫兩次,而兩份的正規化遲早會分岔:履歷側解析到一個
    skill_id、職缺側解析到另一個,然後壓迫度算出來不知道錯在哪。
    所以用同一支。`with_heads=False` 是因為詞頭索引是為了掃逐字稿補洞的,
    職缺要求是短條目,補洞只會製造誤命中。

    代價是掃描本質上是子字串掃描:一條職缺要求可能只有一部分命中。
    那種情況記進 `partial`,不當成命中也不當成漏掉 —— 它是第三種結果。
    """
    index = load_index(vocab_path)
    in_pool = {s.skill_id for s in pool}

    coverable: list[str] = []
    unresolved: list[str] = []
    out_of_pool: list[str] = []
    partial: list[str] = []

    for raw in required:
        text = str(raw or "").strip()
        if not text:
            continue
        hits = index.scan(text)
        if not hits:
            unresolved.append(text)
            continue
        start, end, _surface, sid = hits[0]
        scannable, _offsets = normalize_for_scan(text)
        if start != 0 or end < len(scannable):
            partial.append(text)
            continue
        if sid not in in_pool:
            out_of_pool.append(sid)
            continue
        if sid not in coverable:
            coverable.append(sid)

    return JDCore(
        coverable=tuple(coverable),
        requested=len([r for r in required if str(r or "").strip()]),
        unresolved=tuple(unresolved),
        out_of_pool=tuple(out_of_pool),
        partial=tuple(partial),
    )


# ---------------------------------------------------------------------------
# 構造
# ---------------------------------------------------------------------------


def homogeneity_ladder(k: int = SKILLS_PER_PEER) -> tuple[float, ...]:
    """k 個技能時同質度能落在哪些值上。c = 0..k 對應的 J。

    這個階梯是整數造成的,不是演算法不夠好。印出來是為了讓
    「目標 0.6 拿到 0.5」不被當成 bug —— 0.6 在 k=6 的階梯上不存在。
    """
    return tuple(c / (2 * k - c) for c in range(k + 1))


def _pick_shared_size(target: float, k: int) -> int:
    """挑一個 c 讓 c/(2k-c) 最接近 target。

    不用封閉解 `round(2kJ/(1+J))`:那個式子在整數邊界上的捨入是對 c 捨入,
    而我們要的是對 J 最近。k=6、target=0.6 的時候兩者剛好分歧
    (封閉解給 4.5,Python 的銀行家捨入給 4;而 J 距離上 c=4 的 0.500
    確實比 c=5 的 0.714 近)—— 直接搜就不用討論這件事。
    k 最多 6、7,搜七個值的成本是零。封閉解留在檔頭解釋階梯長這樣的原因。
    """
    ladder = homogeneity_ladder(k)
    return min(range(k + 1), key=lambda c: (abs(ladder[c] - target), c))


def _distinct_jd_needed(c: int, j: int, peers: int) -> int:
    """c 個共用、每人 j 個職缺技能時,總共要用掉幾個不同的職缺技能。

    j ≤ c 的時候共用那塊就夠了,一個都不用額外拿(所以永遠可行)。
    j > c 的時候超出的部分每人一個不同的,所以乘 peers。
    """
    shared_jd = min(c, j)
    return shared_jd + peers * (j - shared_jd)


def _off_jd_count(c: int, j: int, peers: int, k: int) -> int:
    """這個構造要幾個分母外的技能。

    共用那塊有 c 個,其中 min(c,j) 個來自分母,所以共用要 c-min(c,j) 個分母外的。
    每個人各自那塊有 k-c 個,其中 j-min(c,j) 個來自分母,其餘乘人數。

    宣告成一支函式是因為 `plan_skills` 與 `off_jd_needed` 都要這個數字,
    而它算錯的後果是兩種相反的錯:算多了會讓模型挑的 deck 被亂挑的稀釋,
    算少了會在建構到一半的時候拋錯。
    """
    shared_jd = min(c, j)
    return (c - shared_jd) + peers * (k - c - (j - shared_jd))


def _pick_jd_count(target: float, c: int, k: int, peers: int, d: int) -> int:
    """挑一個 j 讓 j/d 最接近 target,且構造得出來。

    回傳的 j 可能比不受限時的最佳解小 —— 可行域是硬的,見檔頭。
    呼叫端拿實際值與目標值比對就知道有沒有被夾,所以這裡不另外回旗標。
    """
    if d <= 0:
        return 0
    options = [j for j in range(min(k, d) + 1) if _distinct_jd_needed(c, j, peers) <= d]
    return min(options, key=lambda j: (abs(j / d - target), -j))


def off_jd_needed(
    core: JDCore,
    *,
    peers: int,
    homogeneity: float = 0.5,
    pressure: float = 0.5,
    k: int = SKILLS_PER_PEER,
    n: int = JD_PREFIX,
) -> int:
    """這一組設定要幾個分母外的技能。呼叫 `plan_skills` 之前就算得出來。

    存在的理由是模型那一側:挑相鄰技能的那次呼叫在排技能集合**之前**,
    所以它要先知道最少要幾個,不夠才補。

    不能拿最壞情況 `peers × k` 當這個數字。預設設定(同質度 0.5)只要 6 個,
    最壞情況是 18;用 18 的話模型挑了 10 個也會被判定不足,
    然後補 8 個亂挑的進去稀釋掉它挑的——實跑看過一次,
    前端職缺的競爭者因此拿到「規劃與管理防火牆」。
    """
    d = len(core.denominator(n))
    c = _pick_shared_size(homogeneity, k)
    j = _pick_jd_count(pressure, c, k, peers, d)
    return _off_jd_count(c, j, peers, k)


@dataclass(frozen=True)
class PeerPlan:
    """四個競爭者各自的技能 id,加上**實際**挑出來的三個指標。

    三個指標一起看才有意義:

        homogeneity     他們彼此多像
        mean_pressure   每個人單獨對這個職缺多強
        union_pressure  這一組加起來把職缺蓋掉多少

    低同質度會壓低 mean_pressure 而推高 union_pressure(檔頭有算術)。
    只看 mean_pressure 會以為生成器弱,只看 union_pressure 會以為每個人都很強。
    """

    peers: tuple[tuple[str, ...], ...]
    shared: tuple[str, ...]
    denominator: tuple[str, ...]
    homogeneity: float
    mean_pressure: float
    union_pressure: float
    seed: int
    notices: tuple[str, ...] = ()

    @property
    def off_jd(self) -> tuple[tuple[str, ...], ...]:
        """每個人身上不在分母裡的技能。

        這些是「他還會別的」。給模型寫背景敘述的時候需要,
        因為一份只寫職缺要求的履歷讀起來像是照職缺抄的。
        """
        d = set(self.denominator)
        return tuple(tuple(s for s in p if s not in d) for p in self.peers)


def plan_skills(
    core: JDCore,
    pool: Sequence[Skill],
    *,
    peers: int,
    homogeneity: float = 0.5,
    pressure: float = 0.5,
    k: int = SKILLS_PER_PEER,
    n: int = JD_PREFIX,
    seed: int = 0,
    off_jd_deck: Sequence[str] | None = None,
) -> PeerPlan:
    """挑出每位競爭者的技能 id,並回報挑出來的實際指標。

    `peers` 必填,見檔頭「人數不是四」:預設值會讓人數錯了也不報錯。

    `homogeneity` 與 `pressure` 是**目標**;回傳的三個數字是實際值。
    兩者會差,理由有三種,都寫進 `notices`:

        整數階梯   同質度只能落在 c/(2k-c) 上
        可行域     低同質度 × 高單人壓迫度 互斥(檔頭有算術)
        分母不足   職缺可覆蓋的要求少於 n,或一條都沒解析出來

    預設 0.5 / 0.5 在 k=6、d=8 時剛好 c=4、j=4,兩個目標都精確達成
    而且不碰可行域邊界(三位競爭者的上限是 0.625)—— 選這組當預設
    是因為它不需要解釋。

    `off_jd_deck` 是分母外那幾格的候選 id(模型提的,已解析)。不傳就從
    整個池子均勻亂挑,而那條路產出的競爭者不可信,見檔頭的實跑例子。
    傳進來的 id 一律要在池子裡、且不在職缺要求裡,不然拋 `ValueError` ——
    混用兩個池子會讓指標對不上,而那種錯不該默默吃掉。

    技能池不夠大的時候拋 `ValueError`。這裡不默默給短少的集合:
    一個只有四個技能的「六技能競爭者」會讓三個指標全部算錯,
    而錯的方向是壓迫度偏低 —— 看起來像競爭者不夠強,不像程式壞了。
    """
    rng = random.Random(seed)
    notices: list[str] = []

    denom = core.denominator(n)
    d = len(denom)
    if d == 0:
        notices.append(
            "職缺一條要求都沒解析進技能池,壓迫度沒有分母,兩個壓迫度數字一律是 0"
        )
    elif d < n:
        notices.append(f"可覆蓋的職缺要求只有 {d} 條,分母用 {d} 不是 {n}")
    if core.out_of_pool:
        notices.append(
            f"職缺有 {len(core.out_of_pool)} 條要求對到詞彙表但不在技能池裡,"
            "不算進分母(沒有競爭者可能踩到)"
        )
    if core.unresolved:
        notices.append(f"對不到詞彙表的職缺要求 {len(core.unresolved)} 條")
    if core.partial:
        notices.append(f"只命中一部分的職缺要求 {len(core.partial)} 條,當成沒解析到")

    c = _pick_shared_size(homogeneity, k)
    ladder = homogeneity_ladder(k)
    if abs(ladder[c] - homogeneity) > 1e-9:
        notices.append(
            f"同質度目標 {homogeneity:.3f} 不在 k={k} 的階梯上,取最近的 {ladder[c]:.3f}"
        )

    j = _pick_jd_count(pressure, c, k, peers, d)
    if d:
        unconstrained = min(range(min(k, d) + 1), key=lambda x: abs(x / d - pressure))
        if j < unconstrained:
            notices.append(
                f"單人壓迫度目標 {pressure:.3f} 在同質度 {ladder[c]:.3f} 下做不出來,"
                f"可行上限 {j / d:.3f}(分母 {d} 分不夠 {peers} 個人各拿 {unconstrained} 個)"
            )
        elif abs(j / d - pressure) > 1e-9:
            notices.append(
                f"單人壓迫度目標 {pressure:.3f} 取最近的可達值 {j / d:.3f}(分母 {d})"
            )

    shared_jd_count = min(c, j)
    private_jd_each = j - shared_jd_count

    # 分母以外的池子。排掉**整份**職缺要求而不只是分母,這樣競爭者身上
    # 任何一個職缺技能都是刻意放的,壓迫度就是職缺重疊的全部真相。
    # 代價是排名第 9 以後的職缺要求不會出現在競爭者身上(d 中位數 10,
    # 影響約兩筆),換來的是數字可以直接讀。
    excluded = set(core.coverable) | set(core.out_of_pool)
    if off_jd_deck is None:
        deck = [s.skill_id for s in pool if s.skill_id not in excluded]
        notices.append(
            "分母外的技能是從池子均勻亂挑的,領域不會貼合職缺,只適合測試與量測"
        )
    else:
        in_pool = {s.skill_id for s in pool}
        bad = [i for i in off_jd_deck if i not in in_pool]
        clash = [i for i in off_jd_deck if i in excluded]
        if bad:
            raise ValueError(f"off_jd_deck 有 {len(bad)} 個 id 不在技能池裡:{bad[:3]}")
        if clash:
            raise ValueError(
                f"off_jd_deck 有 {len(clash)} 個 id 是職缺要求,會讓壓迫度算錯:{clash[:3]}"
            )
        deck = list(dict.fromkeys(off_jd_deck))
    rng.shuffle(deck)

    need_off = _off_jd_count(c, j, peers, k)
    if need_off > len(deck):
        raise ValueError(
            f"分母外的技能不夠:要 {need_off} 個,可用的只有 {len(deck)} 個"
            f"(peers={peers} k={k} c={c} j={j})"
        )
    if private_jd_each and shared_jd_count + peers * private_jd_each > d:
        raise ValueError(
            f"構造不成立:分母只有 {d} 個,卻要 {shared_jd_count} 共用 + "
            f"{peers}×{private_jd_each} 各自"
        )

    shared = list(denom[:shared_jd_count])
    shared += [deck.pop() for _ in range(c - shared_jd_count)]

    jd_rest = list(denom[shared_jd_count:])
    built: list[tuple[str, ...]] = []
    for i in range(peers):
        own = list(shared)
        own += jd_rest[i * private_jd_each : (i + 1) * private_jd_each]
        own += [deck.pop() for _ in range(k - c - private_jd_each)]
        if len(set(own)) != k:
            raise ValueError(
                f"第 {i} 個競爭者拿到 {len(set(own))} 個不同技能,應該是 {k}"
            )
        built.append(tuple(sorted(own)))

    # 每一對的交集都該恰好是 c,不然封閉解不成立而三個指標會說謊。
    for a in range(peers):
        for b in range(a + 1, peers):
            got = len(set(built[a]) & set(built[b]))
            if got != c:
                raise ValueError(f"第 {a} 與第 {b} 個競爭者交集 {got},應該是 {c}")

    got_h, got_mean, got_union = verify(built, denom)
    return PeerPlan(
        peers=tuple(built),
        shared=tuple(sorted(shared)),
        denominator=denom,
        homogeneity=got_h,
        mean_pressure=got_mean,
        union_pressure=got_union,
        seed=seed,
        notices=tuple(notices),
    )


def verify(
    peers: Sequence[Iterable[str]], denominator: Iterable[str]
) -> tuple[float, float, float]:
    """任何一組競爭者技能 → (同質度, 單人壓迫度, 聯集壓迫度)。

    `plan_skills` 自己就是用這個算回報值的,所以量測工具拿真輸出重跑
    走的是同一條路。成員 A 要的「門檻拿真輸出重跑」就是餵這支。

    它也是模型填完分母外那幾格之後的檢查點:填完重算一次,
    對不上目標就是填充撞車了(兩個競爭者拿到同一個技能),不是模型寫壞文案。
    """
    denom = tuple(denominator)
    return (
        mean_pairwise_jaccard(list(peers)),
        mean_coverage(list(peers), denom),
        union_coverage(list(peers), denom),
    )


def generate(
    required: Sequence[str],
    *,
    group_size: int = 4,
    group_interviewers: int = 1,
    homogeneity: float = 0.5,
    pressure: float = 0.5,
    seed: int = 0,
    off_jd_deck: Sequence[str] | None = None,
    pool: Sequence[Skill] | None = None,
    vocab_path: Path | str = VOCAB_PATH,
) -> PeerPlan:
    """職缺要求字串 → 每位競爭者的技能 id。解析與挑選串起來的那一層。

    `group_size` 含使用者,跟 `InterviewContext.group_size` 同一個意思,
    人數由 `peer_count` 從 persona 層數出來。

    `pool` 留著是為了測試可以餵小池子,正式路徑不傳。
    """
    pool = load_pool() if pool is None else pool
    core = resolve_jd(required, pool, vocab_path=vocab_path)
    return plan_skills(
        core,
        pool,
        peers=peer_count(group_size, group_interviewers),
        homogeneity=homogeneity,
        pressure=pressure,
        seed=seed,
        off_jd_deck=off_jd_deck,
    )


def names_for(ids: Iterable[str], pool: Sequence[Skill]) -> tuple[str, ...]:
    """skill_id → 寫在履歷上的名字,照傳入的順序。

    對不到的 id 直接跳過而不是放一個佔位字串:技能集合是程式挑的、
    來源就是這個池子,所以對不到代表呼叫端混用了兩個池子。
    放佔位字串會讓那個錯誤變成競爭者履歷上的一行怪字。
    """
    by_id = {s.skill_id: s.display for s in pool}
    return tuple(by_id[i] for i in ids if i in by_id)
