"""競爭者可以用的技能池:從 234 筆詞彙表切出「像履歷」的那一批。

【為什麼不能用 skill_id 的前綴切】
`sk:` 與 `skm:` 看起來像「權威 vs 自鑄」,實際上它編碼的是
`provenance.zhang_matched`——名稱有沒有被 O*NET / UCAN 權威表背書。
那跟「像不像履歷」是正交的兩件事。

實際跑 234 版驗過(成員 A 的結論,數字我重算過一致):

    純 L1_onet 那 34 筆      全部是 sk: 開頭    最不像履歷,但前綴看起來最「可信」
    skm: 裡 zhang_matched    0 筆               前綴與背書完全一對一
    L0 ∪ L3 這 152 筆        只有 20 筆被背書    拿背書當主篩會砍掉 132 筆

所以用前綴篩等於用背書篩,而那會把最像履歷的工具名全濾掉、
把「主動聆聽」「判斷與決策」這類 O*NET 抽象職能全撈進來。方向剛好反。

【所以用 layer 切】
    L0_curated     32   團隊手寫種子,人工確認過       收
    L1_onet        35   O*NET 可轉移技能,抽象職能     排除
    L2_ucan        22   UCAN 目標途徑職能,半具體      可選,預設不收
    L3_market     132   市場爬蟲頻次,最像履歷         收(主力)
    L4_merge_alias   4   別名合併產物,不是獨立技能     排除
    L5_interview_coverage 29  面試黃金集覆蓋層        只挑工具名型

L5 要特別處理:它是為了讓面試黃金測試集的標記鍵對得上才補的,措辭直接取自
真實逐字稿,所以裡面有「Google Data Studio」這種工具名,也有
「電話接聽與人員接待事項」這種整句。整層收會讓競爭者履歷出現一整句話當技能名。

【已知的品質風險】
L3_market 佔了 152 裡的 132,而那一層是爬蟲頻次來的、沒有人逐筆看過。
意思是這個池的品質幾乎等於 L3 的品質。這件事會在生成出奇怪的技能名時
才被發現,所以 `audit()` 把分層組成印出來,不要等到看到輸出才想起這件事。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

VOCAB_PATH = Path("fixtures/vocab/skills_v1.json")
PROVENANCE_PATH = Path("fixtures/vocab/skills_v1.provenance.json")

RESUME_LIKE_LAYERS: tuple[str, ...] = ("L0_curated", "L3_market")
"""預設收哪幾層。成員 A 的建議:L0 ∪ L3,排除 L1_onet。"""

OPTIONAL_LAYERS: tuple[str, ...] = ("L2_ucan",)
"""半具體,要放寬的時候才加。加進來是 171 筆。"""

NOT_A_SOURCE_LAYERS: tuple[str, ...] = ("L1_onet", "L4_merge_alias")
"""不能當來源層的。只屬於這些層的那些不收。

L1_onet 是抽象職能(主動聆聽、判斷與決策、操作監控),拿去當競爭者履歷技能會很假。
L4_merge_alias 標的是「這一筆吸收了別名」(Photoshop 與 Adobe Photoshop 合併),
不是一個獨立技能的來源。

【這裡原本寫成硬排除,而那是錯的】
第一版寫「只要屬於這兩層之一就不收」,結果砍掉四筆:
CSS、HTML、Illustrator、Adobe Photoshop——整個詞彙表裡最像履歷的那幾個。
它們同時屬於 L3_market 與 L4_merge_alias,而 L4 那個標籤講的是別名合併,
跟品質正交。

所以規則是**只屬於**這些層才不收。對預設路徑這條永遠不觸發
(收取條件已經要求有 L0 或 L3),它真正擋的是 `also` 那條路:
有人手動指定一個只有 L4 的 id 進來。

這跟成員 A 警告的「可信不等於你要的」是同一個錯誤形狀:
拿一個看起來合理的標籤當篩子,而篩掉的方向剛好相反。
"""

SELECTIVE_LAYERS: tuple[str, ...] = ("L5_interview_coverage",)
"""只挑工具名型的那一層。預設不收,因為「只挑工具名型」需要人工判斷。

要收的話呼叫端自己給一份 skill_id 清單,不要在這裡寫一條
「長度小於 N 就算工具名」的規則——那種規則看起來能動,
實際上會把「Google Data Studio」排除、把「電話接聽」收進來。
"""

MAX_NAME_CHARS = 20
"""技能名長過這個字數就不收。

**預設池裡沒有一筆會被它擋到**(最長 16 字),所以它對預設路徑是惰性的。
它存在是為了 `also` 那條路:L5_interview_coverage 的措辭取自真實逐字稿,
裡面有「電話接聽與人員接待事項」這種整句。

而它也擋不住那一句(12 個字)——擋整句的主要手段是 L5 預設不收,
這條只是最後一道。`audit()` 會印出池子裡最長的名稱,
所以「這條現在是惰性的」這件事看得見,不用靠人記得。
"""


@dataclass(frozen=True)
class Skill:
    """池子裡的一筆技能。只留生成需要的欄位。"""

    skill_id: str
    name_zh: str
    name_en: str = ""
    layers: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        """寫在競爭者履歷上的名字。中文優先,沒有才用英文。"""
        return self.name_zh or self.name_en


def load_pool(
    *,
    vocab_path: Path | str = VOCAB_PATH,
    provenance_path: Path | str = PROVENANCE_PATH,
    layers: Sequence[str] = RESUME_LIKE_LAYERS,
    also: Iterable[str] = (),
) -> tuple[Skill, ...]:
    """載入技能池。照 layer 篩,排除硬排除層。

    `also` 是額外收的 skill_id(例如從 L5 挑出來的工具名型)。
    它仍然要過 `NOT_A_SOURCE_LAYERS` 與長度檢查——「我特別指定的」
    不等於「可以繞過規則」,不然那條規則遲早被繞光。

    回傳照 skill_id 排序,所以同樣的輸入永遠同樣的順序。
    生成要可重現,而 dict 的順序不是合約。
    """
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    prov = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    want = set(layers)
    extra = set(also)

    out: list[Skill] = []
    for entry in vocab:
        sid = entry.get("skill_id", "")
        p = prov.get(sid) or {}
        got = tuple(p.get("layers") or ())
        if got and set(got) <= set(NOT_A_SOURCE_LAYERS):
            continue
        if not (set(got) & want or sid in extra):
            continue
        name = entry.get("name_zh") or entry.get("name_en") or ""
        if not name or len(name) > MAX_NAME_CHARS:
            continue
        out.append(
            Skill(
                skill_id=sid,
                name_zh=entry.get("name_zh", ""),
                name_en=entry.get("name_en", ""),
                layers=got,
            )
        )
    return tuple(sorted(out, key=lambda s: s.skill_id))


@lru_cache(maxsize=1)
def default_pool() -> tuple[Skill, ...]:
    """預設技能池,只載一次。

    `load_pool` 本身不快取,因為測試要餵不同路徑的 fixture。
    但正式路徑上它在 `create_interview` 裡面,而每一場都重讀兩個 JSON、
    重建索引是白做的工。所以預設那一組單獨包一層。
    """
    return load_pool()


def audit(pool: Sequence[Skill]) -> dict[str, int]:
    """池子的分層組成。印出來是為了讓品質風險看得見。

    L3_market 佔絕大多數這件事不該靠人記得——它決定了這個池的品質上限,
    而那一層沒有人逐筆看過。
    """
    counts: dict[str, int] = {}
    for s in pool:
        for layer in s.layers:
            counts[layer] = counts.get(layer, 0) + 1
    counts["_total"] = len(pool)
    counts["_longest_name"] = max((len(s.display) for s in pool), default=0)
    return counts
