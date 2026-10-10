"""場上每個人的三層狀態。

【為什麼需要這個檔】
現在的 `PersonaSpec` 把所有欄位放在同一層,沒有任何地方說明哪些欄位
整場不該變、哪些該累積、哪些每輪要重算。Qi et al.(2026)把這種寫法
直接點名為 persona drift 的成因:persona 當成單一靜態屬性,結果不是
機械式重複就是角色崩壞。

我們自己的已知限制「AI-邏輯 與 AI-親切 退化成同一種禮貌語氣」就是那個 drift。

【三層的差別是生命週期,不是三個功能】

    Long   整場不變        persona、私有目標、經歷池、目標職缺
    Mid    每輪追加只增不減  我說過什麼、誰忽略了我
    Short  每輪重算不落地    即時狀態

Short 不落地是刻意的:上一輪的狀態不該影響這一輪。存下來就會變成一個
沒有人宣告過生命週期的欄位,而「事實只存在於資料的形狀裡卻沒有被宣告」
這件事在這個專案已經出過五次事(斷句來源、填充詞可信度、輸入方式、
段界、發言者身分)。

【這一版只宣告形狀,不改行為】
三層可以是空的。填哪一層就是做哪個功能:

    Mid    對話記憶(群面競爭者)
    Short  AI 競爭者從處境推導的即時狀態

使用者的情緒判讀**不在這裡**。那一項已改為報告端的逐題回饋,
不進面試現場,所以使用者在面試過程中沒有 Short 層。

【為什麼規則要靠型別而不是靠約定】
三條規則都寫成「不可能違反」而不是「不要違反」:

    identity 是 frozen,而且 advanced() 不接受新的 identity
    memory 只有 appended(),沒有刪除或覆寫的入口
    advanced() 一定重設 affect —— 不傳就是清空,沒有「沿用上一輪」這個選項

規則寫兩次沒生效就不要寫第三次。這三條有唯一正確答案,所以交給型別。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Long —— 身分層
# ---------------------------------------------------------------------------


@runtime_checkable
class Identity(Protocol):
    """身分層的最小介面:一個穩定的識別字串。

    實際型別依角色而不同,這裡不收斂成單一 dataclass:

        AI 競爭者   PersonaSpec(+ 之後的私有目標)
        面試官      InterviewerSpec(尚未實作)
        使用者      經歷池與目標職缺

    三者唯一的共通點就是「整場不變,而且有 id」。多要求一個欄位,
    就會逼其中一種角色塞一個它沒有的東西進來。
    """

    id: str


# ---------------------------------------------------------------------------
# Mid —— 累積層
# ---------------------------------------------------------------------------

MEMORY_KINDS: tuple[str, ...] = ("said", "ignored", "conflict")
"""這一層記三件事,各自有明確的用途:

    said      我說過什麼       —— 不然會重複自己的主張
    ignored   我被誰跳過       —— 被忽略兩次要更堅持
    conflict  誰講的跟我衝突   —— 競爭者要能接上分歧,不是各說各話

刻意不記「發言次數」。那個用 `spoken_by` 數就有,而且數得比模型準。
"""


@dataclass(frozen=True)
class MemoryEntry:
    """一筆記憶。turn_idx 是它發生在第幾輪,用來排序與對帳。"""

    turn_idx: int
    kind: str
    text: str
    """盡量是逐字稿的原句。報告端的引用要能機械驗證,記憶這邊先留一樣的習慣。"""


@dataclass(frozen=True)
class SessionMemory:
    """Mid 層。只增不減,TTL 到面試結束。

    【為什麼不做跨場記憶】
    跨場會讓狀態逃出單場的生命週期,而目前沒有任何功能需要它。
    真的要做的時候,那是另一個層,不是把這一層的 TTL 拿掉。
    """

    entries: tuple[MemoryEntry, ...] = ()

    def appended(self, *new: MemoryEntry) -> SessionMemory:
        """回一份新的記憶。沒有刪除或覆寫的入口是刻意的。"""
        return SessionMemory(entries=self.entries + tuple(new))

    def of_kind(self, kind: str) -> tuple[MemoryEntry, ...]:
        return tuple(e for e in self.entries if e.kind == kind)

    def ignored_count(self) -> int:
        """被跳過幾次。『被忽略兩次要更堅持』唯一該讀的數字。"""
        return len(self.of_kind("ignored"))


# ---------------------------------------------------------------------------
# Short —— 即時層
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Affect:
    """Short 層。每輪重算,不落地,不跨輪沿用。

    【目前只有 AI 競爭者用得到】
    而且它是**從處境推導**的,不是判讀出來的 —— 被忽略兩次就更堅持,
    那是讀 Mid 層算出來的,不需要呼叫模型。

    使用者的情緒判讀不走這裡,它在報告端逐題標註。
    """

    label: str = ""
    """空字串代表「這一輪沒有推導出任何狀態」,不是「平穩」。"""

    note: str = ""
    """為什麼推導成這個狀態。空的 label 不該有 note。"""

    def is_empty(self) -> bool:
        return not self.label


EMPTY_AFFECT = Affect()


# ---------------------------------------------------------------------------
# 三層合一
# ---------------------------------------------------------------------------

AGENT_KINDS: tuple[str, ...] = ("user", "interviewer", "peer")


@dataclass(frozen=True)
class AgentState:
    """場上一個人的完整狀態。每個人一份,使用者也算。

    【為什麼使用者也要有一份】
    使用者的 Long(經歷池、目標職缺)與 Mid(講過什麼)其實已經在傳了,
    只是散在 request 的不同欄位裡。收成同一個結構之後,
    「使用者」與「AI」在型別上變成同一種東西,下游不必為兩者各寫一套。
    """

    agent_id: str
    kind: str
    identity: Identity
    memory: SessionMemory = field(default_factory=SessionMemory)
    affect: Affect = EMPTY_AFFECT

    def advanced(
        self, *, remembered: tuple[MemoryEntry, ...] = (), affect: Affect = EMPTY_AFFECT
    ) -> AgentState:
        """推進一輪。這是唯一的更新入口。

        三層在這裡各走各的路:

            identity   原封不動 —— 這個方法不收新的 identity
            memory     追加 remembered
            affect     **一律換成參數給的值**,不傳就是清空

        最後一條是 Short 層不累積的實作。不傳 affect 卻沿用上一輪的,
        在這個介面上做不到 —— 那正是重點。
        """
        return replace(self, memory=self.memory.appended(*remembered), affect=affect)
