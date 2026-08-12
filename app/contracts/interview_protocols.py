"""面試模組的內部國界:成員 A 與成員 B 的交接介面。

【切線原則】
沿用既有分工:A 管正規化、檢索、確定性計分;B 管 LLM 擷取與端點。

漏講點的本質是三個集合的差集:

    履歷技能集 ∩ JD 需求集 − 逐字稿提及集 = 漏講候選

三個集合都要經過正規化才能比對,這整段屬於 A。
B 拿到候選清單之後,只負責替每一條寫那句 why。

【為什麼要凍結成 Protocol】
B 從第一天就用 Fake 實作開工,不必等 A 交件;A 依 Protocol 寫真實作,
不必等 B 的 prompt 定稿。兩邊只在同步點換一行 import。
這跟 W2 用 FakeRetriever 頂著的做法是同一套。

【本檔在 D1 凍結,之後不改】
要改先在群組提出,兩人都同意才動。因為兩邊會同時依賴它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# 交換用的資料結構(dataclass,不是 pydantic —— 這是內部型別,不出 HTTP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextStats:
    """逐字稿的確定性文字統計。

    用途:當 LLM 評分的錨點。填充詞次數是硬數字,不讓 LLM 自己數——
    LLM 數數字不可靠,而且同一段逐字稿每次數出來還不一樣,分數就不穩定。

    注意 Android 中文 STT 通常不輸出標點,sentence_count 與
    avg_sentence_len 在無標點時要用其他方式斷句(語氣詞、連接詞),
    或直接回 0 表示不可用。實作時以怡君提供的真實逐字稿樣本為準。
    """

    char_count: int = 0
    filler_count: int = 0  # 填充詞總次數
    filler_detail: dict[str, int] = field(default_factory=dict)  # 嗯/那個/就是 各幾次
    quantifier_count: int = 0  # 數字與量化詞出現次數
    sentence_count: int = 0  # 0 代表無法斷句
    avg_sentence_len: float = 0.0  # 0 代表不可用


@dataclass(frozen=True)
class GapCandidate:
    """一個漏講候選。

    evidence 兩個欄位是 B 寫 why 時的唯一素材來源。
    沒有 evidence,B 只能靠 LLM 自由發揮,那就會捏造使用者沒有的經歷——
    這是 B1 已經封死的路,面試這邊沒有理由開後門。
    """

    skill: str  # 正規化後的技能詞,前端顯示用
    resume_experience_id: str  # 來自哪一筆經歷(ExperienceDTO.id)
    resume_evidence: str  # 履歷原文片段,證明使用者確實有這項
    jd_evidence: str  # JD 原文片段,證明這份工作確實重視
    weight: float = 0.0  # 0–1,JD 重視程度,用於排序取前 N 名


@dataclass(frozen=True)
class CollabRaw:
    """協作維度的確定性分數。僅 group 模式使用。

    name 必須是 COLLAB_DIM_NAMES 四個之一。
    evidence 給 B 寫 hint 用,例如「共發言 4 次,其中 1 次接續他人論點」。
    """

    name: str
    score: int  # 0–100
    evidence: str = ""


# ---------------------------------------------------------------------------
# 三個 Protocol —— 成員 A 的交付項目
# ---------------------------------------------------------------------------


@runtime_checkable
class TranscriptAnalyzer(Protocol):
    """逐字稿的正規化與統計。成員 A 負責。

    可直接沿用既有的 VocabNormalizer,換一個輸入源即可。
    """

    def mentioned_skills(self, transcript: str) -> set[str]:
        """逐字稿 → 使用者實際講出來的技能集合(已正規化)。

        正規化的重點:使用者口語會說「我會寫 SQL」「用資料庫查資料」,
        兩者都應該歸到同一個正規化詞。沒有這一層,差集會全部誤報。
        """
        ...

    def text_stats(self, transcript: str) -> TextStats:
        """逐字稿 → 確定性文字統計。"""
        ...


@runtime_checkable
class GapComputer(Protocol):
    """三集合差集運算。成員 A 負責。

    可沿用 WeightedScorer 的覆蓋率計算,方向反過來取即可。
    """

    def compute(
        self,
        resume: list[dict],  # ExperienceDTO.model_dump() 的清單
        jd: str,
        transcript: str,
    ) -> list[GapCandidate]:
        """回傳漏講候選,依 weight 由高到低排序。

        履歷為空時回空清單,不要拋例外——沒建經歷是正常情況,
        端點要能優雅降級。
        """
        ...


@runtime_checkable
class CollabScorer(Protocol):
    """群面協作四項的確定性分數。成員 A 負責。

    輸入是 InterviewSession.groupSays,跟 turns 是不同的資料流。
    """

    def score(self, group_says: list[str]) -> list[CollabRaw]:
        """回傳四筆,順序對齊 COLLAB_DIM_NAMES。

        建議的確定性指標(實作細節由 A 決定):
          參與主動性 —— 發言次數、首次發言時機
          傾聽與回應 —— 發言中出現前一位發言者關鍵詞的比例
          論點建構   —— 因果連接詞密度、平均發言長度
          協作姿態   —— 打斷次數、同意詞與反對詞的比例
        """
        ...


# ---------------------------------------------------------------------------
# Fake 實作 —— 成員 B 在 A 交件前的替身
# ---------------------------------------------------------------------------


class FakeTranscriptAnalyzer:
    """固定回傳。只為了讓 B 的 pipeline 跑得起來。"""

    def mentioned_skills(self, transcript: str) -> set[str]:
        found = set()
        for kw in ("SQL", "Python", "Excel", "簡報", "行銷"):
            if kw.lower() in transcript.lower():
                found.add(kw)
        return found

    def text_stats(self, transcript: str) -> TextStats:
        detail = {w: transcript.count(w) for w in ("嗯", "那個", "就是") if w in transcript}
        return TextStats(
            char_count=len(transcript),
            filler_count=sum(detail.values()),
            filler_detail=detail,
            quantifier_count=sum(c.isdigit() for c in transcript),
            sentence_count=0,
            avg_sentence_len=0.0,
        )


class FakeGapComputer:
    """固定回傳三個候選。履歷為空時回空清單,這個降級行為要跟真實作一致。"""

    def compute(self, resume: list[dict], jd: str, transcript: str) -> list[GapCandidate]:
        if not resume:
            return []
        first_id = str(resume[0].get("id", ""))
        return [
            GapCandidate(
                skill="SQL",
                resume_experience_id=first_id,
                resume_evidence="用 SQL 重寫週報查詢",
                jd_evidence="需具備 SQL 撰寫能力",
                weight=0.9,
            ),
            GapCandidate(
                skill="報表自動化",
                resume_experience_id=first_id,
                resume_evidence="將週報產出時間由 4 小時縮短為 1 小時",
                jd_evidence="能建立自動化報表流程",
                weight=0.75,
            ),
            GapCandidate(
                skill="跨部門溝通",
                resume_experience_id=first_id,
                resume_evidence="協調行銷與工程兩邊的需求排程",
                jd_evidence="需與產品、工程團隊密切協作",
                weight=0.6,
            ),
        ]


class FakeCollabScorer:
    """固定回傳四筆,順序正確。"""

    def score(self, group_says: list[str]) -> list[CollabRaw]:
        n = len(group_says)
        return [
            CollabRaw("參與主動性", min(100, 40 + n * 10), f"共發言 {n} 次"),
            CollabRaw("傾聽與回應", 68, "有 1 次接續他人論點"),
            CollabRaw("論點建構", 71, "平均發言 32 字"),
            CollabRaw("協作姿態", 70, "無打斷,同意詞 2 次"),
        ]
