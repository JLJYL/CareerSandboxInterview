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
class JDInput:
    """JD 的結構化輸入。

    【為什麼不是純字串】
    成員 A 以 jobs_all.jsonl 全量實測:1,010 筆同時具備 requiredSkills 與
    description 的 JD 中,5,988 條 requiredSkills 只有 332 條(5.5%)字面出現在
    description 裡;82.2% 的 JD 完全沒有任何 requiredSkill 出現在散文中,
    每份 JD 的散文涵蓋率中位數為 0.0%。

    只給 description 等於要求從一份幾乎不提需求的散文裡反推需求,
    JD 側召回率上限就是 5.5%,之後校準怎麼調都救不回來。

    【required_skills 從哪裡來】
    兩條路,交給 A 的形狀相同:
      黃金測試集 —— 直接取型錄職缺的 requiredSkills
      正式環境   —— 前端只有使用者手貼的散文(InterviewConfig.customJd,
                    且該欄位標示為選填),由 B 的 LLM 抽取器產出

    B 的抽取器不可省略。省略的話 A 會在一個正式環境不存在的分佈上校準:
    型別漂亮,但 required_skills 恆空。

    順序即優先序:104 的 requiredSkills 排前面的通常較關鍵,
    A 的 JD_POSITION_DECAY 依賴這個順序,重排會破壞權重。
    """

    required_skills: list[str] = field(default_factory=list)
    description: str = ""
    source: str = "extracted"
    """required_skills 的來源:"catalog"(型錄結構化欄位)或 "extracted"
    (B 的 LLM 從散文抽取)。用於區分校準分佈與正式分佈,
    評測報表應分開統計這兩種來源的命中率。"""


@dataclass(frozen=True)
class TextStats:
    """逐字稿的確定性文字統計。

    用途:當 LLM 評分的錨點。填充詞次數是硬數字,不讓 LLM 自己數——
    LLM 數數字不可靠,而且同一段逐字稿每次數出來還不一樣,分數就不穩定。
    """

    char_count: int = 0
    filler_count: int = 0
    filler_detail: dict[str, int] = field(default_factory=dict)
    quantifier_count: int = 0
    sentence_count: int = 0
    avg_sentence_len: float = 0.0
    filler_reliability: str = "unknown"
    """filler_count 是否可信,決定 B 能不能拿它當流暢度的錨點。

      "measured"        逐字稿裡確實有填充詞,計數反映使用者的口語習慣
      "suppressed"      引擎做了 disfluency removal,計數反映的是後處理不是使用者
      "unknown"         尚未由 STT_TERM_PROBE 確認
      "not_applicable"  這段是打字輸入,填充詞計數本來就無意義

    "not_applicable" 由 B 側設定,不是 A 判定的——A 只拿得到 transcript,
    看不出它是講的還是打的。B 依 HTTP 層的 input_mode 以 dataclasses.replace()
    覆寫。A 的 text_stats 簽章不變。

    【為什麼需要這一欄】
    多數語音辨識引擎預設移除「嗯」「那個」「就是」。若 Android 也如此,
    filler_count 會恆為接近 0,而 B 的 subScores 若拿它當「表達流暢度」的錨點,
    每個人都會拿到一樣的分數——而且看起來很客觀,因為背後有硬數字。
    那比沒有這個維度更糟。

    【B 側規則】
    不是 "measured" 時,流暢度不得以填充詞為主要依據,改用不受 disfluency
    removal 影響的文字訊號:連接詞單調性(然後/所以 的重複率)、句長變異、
    答案長度與題目複雜度的落差。並在 verdict 措辭上不宣稱量到了口語習慣。

    表達流暢度是 SUB_SCORE_NAMES 的固定值域,不可移除,只能換錨點。

    這一欄與 segmentation 是同一種設計:讓不可信的測量值標示自己不可信,
    而不是讓 prompt 去猜。
    """

    segmentation: str = "unavailable"
    """斷句方式,決定 sentence_count 與 avg_sentence_len 的可信度。

      "punctuation"       有標點,精確值
      "discourse_marker"  無標點,以語氣詞估算,是估算值
      "unavailable"       無法斷句,兩個欄位皆為 0

    B 側規則:segmentation 不是 "punctuation" 時,prompt 不得把平均句長
    講得像量出來的。Android 中文 STT 通常不輸出標點,正式環境的常態會是
    "discourse_marker",兩態設計(有值 / 0)會讓估算值無處安放。
    """


@dataclass(frozen=True)
class GapCandidate:
    """一個漏講候選。

    evidence 欄位是 B 寫 why 時的唯一素材來源。
    沒有 evidence,B 只能靠 LLM 自由發揮,就會捏造使用者沒有的經歷——
    這是 B1 已經封死的路,面試這邊沒有理由開後門。
    """

    skill_id: str
    """三個集合 join 的鍵,帶命名空間前綴(sk: / skm:)。

    不可用顯示字串當鍵:兩個命名空間若有同名條目會靜默合併,
    且 B 側拿到字串之後無法回查詞彙表。
    """

    display: str
    """前端顯示與 log 用,格式 name_zh(skill_id)。"""

    kind: str = "hard"
    """"hard" 或 "soft"。決定 B 的 why 走哪一路。

    【語意:這是可偵測性,不是技能分類學】
    kind 回答的問題是「如果使用者講了,我抓不抓得到」,不是「這是不是軟技能」。

      抓得到 → 沒抓到就代表他真的沒講 → 否定判斷有意義 → "hard"
      抓不到 → 沒抓到什麼都不代表 → 這條 gap 是雜訊     → "soft"

    這是詞彙表別名覆蓋度的函數,不是技能本身的屬性。同一個「溝通協調」:
    詞彙表有「溝通」這個口語別名時它是 hard,只有正式全名時它是 soft。
    分類學前綴(sk: / skm:)編碼不了這件事,不可用前綴推導。

    保留 hard/soft 這組字面是為了不讓已對齊的兩側重跑,語意以本段為準。

    【一個實測踩過的坑】
    不可用「含拉丁字母即為 hard」判別。詞彙表每個條目都有 name_en,
    「溝通協調」的 name_en 是 Communication,會讓每個軟技能都誤判成 hard。
    判別只看 name_zh,英文名不是「口語會講出來」的證據。

    【B 側規則】
    kind == "soft" 時直接濾掉,不要改用委婉語氣保留。一份漏講清單裡混進
    假指控,比清單短一點糟得多——使用者會照著一個他其實已經講過的點去改,
    而且是以同樣的自信度被告知的。

    但這道過濾是第二道防線,不是常開。見 GapComputer.compute 的 emit_soft 說明。

    【A 側】
    W2 D3–D5 以黃金集實測取代啟發式:對每個已標記為「有講」的技能,
    查表面掃描抓不抓得到,那才是可偵測性的實測值。
    無法判定時一律填 "hard"(保守,寧可少濾)。
    """

    resume_experience_ids: list[str] = field(default_factory=list)
    """命中的經歷 id,可為空,可為多筆。

    刻意用複數:實測 persona 履歷的技能標籤全部集中在 experiences[0].tags,
    包含對無經歷者補一筆空殼來掛 tags。用單數會讓 why 寫成
    「你在圖書館志工經歷中展現了進度控管」這類錯誤歸因——
    比沒有 evidence 更危險,因為它讀起來很具體。

    B 側鐵則見下方 WHY_MUST_BE_JD_SIDE。
    """

    resume_evidence: str = ""
    """履歷原文片段或標籤,證明使用者確實有這項。"""

    jd_evidence: str = ""
    """JD 原文片段,證明這份工作確實重視。這是 why 唯一允許引用的素材。"""

    weight: float = 0.0
    """0–1,JD 重視程度,用於排序取前 N 名。內部用,不出 HTTP。"""


WHY_MUST_BE_JD_SIDE = """MissingPoint.why 的鐵則:

why 是「為什麼這一點對這份 JD 重要」,是 JD 側的重要性論證,
不是履歷側的敘事。

禁止在 why 裡敘述使用者在哪一段經歷做了什麼。
理由:技能標籤與經歷的對應關係不可靠(標籤集中在第一筆經歷),
一旦敘事就會產生錯誤歸因,而且錯得很具體、很有說服力。

經歷標題只當弱線索,用於決定 point 的措辭,不可寫進 why。
why 只能引用 GapCandidate.jd_evidence。"""


@dataclass(frozen=True)
class Utterance:
    """群面的一則發言。

    【為什麼不是 list[str]】
    協作四項有三項需要時序與發言者身分:
      參與主動性 —— 需要首次發言時機
      傾聽與回應 —— 需要比對前一位發言者說了什麼
      協作姿態   —— 需要時間重疊才算得出打斷

    尤其「傾聽與回應」:若輸入只有使用者自己的發言,
    前一位發言者的內容根本不在場,這不是難,是資訊缺失。

    【前端配合事項】
    目前 InterviewLiveGroupScreen.submitGroup 只呼叫
    recordGroupSay(使用者的話),AI 同儕的發言進了畫面的 messages
    但從未寫入 session。需請前端改為記錄所有發言者。
    messages 已經帶 speaker 欄位,改動不大。
    """

    speaker_id: str
    """"user" 或 persona 的顯示名稱(主考官 / AI-邏輯 / AI-強勢 / AI-親切)。"""

    text: str
    start_ms: int = 0
    """毫秒。沒有計時資料時全部給 0,至少保住陣列順序。"""

    end_ms: int = 0
    """毫秒。有值才算得出打斷。"""


@dataclass(frozen=True)
class CollabRaw:
    """協作維度的確定性分數。僅 group 模式使用。

    name 必須是 COLLAB_DIM_NAMES 四個之一。
    evidence 給 B 寫 hint 用,例如「共發言 4 次,其中 1 次接續他人論點」。

    注意尺度:score 是 int 0–100,直接對應 Kotlin CollabDim.score 的 UI 欄位;
    GapCandidate.weight 是 float 0–1,內部排序用不出 HTTP。
    兩者用途不同,刻意不統一。
    """

    name: str
    score: int
    evidence: str = ""


# ---------------------------------------------------------------------------
# 三個 Protocol —— 成員 A 的交付項目
# ---------------------------------------------------------------------------


@runtime_checkable
class TranscriptAnalyzer(Protocol):
    """逐字稿的正規化與統計。成員 A 負責。

    【生命週期】
    mentioned_skills 需要語意比對(「我會寫 SQL」與「用資料庫查資料」要歸到
    同一詞),會載入 bge-m3,模型體積約 2.3GB。實作必須是 app 啟動時載入的
    singleton,不可每個 request 建立新實例,否則第一次呼叫會逾時。
    端點組裝時以 FastAPI lifespan 處理。
    """

    def mentioned_skills(self, transcript: str) -> set[str]:
        """逐字稿 → 使用者實際講出來的技能集合。

        回傳 skill_id(帶 sk: / skm: 前綴),不是顯示字串。
        這是三個集合 join 的鍵,顯示字串走 GapCandidate.display。
        """
        ...

    def text_stats(self, transcript: str) -> TextStats:
        """逐字稿 → 確定性文字統計,含 segmentation 標示。"""
        ...


@runtime_checkable
class GapComputer(Protocol):
    """三集合差集運算。成員 A 負責。"""

    def compute(
        self,
        resume: list[dict],
        jd: JDInput,
        transcript: str,
    ) -> list[GapCandidate]:
        """回傳漏講候選,依 weight 由高到低排序。

        履歷為空時回空清單,不要拋例外——沒建經歷是正常情況,
        端點要能優雅降級。

        【soft 候選的發放時機】
        實作以 emit_soft 開關與語意比對連動:

          語意段關閉(W1)     不發 soft。發出去的每一條都是表面或詞頭搆得到的
          語意段開啟(W2 校準後) 發 soft,帶 kind="soft"

        理由:語意段沒開時,軟技能的漏講判定全部是低信心,不是有些。
        把一批本來就沒有一條可信的候選交給 B 逐條判斷,B 沒有資訊可以做那個判斷。

        對應的,B 側的 kind 過濾是第二道防線,不可設成永久開啟——
        永久開啟等於把軟技能整類砍掉,整個功能會退化成關鍵字比對。
        語意段上線後,判斷交還給信心度。

        兩道防線不衝突:W1 沒有 soft 流過來,B 的過濾器空轉;
        FakeGapComputer 固定帶一筆 soft,讓 B 在 W1 就測得到過濾路徑。
        """
        ...


@runtime_checkable
class CollabScorer(Protocol):
    """群面協作四項的確定性分數。成員 A 負責,W2 交付。"""

    def score(self, utterances: list[Utterance]) -> list[CollabRaw]:
        """回傳四筆,順序對齊 COLLAB_DIM_NAMES。

        建議的確定性指標(實作細節由 A 決定):
          參與主動性 —— 發言次數、首次發言時機(需 start_ms 或陣列順序)
          傾聽與回應 —— 發言中出現前一位發言者關鍵詞的比例
          論點建構   —— 因果連接詞密度、平均發言長度
          協作姿態   —— 打斷次數(需 start_ms/end_ms)、同意與反對詞比例

        時間資料全為 0 時,前兩項退化為依陣列順序估算,第四項的打斷偵測
        不可用,應在 evidence 註明,並由 B 在 hint 措辭上避免過度宣稱。
        """
        ...


# ---------------------------------------------------------------------------
# Fake 實作 —— 成員 B 在 A 交件前的替身
# ---------------------------------------------------------------------------


class FakeTranscriptAnalyzer:
    """固定回傳。只為了讓 B 的 pipeline 跑得起來。"""

    def mentioned_skills(self, transcript: str) -> set[str]:
        found = set()
        for kw, sid in (("SQL", "sk:sql"), ("Python", "sk:python"), ("Excel", "sk:excel")):
            if kw.lower() in transcript.lower():
                found.add(sid)
        return found

    def text_stats(self, transcript: str) -> TextStats:
        detail = {w: transcript.count(w) for w in ("嗯", "那個", "就是") if w in transcript}
        has_punct = any(c in transcript for c in "。,,!?")
        return TextStats(
            char_count=len(transcript),
            filler_count=sum(detail.values()),
            filler_detail=detail,
            quantifier_count=sum(c.isdigit() for c in transcript),
            sentence_count=0,
            avg_sentence_len=0.0,
            filler_reliability="measured" if detail else "unknown",
            segmentation="punctuation" if has_punct else "unavailable",
        )


class FakeGapComputer:
    """固定回傳三個候選。履歷為空時回空清單,這個降級行為要跟真實作一致。"""

    def compute(self, resume: list[dict], jd: JDInput, transcript: str) -> list[GapCandidate]:
        if not resume:
            return []
        first_id = str(resume[0].get("id", ""))
        return [
            GapCandidate(
                skill_id="sk:sql",
                display="SQL(sk:sql)",
                kind="hard",
                resume_experience_ids=[first_id],
                resume_evidence="用 SQL 重寫週報查詢",
                jd_evidence="需具備 SQL 撰寫能力",
                weight=0.9,
            ),
            GapCandidate(
                skill_id="sk:report_automation",
                display="報表自動化(sk:report_automation)",
                kind="hard",
                resume_experience_ids=[first_id],
                resume_evidence="將週報產出時間由 4 小時縮短為 1 小時",
                jd_evidence="能建立自動化報表流程",
                weight=0.75,
            ),
            GapCandidate(
                skill_id="skm:cross_team_comm",
                display="跨部門溝通(skm:cross_team_comm)",
                kind="soft",
                resume_experience_ids=[first_id],
                resume_evidence="協調行銷與工程兩邊的需求排程",
                jd_evidence="需與產品、工程團隊密切協作",
                weight=0.6,
            ),
        ]


class FakeCollabScorer:
    """固定回傳四筆,順序正確。"""

    def score(self, utterances: list[Utterance]) -> list[CollabRaw]:
        mine = [u for u in utterances if u.speaker_id == "user"]
        n = len(mine)
        return [
            CollabRaw("參與主動性", min(100, 40 + n * 10), f"共發言 {n} 次"),
            CollabRaw("傾聽與回應", 68, "有 1 次接續他人論點"),
            CollabRaw("論點建構", 71, "平均發言 32 字"),
            CollabRaw("協作姿態", 70, "無打斷,同意詞 2 次"),
        ]
