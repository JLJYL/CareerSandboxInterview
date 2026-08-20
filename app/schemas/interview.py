"""面試模組 HTTP 合約(A1–A5)。

【本檔的地位】
這是前端 Kotlin 介面的 Python 鏡射。每一個 class 都對應 repo 內
app/src/main/java/com/careersandbox/app/data/mock/ 底下的一個 data class 或 interface,
欄位名稱、型別、順序皆逐檔核對過。改這裡之前先確認 Kotlin 那側也要改。

【對照表】
  InterviewConfig.kt        -> InterviewContext
  InterviewSession.Turn     -> TurnDTO
  ExperienceDtos.kt         -> ExperienceDTO
  MissingPointsAnalyzer.kt  -> MissingPointDTO           (A5)
  InterviewReportData.kt    -> FaceDimensionDTO / SubScoreDTO / QuestionFeedbackDTO
                               StarPartDTO / VideoDimDTO / CollabDimDTO  (A4)
  InterviewProber.kt        -> TurnResponse              (A1, mode=single)
  PanelDispatcher.kt        -> TurnResponse              (A2, mode=panel)
  GroupDispatcher.kt        -> TurnResponse + 開場預生成  (A3, mode=group)

【一條鐵則】
request 嚴格、response 寬鬆。
request 欄位由前端控制,值域封閉,驗證嚴格可以及早發現前後端不同步。
response 欄位有 LLM 產出成分,值域無法保證,一律用開放型別 + 機械修復,
不用 Literal。理由見 CareerCategory 事故:把 LLM 產出的欄位鎖成 Literal
會在正式環境噴 500,而不是回一個略微不完美的結果。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

# ---------------------------------------------------------------------------
# 基底:JSON 走 camelCase(對齊 Kotlin/Gson),Python 內部走 snake_case
# ---------------------------------------------------------------------------
# 【上線前必須確認】B1/B2/C1 既有端點若不是用 camelCase,把 alias_generator
# 拔掉即可,只需要改這一個地方。


class _Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# 值域常數
# ---------------------------------------------------------------------------

InterviewMode = Literal["single", "panel", "group"]
"""三種面試模式。single 一對一、panel 主管三人、group 團體。

值由前端路由決定(Routes.INTERVIEW_LIVE_INDIVIDUAL / _PANEL / _GROUP),
封閉集合,用 Literal 安全。
"""

# 三個面向的固定標記與名稱,前端 UI 照 letter 排版,不可增減也不可改名。
FACE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("內", "內容"),
    ("構", "結構"),
    ("達", "表達"),
)

# 六項細分分數的固定名稱與順序,前端雷達圖照這個順序畫。
SUB_SCORE_NAMES: tuple[str, ...] = (
    "內容深度",
    "邏輯清晰度",
    "表達流暢度",
    "互動能力",
    "應變能力",
    "自信程度",
)

# STAR 四段的固定鍵值與名稱。
STAR_PARTS: tuple[tuple[str, str], ...] = (
    ("S", "情境 Situation"),
    ("T", "任務 Task"),
    ("A", "行動 Action"),
    ("R", "結果 Result"),
)

# 協作維度的四項固定名稱,僅 group 模式產出。
COLLAB_DIM_NAMES: tuple[str, ...] = (
    "參與主動性",
    "傾聽與回應",
    "論點建構",
    "協作姿態",
)

# 表達面向的語音指標標籤。本期只做「填充詞」,其餘等前端補時間資料。
PROSODY_LABELS: tuple[str, ...] = (
    "語速",
    "停頓",
    "語調",
    "填充詞",
    "開口前思考",
)


# ---------------------------------------------------------------------------
# 共用資料結構
# ---------------------------------------------------------------------------


class TurnDTO(_Base):
    """一輪問答。鏡射 InterviewSession.Turn。"""

    question: str = ""
    answer: str = ""
    input_mode: str = "unknown"
    """這段回答怎麼產生的:"voice" / "typed" / "unknown"。

    四個測量的有效性取決於這一欄,見 INPUT_MODE_RULE。
    """

    answer_segments: list[str] = Field(default_factory=list)
    """STT 逐段送出的原始片段。有值時以此為準,answer 僅供顯示。

    【為什麼要顯式帶,而不是靠 answer 用換行接的慣例】
    Android STT 停頓 1–2 秒就自動送出,一次回答必然被切成數段,
    實測每段 69–86 字。段界是實測的語流邊界,是 TextStats 的
    segmentation="stt_segment" 的來源。

    若只靠「answer 用 \n 接」的口頭約定,有人改成用空字串或空白接時,
    邊界資訊會靜默消失,而且要到校準數字對不上才會發現。
    這是同一類問題的第四次:測量的有效性取決於一個沒有被宣告的來源。

    為空時後端退回以 \n 切分 answer。
    """


class ExperienceDTO(_Base):
    """一筆使用者經歷。鏡射 data/remote/ExperienceDtos.kt 的 ExperienceResponse。

    A5 與 questionFeedbacks.better 的事實來源。除 id/title/category 外
    全部給預設值,因為使用者常常只填一半。
    """

    id: str = ""
    title: str = ""
    category: str = ""
    period: str = ""
    role: str = ""
    action: str = ""
    result: str = ""
    learning: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class InterviewContext(_Base):
    """面試設定。鏡射 InterviewConfig.kt。

    注意 InterviewConfig 目前沒有 format 欄位(single/panel 是畫面內的區域變數),
    所以本合約改用頂層 mode 傳遞,前端需補一行把 format 寫進 config。
    """

    round: str = "初試"  # 初試 / 複試 / 主管面
    language: str = "中文"  # 中文 / English
    type: str = "行為"  # 行為 / 技術 / 情境
    difficulty: str = "中等"  # 新手 / 中等 / 困難
    group_interviewers: int = 1  # group 模式:1 位主持 / 3 位 panel
    custom_role: str = ""
    custom_company: str = ""
    custom_seniority: str = "新鮮人"  # 新鮮人 / 1-3年 / 資深
    custom_industry: str = ""
    custom_jd: str = ""


INPUT_MODE_RULE = """input_mode 的用途與規則。

值域:"voice"(裝置端 SpeechRecognizer)/ "typed"(鍵盤)/ "unknown"。

【為什麼需要】
後端永遠只收到文字,分不出這段是講出來的還是打字打的。但有四個測量的
有效性完全取決於這件事:

  filler_count  語音時反映口語習慣;打字時恆為 0,反映的是「打字不會打嗯」
  segmentation  語音時是 unavailable(中文 STT 無標點);打字時是 punctuation,
                但那是鍵盤打的不是講的
  表達流暢度     打字時量到的是打字習慣
  prosody       打字時完全無意義

最嚴重的後果是評分偏誤:打字的答案沒有填充詞、標點乾淨,會拿到「高」流暢度;
認真用講的反而拿低分。在面試模擬器裡這是在獎勵錯誤的行為。

【B 側規則】
input_mode == "typed" 時:
  - 表達流暢度不得以填充詞或標點為依據,改用內容面訊號
  - prosody 一律不產出
  - verdict 措辭不得宣稱評估了口語表現
  - TextStats.filler_reliability 視為 "not_applicable",不採用 A 給的值

input_mode == "unknown" 時比照 "typed" 保守處理。寧可少講一個維度,
也不要用一個量錯東西的分數去指導使用者。

【前端現況】
資訊已經存在但被丟棄。群面兩個呼叫點傳的參數相同:

    line 129  rememberInPageVoice(...) { t -> submitGroup(t, t) }   語音
    line 210  submitGroup(said, said)                               打字

只需在呼叫點各傳一個常數即可。一對一與 panel 目前沒有文字輸入,
恆為 "voice";若之後補上文字備援(裝置不支援辨識時的降級路徑),
這一欄就會同時有兩種值。
"""


class UtteranceDTO(_Base):
    """群面的一則發言。鏡射前端需新增的 groupSays 結構。

    【前端需配合修改】
    目前 InterviewSession.groupSays 是 List<String>,且
    InterviewLiveGroupScreen.submitGroup 只記錄使用者自己的發言,
    AI 同儕的話進了畫面的 messages 但從未寫入 session。

    協作四項有三項需要發言者身分與時序才算得出來,尤其「傾聽與回應」
    要比對前一位發言者說了什麼——只記使用者的話,那份資訊根本不在場。

    需請前端改為記錄所有發言者。messages 已帶 speaker 欄位,改動不大。
    """

    speaker: str = "user"  # "user" 或 persona 顯示名稱
    text: str = ""
    input_mode: str = "unknown"  # "voice" / "typed" / "unknown",僅 user 的發言有意義
    start_ms: int = 0  # 毫秒,無計時資料時給 0,至少保住順序
    end_ms: int = 0  # 毫秒,有值才算得出打斷


class SpeechStats(_Base):
    """語音時間資料。前端目前尚未提供,預留欄位。

    Android SpeechRecognizer 的 onBeginningOfSpeech / onEndOfSpeech 目前是空實作,
    時間資料被丟棄。前端補齊後即可帶入,後端不需改合約。
    每個 list 的索引對齊 turns 的索引。
    """

    lead_in_seconds: list[float] = Field(default_factory=list)  # 開口前延遲
    speaking_seconds: list[float] = Field(default_factory=list)  # 發話總秒數
    pause_counts: list[int] = Field(default_factory=list)  # 句間停頓次數


class Persona(_Base):
    """一個面試場上的角色。

    display_name 是開放字串,不用 Literal。panel 的三位主管、group 的三位
    AI 應徵者名稱都可能隨 persona 調校變動,鎖死會讓調 prompt 變成改合約。
    """

    id: str  # 穩定鍵,後端內部用
    display_name: str  # 前端直接顯示
    role: str = "interviewer"  # interviewer / peer / moderator
    blurb: str = ""  # 一句人設說明,可空


# ---------------------------------------------------------------------------
# 端點一:POST /interviews  —— 開場
# ---------------------------------------------------------------------------


class StartInterviewRequest(_Base):
    mode: InterviewMode
    context: InterviewContext = Field(default_factory=InterviewContext)
    experiences: list[ExperienceDTO] = Field(default_factory=list)


class StartInterviewResponse(_Base):
    """開場回應。

    interrupt_lines 在此一次生成完的理由:GroupDispatcher.interruptLine(index)
    是前端偵測打字停頓時同步呼叫的,那個時機發 HTTP 會卡住 UI。改成開場預生成
    一組台詞、前端按 index 取用,Kotlin 介面簽章完全不變。

    fallback_probes 對應 InterviewProber.probe(fallback: List<String>)。
    目前那個池是前端寫死的,改由後端依 JD 生成會更貼題,呼叫點一樣不用改。
    """

    session_id: str
    mode: InterviewMode
    opening_question: str
    opening_speaker: str = ""  # single 可空;panel/group 帶說話者
    opening_topic: str = ""  # 開場題所屬領域,前端由此開始累積 asked_topics
    personas: list[Persona] = Field(default_factory=list)
    interrupt_lines: list[str] = Field(default_factory=list)  # A3 搶話預生成
    interrupt_cap: int = 0  # 等於 len(interrupt_lines)
    fallback_probes: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)  # 降級說明


# ---------------------------------------------------------------------------
# 端點二:POST /interviews/{session_id}/turns  —— 每輪(A1 / A2 / A3)
# ---------------------------------------------------------------------------


class TurnRequest(_Base):
    """一輪送出。

    question 與 fallback 帶進來的理由:讓後端在 session 尚未落地(資料庫組的
    Interview_Turns 還沒好)時仍能無狀態運作。session 通了之後這兩個欄位
    可以省略,合約不用改。
    """

    answer: str
    follow_up_idx: int = 0
    question: str = ""  # 本輪的題目
    fallback: list[str] = Field(default_factory=list)  # 對齊 A1 簽章
    input_mode: str = "unknown"
    """"voice"(裝置端 STT)/ "typed"(鍵盤)/ "unknown"。見 INPUT_MODE_RULE。"""
    asked_topics: list[str] = Field(default_factory=list)
    """本場已涵蓋的領域,前端累積 TurnResponse.topic 後回傳。

    「追問不得重複已問領域」這條禁則沒有清單就無法執行——只在 prompt 寫
    「不得重複」,模型會自行想像哪些問過了,結果照樣重複。
    session 落地後這個欄位可由後端自行查詢,屆時前端可省略,合約不用改。
    """


class TurnResponse(_Base):
    """一輪回應。三種模式共用同一個形狀。

    speaker 一律開放字串。single 模式回空字串,前端不顯示說話者;
    panel 回「HR 主管 / 技術主管 / 用人主管」;group 回「主考官 / AI-邏輯 /
    AI-強勢 / AI-親切」。這些字串會隨 persona 調校變動,不可鎖 Literal。
    """

    speaker: str = ""
    next_question: str
    reaction: str = ""  # A1.reaction / A2.reaction;group 不使用
    is_follow_up: bool = False  # 這句是追問還是新的主問題
    should_advance: bool = False  # 前端是否該推進到下一主問題
    topic: str = ""
    """next_question 所屬的領域標籤,例如「資料分析」「團隊衝突」。

    由後端標記、前端累積,下一輪回傳 TurnRequest.asked_topics。
    讓前端不必自己判斷題目屬於哪個領域。追問(is_follow_up=True)沿用
    上一題的 topic,不算新領域。
    """
    notices: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 端點三:POST /interviews/{session_id}/report  —— 賽後(A4 + A5)
# ---------------------------------------------------------------------------


class ReportRequest(_Base):
    """報告請求。

    A4 與 A5 合併成一個端點的理由:兩者吃的輸入完全相同(逐字稿 + JD + 履歷),
    拆兩個端點等於同一份資料送兩次、LLM 跑兩趟。前端拿一份 response 餵
    InterviewReportProvider 與 MissingPointsAnalyzer 兩個 provider 即可。
    """

    mode: InterviewMode
    context: InterviewContext = Field(default_factory=InterviewContext)
    experiences: list[ExperienceDTO] = Field(default_factory=list)
    turns: list[TurnDTO] = Field(default_factory=list)
    group_says: list[UtteranceDTO] = Field(default_factory=list)
    """群面逐則發言,含發言者與時序。僅 group 模式使用。

    元素型別由 str 改為 UtteranceDTO 的理由見 UtteranceDTO 說明。
    前端 InterviewSession.groupSays 需同步改型別並記錄所有發言者。"""
    speech_stats: SpeechStats | None = None  # 前端補齊後才有


class ProsodyItem(_Base):
    """鏡射 Kotlin 的 Pair<String, String>。

    Gson 序列化 Pair 會產生 {"first":…,"second":…},不好讀也不好擴充,
    所以合約層改用具名欄位,前端轉成 Pair 只需要一行 map。
    """

    label: str
    value: str


class FaceDimensionDTO(_Base):
    """三大面向之一。鏡射 FaceDimension。"""

    letter: str  # 內 / 構 / 達
    name: str  # 內容 / 結構 / 表達
    score: int = Field(ge=0, le=100)
    verdict: str  # 一句總評
    points: list[str] = Field(default_factory=list)  # 可行動建議
    prosody: list[ProsodyItem] | None = None  # 僅「表達」面向

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> object:
        """LLM 偶爾會回 105 或 -3。夾回範圍,不要噴 500。"""
        if isinstance(v, (int, float)):
            return max(0, min(100, int(v)))
        return v


class SubScoreDTO(_Base):
    """六項細分分數之一。鏡射 SubScore。"""

    name: str
    score: int = Field(ge=0, le=100)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> object:
        if isinstance(v, (int, float)):
            return max(0, min(100, int(v)))
        return v


class QuestionFeedbackDTO(_Base):
    """逐題回顧。鏡射 QuestionFeedback。

    better 有兩種模式,由 ReportResponse.resume_grounded 標示:
      True  —— 有履歷,可以補入履歷中真實存在的量化成果
      False —— 無履歷,只能重組措辭(加 STAR 結構、去冗詞、結論提前),
               嚴禁生成任何履歷中不存在的數字或成果
    """

    question: str
    answer: str
    comment: str = ""  # 教練點評
    better: str = ""  # 更好的講法


class StarPartDTO(_Base):
    """STAR 四段之一。鏡射 StarPart。

    from_answer 必須是逐字稿中真實出現的字串,不可改寫。
    這是 verbatim 引用,需要以 gpt-4o 產出,gpt-4o-mini 做不到跨行原文引用。
    上線前以機械檢查驗證:from_answer 必為 answer 的子字串。
    """

    key: str  # S / T / A / R
    name: str
    present: bool
    from_answer: str = ""  # present 時:從回答中對到的句子
    hint: str = ""  # 缺漏時:該補什麼


class VideoDimDTO(_Base):
    """影像維度。鏡射 VideoDim。

    本期不做。A6 FaceMetricsProvider 是裝置端 MediaPipe,不經過後端。
    此類別存在只為了讓 response 形狀完整,本期恆回空陣列。
    """

    name: str
    score: int = Field(ge=0, le=100)
    hint: str = ""


class CollabDimDTO(_Base):
    """協作維度。鏡射 CollabDim。僅 group 模式產出,其餘模式恆空。"""

    name: str
    score: int = Field(ge=0, le=100)
    hint: str = ""


class MissingPointDTO(_Base):
    """一個漏講加分點。鏡射 MissingPoint(A5)。"""

    point: str  # 該提的亮點(履歷有、本場沒講到)
    why: str  # 為什麼這段對這份 JD 重要


class ReportResponse(_Base):
    """報告回應。一次涵蓋 A4 六個方法與 A5。

    前端對接方式:
      InterviewReportProvider.faceDimensions()   <- face_dimensions
      InterviewReportProvider.subScores()        <- sub_scores
      InterviewReportProvider.questionFeedbacks()<- question_feedbacks
      InterviewReportProvider.starParts()        <- star_parts
      InterviewReportProvider.videoDims()        <- video_dims(本期恆空)
      InterviewReportProvider.collabDims()       <- collab_dims(非 group 恆空)
      InterviewReportProvider.improvements()     <- improvements
      MissingPointsAnalyzer.analyze()            <- missing_points
    """

    mode: InterviewMode
    face_dimensions: list[FaceDimensionDTO] = Field(default_factory=list)
    sub_scores: list[SubScoreDTO] = Field(default_factory=list)
    question_feedbacks: list[QuestionFeedbackDTO] = Field(default_factory=list)
    star_parts: list[StarPartDTO] = Field(default_factory=list)
    video_dims: list[VideoDimDTO] = Field(default_factory=list)
    collab_dims: list[CollabDimDTO] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    missing_points: list[MissingPointDTO] = Field(default_factory=list)

    resume_grounded: bool = False
    """本次是否有履歷可依據。False 時 better 只做措辭重組、
    missing_points 恆為空陣列,並在 notices 說明原因。"""

    notices: list[str] = Field(default_factory=list)
