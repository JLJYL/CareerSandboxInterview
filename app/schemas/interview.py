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

    segment_starts_ms: list[int] = Field(default_factory=list)
    """每一段開始聆聽的時點,毫秒,以本輪回答開始為原點。

    【用途】
    段界的語意在前端修好截斷之後改變了:原本是「STT 自動送出點」,
    現在是「引擎重啟點」。兩者的間隔性質不同,而黃金集那六段是舊行為錄的,
    所以新格式要重新驗證一次。沒有這一欄就沒得驗。

    【筆數可能比 answer_segments 多 1】
    多的那一筆代表:使用者按下結束時,系統正要開始聽新的一段,但那一段沒有內容。
    這個長度差的意義由 ended_by 表達,見下。

    【前端】
    怡君已完成並合進主 repo。為空時視為無時間資料,
    依賴時間的量一律不可用,由 TextStats.segmentation 反映。
    """

    ended_by: str = "unknown"
    """本輪怎麼結束的:"user"(使用者按下結束)/ "timeout"(引擎逾時)/ "unknown"。

    【用途:判斷這段回答有沒有講完】
    被引擎切斷跟自己講完是兩回事,而下游有三處會因為分不出來而做錯:

        starParts   被切斷導致缺 R 段,報告寫「缺結果,補上這件事最後怎麼了」
                    ——那是責備一個被打斷的人
        missingPoints 他正要講那個亮點就被切了,報告說他漏講
        live 引擎    該追問「你剛剛還沒講完的部分是什麼」,而不是換新題

    第三個最有價值,因為它是即時的——面試進行中就能補救,不用等報告。

    【不是為了消歧義長度差】
    早期版本的理由是「segmentStartsMs 比 answerSegments 多一筆時,
    那筆的意義藏在長度差裡」。那件事用註解就能解決,不值得為它加一個欄位。
    加了沒人讀的欄位,就是這份合約一路在避免的東西。
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
    transcription_engine: str = "api"
    """逐字稿是怎麼產生的:"api"(雲端轉錄)/ "device"(裝置端即時辨識)。

    【為什麼要知道】
    兩種引擎的失真方式相反,而 prompt 要據此改變行為:

        device  會漏字、沒標點、保留詞彙型填充詞
                → 追問時要假設「沒講到可能是被吃掉的」
        api     不漏字、有標點、填充詞被模型清掉
                → 可以相信「沒講到就是真的沒講」,追問才追得下去

    寫成同一段說明會讓模型做錯事:告訴它「填充詞保留」而逐字稿一個都沒有,
    它會推論使用者講話很流暢——那是把「模型清掉了」誤讀成「使用者沒講」。

    【預設 api】
    前端已於 2026-09 改用 MediaRecorder + Whisper,舊的 SpeechRecognizer
    整支移除。預設值跟著現況走,舊資料(黃金測試集那六段)要明確傳 "device"。

    填充詞可不可用不由這一欄決定,那由 TextStats.filler_reliability 決定——
    這一欄影響的是 prompt 的措辭,不是計分。
    """

    language: str = "中文"  # 中文 / English
    type: str = "行為"  # 行為 / 技術 / 情境
    difficulty: str = "中等"  # 新手 / 中等 / 困難
    group_interviewers: int = 1
    """群面的面試官配置:1 位主持 / 3 位主管。

    3 的時候主考官不出現,改由用人主管、技術主管、HR 主管輪流主持。
    對齊前端 InterviewLiveGroupScreen 的 panelRoster。
    """

    group_size: int = 4
    """群面的小組人數,含使用者本人,3–5。決定出場幾位 AI 應徵者。

    【前端目前沒有帶這一欄】
    InterviewSetupGroupScreen 有「小組人數」選項,但選完只存在畫面的區域變數,
    沒有寫進 InterviewConfig。所以後端收到的一律是預設值 4。
    這是前端缺口,列入交接清單。
    """

    group_role: str = "一般應徵者"
    """使用者在群面裡的相對定位:一般應徵者 / 較資深應徵者 / 較資淺應徵者。

    這不是裝飾——「較資深」的說明是「其他人比你新鮮,你會被期待多分享經驗」,
    它會改變 AI 應徵者對使用者的態度。

    【前端目前沒有帶這一欄】
    同 group_size,選完沒有存進 InterviewConfig。
    """
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

    speaker: str = "user"
    """發言者。"user" 或 persona 的顯示名稱。"""

    content: str = ""
    """發言內容。

    【欄位名跟著前端】
    前端的 GroupUtterance 用 content,早期版本這裡叫 text——
    名字不同會讓反序列化拿到空字串,而且不會報錯:
    協作維度會收到一批空發言,分數照樣算得出來,只是全部沒有依據。
    """

    is_user: bool = False
    """這句是不是使用者本人講的。

    協作評分只評使用者。沒有這一欄就得靠 speaker == "user" 判斷,
    而 speaker 是 persona 的顯示名稱、可能隨調校變動——
    用一個會變的欄位去判斷「這是不是本人」不安全。
    """

    segments: list[str] = Field(default_factory=list)
    """STT 逐段送出的原始片段。理由同 TurnDTO.answer_segments。"""

    segment_starts_ms: list[int] = Field(default_factory=list)
    """每段開始聆聽的時點,毫秒。理由同 TurnDTO.segment_starts_ms。"""

    input_mode: str = "unknown"
    """"voice" / "typed" / "unknown"。僅 is_user 為真的發言有意義。

    群面同時有語音與文字輸入(搶話偵測需要打字停頓),所以逐句都要標。
    """

    start_ms: int = 0
    """這句話開始的時點,毫秒。

    跟 segment_starts_ms 的差別:那是段落層級,這是整句層級。
    前端目前只提供 segment_starts_ms,所以這一欄通常是 0——
    需要時取 segment_starts_ms[0]。保留是為了讓打斷偵測有明確的欄位可用。
    """

    end_ms: int = 0
    """這句話結束的時點,毫秒。有值才算得出打斷。"""


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
    mode: InterviewMode | None = None
    """這一場的面試模式。前端在開場時就知道,每輪帶過來即可。

    【為什麼是選填】
    早期版本漏了這一欄,後端只能從 spokenBy 反推——出現過的說話者屬於哪一組
    persona。那個反推在兩種情況下會錯:

        第一輪 spokenBy 還是空的,一律當成 single
        一對一本來就不回 speaker,所以永遠推不出 panel 與 group 的差別
                                (要靠開場的 openingSpeaker 被放進 spokenBy)

    現在改成前端直接帶。留成選填而不是必填,是因為前端還沒改——
    None 時退回反推,前端補上之後反推就不會被用到,合約不用再改一次。

    交接清單:請前端在每次 turns 請求帶上 mode,值同開場時送的那個。
    """

    input_mode: str = "unknown"
    """"voice"(裝置端 STT)/ "typed"(鍵盤)/ "unknown"。見 INPUT_MODE_RULE。"""

    ended_by: str = "unknown"
    """這段回答怎麼結束的:"user" / "timeout" / "unknown"。

    "timeout" 時追問要先把話接完,不要換新題——他還在講就被引擎送出了。
    這是三個消費點裡唯一即時的一個:面試進行中就能補救,不用等報告。
    """

    context: InterviewContext = Field(default_factory=InterviewContext)
    """面試設定。每輪都帶的理由跟 question/askedTopics 一樣:
    session 尚未落地時後端查不到,只能由前端帶。

    難度、群面配置、使用者定位都在這裡——沒有它,每一輪的追問深度與
    出場的 persona 都會退回預設值。"""
    spoken_by: list[str] = Field(default_factory=list)
    """本場已經開口過的說話者,依序累積 TurnResponse.speaker。

    僅 panel 與 group 需要。single 恆為空——只有一位面試官,沒有派發問題。

    【為什麼要帶】
    派發規則有一條「整場不要只有一位主管在講,某位完全沒開口時優先給他」,
    但模型看不到誰開過口,那條規則就永遠不會生效。

    實測:panel 五輪裡 HR 主管講了 4 次、用人主管 0 次;
    group 四輪裡 AI-強勢與 AI-親切完全沒出現。

    這是同一類問題的第三次:規則要求模型知道它拿不到的資訊。
    前兩次是 asked_topics 與已問問題原文。
    """

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

    evidence: str = ""
    """支持這個等第的逐字稿原句。

    【為什麼要這一欄】
    協作是唯一零量化錨點的評分——rubric 明文禁用發言次數、字數、
    時間佔比、首次發言早晚、發言長度,四個維度全靠語意判斷。

    沒有 evidence 就沒有辦法檢查這個等第是根據什麼給的,
    也沒有辦法用 starParts 那種機械檢查驗證它不是編的。

    必須是逐字稿裡真的出現過的字串。找不到對應的句子時留空,
    不要湊一句——那跟 starParts.fromAnswer 是同一條規則。

    前端的 CollabDim 沒有這一欄,所以它不會顯示在畫面上,
    只用於驗收與 notices。"""


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
