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

      "measured"        計數反映使用者的口語習慣,但只涵蓋「詞彙型」填充詞

    實測 Android 會全數移除非詞彙填充音(「嗯」5→0、「呃」5→0,部分轉成「而」),
    詞彙型完整保留(「那個」2→2、「就是」7→8、「然後」1→3)。
    所以 "measured" 的正確讀法是「詞彙型填充詞已量到」,不是「口語習慣已完整量到」。
    B 側措辭規則:不得宣稱涵蓋全部口語習慣。

    鑑別力實測(每百字):正常發揮 3.0 / 多講故事 6.4 / 刻意講亂 9.8——
    單調且間距明顯,目前整個模組唯一乾淨的量化訊號。
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
      "stt_segment"       無標點,但邊界來自 STT 自動送出點,是實測值
      "discourse_marker"  無標點,以語氣詞估算,是估算值
      "unavailable"       無法斷句,兩個欄位皆為 0

    "stt_segment" 是正式環境的常態,前提是前端把多段以換行接起來
    (見 TurnDTO.answer_segments)。它量的不是句長,是「停頓前講了多長」,
    也就是語流連續性。實測 69–86 字。
    B 側措辭規則:不得講成「句子長度」。

    B 側規則:segmentation 不是 "punctuation" 時,prompt 不得把平均句長
    講得像量出來的。Android 中文 STT 通常不輸出標點,正式環境的常態會是
    "discourse_marker",兩態設計(有值 / 0)會讓估算值無處安放。
    """

    connector_rate: float = 0.0
    """每百字的**因果**連接詞數（因為/所以/因此/由於/導致/才能/為了⋯）。

    ★ 只計因果一類。原本規劃三類,實測六段逐字稿(2,477 字)後兩類不成立:
        轉折(但是/不過/然而/雖然) 出現 **0 次**——不是稀少,是完全沒有。
        序列 扣掉「然後」只剩 10 次,而「然後」39 次且已在 FILLERS 裡。
             同時算連接詞與填充詞的話,講話越不流暢的人邏輯分數越高,
             兩個指標會反向打架。因果類不含「然後」,所以不會發生。
      做一個永遠是 0 的欄位比沒有欄位更容易被誤用。

    ★★ **錨點不可靠,建議當背景資訊而非公式輸入。** 實測六段每百字:
          0.26 Helen 正常   0.00 Helen 講滿    0.00 Victor 正常
          1.01 Victor 講故事 1.47 Ken 正常     0.72 Ken 刻意講亂
      問題在兩段同樣「正常發揮」卻是 0.00 與 1.47,橫跨整個量程——
      那個差異來自個人講話習慣,不是邏輯清晰度。
      填充詞那組是 3.0/6.4/9.8 單調對應錄音指示,這組不是。
    """

    connector_detail: dict[str, int] = field(default_factory=dict)
    """分類計數。目前只有 {"causal": n}，見上。"""


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


DEMONSTRATION_THRESHOLD = """展演的認定門檻(標記與 why 否決共用)。

一個技能算不算「講到了」,分三種情況:

  指名        原字出現。「我用 Angular 寫的」                    → 算
  展演        描述具體行為,且該行為不做這件事就做不到。
              「我排了每週進度表,有人拖延就設 deadline」= 專案時間控管  → 算
  自我宣稱    只有形容或評價,沒有可查證的行為。
              「我很注重時程管理」                              → 不算
  只能推論    要靠常識補完才成立。「我在餐廳打工過」→ 抗壓性        → 不算

門檻線在「具體行為」與「自我宣稱」之間,不在「有沒有講到技能名稱」。

【為什麼要兩邊共用同一個定義】
黃金集用它標 said,B 的 why 否決用它判斷要不要丟掉候選。
兩邊用不同的門檻,評測結果就無法解釋——測試集說「這條算講到」,
線上卻報成漏講,而數字看起來只是準確率低,查不出是定義不一致。
"""


COLLAB_PROHIBITED_INDICATORS = """協作評分明確禁用的指標。

【禁用】發言次數、發言總字數、說話時間佔比,以及任何以「量」為主的指標。

理由:babble 假說。小組討論裡講最多話的人會被認為是領導者,
與講的內容好不好無關;MacLaren et al. (2020) 在控制智力、人格、性別之後
該效應依然成立,同一份研究並發現男性平均只因性別就多得一票,
且該偏誤與任何可觀察的參與品質指標都無關聯。

我們是回饋產品。使用者看到「參與主動性偏低」就會去多講話——
用發言量計分等於在訓練他 babble,並把一個已知的性別偏誤寫進評分公式。

發言量指標**一律不可出現**,連當背景資訊都不行。

早期版本允許加 `bg_` 前綴當背景資訊。rubric 到手後這個折衷被推翻:
禁令是「一律不可作為評分依據」,而且 babble 假說有性別偏誤
(MacLaren 2020),那是「不可用於任何選拔」的等級。

放到 LLM 面前就是風險,前綴擋不住——這個專案反覆證實
prompt 裡的數字會被錨定。

【為什麼要前綴,不能只寫在註解裡】
只在 docstring 寫「可當背景資料但不得單獨決定等第」,那個保護很弱——
發言次數是那堆數字裡最直觀的一個,LLM 會錨定上去。
前綴讓 B 的 prompt 可以機械地把 `bg_` 開頭的放進「背景」而不是「依據」,
而且這條界線可以被測試檢驗。

實測佐證:A 側原本把 first_speak_position 的分母寫成總則數,
結果同樣在第 2 則開口、後面多講 5 次,位置就從 1.00 變成 0.17,
看起來「開口早很多」——發言量從後門混進了那個訊號。
測試抓到了,分母改成「別人給了幾次機會」。

**禁令要能被測試檢驗才是真的禁令。**

BARS 錨點的文字一律不得以次數描述等第。

【禁用】打斷次數。
群面畫面在 AI 發言時以 isTyping 擋住輸入,使用者實際上打斷不了,
這個指標結構上恆為 0。

【禁用】任何需要影像、音訊、語速、語調的指標。
我們只有純文字逐字稿。

BARS 錨點本體不放在這裡,等後端組的文獻整理回來後放
app/prompts/collab_rubric.py。
"""


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
    """發言者的顯示名稱。

    **不要用它判斷是不是使用者本人。** 前端實際送的是「你」不是「user」
    (InterviewLiveGroupScreen 的 recordGroupUtterance(speaker = "你")),
    而且 persona 的顯示名稱會隨調校變動。用 is_user。
    """

    text: str

    is_interviewer: bool = False
    """這句是不是面試官(主持人)講的。

    三種身分:使用者(is_user)、面試官(is_interviewer)、同儕競爭者(兩者皆否)。

    【為什麼要分開面試官與同儕】
    主考官宣布題目不是在提論點。把它拿去配對「傾聽與回應」,
    使用者的開場發言會被評成 level 1(完全沒有回應前一位)——
    那不是傾聽失敗,那是開場。

    「參與主動性」同理:使用者在主考官出題之後第一個開口,
    行為上是起頭,不是接續。

    【值從哪裡來】
    routes.py 依 speaker 查 personas_for() 的 role 欄位推導,不需要前端送。
    那份 persona 註冊表就是定義顯示名稱的地方——改名會同時改到兩邊,
    所以這不是「用會變的字串判斷身分」,是查它自己的定義。

    查不到的名稱視為同儕(is_interviewer=False)。那是保守的方向:
    誤判成同儕只是回到加這一欄之前的行為,誤判成面試官會讓
    真正的同儕發言從配對裡消失。
    """

    is_user: bool = False
    """這句是不是使用者本人講的。

    【為什麼不能靠 speaker_id 推】
    前端送的是「你」,早期版本的切片邏輯寫 speaker_id == "user",
    照前端實際的 payload 會讓四個維度全部切錯——
    使用者的發言一則都取不到,而 AI 的發言全部被當成使用者的。

    而且不會報錯:切片是空的,LLM 收到空切片回 level 0,
    報告顯示「這次沒有可觀察的內容」。看起來像正常降級,實際上是切錯了。
    """

    start_ms: int = 0
    """毫秒。沒有計時資料時全部給 0,至少保住陣列順序。"""

    end_ms: int = 0
    """毫秒。有值才算得出打斷。"""


@dataclass(frozen=True)
class CollabSlice:
    """一個協作維度對應的逐字稿切片。僅 group 模式使用。

    【這裡沒有數字,原因是 rubric 到了之後才看清楚的】
    原設計是 A 交四組可觀察值(發言次數、首次發言時機、連接詞密度、打斷次數),
    B 的 LLM 對照 BARS 指派等第。rubric 到手後發現那四項有三項被明文禁用:

        發言次數、字數、時間佔比、**首次發言的絕對早晚** —— babble 假說,
                                    且該效應有性別偏誤(MacLaren 2020)
        發言長度、長度變異             —— 論點建構明訂不看長度
        打斷次數                       —— 介面擋住輸入,結構上恆為 0

    剩下的線索全部是語意判斷:「是否推動討論往可行方向收斂」
    「是否正確指涉特定他人講的內容」「面對分歧是先肯定再轉折還是直接否定」。
    那些確定性規則做不到,照切線原則本來就該由 LLM 判斷。

    所以 A 側的價值在**切片**不在**打分**:把逐字稿依維度切成四份,
    每份只含該維度需要看的內容。切片是確定性的、可測試的、跟 LLM 無關。

    【為什麼連 causal_connector_count 都不留】
    它是 rubric 論點建構三條線索中唯一機械可判的,但 A 實測過它沒有鑑別力:
    兩段同樣「正常發揮」的錄音,每百字 0.00 vs 1.50,橫跨整個量程。

    而且這個專案反覆證實 prompt 裡的數字會被 LLM 錨定——
    表達流暢度給區間就輸出邊界值、內容深度十格有七格是 68。
    用一個弱數字去壓兩個強的語意線索,比完全不給更糟。

    【為什麼不是「signals 可以為空」】
    留一個永遠是空 dict 的欄位,等於宣告一件不存在的事。
    改型別才誠實。
    """

    name: str
    """四個固定名稱之一,見 COLLAB_DIM_NAMES。"""

    excerpts: list[str] = field(default_factory=list)
    """這個維度要看的逐字稿片段,依原順序。

    每個維度看的東西不同(rubric 第一節):
        參與主動性  使用者的每一則發言,帶「他人之後/自己起頭」標記
        傾聽與回應  使用者發言 ＋ 緊鄰的前一則他人發言(成對)
        論點建構    使用者的每一則發言,單則獨立(rubric 明訂看「一段發言內部」)
        協作姿態    整段討論(立場衝突可能出現在任何地方,切片會漏掉)

    綁定不同切片是壓 halo 的關鍵:rubric 第八節第 2 點實測,
    一次評四維的維度間相關 r̄ ≈ .86–.92(幾乎分不開),
    分開評掉到 .26–.35(接近人類評審的 .34)。
    """

    note: str = ""
    """這個切片的產生方式與已知限制,寫給 B 放進 prompt 或 notices。

    例:「協作姿態未做切片,整段給出——立場衝突的偵測是語意判斷,
    寬鬆初篩會漏掉沒有標記詞的分歧,也會誤收單純的轉折。」
    """


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

    def mentioned_skills(
        self,
        transcript: str,
        candidates: set[str] | None = None,
    ) -> set[str]:
        """逐字稿 → 使用者實際講出來的技能集合。

        回傳 skill_id(帶 sk: / skm: 前綴),不是顯示字串。
        這是三個集合 join 的鍵,顯示字串走 GapCandidate.display。

        【candidates 的用途:W2 的履歷條件式模糊匹配】
        None 時掃全詞彙表(W1 行為,向後相容)。
        給定時只在這個集合內比對。

        存在理由:實測 STT 對英文技術詞的錯誤不是穩態的——Angular 三次分別
        轉成 Android、整段消失、Andrew;Git 是 gate、gats、Gate。靜態別名表
        補了一種下次來另一種,走不通。

        但把搜尋範圍限縮到「這個人履歷上的 8 個技能」之後,問題就變窄了:
        掃全表時「加巴screen」什麼都不是,已知候選有 JavaScript 時它是強候選。
        誤傷面積小,而且不需要預先知道會錯成什麼——這是唯一能對付非穩態
        錯誤的方法。

        呼叫端(GapComputer.compute)以履歷技能集當 candidates。
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
class CollabObserver(Protocol):
    """群面協作維度的逐字稿切片。成員 A 負責。

    原本叫 CollabScorer、後來改成回可觀察值,現在只回切片——
    每一次改名都是因為它做的事變小了,而每一次變小都是因為
    「這件事確定性規則做不到」被看清楚。

    最終的形狀:A 決定「看哪一段」,B 的 LLM 決定「這一段算幾級」。
    """

    def observe(self, utterances: list[Utterance]) -> list[CollabSlice]:
        """回傳四份切片,順序對齊 COLLAB_DIM_NAMES。

        【純機械操作,沒有語意判斷】
        前三個維度的切法是確定的:過濾 is_user、成對取前一則、單則獨立。
        第四個(協作姿態)不切,整段給出——立場衝突的偵測是語意判斷。

        【已知限制要寫進 note】
        沒有 speaker 身分時(前端尚未提供)「傾聽與回應」切不出成對片段,
        那時回空 excerpts 並在 note 說明。空切片不是失敗,是誠實的降級——
        B 收到空的會在報告裡標明該維度不可用,而不是給一個沒有依據的分數。
        """
        ...


# ---------------------------------------------------------------------------
# Fake 實作 —— 成員 B 在 A 交件前的替身
# ---------------------------------------------------------------------------


class FakeTranscriptAnalyzer:
    """固定回傳。只為了讓 B 的 pipeline 跑得起來。"""

    def mentioned_skills(
        self, transcript: str, candidates: set[str] | None = None
    ) -> set[str]:
        found = set()
        for kw, sid in (("SQL", "sk:sql"), ("Python", "sk:python"), ("Excel", "sk:excel")):
            if kw.lower() in transcript.lower():
                found.add(sid)
        return found if candidates is None else (found & candidates)

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


class FakeCollabObserver:
    """固定回傳四份切片,順序正確。

    【為什麼 Fake 要示範降級】
    第二份(傾聽與回應)在輸入只有使用者發言時回空 excerpts。
    B 的「收到空切片就標明該維度不可用」那條路徑必須在 W1 就測得到,
    不能等 A 交件才發現沒處理。
    """

    def observe(self, utterances: list[Utterance]) -> list[CollabSlice]:
        mine = [u for u in utterances if u.is_user]
        # 只跟同儕配對。面試官宣布題目不是在提論點,
        # 拿它配對會讓使用者的開場發言被評成「完全沒有回應前一位」。
        peers = [u for u in utterances if not u.is_user and not u.is_interviewer]

        # 傾聽與回應:使用者發言 ＋ 最近一則**同儕**發言,成對。
        # 用浮動視窗不是嚴格相鄰:使用者連續講兩則,兩則都配對同一位同儕。
        pairs: list[str] = []
        seen_peer: Utterance | None = None
        first_user_after_peer = False
        for u in utterances:
            if u.is_user:
                if seen_peer is not None:
                    pairs.append(
                        f"他人({seen_peer.speaker_id})「{seen_peer.text}」\n"
                        f"你「{u.text}」"
                    )
                    first_user_after_peer = True
            elif not u.is_interviewer:
                seen_peer = u

        # 參與主動性:接續/起頭同樣看最近一則同儕,不是任何非使用者。
        marks: list[str] = []
        seen_peer = None
        for u in utterances:
            if u.is_user:
                tag = "接續他人之後" if seen_peer is not None else "自己起頭"
                marks.append(f"[{tag}]「{u.text}」")
            elif not u.is_interviewer:
                seen_peer = u

        return [
            CollabSlice(
                "參與主動性",
                marks,
                "方括號是接續/起頭的標記,「」裡面才是逐字稿原文,引用時只取「」裡面的內容。"
                "標記只是脈絡,不是評分依據——接續也可以是主動推進。"
                "面試官出題不算「他人」,使用者在出題後第一個開口仍是起頭。",
            ),
            CollabSlice(
                "傾聽與回應",
                pairs,
                ""
                if pairs
                else (
                    "沒有同儕發言,無法配對,此維度不可評"
                    if not peers
                    else "有同儕發言但使用者沒有接在後面,此維度不可評"
                ),
            ),
            CollabSlice(
                "論點建構",
                [f"「{u.text}」" for u in mine],
                "單則獨立,rubric 明訂看一段發言內部的理由與結論結構。「」裡面是原文。",
            ),
            CollabSlice(
                "協作姿態",
                [
                    f"[{'你' if u.is_user else u.speaker_id}]「{u.text}」"
                    for u in utterances
                ],
                "未切片,整段給出——立場衝突可能出現在任何地方,切片會漏掉。"
                "方括號是講者,「」裡面才是原文,引用時只取「」裡面的內容。",
            ),
        ]
