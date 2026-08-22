# 面試模組合約(A1–A5)

凍結日:W1 D1。此後修改需前後端雙方同意。

## 一、五個接點如何對應到三個 HTTP 端點

| 接點 | Kotlin 介面 | HTTP 端點 | 說明 |
|---|---|---|---|
| A1 | `InterviewProber` | `POST /interviews/{id}/turns`(mode=single) | 一對一追問 |
| A2 | `PanelDispatcher` | `POST /interviews/{id}/turns`(mode=panel) | 主管三人派發 |
| A3 | `GroupDispatcher` | `POST /interviews/{id}/turns`(mode=group)<br>+ `POST /interviews` 帶搶話台詞 | 群面同儕 |
| A4 | `InterviewReportProvider` | `POST /interviews/{id}/report` | 報告六件 |
| A5 | `MissingPointsAnalyzer` | `POST /interviews/{id}/report` | 漏講點 |
| A6 | `FaceMetricsProvider` | 無 | 裝置端 MediaPipe,不經後端 |

另有一個接點清單未列、但實際不存在的缺口:**主問題生成**。目前開場題與追問池
寫死在 `InterviewLiveIndividualScreen`,`InterviewConfig` 的五個職位脈絡欄位
無人使用。由 `POST /interviews` 補上。

### 為什麼是三個端點而不是五個

A1/A2/A3 三者輸入輸出形狀完全相同,只有 persona 組成與路由層不同。
合併成一個端點加 `mode` 欄位,後端就是「一套 pipeline + 三組 persona 設定」,
不必蓋三次管線。這是三週期程能成立的主要原因。

A4/A5 吃的輸入完全相同(逐字稿 + JD + 履歷)。拆兩個端點等於同一份資料送兩次、
LLM 跑兩趟。前端拿一份 response 餵兩個 provider 即可。

---

## 二、端點規格

### `POST /interviews` —— 開場

| 方向 | 型別 |
|---|---|
| Request | `StartInterviewRequest` |
| Response | `StartInterviewResponse` |

**兩個設計要點:**

`interruptLines` 在開場一次生成完。`GroupDispatcher.interruptLine(index)` 是前端
偵測打字停頓時同步呼叫的,那個時機發 HTTP 會卡住 UI。改成開場預生成一組台詞、
前端按 index 取用,Kotlin 介面簽章完全不變。

`fallbackProbes` 對應 `InterviewProber.probe(fallback: List<String>)`。
目前那個池是前端寫死的,改由後端依 JD 生成會更貼題,呼叫點一樣不用改。

### `POST /interviews/{sessionId}/turns` —— 每輪(A1/A2/A3)

| 方向 | 型別 |
|---|---|
| Request | `TurnRequest` |
| Response | `TurnResponse` |

`speaker` 依 mode 填入:

| mode | speaker | 對應 |
|---|---|---|
| single | 空字串 | 前端不顯示說話者 |
| panel | HR 主管 / 技術主管 / 用人主管 | `Pair.first` |
| group | 主考官 / AI-邏輯 / AI-強勢 / AI-親切 | `Pair.first` |

`TurnRequest` 帶 `question`、`fallback`、`askedTopics` 的理由:讓後端在
session 尚未落地(資料庫組的 `Interview_Turns` 還沒好)時仍能無狀態運作。
session 通了之後這三個欄位可以省略,合約不用改。

**已問領域的往返**:後端在 `TurnResponse.topic` 標記本題所屬領域,前端累積後
於下一輪回傳 `TurnRequest.askedTopics`。開場題的領域由
`StartInterviewResponse.openingTopic` 提供,前端由此開始累積。

這個往返存在的理由是「追問不得重複已問領域」這條禁則。只在 prompt 寫
「不得重複」而不給清單,模型會自行想像哪些問過了,結果照樣重複。
由後端標記領域而非讓前端自行判斷,是因為前端沒有題目的語意資訊。

追問(`isFollowUp=true`)沿用上一題的 `topic`,不計為新領域。

### `POST /interviews/{sessionId}/report` —— 賽後(A4+A5)

| 方向 | 型別 |
|---|---|
| Request | `ReportRequest` |
| Response | `ReportResponse` |

前端對接:

```
InterviewReportProvider.faceDimensions()    <- faceDimensions
InterviewReportProvider.subScores()         <- subScores
InterviewReportProvider.questionFeedbacks() <- questionFeedbacks
InterviewReportProvider.starParts()         <- starParts
InterviewReportProvider.videoDims()         <- videoDims(本期恆空)
InterviewReportProvider.collabDims()        <- collabDims(非 group 恆空)
InterviewReportProvider.improvements()      <- improvements
MissingPointsAnalyzer.analyze()             <- missingPoints
```

---

## 三、三條鐵則

### 鐵則一:request 嚴格、response 寬鬆

`mode` 由前端路由決定,值域封閉,用 `Literal` 嚴格驗證。

`speaker`、`subScores.name`、`collabDims.name` 有 LLM 產出成分,一律用開放
字串,絕不用 `Literal`。

理由是 `CareerCategory` 事故:把有 LLM 產出成分的欄位鎖成 `Literal`,
會在模型回「邏輯清楚度」(少一個字)時直接噴 500,而不是回一個略微不完美的
結果。值域靠 `interview_repair.py` 的機械修復維持,不靠型別系統。

### 鐵則二:固定值域靠機械修復,不靠 LLM 自律

| 欄位 | 固定值 |
|---|---|
| `faceDimensions.letter` | 內 / 構 / 達 |
| `subScores.name` | 內容深度、邏輯清晰度、表達流暢度、互動能力、應變能力、自信程度 |
| `starParts.key` | S / T / A / R |
| `collabDims.name` | 參與主動性、傾聽與回應、論點建構、協作姿態 |

`repair_report()` 在回傳前執行:缺的補上、多的丟掉、順序歸位、名稱近義變體
依位置歸位。每次修復都寫進 `notices`,問題看得見但不讓使用者看到 500。

### 鐵則三:沒有履歷就不得補事實

`resumeGrounded` 標示本次是否有履歷可依據。

| 值 | `better` 行為 | `missingPoints` |
|---|---|---|
| `true` | 可補入履歷中真實存在的量化成果 | 正常產出 |
| `false` | 只能重組措辭(加 STAR 結構、去冗詞、結論提前) | 恆為空陣列 |

面試教練建議使用者去講一個他其實沒有的成果,比履歷造假更嚴重。
`repair_report()` 在 `resumeGrounded=false` 時強制清空 `missingPoints`,
這是防捏造的最後一道。

`starParts.fromAnswer` 必須是逐字稿中真實出現的字串。以
`verify_star_verbatim()` 機械檢查,**不做自動修復**——引用不存在的原文是嚴重
錯誤,應該讓測試紅燈。此欄位需以 `gpt-4o` 產出,`gpt-4o-mini` 做不到跨行原文引用。

---

## 四、待前端配合的三件事

| # | 項目 | 影響 |
|---|---|---|
| 1 | 五個介面抽出可切換的 Provider 插座 | 擋所有面試端點的驗收 |
| 2 | `format` 寫進 `InterviewConfig` | 擋 panel 模式(後端分不出 single/panel) |
| 3 | 一段真實語音逐字稿樣本 | 擋斷句處理決策,W1 D3 前需要 |

目前畫面是直接寫死呼叫 Mock 單例:

```
InterviewLivePanelScreen.kt:114   MockPanelDispatcher.dispatch(...)
InterviewReportScreen.kt:626      MockMissingPointsAnalyzer.analyze()
InterviewReportScreen.kt:194      MockInterviewReportProvider.faceDimensions()
```

`InterviewConfig` 有 `lastWasVideo` 與 `lastWasGroup`,但沒有 `format`
(single/panel 是 `InterviewSetupScreen` 內的區域變數,用完即棄)。

---

## 五、待確認事項

| # | 事項 | 對象 |
|---|---|---|
| 1 | JSON 命名是否為 camelCase(需與 B1/B2/C1 既有端點一致) | 自查 |
| 2 | `ApiClient.BASE_URL` 的 8000 埠是打模型服務還是資料庫組 | 怡君 |
| 3 | `GET /experiences` 後端是否已實作 | 資料庫組 |
| 4 | session 狀態存放位置(`Interview_Sessions` / `Interview_Turns`) | 資料庫組 |
| 5 | prosody 時間資料由前端提供的時程 | 怡君 |

第 1 項若不一致,把 `_Base` 的 `alias_generator` 拔掉即可,只需改一個地方。

### 待前端配合(D1 異議定案後新增)

`ReportRequest.groupSays` 的元素型別由 `String` 改為物件:

```kotlin
// 現況:InterviewSession.groupSays 是 List<String>,且只記使用者自己的發言
fun submitGroup(visible: String, analyzed: String) {
    InterviewSession.recordGroupSay(visible)   // AI 同儕的話只進 messages
    ...
}

// 需改為記錄所有發言者
data class Utterance(val speaker: String, val text: String,
                     val startMs: Long = 0, val endMs: Long = 0)
```

理由:協作四項有三項需要發言者身分與時序。「傾聽與回應」要比對前一位
發言者說了什麼,只記使用者的話,那份資訊根本不在場。
`messages` 已帶 speaker 欄位,改動不大。W3 D3 群面接線前必須完成。

### 輸入方式必須宣告(inputMode)

後端只收到文字,分不出這段是講出來的還是打字打的。但有四個測量的有效性
完全取決於這件事:`fillerCount`、`segmentation`、表達流暢度、`prosody`。

最嚴重的後果是評分偏誤:打字的答案沒有填充詞、標點乾淨,會拿到「高」流暢度;
認真用講的反而拿低分。在面試模擬器裡這是在獎勵錯誤的行為。

已在三處加入 `inputMode`("voice" / "typed" / "unknown"):
`TurnRequest`、`TurnDTO`、`UtteranceDTO`。預設是 `"unknown"`,不是 `"voice"`——
前端未宣告時比照打字保守處理。

**前端配合**:資訊已經存在但被丟棄,群面兩個呼叫點傳的參數相同。

```kotlin
// InterviewLiveGroupScreen.kt
line 129  rememberInPageVoice(...) { t -> submitGroup(t, t) }   // 語音
line 210  submitGroup(said, said)                              // 打字
```

只需在呼叫點各傳一個常數。一對一與 panel 目前沒有文字輸入,恆為 "voice";
補上文字備援之後就會同時有兩種值。

### 逐字稿分段必須保留(answerSegments)

Android STT 停頓 1–2 秒就自動送出,一次回答必然被切成數段,實測每段 69–86 字。

**前端目前只收第一段。** `InterviewLiveIndividualScreen.kt:173` 與
`InterviewLivePanelScreen.kt:110` 的守衛是 `answer.isNotBlank()`,
而 `InPageVoice.onResults` 之後沒有重啟聆聽。實測損失 75–87%,
且技術詞幾乎全部落在第 2 段以後(第 1 段是「考官好我是誰、哪間學校」的開場白)。

三處要改:

```kotlin
// 1. onResults 累加片段,使用者仍在錄音就重啟聆聽
// 2. 守衛從「已有答案」改成「已按下送出」
// 3. 面試官不可在第一段之後就開始追問(目前 delay(1500) 就回應了)
```

送到後端時以 `answerSegments` 帶原始片段,`answer` 用 `\n` 接起來供顯示。
段界是 `TextStats.segmentation = "stt_segment"` 的來源,用空字串接會靜默遺失。

### 段界時點與結束方式(W2 增補)

`TurnDTO` 加了兩格:

| 欄位 | 用途 |
|---|---|
| `segmentStartsMs` | 每段開始聆聽的時點(毫秒),前端已完成 |
| `endedBy` | `"user"` / `"timeout"` / `"unknown"` |

前端修好截斷之後,段界的語意從「STT 自動送出點」變成「引擎重啟點」,
兩者的間隔性質不同。黃金集那六段是舊行為錄的,新格式要重新驗證一次。

`segmentStartsMs` 的筆數可能比 `answerSegments` 多 1——那筆代表使用者按下結束時
系統正要開始聽新的一段。這個意義由 `endedBy` 表達,不要藏在長度差裡:
藏在長度差裡的意義會消失,任何人 `zip()` 起來就安靜砍掉多的那筆。

### 協作訊號的 bg_ 前綴(W2 增補)

發言量指標放進 `CollabSignal.signals` 時必須加 `bg_` 前綴。
只在註解寫「不得當作依據」的保護太弱,前綴讓這條界線可以被測試檢驗。

實測佐證:`first_speak_position` 原本用總則數當分母,結果同樣在第 2 則開口、
後面多講 5 次,位置就從 1.00 變成 0.17——發言量從後門混進了那個訊號。

### STT 引擎(已定案,無選擇題)

前端使用 Android 裝置端 `SpeechRecognizer`(`ui/components/InPageVoice.kt`,
`languageTag = "zh-TW"`),音訊不出手機,後端永遠只收到文字。

全 repo 無任何 whisper、雲端 STT、音訊上傳的跡象。不需要也不可能在
Python 端安裝對應引擎——那是 Android 系統 API。

**黃金測試集的逐字稿必須是構造的,不可用錄音產生。** 測試集要求
`expected_missing` 精確,真人錄音內容不受控,答案卡不成立。
真實樣本的角色是格式規格,不是資料來源。

需從真實樣本讀出的特性:標點(預期無)、數字呈現方式、英文專有名詞轉寫
(SQL / Sql / 音譯)、填充詞是否被吃掉、中英夾雜的空格處理。
第三項對表面掃描命中率影響最大。

### 已知缺口:認證

本服務與 CareerSandboxModule 獨立部署。前端 `ApiClient` 的攔截器會在每個請求
附上 `Authorization: Bearer <token>`,但**本服務目前不驗證這個 token**。

後果:面試端點對外無保護。校內展示可接受,正式上線前必須補上。

不在三週範圍內處理的理由:驗證 token 需要與後端組共用 JWT 密鑰,等於重新建立
跨服務耦合,成本高於三週能負擔的範圍。此處明文記錄,避免展示時被問到答不出來,
也避免之後沒有人記得補。

---

## 六、本期不做

| 項目 | 原因 |
|---|---|
| A6 影像指標 | 裝置端 MediaPipe,不經後端 |
| `videoDims` | 依賴 A6 |
| prosody 的語速/停頓/語調/開口前思考 | `SpeechRecognizer` 的時間 callback 目前是空實作 |
| 快速面試端點 | `InterviewQuickScreen` 題庫寫死,不呼叫任何接點 |
| 反問環節 | 選項與回答寫死在畫面 |

`prosody` 本期只產出「填充詞次數」,純文字統計可得。
語調需要基頻,`SpeechRecognizer` 只給音量(`rmsdB`),建議永久砍掉或改用
音量變異數當代理指標並在 UI 誠實標示。
