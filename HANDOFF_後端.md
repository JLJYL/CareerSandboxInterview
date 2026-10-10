# 面試報告的儲存 — schema 對齊(模型組 B → 資料庫組)

看過你們的 `interview_sessions` 設計,整體結構對得上。

**這份以模型現在實際回傳的格式為準。** 你們 schema 裡缺的欄位請照現有格式補,
有衝突的地方我標了 **⚠ 請改**,那幾處不改會讓功能失效或靜默拿到 null。

---

## ⚠ 三處衝突,請後端修改

### ⚠ 1. `groupSays` 的型別 — 不改的話群面兩個維度評不出來

```
你們的 schema   groupSays: [String]              // 只有使用者的發言
請改成          groupSays: [Object]              // 所有發言者
```

實際的物件結構:

```json
{
  "speaker": "AI-邏輯",
  "content": "母數是多少",
  "isUser": false,
  "segments": ["母數是多少"],
  "segmentStartsMs": [3200],
  "inputMode": "voice"
}
```

**為什麼不能只存使用者的發言:**

協作四維裡有兩維需要完整的對話脈絡——

| 維度 | 需要什麼 |
|---|---|
| 傾聽與回應 | 使用者發言 **+ 往前最近一則同儕發言**,成對評估 |
| 協作姿態 | **整段對話**,要看出分歧發生在哪、他怎麼回應 |

只有使用者的發言,那兩維完全評不出來。

`isUser` 也不能省:前端送的 `speaker` 是「你」不是 `"user"`,
而 persona 的顯示名稱會隨調校變動,用名稱判斷身分不安全。

**前端那邊已經改好了**(`GroupUtterance` 有這些欄位),只有 schema 還是舊的。

### ⚠ 2. `starBreakdown` → 請改成 `starParts`

```
你們的 schema   starBreakdown
前端介面        fun starParts(): List<StarPart>
模型回傳        starParts
```

前端已經叫 `starParts`(`InterviewReportData.kt`),模型是照前端命名的。
**三方裡只有 schema 不一致**,改這一個比改另外兩個省事。

名稱不一致的後果是靜默的——欄位對不上會拿到 null,不會報錯。

同時請補 `hint` 欄位:

```json
{ "key": "R", "name": "結果", "present": false,
  "fromAnswer": "", "hint": "缺結果:補上這件事最後怎麼了" }
```

`present` 為 false 時,`hint` 是唯一可行動的內容——**沒存的話使用者看不到該怎麼補**。

### ⚠ 3. `prosody` 的格式

```
你們的 schema   prosody: [ [key, value] ]        // 二元陣列
模型回傳        prosody: [ {label, value} ]      // 物件陣列,或 null
```

請改成物件陣列,跟其他欄位一致。

注意它**可能是 `null` 不是空陣列**——只有「表達」那一維、而且填充詞可測時才有值。
打字輸入或雲端轉錄(填充詞被模型清掉)時是 `null`。

---

## 一、分數:模型目前不產出 `overallScore`

**你們 schema 裡的 `overallScore` 目前拿不到值。**

原因:權重沒有依據。「表達流暢度」跟「協作姿態」誰比較重要,取決於職位、
公司、面試階段——沒有資料可以訂那個權重,硬訂一個就是憑感覺。

模型回的是**三組獨立的分數**,彼此沒有加權也沒有合計:

```
faceDimensions   三維面向(內/構/達),分數取自 subScores
subScores        六項細分,0–100
collabDims       協作四維,0–100,僅群面
```

**這件事要三方一起決定**,因為歷史頁的 `InterviewRecord` 有 `score: Int`,
列表上要顯示一個數字。三種做法:

```
a. 模型加      六項細分取平均,我這邊加,一句話的事
b. 你們算      同樣的公式,存的時候算
c. 不顯示總分  歷史頁改成顯示其他指標
```

如果選 a 或 b,要先講定文案:**那是摘要數字,不是評鑑分數**。
六項裡兩項是確定性公式、四項是模型的印象分,平均之後兩種性質混在一起,
不要寫成「你的面試得分」。

**在決定之前,那個欄位先留著給 null。**

---

## 二、你們 schema 缺的欄位(請照現有格式補)

### `improvements: [String]` — 必須補

「下次可以試試」的建議,最多四條。前端介面有 `fun improvements()`,
是報告六個區塊之一。**不存的話使用者看得到的內容會遺失。**

```json
"improvements": [
  "在專題中加入具體的數據或成果數字。",
  "每段開頭先給出結論,再展開細節。"
]
```

### `resumeGrounded: Boolean` — 建議補

使用者有沒有建立經歷。為 `false` 時 `missingPoints` 必定是空的——
不是「他沒漏講」,是「沒履歷所以算不出來」。

不存的話之後看報告分不出這兩種情況。

### `notices: [String]` — 建議補

系統說明,診斷用。例子:

```
"報告:表達流暢度 82 由系統計算(詞彙型填充詞每百字 3.0)"
"報告:第 3 題的回答被語音辨識切斷,那幾題缺少的段落不代表使用者沒講"
"協作:協作姿態這場討論沒有出現立場分歧,觀察不到"
```

之後查「這份報告的分數為什麼是這樣」,答案通常在這裡。

### `collabDims` 的內容(你們 schema 沒展開)

```json
{ "name": "參與主動性", "score": 76,
  "hint": "開場就先表態,搶到定錨位置",
  "evidence": "我覺得先做客群分析" }
```

固定四個名稱:參與主動性、傾聽與回應、論點建構、協作姿態。

**但陣列可能是空的或少於四筆**,那是正常的:

- 非群面模式 → 恆為空
- 這場沒有出現立場分歧 → 協作姿態不評
- 使用者沒跟同儕互動 → 傾聽與回應不評

`evidence` 是支持這個等第的逐字稿原句,前端不顯示,但存起來有用。

### `turns` 請補四個欄位

```
你們的       turns: [{ turnOrder, speaker, question, answer }]
請補上       inputMode, answerSegments, segmentStartsMs, endedBy
```

| 欄位 | 用途 |
|---|---|
| `inputMode` | `voice` / `typed` / `unknown`。打字的答案沒有填充詞,不能拿去評流暢度 |
| `answerSegments` | 語音的原始分段。有分段時斷句是實測值,沒有只能估算 |
| `segmentStartsMs` | 每段開始的時點,毫秒 |
| `endedBy` | `user` / `timeout` / `unknown`。分得出「講完了」與「錄音被切斷」 |

**不補的話報告可以產出,但之後無法重跑。** 拿舊的 session 重新分析時,
少了這四個欄位結果會跟當初不一樣。

`turnOrder` 與 `speaker` 是你們要的,模型不需要但也不衝突,**兩邊都存**。

---

## 三、直接對得上的(不用動)

| 欄位 | 說明 |
|---|---|
| `faceDimensions: [{letter, name, score, verdict, points}]` | 固定三筆:內、構、達 |
| `subScores: [{name, score}]` | 固定六筆,名稱固定 |
| `questionFeedbacks: [{question, answer, comment, better}]` | 筆數 = 題數 |
| `missingPoints: [{point, why}]` | 最多五筆 |

### 固定值域(可以當 enum 處理)

```
faceDimensions.letter   內 / 構 / 達
subScores.name          內容深度、邏輯清晰度、表達流暢度
                        互動能力、應變能力、自信程度
starParts.key           S / T / A / R
collabDims.name         參與主動性、傾聽與回應、論點建構、協作姿態
```

順序也固定,不會變。

---

## 四、兩處待對齊(不急,但要有共識)

### `type` 的值域

```
你們的      "個人" / "panel" / "團體" / "影像" / "快速"
模型用的    "single" / "panel" / "group"
```

模型只實作三種,影像與快速面試沒有。

在你們那邊做對應(存中文、收英文)或直接用模型的值域都可以,
**但要選一個**,而且前端要知道。

### `videoMetrics` 的結構

```
你們的      videoMetrics: { eyeContact, stability, expression, pace }
前端介面    fun videoDims(): List<VideoDim>    VideoDim(name, score, hint)
模型回傳    videoDims: []                      恆為空
```

影像指標**還沒實作**,模型一律回空陣列。

你們的四個欄位名對應的是 `VideoDim.name` 的值,不是物件的 key。
**建議現在不用建**,之後實作時再對齊。

---

## 五、`sessionId` — 目前由模型產生

你們還沒建 session 表,所以現在的做法是:

```
開場   模型產生 itv_a3f9c2e81b04  →  回給前端
每輪   前端從網址帶回來            →  後端收下,不查也不存
報告   同上
```

模型這邊完全無狀態,那個字串對後端沒有意義——**但對你們有**。

### 存報告時請帶上 `sessionId`

```json
{
  "sessionId": "itv_a3f9c2e81b04",
  "userId": "...",
  "report": { ... },
  "createdAt": "..."
}
```

沒有它的話,同一個使用者練了五場,那五份報告分不出誰是誰。
歷史頁的 `InterviewRecord.id` 也是用它。

### 之後要改的話請先說

你們之後建 `interview_sessions` 時,可能會希望由你們產生 id
(格式跟其他表一致、或用 MongoDB 的 `_id`)。

**那個改動可以做,但要先講。** 模型的開場函式收 `sessionId` 參數——
傳進來就用你們的,不傳才自己產。所以改法是:

```
前端先打你們的端點建 session,拿到 id
再打模型的開場端點,把 id 一起送進來
```

模型這邊不用改程式,但**前端的呼叫順序要調整**,所以是三方的事。

現在不用決定。等你們真的要建表時說一聲即可。

---

## 六、`groupParticipants`

```
你們的      groupParticipants: [{ personaId, name, style }]
模型回的    開場的 personas: [{ id, displayName, role, blurb }]
```

**不在報告裡**,在 `POST /interviews`(開場)的回應中。
前端拿到之後可以轉交給你們。

| 你們的 | 模型的 |
|---|---|
| `personaId` | `id`(`peer_logic`、`peer_assertive`…) |
| `name` | `displayName`(`AI-邏輯`、`AI-強勢`…) |
| `style` | `blurb` |
| — | `role`(`moderator` / `peer`) |

**建議把 `role` 一起存**——之後看報告時要分得出哪幾位是競爭者、
哪幾位是面試官。

---

## 七、處理順序

| 優先 | 項目 |
|---|---|
| ⚠★★★ | `groupSays` 改成物件陣列 |
| ★★ | 補 `improvements` |
| ⚠★ | `starBreakdown` → `starParts`,並補 `hint` |
| ★ | `turns` 補四個欄位 |
| ★ | `overallScore` 三方討論,在那之前留 null |
| ★ | 存報告時帶上 `sessionId`(目前由模型產生) |
| — | 補 `resumeGrounded` / `notices` |
| ⚠— | `prosody` 改成物件陣列 |
| — | `type` 值域、`videoMetrics` 之後再對齊 |

---

## 八、我可以配合的

**範例資料我可以產一份真實的報告 JSON** 給你們建 schema 用,
跑一場真實面試的輸出,所有欄位都有值,比照著這份文件猜快。

**合約在 `app/schemas/interview.py`**,每個欄位都有 docstring
寫明它存在的理由與踩過的坑。有疑問直接看那裡或問我。

**`overallScore` 決定要模型算的話,跟我說一聲就加。**
