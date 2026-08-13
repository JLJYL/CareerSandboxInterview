# D1 合約異議定案（B → A）

回覆日：W1 D1。三條阻斷級全數採納，三條非阻斷採納兩條半。
合約已改，`pytest` 57 passed + 4 skipped。

---

## 阻斷級

### 一、`jd: str` → `JDInput`　採納，並補一個你沒提到的前置步驟

你的數字我接受。但我核對前端後發現問題比你量到的更深一層：

**面試的 JD 不是型錄職缺，是使用者手貼的自由文字，而且是選填欄位。**

```kotlin
// InterviewSetupScreen.kt:386
CtxField("貼上 JD(選填,越完整題目越準)", jd, onJd, ...)
```

所以正式環境的 `required_skills` 恆為空，甚至整個 JD 可能是空字串。
只把型別換成 `JDInput` 不解決問題，只是把 0% 涵蓋率藏在一個型別漂亮、
但正式環境永遠空著的欄位後面——你會在一個生產環境不存在的分佈上校準。

真正的缺口是交接文件裡 A5 的第一步「從 JD 抽出重視的能力與關鍵字」在我的
合約裡被弄丟了。我把生字串丟給你，等於要你從散文反推需求，那一步本來是
LLM 抽取，是我的工作。

**定案的結構：**

```
使用者手貼的散文 JD
      ↓  B 的 LLM 抽取器（新增，W1 D3–D5）
JDInput.required_skills, source="extracted"
      ↓  A 的 GapComputer
漏講候選

型錄職缺（黃金測試集）
      ↓  直接取 requiredSkills
JDInput.required_skills, source="catalog"
```

兩條路交給你的形狀相同，校準才轉移得過去。`source` 欄位讓你在評測報表裡
把兩種來源分開統計——這兩個分佈的命中率一定不同，混在一起會看不出真相。

**順帶給你一個現成的評測集**：你手上那 1,010 筆同時有 `requiredSkills` 與
`description` 的 JD，正好是我的抽取器的標準答案卡——輸入用 description、
答案用 requiredSkills，召回率可以直接量。我 W1 D5 會用它出一份數字給你，
你就知道 `source="extracted"` 這條路的實際涵蓋率是多少，不用猜。

`required_skills` 的順序我在 docstring 寫明不得重排，`JD_POSITION_DECAY` 受保護。

### 二、`skill_id` 與 `display` 分離　完全採納

你對，三處指向兩種答案是我的疏漏。已定案：

- `mentioned_skills` 回 `skill_id`（帶 `sk:` / `skm:` 前綴）
- `GapCandidate.skill_id` 是 join 鍵，`GapCandidate.display` 是 `name_zh(skill_id)`
- `FakeTranscriptAnalyzer` 已改回 `{"sk:sql", "sk:python"}`，不再回顯示字串

新增兩條測試：`mentioned_skills` 的回傳一律要有命名空間前綴、
`GapCandidate.skill_id` 與之同一命名空間。差集算錯的話這兩條會先紅燈。

### 三、`CollabScorer` 改吃 `Utterance`　採納，而且這是前端缺口

你說「資訊不在場」——我核對後確認比你說的更嚴重。不是型別問題，是資料
根本沒被記錄：

```kotlin
// InterviewLiveGroupScreen.kt:107
fun submitGroup(visible: String, analyzed: String) {
    InterviewSession.recordGroupSay(visible)     // 只有使用者的話
    messages.add(ChatMessage(..., isUser = true))
    val (rawWho, line) = MockGroupDispatcher.dispatch(...)  // AI 的話只進 messages
}
```

AI 同儕的發言在畫面的 `messages` 裡（已帶 `speaker` 欄位），但從未寫入
session。所以即使你的簽章改了，輸入裡還是不會有前一位發言者的內容。

已採納你的 `Utterance`，並同步改了 HTTP 合約：
`ReportRequest.group_says` 的元素型別由 `str` 改為 `UtteranceDTO`。
前端需求已開給怡君，`messages` 已有 speaker，改動不大。

W3 D3 群面接線前這件事必須完成，我會盯。

---

## 非阻斷

### 四、`resume_experience_ids`　採納，並加一條 prompt 鐵則

改成複數、可為空，已定案。

但只改型別不夠。回去看 Kotlin 的定義：

```kotlin
val why: String,    // 為什麼這段對這份 JD 重要
```

`why` 是 **JD 側的重要性論證**，不是履歷側的敘事。所以我加了一條硬規則
（`WHY_MUST_BE_JD_SIDE`，已寫進 protocols 檔並有測試）：

> `why` 只能引用 `jd_evidence`。禁止敘述使用者在哪一段經歷做了什麼。
> 經歷標題只當弱線索，用於決定 `point` 的措辭，不可寫進 `why`。

這樣即使標籤全掛在 `e1`，也不會產生「你在圖書館志工經歷中展現了進度控管」
這種錯誤歸因。你說得對，那比沒有 evidence 更危險。

你說這個欄位黃金集測不到、報表不給數字——同意，不用勉強。

### 五、履歷側兩個參數合併成常數　採納。`provenance` 不加

你的處理正確，照做。

`provenance` 我決定不加，理由是面試路徑上沒有這個區分：履歷來自
`GET /experiences`，`tags` 和 `description` 都是使用者自己打的，
面試側沒有任何 LLM 抽取步驟。加了也是常數，等於多一個永遠不變的欄位。

若之後 B1 的擷取器要餵進來再說，屆時是新的討論。

### 六、`segmentation` 三態　採納

已加。`"punctuation" | "discourse_marker" | "unavailable"`，
並在 docstring 寫明 B 側規則：不是 `punctuation` 時，prompt 不得把平均句長
講得像量出來的。

**尺度不一致我不改**：`CollabRaw.score` 是 int 0–100 因為它直接對應
Kotlin `CollabDim.score` 的 UI 欄位；`GapCandidate.weight` 是 float 0–1
因為它是內部排序用、不出 HTTP。兩者用途不同，統一反而會讓其中一邊需要轉換。
已在 docstring 註明。

---

## 我加的一項：`GapCandidate.kind`

你的「硬技能被指名、軟技能只被展演」是整份回覆對我最有價值的一點。
但我的處理跟你的建議不同。

你說 prompt 分兩路。我認為**應該直接過濾，不要改用委婉語氣保留**。
理由：一份漏講清單裡混進假指控，比清單短一點糟得多。使用者會照著一個
他其實已經講過的點去改，而且是以同樣的自信度被告知的。委婉語氣不會
降低傷害，只會讓錯誤看起來比較有禮貌。

所以新增：

```python
kind: str = "hard"   # "hard" | "soft"
```

**要問你一件事**：`sk:` 與 `skm:` 這兩個命名空間本身是不是已經編碼了
硬軟技能的區分？是的話你零成本就能填。無法判定時一律填 `"hard"`（保守，
寧可少濾），我這邊的過濾邏輯會處理。

`FakeGapComputer` 已經涵蓋一筆 `soft`，讓我測得到過濾路徑。

---

## 營運事項

**singleton 已寫進合約。** `TranscriptAnalyzer` 的 docstring 註明必須是
app 啟動時載入的 singleton，bge-m3 約 2.3GB，我端點組裝時用 FastAPI
lifespan 處理。謝謝提醒，這個我原本沒想到。

**STT 定案**：同意這件事不能等 W3 D1。我已經跟怡君要真實逐字稿樣本，
拿到第一時間轉你。這是目前唯一同時卡住我們兩個人的東西。

---

## 時程

| 時間 | 事項 |
|---|---|
| 現在 | 合約已改，pull 下來跑 `pytest -q`，應為 57 passed + 4 skipped |
| W1 D3 | 我交黃金測試集給你 |
| W1 D5 | 你交真實作；我交 JD 抽取器召回率報告 |
| W3 D3 前 | 前端 `groupSays` 結構改完（我盯） |

你說「改簽章我改三行」——簽章已經改好了，`JDInput`、`Utterance`、
`skill_id`/`display`、`kind`、`segmentation` 都在裡面。
Fake 三件也同步更新，可以直接對著寫。
