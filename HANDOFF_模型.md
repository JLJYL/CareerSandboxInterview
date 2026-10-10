# 面試模組 — 後端交接(模型組 B → 資料庫組 / JL)

面試模組是獨立的 FastAPI 服務,**不連資料庫**。
這份說明它跟你們那支的邊界、目前靠前端補的東西,以及兩件要確認的事。

---

## 一、邊界

```
Android
   ├──→  :8000   資料庫組(使用者、履歷、職缺、面試紀錄)
   └──→  :8001   面試模組(出題、追問、報告)  ← 這支
```

| | 面試模組 | 資料庫組 |
|---|---|---|
| 狀態 | 無狀態,不存任何東西 | 所有持久化 |
| 資料來源 | 前端在 request 裡帶 | MongoDB |
| 認證 | **目前不驗** | Bearer Token |
| 部署 | uvicorn, 8001 | 現有 |

啟動方式:

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8001
```

啟動時會載入 bge-m3(約 2.3GB),第一次比較慢。
`GET /health` 回 200 代表服務起來了。

---

## 二、目前靠前端補的東西

面試模組沒有資料庫,所以 session 狀態**全部由前端在每次請求裡帶回來**:

| 欄位 | 內容 | 之後可以改成後端查 |
|---|---|---|
| `question` | 上一個問題 | `Interview_Turns` |
| `askedTopics` | 已問過的問題原文 | 同上 |
| `spokenBy` | 已開口過的說話者 | 同上 |
| `followUpIdx` | 整場問到第幾題 | 同上 |
| `mode` | 面試模式 | `Interview_Sessions` |
| `context` | 職位設定、難度、群面配置 | 同上 |
| `experiences` | 使用者的履歷經歷 | 履歷表 |

`sessionId` 目前只是識別字串,**後端收下來原樣回傳,不查也不存**。

**你們的 `Interview_Sessions` / `Interview_Turns` 落地之後**,
這些欄位可以改成後端自己查——合約不用改,前端也不用改,
只要在 `app/api/routes.py` 把 `req.xxx` 換成查詢即可。

---

## 三、兩件要確認的事

### 1. 轉錄端點有沒有回傳 segments ★

你們的 `POST /interview/transcribe` 目前回傳的是什麼?

我這邊的合約有 `segments` 與 `segmentStartsMs` 兩個欄位,
前端送過來的一直是空陣列。怡君說換架構時舊資料來源被移除,
當時只是先補空值讓程式能跑。

**這件事決定三個功能的命運:**

| 有 segments | 沒有 |
|---|---|
| 斷句來源是實測值(`stt_segment`) | 退回估算值(`discourse_marker`) |
| 平均段長可以講成「量出來的」 | 只能講成估算 |
| 被切斷的輪次偵測得到 | 偵測不到 |

Whisper API 的 `verbose_json` 格式會帶 `segments`,每段含 `start` / `end`。
如果你們呼叫時用的是 `response_format="json"`(只回文字),
改成 `verbose_json` 就有了。

**如果拿不到,跟我說一聲**——那三個欄位可以直接標成不可用,
不用一直掛著「進行中」。

### 2. 面試端點要不要驗 token

目前 `POST /interviews`、`/turns`、`/report` **完全不驗 Bearer Token**。
任何人打得到就能用,而每次呼叫都會花 OpenAI 的額度。

三種做法:

```
a. 面試模組自己驗   需要拿到你們的 JWT secret 或驗證端點
b. 走你們的服務代理  前端只打 8000,你們轉發到 8001
c. 網路層隔離       8001 只允許內網,對外不開
```

**我傾向 b**,前端只需要維護一組 BASE_URL,而且認證邏輯集中在一處。
但那要你們加一層轉發。你們判斷哪個成本低。

---

## 四、額度與成本

一場面試的 OpenAI 呼叫次數:

| 階段 | 呼叫數 | 模型 |
|---|---|---|
| 開場 | 1(群面 2) | gpt-4o-mini |
| 每輪 | 1(群面/panel 2:派發 + 生成) | gpt-4o-mini |
| 報告 | 4 | gpt-4o-mini × 3 + gpt-4o × 1 |
| 漏講點 | 1 | gpt-4o-mini |
| JD 抽取 | 1 | gpt-4o-mini |
| 協作評分 | 4(僅群面) | gpt-4o-mini |

一對一五輪約 12 次,群面五輪約 21 次。

**唯一用 gpt-4o 的是報告的 STAR 拆解**——它要跨行原文引用,
小模型做不到(實測)。其餘全部 mini。

`OPENAI_API_KEY` 走 `.env`,`OPENAI_MODEL` 與 `OPENAI_MODEL_VERBATIM`
可以用環境變數覆蓋。

---

## 五、部署時要注意的三件

### bge-m3 的記憶體

啟動時載入一次,約 2.3GB,之後所有請求共用。
**不要用多個 worker**——每個 worker 會各載一份。
單 worker + async 已經夠用,因為瓶頸在 OpenAI 的回應時間不在 CPU。

```bash
uvicorn app.main:app --port 8001 --workers 1
```

### 語意比對預設關閉

`ENABLE_SEMANTIC=0` 是預設值,啟動會快很多。
要開的話設成 `1`,但那會讓第一次啟動多花時間載模型。

### 詞彙表路徑

`VOCAB_PATH` 預設指向 repo 內的 `fixtures/vocab/skills_v1.json`。
如果部署時檔案不在那個相對位置,用環境變數覆蓋。

---

## 六、已知的未解問題

**職缺散文與結構化欄位對不上。** 測試集的職缺需求來自 104 的結構化欄位,
正式環境是使用者手貼的散文。兩者用詞層級不同,
端到端的 recall 從 1.000 掉到 0.375——8 個該給的建議只給得出 3 個。

系統不會給錯建議,但報告會空洞化。正在調整抽取方式。

**這件事跟你們沒有直接關係**,但如果之後要把 104 的結構化資料接進來,
那會直接解決這個問題——列在這裡讓你們知道有這個需求。

---

## 附:我需要你們回覆的

1. `POST /interview/transcribe` 現在回傳的 JSON 結構(有沒有 segments)
2. 認證要走哪一種做法(a / b / c)
3. 部署位址(前端要設 BASE_URL)

前兩項擋著功能,第三項擋著整合測試。
