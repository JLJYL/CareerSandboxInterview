# 外部參考:AI-Interview-System 評估筆記

來源:`github.com/manthan89-py/AI-Interview-System`
評估日:W1 D1 | 規模:1358 行 / 11 個 Python 檔 / 零測試 | 最後 commit:2025-06-12

## 結論

**架構不可參考,prompt 與工程手法有兩項可用。**

差距不在規模,在種類。它是 Streamlit 單體桌面 app,狀態放 `st.session_state`,
STT 用 Speechmatics 雲端 API,TTS 在伺服器端播音,儲存寫本地 JSON 檔。
我們是 FastAPI 服務加 Android 前端,STT 在裝置端,儲存在 MongoDB。

功能覆蓋度也低一階:**只有一對一,沒有 panel、沒有群面、沒有漏講點分析**。
我們最核心的 A5 在它那邊完全沒有對應物。

不 fork、不照它的檔案結構重組。我們的合約已凍結、測試已綠燈,形狀比它完整,
往回對齊是退步。

---

## 可用項目一:追問生成的禁則(用於 A1,W2 實作)

它的 `next_question_generation` prompt 輸入是「前一題 + 這次回答 + JD +
履歷亮點」,跟 A1 幾乎一致。真正值得借用的是它的 **Avoid 段落**——
這些是實務上真的會犯、而且不寫進 prompt 就一定會發生的錯。

已整理成規則常數,見 `app/prompts/probe_rules.py`。

要點:
- 寫成**模式**而非具體例句。列舉例句會讓模型只避開那幾句,
  換個說法照樣違規。
- 「不得重複已問領域」需要傳入已問清單才能執行,不能只靠 prompt 自律。
  這是 `asked_topics` 參數存在的理由。
- 「不得誘導」對面試教練特別重要:誘導式追問會讓使用者答出他其實沒有的
  經歷,那份逐字稿再拿去做 A5,漏講點分析就整個失真。

它的 prompt 有一個我們**不採用**的設定:`feedback` 限制 90 字。
中文 90 字太短,而且散文式回饋不可行動。我們用 `verdict` 加 `points` 的
結構化形式。

---

## 可用項目二:LLM 呼叫並行化(用於 A4,W2 實作)

它在 `analyze_candidate.py` 用 `asyncio.gather` 把「產生回饋」與「產生下一題」
同時發出,外層加 30 秒 timeout。這是整個 repo 唯一值得直接借用的工程手法。

### 為什麼有效

LLM 呼叫的時間幾乎全部花在等待網路回應,那段時間本機 CPU 閒置。
序列寫法把這段閒置重複浪費 N 次。並行之後總時間等於最慢的那一個。

實測(每次呼叫固定 2 秒,六個區塊):

```
序列(一個接一個): 12.0 秒
並行(同時發出)  : 2.0 秒
```

真實 GPT-4o 呼叫約 3–8 秒。A4 六個區塊序列跑是 20–50 秒,並行後約 8 秒。

### 我們的版本比它硬化的三處

實作見 `app/pipeline/parallel.py`,已測試。

| 問題 | 參考 repo | 我們 |
|---|---|---|
| 一個區塊失敗 | 整包 `gather` 失敗,五個成功的結果全部丟掉 | `return_exceptions=True`,失敗的單獨記入 notices,其餘保留 |
| timeout 位置 | 包在 `gather` 外層,一個慢就全部取消 | 每個工作各自計時,慢的單獨逾時,其餘正常回來 |
| 速率限制 | 無控制 | 號誌限制同時數,`interview_eval.py` 跑 10 份測試集才不會撞上限 |

它另外用 `ThreadPoolExecutor` 加 `run_in_executor`,原因是 `litellm.completion`
是同步函式。**我們不需要這一層**,OpenAI SDK 的 `AsyncOpenAI` 本身就是非同步的。

### A4 的三層結構

有依賴的不能並行。實際順序是:

```
第一層 確定性計算(本地,毫秒級)
       text_stats / GapComputer / CollabScorer
          ↓
第二層 五個獨立 LLM 呼叫(並行)
       三維面向 / 六項細分 / 逐題回顧 / STAR 拆解 / 漏講點 why
          ↓
第三層 依賴第二層
       改進建議 <- 由三維面向的 points 彙整,不再打 LLM
```

`improvements` 不另外呼叫 LLM 的理由有二:省一次呼叫;讓建議與評語必然一致。

### 兩個提醒

**成本不變。** 並行不減少呼叫次數,省的只有等待時間。

**開場端點也適用。** `POST /interviews` 要同時產出開場題、`fallbackProbes`、
`interruptLines`、personas 說明,四項彼此獨立。開場等待時間直接影響使用者
對「這 app 會不會卡」的第一印象。

A1/A2/A3 每輪只發一個呼叫,並行沒有用武之地。

---

## 明確不採用的三處

| 項目 | 它的做法 | 問題 |
|---|---|---|
| JSON 解析失敗 | `return None` | 上層拿 `None` 去用會在別處炸開,追不到源頭。我們用 `repair_report()` 機械修復加 notices |
| 總分計算 | 各題分數取平均,全檔 8 行 | 無加權、無維度。我們是三維加六項細分 |
| 分數尺度 | 1–10 | 粒度太粗。我們用 0–100 |
