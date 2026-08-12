# 面試模組(A1–A5)

CareerSandbox 面試模擬功能的模型端開發區。

## 與 CareerSandboxModule 的關係

本 repo 為**獨立服務**,與 CareerSandboxModule(B1/B2/C1)分開部署。

| 項目 | 處理方式 |
|---|---|
| 履歷資料 | 由前端在 request body 帶入(`ReportRequest.experiences`),本服務不連 MongoDB |
| 正規化元件 | `VocabNormalizer` 等複製自 CareerSandboxModule,**三週內不同步**,檔頭需標明是複本 |
| 認證 | 目前不驗 Bearer token,為已知缺口,見 `CONTRACTS_interview.md` 第五節 |
| 前端接法 | 需另建一組 Retrofit 實例指向本服務,現有 `ApiClient` 是單一 BASE_URL |

複製正規化元件時注意:面試的正規化問題與履歷不同。履歷是書面語、有標點;
逐字稿是口語、中文 STT 不輸出標點,且會出現「我會寫 SQL」這類說法。
直接搬過來很可能命中率不足,需預留調整時間,並以怡君提供的真實逐字稿樣本校準。

## 快速開始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入金鑰,.env 不進版控
pytest -q                   # 應為 47 passed
```

`pytest` 不打真 LLM、不連網,全部是形狀與值域檢查,任何人 clone 下來都應該全綠。
如果紅燈,先確認是不是自己改了共用凍結區。

## 檔案歸屬

改動別人的區域之前先講一聲,共用凍結區則需雙方同意。

### 共用凍結區(改動需雙方同意)

| 檔案 | 內容 |
|---|---|
| `app/schemas/interview.py` | 五個接點的 HTTP 合約 |
| `app/schemas/interview_repair.py` | 固定值域的機械修復 |
| `app/contracts/interview_protocols.py` | 兩人之間的介面 |
| `fixtures/interview/golden/` | 回歸基準,由 `scripts/make_interview_golden.py` 生成 |
| `CONTRACTS_interview.md` | 合約說明 |
| `tests/test_interview_contracts.py` | 合約測試 |

這些檔案任何一方單方面修改,另一方的測試會紅燈。要改先在群組提出。

### 成員 B(LLM 與端點)

```
app/prompts/            所有 prompt
app/pipeline/missing_points.py     A5
app/pipeline/interview_report.py   A4
app/pipeline/interview_live.py     A1/A2/A3
app/pipeline/parallel.py           並行工具(共用,但由 B 維護)
app/api/                端點(尚未落地)
scripts/                測試集生成與評測
tests/test_missing_points.py 等
```

### 共用元件複本(唯讀)

```
app/vendor/             從 CareerSandboxModule 複製,禁止修改
app/vendor/README.md    來源與 commit hash 紀錄
app/vendor/MANIFEST.json 指紋,由 scripts/vendor_pin.py 產生
```

面試側需要調整時,在 `app/pipeline/` 寫擴充層包住複本,不要改複本本身。
`tests/test_vendor_integrity.py` 會檢查,改到會紅燈。

### 成員 A(正規化、檢索、確定性計分)

```
app/pipeline/transcript_analyzer.py   TranscriptAnalyzer
app/pipeline/gap_computer.py          GapComputer
app/pipeline/collab_scorer.py         CollabScorer
tests/test_gap_computer.py 等
```

A 只需要讀 `app/contracts/interview_protocols.py`,不需要讀 schema 層。
Protocol 刻意用 `list[dict]` 而非 Pydantic 型別,就是為了讓 A 不必依賴 B 的 schema。

## 分支

```
feat/b-a5  → feat/a-gap  → feat/b-a4  → feat/b-live
```

依序 squash merge。每個分支合併前 `pytest` 必須全綠。

## 交接點

| 時間 | 誰交給誰 | 內容 |
|---|---|---|
| W1 D3 | B → A | `fixtures/interview/transcripts_v1.jsonl`(黃金測試集) |
| W1 D5 | A → B | `GapComputer` 與 `TranscriptAnalyzer.mentioned_skills` 真實作 |
| W2 D3 | A → B | `text_stats` 與 `CollabScorer` |

W1 D3 那一筆是唯一會卡住 A 的東西。沒有測試集,A 從 D3 到 D5 只能盲寫。

## 驗收方式

用 `FakeGapComputer` 換成真實作,跑:

```bash
python scripts/interview_eval.py
```

命中率過門檻且 exit code 0 即為收貨。不用逐案討論哪一項算不算漏講。

## 紀律

- `conftest.py` 全部 mock,測試不打真 LLM
- 每個 bug 先在測試裡重現,再修
- 固定值域(六項細分名稱、STAR 四鍵、協作四項)零偏移,已寫成測試
- `resume_grounded=False` 時 `missing_points` 強制清空,這是防捏造的最後一道

## 不做的範圍

A6 影像指標(裝置端 MediaPipe)、`videoDims`、prosody 的語速/停頓/語調/開口前思考、
快速面試端點、反問環節。理由見 `CONTRACTS_interview.md` 第六節。
