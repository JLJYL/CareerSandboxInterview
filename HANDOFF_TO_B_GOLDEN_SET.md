# W1 A 側交接：黃金測試集與詞彙表擴充（A → B）

`feat/a-gap` 已推上 `CareerSandboxInterview`。
`feat/vocab-expand-286` 已推上 `CareerSandboxModule`。

黃金集是**完整草稿**。缺的不是工作量，是**第二個標記者**——
`labeled_by` 目前只有我，自己標完自己確認等於沒有覆核。

原本分工表上黃金集是你 W1 D2–D3 的欄位。我做掉了大部分（選 JD、查技能字串
本來就要動詞彙表），但覆核這一步必須是你——這也是我們原本講好的
「前 3 份兩人各標一次比一致率」。

---

# 一、先做這一件（30 秒）

```bat
git fetch && git checkout feat/a-gap
python tools/interview_eval.py --cases fixtures/golden/interview ^
    --vocab fixtures/vocab/skills_v1.json --draft --by-case
```

**不需要 embedding、不需要 `jobs_all.jsonl`、不需要 Module repo。**
語意段預設關閉，這條路是純字面與別名比對，秒跑完。

（`build_interview_golden.py` 需要 `jobs_all.jsonl`，但你不用跑它——10 格已經生好。）

---

# 二、repo 相關：我做的三個決定

這一節是**告知**，不需要你回覆，但你要知道。

## 1. 兩個 repo 維持分開

我一開始把 `CareerSandboxModule` 整包帶進 `feat/a-gap`（`scorer.py`、`recommend.py`、
`retrieval/`、`schemas/domain.py` 等 20 幾個檔），原因是驗證詞彙表擴充要跑
`run_golden_pairs`，而它相依 `VocabNormalizer` 整條。

**已經全部拆掉了。** `feat/a-gap` 現在只含面試模組的東西，你原本推的
（`parallel.py`、`probe_rules.py`、`schemas/interview*.py`、`vendor/`、
`fixtures/interview/golden/`）一個都沒被動到。

不合併的理由：合併要動你已經 push 的東西（PR #1），而你正在寫 A5 端點。
為了少切一次目錄去打斷你，不划算。

`app/contracts/protocols.py` 特別拆掉了——它跟 `interview_protocols.py` 是
**兩份不同的合約**，並存的話之後沒人知道該改哪一份。

## 2. `skills_v1.json` 正本在 Module repo

Interview 那份（`fixtures/vocab/skills_v1.json`）是**唯讀鏡射**。

留鏡射的理由是摩擦成本：你要覆核就得跑 `interview_eval`，跑它就要詞彙表。
要求你 checkout 兩個 repo 才能看一份黃金集，那個覆核大概就不會發生。

代價是兩份會分岔，用一條規矩壓住：**所有編輯只在 Module repo 進行，改完同步過來。**

## 3. `feat/a-gap` 上有哪些檔案

```
app/pipeline/transcript.py          TranscriptAnalyzer 真實作（表面／詞頭／語意三段）
app/pipeline/gap.py                 GapComputer 真實作
tools/interview_eval.py             評測器（就緒閘門、覆蓋率閘門、--by-case）
tools/label_interview_golden.py     ★ 你要覆核的 53 筆裁決都在這裡
tools/build_interview_golden.py     10 格產生器
tools/expand_vocab.py               詞彙表擴充工具
fixtures/golden/interview/*.json    10 格黃金集
fixtures/golden/interview/raw/      6 段 STT 原始逐字稿
fixtures/vocab/skills_v1.json       286 條（唯讀鏡射）
data/stt_confusions.v1.json         STT 轉寫對照（價值有限，見第五節）
tests/test_interview_w1.py          42 tests，全部不依賴模型
```

---

# 三、黃金測試集內容

## 10 格的構成

3 個 persona × 3–4 個 JD = 10 格，由 **6 段逐字稿**支撐。
素材全部來自 `persona_150` 與 `jobs_all`，零生成內容（沿用 `golden_pairs.v1` 的規矩）。

| case | 錄音 | persona | JD | 段/字 | 分層 |
|---|---|---|---|---|---|
| ivw-001 | T1 | resume-013 資工 | 軟體工程師 | 5 / 380 | hard-skill-dominant |
| ivw-002 | T1 | resume-013 | TypeScript前端 | 5 / 380 | ＋jd-prose-heavy |
| ivw-003 | T1 | resume-013 | PM 專案管理師 | 5 / 380 | mismatched-pair（交集應為空）|
| ivw-004 | T2 | resume-013 | 前端網頁工程師 | 3 / 254 | ＋near-perfect-mention, stt-letter-split |
| ivw-005 | T3 | resume-140 行流 | 專案管理師 | 6 / 468 | **soft-skill-dominant ★核心** |
| ivw-006 | T3 | resume-140 | 產品行銷專案經理 | 6 / 468 | ＋huge-required-list（31 項）|
| ivw-007 | T4 | resume-140 | 遊戲營運企劃助理 | 8 / 690 | near-zero-mention |
| ivw-008 | T5 | resume-079 會計 | 會計專員 | 4 / 408 | dense-jargon, truncated-surface |
| ivw-009 | T6 | resume-079 | 行政會計專員 | 4 / 277 | ＋heavy-disfluency |
| ivw-010 | T5 | resume-079 | 會計行政助理 | 4 / 408 | cross-domain-overlap |

分層有兩軸，不要混：**配對軸**由（履歷×JD）決定，**逐字稿軸**由錄音指示決定。
「幾乎全講到」不是配對的性質，是對受試者下的指令。

## 逐字稿怎麼來的

**怡君錄的真實 Android STT 輸出，一個字都沒修。**

你原本規劃構造，我改用錄的，理由有二：分層照樣精準（給要點不給稿，
要點決定哪些技能會被提到），以及構造出來的術語轉寫是想像的，
而想像跟實測差很遠——見第五節第一項。

裝置：Galaxy S24 / Android 16 / Google App 17.44.15。**單一講者。**

`transcript_segments` 是陣列不是字串——實測停頓 1–2 秒 STT 就自動送出，
一次回答必然被切成數段。下游用 `"\n".join()` 餵給 `compute()`，
**換行本來就是斷句符號，所以送出邊界直接變成句界**，`sentence_count`
從估算值變成實測值。這是分段送出白撿的好處。

**你那側要確認**：`InterviewSession.turns` 怎麼壓成 `transcript: str`。
直接串接無分隔的話邊界資訊就丟了。建議用換行接，合約不用改。

## 標記結構

每個技能標三格，**漏講正解由三格推出，機器完全不參與**（所以不循環）：

| 格 | 問題 | 2 | 1 | 0 |
|---|---|---|---|---|
| `has` | 履歷真的有嗎 | 明確有 | 模糊（不計分）| 沒有 |
| `wants` | JD 真的要嗎 | 明確要 | nice-to-have（不計分）| 沒要 |
| `said` | 逐字稿講到了嗎 | 明確講到 | 判不出（不計分）| 沒講到 |

**漏講 = `has==2 and wants==2 and said==0`。**
三個 collector 的準確率因此各自可測——壞了知道去修哪個。

---

# 四、第一組真實數字

```
集合          precision   recall   TP   FP   FN   不計分   未標記
履歷集             0.973    1.000   72    2    0     18      0
JD 集             0.975    1.000  116    3    0      3     24
提及集             0.952    0.488   20    1   21      3      5
漏講(端到端)        0.467    1.000    7    8    0     24      0
```

## 對你影響最大的一個數字

**漏講 precision 0.467 —— 我們報出去的漏講，超過一半是假指控。**

這不只是我的問題。**你的 why prompt 會替這些假指控寫出有說服力的理由**，
而且語氣跟真漏講一模一樣。使用者會照著一個他其實已經講過的點去改。

八筆假指控，兩種成因，修法完全不同：

| 假指控 | 成因 | 修法（W2）|
|---|---|---|
| `Angular`／`Git`／`JavaScript` | STT 把英文詞轉爛，且**每次錯得不一樣** | 履歷條件式模糊匹配 |
| `軟體程式設計`／`專案時間╱進度控管` | **展演式陳述**——做了但沒說出名字 | 語意段（目前沒開）|

## 三個好消息

**履歷集與 JD 集 precision/recall 都在 0.97 以上** —— 管線結構是對的，
壞的只有提及那一段。

**漏講 recall 1.000** —— 真漏講一個都沒放過，問題純粹在多報。

**排序 0.800** —— 人工首選八成落在機器前三名，權重公式方向正確。

---

# 五、實測發現（按重要性）

## 1. 轉寫不穩定，靜態別名表對英文技術詞走不通

同一個詞、同一個人、同一支手機：

| 技能 | 探測 | T1 | T2 |
|---|---|---|---|
| Angular | Android | **整段消失**（只剩「這個框架」）| Andrew |
| Git | gate | gats | Gate |
| JavaScript | JavaScript ✓ | **加巴screen** | JavaScript ✓ |
| HTML | HTML ✓ | HTML ✓ **與 HDMI 並存於同一段** | HTML ✓ |
| Kotlin | — | Collin | Collin |

問題不是「有一個固定的錯誤形式」，是「**每次都錯得不一樣**」。
所以 `stt_confusions.v1.json` 記的是**已觀察到的變體**，不是映射表——
補了 `gate`，下次來的是 `gats`。只有 `Kotlin→Collin` 這種穩定的才真有用。

**中文詞 100% 存活，英文 36%。** T3/T4/T5 三段的要求技能全部抓到。

## 2. 詞彙表覆蓋率不是提及偵測的瓶頸

補了 176 條，提及集 recall 完全沒動（0.488 → 0.488，TP/FN 一個數字都沒變）。
漏抓的是 `Angular`／`Git`，它們在逐字稿裡是「Andrew」「gats」——
詞彙表有沒有那條根本不相干。

**但補詞彙表會讓假指控變多。** 交集因為兩側都解析得更好而變大，提及偵測卻沒跟上，
於是 gap 變多、多出來的全是假指控（FP 6 → 8）。

**結論：詞彙表和提及偵測要一起走。在提及偵測修好之前，不要再往下補詞彙表。**

## 3. `filler_count` 可以用，鑑別力很漂亮

實測每百字：T3 正常發揮 **3.0**、T4 多講故事 **6.4**、T6 刻意講亂 **9.8**。
單調、間距明顯、完全吻合錄音指示。**這是目前整個模組唯一乾淨的量化訊號。**

但範圍變窄：Android 會**全數移除**非詞彙填充音（「嗯」5→0、「呃」5→0，
部分被轉成「而」），詞彙型完整保留（「那個」2→2、「就是」7→8、「然後」1→3）。

所以 `FILLERS` 已縮減為只含詞彙型。**你的 prompt 不要宣稱它涵蓋全部口語習慣**，
它涵蓋的是詞彙型填充詞。`filler_reliability` 會回 `measured`。

## 4. `avg_sentence_len` 不是句長

實測 69–86 字。那是她**停頓前講了多長**（段界＝STT 自動送出點）。
這其實更有用——它測語流連續性——但措辭不要講成「句子長度」。

## 5. 語意反轉：你比我更該擔心

「蠻有興趣」→「沒有興趣」，兩次獨立測試都出現，非偶發。

漏講判定受影響有限（它看技能詞不看情態），但你的 `starParts`／
`questionFeedbacks`／`subScores` 是直接讀語意的——**一句反轉會讓回饋整段錯，
而且錯得很有說服力。**

值得跟怡君討論：**讓使用者看得到自己的逐字稿**。他一眼就能發現
「我明明說蠻有興趣」。這比後端做任何補救都有效，成本只是多一個畫面。

## 6. 中文詞會被空白切開

逐字稿出現 `做財務報 表的工作`——表面掃描直接失效。
已在 `normalize_for_scan` 移除中日韓字元之間的空白（英文之間的保留，
`data analysis` 不能黏成一團）。

---

# 六、詞彙表擴充：110 → 286 條

`jobs_all` 前 200 名 ＋ 黃金集缺的 23 條。
**已驗證 `run_golden_pairs` 15/15 格通過、逐技能 75/75，沒有退步。**
退路在 `fixtures/vocab/skills_v1.pre_expand.json`（Module repo）。

為什麼要做：擴充前，黃金集 87 個技能字串只有 28 個對得上（**32%**）。
對不上的會被排除計分，而它們正好是最難的那批（`AJAX`／`ASP.NET`／`Angular`／
`C#`／`MES` 全部沒有條目）。於是指標會**上升**——不是變好，是難題被移出考卷。
`interview_eval` 現在有硬閘門擋這件事：覆蓋率低於 80% 時拒絕寫 baseline。

三件擴充時處理掉的事：

**命名空間。** 從既有條目反推規則而不是自己發明。順帶回答你 D1 問我的問題——
`sk:`／`skm:` **編的是資料來源，不是硬軟技能**，而且方向跟直覺相反：
`sk:` 67 條裡只有 1 條含拉丁字母（`使用者體驗設計`、`談判協商`、`說服力`——
O*NET 語彙），`skm:` 43 條裡有 23 條含拉丁字母（`Outlook`、`Java`、
`行政事務處理`——104 語彙）。新條目全部來自 `jobs_all`，所以走 `skm:`。

**詞頭別名。** `報表彙整與管理` 同時登記 `報表彙整`，因為人講話講詞頭。
實測全語料中文技能 54% 抽得出詞頭。

**規格型條目排除。** `中文打字20~50` 在 `jobs_all` 排第 105 名，
`中文打字50~75` 排第 45 名。它們不是技能是任職條件，人不會在面試裡講出這串字，
補進去會變成**永久的假指控**——每一次都出現。

擴充中途抓到一個會咬人的東西：詞頭下限原本是 2 字，會生出
`規劃` ←（`規劃、組織、指導及協調組織內部行政作業`）這種別名。
「我負責活動規劃」「做財務規劃」全部誤中。已把下限提到 4 字——
中文表面形不做邊界檢查（詞間沒有空白可依），短詞頭必然以子字串形式亂命中。

---

# 七、請你做的三件事

## 1. 覆核 53 筆 `said` 裁決 ★ 擋著 baseline

全部在 `tools/label_interview_golden.py` 的 `ADJUDICATED`，每筆附理由。
**不同意就改那個 dict 重跑 `python -m tools.label_interview_golden`**，
不要直接改 JSON——理由要跟標記一起留著，否則三個月後沒人知道當初為什麼那樣判。

### 先確認你同意這三條規則

- **指名算 2** —— 原字出現：「我用 Angular 寫的」
- **展演算 2** —— 具體行為描述：「我排了每週進度表，每次開會確認進度」＝專案時間控管
- **只能推論算 0** —— 「我在餐廳打工過」推不出「抗壓性」

第二條是重心，`ivw-005` 整格的意義都在它上面。

### `said` 的定義

問「**這個人有沒有講**」，不是問「文字裡看不看得出來」。
兩者在 STT 出錯處分岔，而分岔處正是產品風險所在——他講了 Angular、
系統說他沒講，他就不信任這個系統，不管是誰的錯。

### 六筆最該看的

| 我的判定 | 理由 | 可能的爭議 |
|---|---|---|
| T1 `JavaScript` = **2** | 「用加巴screen」——加巴≈java、screen≈script | 走樣得很遠，你可能認為讀不出來 |
| T1 `Angular` = **1（不計分）** | 「這個框架」是懸空指涉，判不出她講了沒有 | 你可能認為證據夠強→2，或不夠→0 |
| T2 `Angular` = **2** | 「用了Andrew了這個框架」——有詞元佔著框架名的位置 | Andrew vs Angular 是不是太遠 |
| T3 `專案時間` = **2** | 展演：排訪談時間、設 deadline、追延遲 | 展演的門檻到哪裡 |
| T3 `Line` = **0** | 「設一個在LINE」其實是「設一個 deadline」 | 刻意留的假陽性測資 |
| T6 `PowerPoint` = **0** | 她**實際沒講**，不是轉寫錯誤 | 要點卡要求提到，但她跳過了 |

### 兩組合併裁決我判「不同技能」

這兩組會改變交集，是整條鏈的上游，錯了下游全錯：

- `專案管理` **≠** `專案管理架構及專案說明`（做專案的能力 vs 能講方法論）
- `帳務處理` **≠** `結帳作業與帳務處理`（日常記帳 vs 期末關帳）

比照 `golden_pairs.v1` 對「財務報表製作≠報表彙整與管理」的保守判法。

## 2. 標 29 筆未標記的

補完詞彙表後，JD 散文掃描新找到、候選池裡沒有的技能：

```
C++、CSS、Claude、ERP 系統、Node.js、React、Vue.js、品牌行銷管理、
報表製作、文書處理軟體操作、時間管理、溝通協調、溝通能力 ⋯
```

它們現在不計分，所以**JD 集那個 0.975 是虛的**。多數 `has=0`（履歷沒有），
標起來快。加進 `case['labels']['skills']` 補三格即可。

## 3. 知道詞彙表變了

Module repo 的 `fixtures/vocab/skills_v1.json` 已從 110 條變 286 條
（branch `feat/vocab-expand-286`）。A1 檢索與 B1 計分都會受影響，
已驗 `golden_pairs` 15/15。

---

# 八、覆核完之後

```bat
:: 三個旗標翻 true 後
python tools/interview_eval.py --cases fixtures/golden/interview ^
    --vocab fixtures/vocab/skills_v1.json ^
    --write-baseline data/interview_baseline.json
```

工具會擋住三種情況：未覆核的草稿、詞彙表覆蓋率低於 80%、含合成逐字稿。
**所以只要它寫得出來，那份 baseline 就是可信的。**

那是之後每次回歸的比較基準，也是 W1 的交付證據。

---

# 九、三個必須寫進報告的限制

**單一講者。** 六段都是怡君錄的，英文技術詞 36% 這個數字要理解成
「這個講者在這支手機上」。

**有效樣本數是 6 不是 10。** 同一段逐字稿服務多格，誤差相關，
不要當成獨立觀測。`interview_eval` 會在報表裡提醒。

**參數仍未校準。** `MENTION_THRESHOLD = 0.55`、`W_JD/W_RESUME = 0.6/0.4`、
`JD_POSITION_DECAY = 0.3` 目前**全是憑常識填的**。
而且 `MENTION_THRESHOLD` 現在是**死參數**——語意段沒開，它一次都沒被用到。
W2 D3–D5 校準才會讓它們有根據。

---

# 十、我 W2 接著做的

**第一個實驗：打開語意段。** `enable_semantic` 從頭到尾是 `False`，
所以 recall 0.488 是「表面掃描 ＋ 詞頭匹配」的天花板。
打開它、重跑、看 recall 動多少——不用寫任何新程式，就能測出目前最大的未知數。
它直接針對 `軟體程式設計`／`專案時間╱進度控管` 那兩筆展演式假指控。

**第二個：履歷條件式模糊匹配。** 現在是拿 286 條詞彙表掃逐字稿，
但真正要回答的問題很窄：「這個人履歷上的這 8 個技能，他討論到哪些？」
掃全表時「加巴screen」什麼都不是；若知道履歷有 JavaScript，
它就是強候選（加巴≈java）。範圍限縮讓誤傷面積很小，
而且**不需要預先知道會錯成什麼**——這是唯一能對付「每次錯得不一樣」的方法。

`CollabScorer` 照分工表 W2 交付，`Utterance` 型別已確認可實作。
