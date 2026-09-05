# 覆蓋包 — 繁簡正規化

基準:上一包已套用、已 commit 的狀態(`feat/a-semantic` 合併 `feat/b-collab-e2e`)。

整包拖進 repo 根目錄覆蓋。**沒有新增任何套件依賴**——對照表抽成資料檔了。

驗證:

    pytest tests/ -q
    → 469 passed, 4 skipped

    python tools/interview_eval.py --cases fixtures/golden/interview \
        --vocab fixtures/vocab/skills_v1.json --draft
    → 數字與上一包完全相同(繁體輸入行為不變)

---

## 為什麼要做

怡君 2026-09-05 的實測報告:同一位講者、同一支後端、同樣的呼叫方式,
五段 App 錄音輸出**繁體**,一段電腦錄音輸出**簡體**。
原因未定(她懷疑是收音鏈路差異),但不影響結論——不能假設逐字稿一定是繁體。

詞彙表是繁體。逐字稿回簡體時,中文技能**全部**比對不到(實測 7 個測 7 個失敗)。
而 gap = 履歷∩JD − 提及,提及變小,交集裡的技能就變成假指控。

拿黃金集實測(只把逐字稿轉簡體、其餘完全不動):

    提及集 recall    0.488 → 0.366
    漏講 precision   0.533 → 0.444
    假指控             7 → 10 筆

不報錯,只是告訴使用者「你什麼都沒講到」。

修正後,簡體輸入的四項指標與繁體**完全相同**。

---

## 檔案清單

### 新增

    data/t2s_chars.v1.json       繁→簡字元對照,4105 筆
    tests/test_transcript_t2s.py 33 個測試

### 修改

    app/pipeline/transcript.py   fold_chars 加繁簡;折疊責任移進 SurfaceIndex.scan()
    app/pipeline/gap.py          呼叫端還原成傳原文(責任收歸 scan)

---

## 三個設計決定

### 為什麼是資料檔不是套件依賴

原本要裝 `opencc-python-reimplemented`。但既然只做**字元層**(為了等長),
就不需要它的詞組字典,只需要字元對照表——所以把 OpenCC 的
`TSCharacters.txt` 抽成資料檔,4105 筆全部 1 字元對 1 字元。

好處:零依賴、等長由建表時保證、跟現有 `data/stt_confusions.v1.json` 同一個模式。
授權:OpenCC 是 Apache-2.0,檔案裡有標來源。

### 為什麼是繁→簡,不是簡→繁

繁→簡是多對一,確定性的;簡→繁是一對多,要消歧義、會猜錯。
比對只需要兩邊落到同一個形,取確定性的方向。

顯示給人看的一律用**原始**逐字稿(evidence 靠位移切原文),
所以轉成簡體不會讓使用者看到簡體。

### 折疊責任移進 SurfaceIndex.scan()

原本折疊放在呼叫端。結果 `tests/test_interview_w1.py` 直接呼叫
`scan()` 時漏掉,整個索引靜默失效(掃不到東西、不報錯)。

責任放進 `scan()` 內部,呼叫端就不可能忘。`gap.py` 的三處包裝跟著還原。
這件事成立的前提是 `fold_chars` 嚴格等長——回傳的位移在原始文字上也成立。
**新增 `_LOOKALIKE_MAP` 或繁簡表條目時,一律 1 字元對 1 字元。**

---

## 一個刻意沒做的地方

`normalize_for_scan` **不做**繁簡轉換,只做 STT 相關的正規化。

理由:它的輸出同時餵給兩條路。表面掃描走 `scan()`(自己會折疊),
但**語意段拿的是它的原始輸出**,而語意段的詞彙表向量用的是
未折疊的表面形——這裡若轉簡體,視窗是簡體、詞彙表是繁體,
相似度會被字形差異污染。

這是實作過程中真的踩到的:先在 `normalize_for_scan` 加了繁簡,
`test_analyse_is_safe_under_concurrent_use` 就掛了,原因正是這個不對稱。

☆ 語意段兩側都折疊會更一致,但那會改變送進 bge-m3 的文字,
  相似度分布跟著變,而 `MENTION_THRESHOLD=0.45` 是在現有向量上校準的。
  **等下一輪重新校準時再一起處理**——反正 STT 換引擎本來就要重新校準。

---

## 這份報告帶來的其他資訊(尚未處理)

### 標點分布跟先前假設的不一樣

先前只有單一佐證樣本,判斷 Whisper 是「標點部分存在」。
五段真實 App 錄音顯示的是:

    A1、A3、C1    完全沒有標點,只有空格
    A2            只有一個頓號
    D1            開頭有半形逗號,中段之後全無
    B1(電腦錄音)  標點完整

App 錄音的**主流是零標點純空格**,不是部分標點。
擱置中的 `_segment` 修正方向仍然對(第三個分支正好處理這個),
但主次要調整。等更多樣本再動。

### 新的 STT 混淆表候選(僅 5 組、單一講者)

    父層 → 副層
    函式庫 → 函釋庫
    技術棧 → 技術站
    Kotlin → Calling(D1)／空靈(B1)
    Jetpack Compose → JPEG Compose

現有 `data/stt_confusions.v1.json` 是照 Android 實測建的,
這幾組是 Whisper 的錯誤形,兩者錯誤模式不同。樣本太少,先不建表。

### 好消息

    雜訊環境    C1 vs A1 幾乎相同,只漏兩個語助詞
    長音檔      B1 超過 3 分鐘,無 timeout 無截斷,60 秒設定足夠

### 已知失敗模式

    400 Invalid file format — 音檔編碼與副檔名不符時
