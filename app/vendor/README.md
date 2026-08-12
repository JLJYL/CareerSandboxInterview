# 複製元件的來源紀錄

每次複製檔案進來,在下表加一列,並執行 `python scripts/vendor_pin.py` 更新指紋。

| 檔案 | 來源 repo | 來源路徑 | commit hash | 複製日 | 複製人 |
|---|---|---|---|---|---|
| (尚未複製) | | | | | |

## 複製步驟

```powershell
# 1. 在 CareerSandboxModule 取得目前的 commit hash
cd <CareerSandboxModule 路徑>
git rev-parse --short HEAD

# 2. 找出要複製的檔案
Get-ChildItem -Recurse -Include *.py | Select-String -Pattern "VocabNormalizer|WeightedScorer" -List | Select-Object -ExpandProperty Path

# 3. 複製到本 repo 的 app/vendor/
# 4. 在複製進來的每個檔案最上方加入來源標頭(格式見下)
# 5. 在上表登記
# 6. 執行 python scripts/vendor_pin.py 記錄指紋
# 7. pytest -q 確認全綠
```

## 來源標頭格式

複製進來的檔案第一行必須是:

```python
# VENDORED from CareerSandboxModule @ <commit hash> on <日期>. DO NOT MODIFY.
# 面試側的調整寫在 app/pipeline/,不要改這個檔案,也不要回推到來源 repo。
```

## 不可修改的原因

修改複本會讓兩個 repo 分岔,而且分岔看不見。需要調整時的正確做法:

```python
# app/pipeline/interview_vocab.py
from app.vendor.vocab_normalizer import VocabNormalizer

class InterviewVocabNormalizer:
    """在複本外面包一層,處理口語逐字稿特有的問題。

    履歷是書面語、有標點;逐字稿是口語、中文 STT 不輸出標點,
    而且會出現「我會寫 SQL」這類說法。這些差異在這裡處理。
    """

    def __init__(self) -> None:
        self._base = VocabNormalizer()
        self._spoken_synonyms = {...}   # 面試特有的口語對應
```
