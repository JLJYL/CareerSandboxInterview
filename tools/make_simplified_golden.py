#!/usr/bin/env python3
"""把黃金測試集的逐字稿轉成簡體,產出一份平行的測試集。

## 用途

Whisper 對中文繁簡輸出沒有一致保證(怡君 2026-09-05 實測:同一位講者、
同一支後端,五段 App 錄音輸出繁體、一段電腦錄音輸出簡體)。

表面掃描那條路已經修好了——`fold_chars` 會把兩邊都折成簡體再比對,
實測繁簡兩種輸入的 draft 指標完全相同。

**但語意段那條路沒有覆蓋到。** 語意段的視窗文字來自 `normalize_for_scan`,
那支刻意不做繁簡轉換(見它的註解:轉了會讓視窗是簡體、詞彙表向量是繁體,
反而製造新的不對稱)。所以:

    逐字稿繁體   視窗繁體 vs 詞彙表繁體   一致 ✓
    逐字稿簡體   視窗簡體 vs 詞彙表繁體   不一致 ← 這支要量的就是它

bge-m3 是多語模型,簡繁同義詞的向量會接近但不相同,所以不會像表面掃描
那樣歸零。問題是**會不會掉到 MENTION_THRESHOLD=0.45 以下**——那要實跑才知道。

## 怎麼用

    python tools/make_simplified_golden.py
    → 產出 fixtures/golden/interview_simplified/

    # 先跑繁體當基準
    python tools/interview_eval.py --cases fixtures/golden/interview \\
        --vocab fixtures/vocab/skills_v1.json --semantic

    # 再跑簡體
    python tools/interview_eval.py --cases fixtures/golden/interview_simplified \\
        --vocab fixtures/vocab/skills_v1.json --semantic

## 怎麼判讀

只看**提及集 recall** 與**漏講的假指控筆數**,其他指標不受逐字稿影響。

    兩邊幾乎相同        語意段對字形不敏感,不用處理
    簡體明顯較差        語意段需要繁簡正規化,下一輪校準時一起做
                        (兩側都折疊,然後重跑門檻掃描)

★ 只轉逐字稿,標記(has/wants/said)與 JD、履歷完全不動。
  變因只有一個,差異才歸得了因。
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "fixtures" / "golden" / "interview"
DST = ROOT / "fixtures" / "golden" / "interview_simplified"
T2S = ROOT / "data" / "t2s_chars.v1.json"


def load_table() -> dict[str, str]:
    raw = json.loads(T2S.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.get("chars", {}).items() if len(k) == 1 == len(v)}


def main() -> int:
    if not SRC.is_dir():
        print(f"找不到來源:{SRC}")
        return 1
    if not T2S.is_file():
        print(f"找不到對照表:{T2S}")
        return 1

    table = load_table()
    print(f"對照表 {len(table)} 筆")

    def conv(s: str) -> str:
        return "".join(table.get(c, c) for c in s)

    DST.mkdir(parents=True, exist_ok=True)
    n = changed = 0
    for p in sorted(SRC.glob("*.json")):
        case = json.loads(p.read_text(encoding="utf-8"))
        n += 1
        before = json.dumps(case, ensure_ascii=False)

        segs = case.get("transcript_segments")
        if segs:
            case["transcript_segments"] = [conv(s) for s in segs]
        if case.get("transcript"):
            case["transcript"] = conv(case["transcript"])

        after = json.dumps(case, ensure_ascii=False)
        if before != after:
            changed += 1
        (DST / p.name).write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"處理 {n} 份,其中 {changed} 份逐字稿有變化")
    print(f"輸出:{DST}")
    print()
    print("接著跑:")
    print(f"  python tools/interview_eval.py --cases {DST.relative_to(ROOT)} \\")
    print("      --vocab fixtures/vocab/skills_v1.json --semantic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
