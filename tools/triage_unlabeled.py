"""未標記候選分流（成員 A，W2 收尾）。

把「機器找到但候選池沒有」的技能分成該標與不該標，只把該標的交給 B。

## 為什麼要分流

語意段開啟後，未標記從 29 筆跳到 224 筆——它會對整份詞彙表做最近鄰，
掃出一堆跟這份履歷/JD 無關的東西。直接把 224 筆丟給 B，他大概會擱著，
而擱著的結果是 JD 集的指標永遠是虛的。

分四類，只有前三類該標：

    A  履歷∩JD 都有   → **會產生 gap**，不標的話漏講指標是錯的     必須標
    B  只在 JD 側     → 影響 JD 集 precision                    該標
    C  只在履歷側     → 影響履歷集 precision                    該標
    D  只在提及集     → 語意段的最近鄰噪音，進不了任何交集        不必標

D 類進不了 `履歷 ∩ JD`，所以**它永遠不會變成漏講**，標了對指標沒有影響。
但它的「量」本身是一個發現：D 佔比很高就代表語意段誤命中嚴重，
只是那些誤命中剛好落在池外——換一批 JD 就未必。那要寫進限制。

## ★ 一定要用真 embedding

假向量的最近鄰是亂的，分類會完全錯（實測 `AJAX`／`C#`／`Word` 被歸成噪音）。
本工具不接受假向量。

用法:
    python tools/triage_unlabeled.py --cases fixtures/golden/interview ^
        --vocab fixtures/vocab/skills_v1.json ^
        --stt-aliases data/stt_confusions.v1.json ^
        --out data/unlabeled_for_review.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline import transcript as T  # noqa: E402
from app.pipeline.gap import GapComputer  # noqa: E402
from tools.interview_eval import (  # noqa: E402
    build_id_names, build_name_index, load_embedding, load_vocab, show,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--stt-aliases", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if os.environ.get("INTERVIEW_EVAL_FAKE_EMBED"):
        print("✗ 這支不接受假向量——最近鄰是亂的，分類會完全錯。")
        return 1

    vocab = load_vocab(args.vocab)
    build_id_names(vocab)
    name2id = build_name_index(vocab)
    stt = None
    if args.stt_aliases and args.stt_aliases.exists():
        stt = json.loads(args.stt_aliases.read_text(encoding="utf-8")).get("safe_aliases")

    print("載入 bge-m3⋯")
    embedding = load_embedding()

    rows: dict[str, dict] = defaultdict(
        lambda: {"cases": set(), "resume": 0, "jd": 0, "said": 0})

    for f in sorted(args.cases.glob("ivw-*.json")):
        case = json.loads(f.read_text(encoding="utf-8"))
        analyzer = T.TranscriptAnalyzer(vocab=vocab, embedding=embedding,
                                        enable_semantic=True, stt_aliases=stt)
        computer = GapComputer(analyzer, emit_soft=True)
        segs = case.get("transcript_segments") or []
        transcript = "\n".join(s for s in segs if s.strip())
        jd = JDInput(required_skills=list(case["jd"].get("requiredSkills") or []),
                     description=case["jd"].get("description") or "", source="catalog")
        machine = computer.explain(case["resume"], jd, transcript)
        pool = {name2id.get(k) for k in case["labels"]["skills"]} - {None}

        for sid in set(machine["resume"]) | set(machine["jd"]) | set(machine["mentioned"]):
            if sid in pool:
                continue
            r = rows[sid]
            r["cases"].add(case["case_id"])
            for key in ("resume", "jd", "said"):
                src = "mentioned" if key == "said" else key
                if sid in machine[src]:
                    r[key] += 1

    buckets = {"A": [], "B": [], "C": [], "D": []}
    for sid, r in rows.items():
        if r["resume"] and r["jd"]:
            buckets["A"].append(sid)
        elif r["jd"]:
            buckets["B"].append(sid)
        elif r["resume"]:
            buckets["C"].append(sid)
        else:
            buckets["D"].append(sid)

    total = len(rows)
    print(f"\n未標記候選共 {total} 個\n")
    labels = {
        "A": ("履歷∩JD 都有 → **會產生 gap，必須標**", True),
        "B": ("只在 JD 側 → 影響 JD 集 precision", True),
        "C": ("只在履歷側 → 影響履歷集 precision", True),
        "D": ("只在提及集 → 最近鄰噪音，進不了交集，不必標", False),
    }
    for key, (desc, must) in labels.items():
        n = len(buckets[key])
        print(f"  {key}  {desc:<44} {n:>4} 個 ({n/total:.0%})")

    need = buckets["A"] + buckets["B"] + buckets["C"]
    print(f"\n★ 實際要標的：{len(need)} 個（原本 {total} 個，"
          f"減少 {1 - len(need)/total:.0%}）\n")
    for key in ("A", "B", "C"):
        if buckets[key]:
            print(f"  [{key}] " + "、".join(sorted(show(x) for x in buckets[key])))

    noise = len(buckets["D"]) / total if total else 0
    print(f"\n  D 類佔 {noise:.0%}。這個比例本身是發現——")
    print("  它代表語意段在放寬的同時誤命中的量，只是那些誤命中剛好落在池外。")
    print("  換一批 JD 或不同履歷就未必。這要寫進限制。")

    if args.out:
        payload = {
            "_note": "A/B/C 需要人工標 has/wants/said；D 是語意段最近鄰噪音，不必標。",
            "_counts": {k: len(v) for k, v in buckets.items()},
            "need_labeling": [
                {"skill_id": s, "display": show(s), "bucket": k,
                 "cases": sorted(rows[s]["cases"])}
                for k in ("A", "B", "C") for s in sorted(buckets[k])
            ],
            "noise_only": sorted(show(s) for s in buckets["D"]),
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\n  已寫入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
