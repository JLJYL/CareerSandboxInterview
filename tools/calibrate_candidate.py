"""履歷候選門檻掃描（成員 A，W2）。

固定視窗 16／一般門檻 0.45（已校準），只掃 `CANDIDATE_THRESHOLD` 一維。

## 這個參數在做什麼

漏講 = 履歷∩JD − 提及。一筆假指控只會發生在**履歷上有、JD 也要**的技能上。
所以對「這個人履歷上的那幾個技能」放寬偵測門檻，正好對準傷害發生的地方，
而範圍只有 8 個詞，誤傷面積很小。

## 為什麼放寬不會製造新的假指控

放寬只會讓 mentioned 變大，而 gap = 交集 − mentioned，
所以它**只能減少 gap，不可能新增**。

代價在另一邊：放太寬會把**真漏講**也吃掉，`漏講 recall`（目前 1.000）會掉。
那才是這張表要盯的欄位——不是 precision 漲了多少，是 recall 掉了沒有。

## 為什麼這次跑得快

`_semantic_pass` 已改成不內建門檻、回傳全部候選，加上詞彙表向量的全域快取，
整輪掃描只算一次向量。先前 4×7 的掃描跑了 2.7 小時，這次應該幾分鐘。

用法:
    python tools/calibrate_candidate.py --cases fixtures/golden/interview ^
        --vocab fixtures/vocab/skills_v1.json --real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline import transcript as T  # noqa: E402
from app.pipeline.gap import GapComputer  # noqa: E402
from tools.interview_eval import (  # noqa: E402
    SetMetric, build_id_names, build_name_index, compare, load_embedding,
    load_vocab, show,
)

#: 0.45 = 不放寬（等同一般門檻）。往下掃到 0.20。
CANDIDATE_THRESHOLDS = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20]


def run(cases, vocab, embedding, cand_th):
    name2id = build_name_index(vocab)
    said_m, gap_m = SetMetric(), SetMetric()

    for case in cases:
        analyzer = T.TranscriptAnalyzer(
            vocab=vocab, embedding=embedding, enable_semantic=True,
            candidate_threshold=cand_th,
        )
        computer = GapComputer(analyzer, emit_soft=True)
        segs = case.get("transcript_segments") or []
        transcript = "\n".join(s for s in segs if s.strip()) or case.get("transcript", "")
        jd = JDInput(required_skills=list(case["jd"].get("requiredSkills") or []),
                     description=case["jd"].get("description") or "", source="catalog")

        labels = {}
        for name, lab in case["labels"]["skills"].items():
            sid = name2id.get(name)
            if sid:
                labels[sid] = lab
        machine = computer.explain(case["resume"], jd, transcript)

        said_truth = {k: (v.get("said") if v.get("said") is not None else 0)
                      for k, v in labels.items()}
        gap_truth = {}
        for sid, lab in labels.items():
            h, w = lab.get("has", 0), lab.get("wants", 0)
            s = lab.get("said") if lab.get("said") is not None else 0
            gap_truth[sid] = 1 if 1 in (h, w, s) else (2 if (h == 2 and w == 2 and s == 0) else 0)

        said_m.merge(compare(said_truth, set(machine["mentioned"])))
        gap_m.merge(compare(gap_truth, set(machine["gap"])))
    return said_m, gap_m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(args.cases.glob("ivw-*.json"))]
    vocab = load_vocab(args.vocab)
    build_id_names(vocab)

    if not args.real:
        print("✗ 這支一定要 --real。假向量沒有語意,掃出來的門檻沒有意義。")
        return 1
    print("載入 bge-m3⋯（詞彙表向量只算一次,之後每個門檻是純過濾）\n")
    embedding = load_embedding()

    print(f"{'候選門檻':>8}{'提及P':>8}{'提及R':>8}{'漏講P':>8}{'漏講R':>8}"
          f"{'真漏講':>7}{'假指控':>7}   假指控的是")
    rows = []
    for th in CANDIDATE_THRESHOLDS:
        said, gap = run(cases, vocab, embedding, th)
        names = "、".join(sorted({show(x) for x in gap.fp_items})) or "—"
        print(f"{th:>8.2f}{said.precision:>8.3f}{said.recall:>8.3f}"
              f"{gap.precision:>8.3f}{gap.recall:>8.3f}{gap.tp:>7}{gap.fp:>7}   {names}")
        rows.append((th, said, gap))

    base = rows[0][2]          # 0.45 = 不放寬
    print(f"\n基準（不放寬）：漏講 P {base.precision:.3f} / R {base.recall:.3f}"
          f"，真漏講 {base.tp}、假指控 {base.fp}")

    # ★ 選點規則：漏講 recall 不得退步。真漏講被吃掉比假指控少一筆嚴重得多——
    #   前者是「該講的沒被告知」,使用者不知道自己漏了什麼;
    #   後者只是多一條可以忽略的建議。
    ok = [(th, s, g) for th, s, g in rows if g.recall >= base.recall]
    best = max(ok, key=lambda r: r[2].precision) if ok else None
    if best and best[2].precision > base.precision:
        print(f"★ 建議 CANDIDATE_THRESHOLD = {best[0]:.2f}"
              f"（漏講 P {base.precision:.3f} → {best[2].precision:.3f}，"
              f"recall 維持 {best[2].recall:.3f}）")
    else:
        print("✗ 沒有任何門檻能在不犧牲漏講 recall 的前提下改善 precision。")
        print("  那本身是結論：剩下的假指控靠放寬門檻解決不了，")
        print("  維持 0.45，把力氣移到別的地方。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
