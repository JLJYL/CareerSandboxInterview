"""語意段校準掃描（成員 A，W2 D3–D5）。

一次跑完視窗大小 × 門檻的組合，輸出掃描表。比照 calibrate_normalizer.py
與 run_golden_pairs.py 的做法。

## 為什麼要掃而不是試

改一個參數跑一次，載一次 bge-m3（2.3GB）要等很久，而且看到數字動了也
分不出是哪個參數造成的。掃描表把整個參數空間一次攤開,選點才有依據。

## 三個要一起看的數字

  提及集 recall     語意段的正面效果:她講了、我抓不抓得到
  提及集 precision  語意段的副作用:她沒講、我卻算她講了
  漏講 precision    最終產品指標:報出去的漏講有多少是假指控

★ 只看 recall 會選出一個門檻低到什麼都算「講到」的點——那時 gap 會變成空的,
  precision 看起來 100%,但那不是準,是什麼都不報。所以三個要一起看。

## 一個實測發現決定了要掃視窗

假 embedding 驗證時發現:預設 SEMANTIC_WINDOW=28 會把「我那時候排了每週的
進度表然後每次開會都會確認大家有沒有跟」這樣一整段話,拿去跟「進度控管」
這 4 個字比相似度。長短懸殊會稀釋餘弦分數——中間那些「我那時候」「然後」
「大家」全是雜訊。

而合約 docstring 舉的例子(「用資料庫查資料」對上 SQL)是**短句對短詞**。
所以視窗大小可能比門檻更關鍵,兩個要一起掃。

用法:
    python tools/calibrate_semantic.py --cases fixtures/golden/interview ^
        --vocab fixtures/vocab/skills_v1.json --real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline import transcript as T  # noqa: E402
from app.pipeline.gap import GapComputer  # noqa: E402
from tools.interview_eval import (  # noqa: E402
    SetMetric, build_id_names, build_name_index, compare, load_vocab,
)

WINDOWS = [12, 16, 20, 28]
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


class HashEmbedding:
    """確定性假向量。只用來驗程式路徑,分數沒有語意,不可拿來選參數。"""

    def encode(self, texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(64, dtype="float32")
            for i, ch in enumerate(t):
                v[(ord(ch) * 7 + i) % 64] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.asarray(out)


def load_embedding(real: bool):
    if not real:
        print("⚠ 使用假向量,分數沒有語意——只驗程式跑不跑得動,不可據此選參數。\n")
        return HashEmbedding()
    from app.providers.embeddings import BgeM3Embedding  # 你 W2 那個實作
    print("載入 bge-m3(約 2.3GB),第一次會很久⋯\n")
    return BgeM3Embedding()


def run_once(cases, vocab, embedding, window, stride, threshold, emit_soft):
    T.SEMANTIC_WINDOW = window
    T.SEMANTIC_STRIDE = stride
    name2id = build_name_index(vocab)
    said_m, gap_m = SetMetric(), SetMetric()

    for case in cases:
        analyzer = T.TranscriptAnalyzer(
            vocab=vocab, embedding=embedding,
            enable_semantic=embedding is not None,
            threshold=threshold,
        )
        computer = GapComputer(analyzer, emit_soft=emit_soft)
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
    ap.add_argument("--real", action="store_true", help="用真 bge-m3;不加則用假向量只驗路徑")
    ap.add_argument("--emit-soft", action="store_true", default=True)
    args = ap.parse_args()

    cases = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(args.cases.glob("ivw-*.json"))]
    vocab = load_vocab(args.vocab)
    build_id_names(vocab)
    embedding = load_embedding(args.real)

    base_w, base_s = T.SEMANTIC_WINDOW, T.SEMANTIC_STRIDE

    print("=== 基準線：語意段關閉 ===")
    said0, gap0 = run_once(cases, vocab, None, base_w, base_s, 1.1, False)
    print(f"  提及集 P={said0.precision:.3f} R={said0.recall:.3f}"
          f"   漏講 P={gap0.precision:.3f} R={gap0.recall:.3f}\n")

    print(f"{'視窗':>4}{'門檻':>7}{'提及P':>8}{'提及R':>8}{'漏講P':>8}{'漏講R':>8}   相對基準線")
    best = None
    for window in WINDOWS:
        for th in THRESHOLDS:
            said, gap = run_once(cases, vocab, embedding, window, max(window // 2, 4),
                                 th, args.emit_soft)
            dr = said.recall - said0.recall
            dp = gap.precision - gap0.precision
            flag = ""
            if dr > 0 and dp >= -0.02:
                flag = "  ← recall 升、漏講 P 沒退"
                if best is None or dr > best[0]:
                    best = (dr, window, th)
            print(f"{window:>4}{th:>7.2f}{said.precision:>8.3f}{said.recall:>8.3f}"
                  f"{gap.precision:>8.3f}{gap.recall:>8.3f}   R{dr:+.3f} P{dp:+.3f}{flag}")

    T.SEMANTIC_WINDOW, T.SEMANTIC_STRIDE = base_w, base_s
    print()
    if best:
        print(f"★ 提及 recall 提升最多且漏講 precision 沒退步："
              f"視窗 {best[1]}、門檻 {best[2]:.2f}（recall {best[0]:+.3f}）")
    else:
        print("✗ 沒有任何組合能在不犧牲漏講 precision 的前提下提升 recall。")
        print("  那本身是結論：語意段對這份資料無效，該把力氣移到模糊匹配。")
    if not args.real:
        print("\n⚠ 本次用假向量，上表僅證明程式跑得動。選參數必須加 --real。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
