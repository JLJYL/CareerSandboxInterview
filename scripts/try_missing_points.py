"""A5 漏講點手跑驗收:用黃金集十格實跑一次,看 why 寫得如何、否決了幾條。

【這支在驗什麼】
成員 A 量的漏講 precision 約 0.47 是**否決前**的數字——那是 GapComputer
的輸出。B 的否決機制是第二道防線,兩個數字要分開看,否則之後沒人知道
改善來自哪一層。

這支同時輸出:
    否決前  GapComputer 給了幾條候選,對照黃金集標記有幾條是真漏講
    否決後  LLM 判斷後保留幾條,其中有幾條是真漏講
    每一條被否決的原因,以及每一條保留下來的 why 原文

【為什麼要人眼看】
自動指標只告訴你數字動了,不告訴你 why 寫得像不像人話、
有沒有把 JD 的話原封不動抄一遍、有沒有偷偷變成建議句。
那些要看過才知道。

執行:
    python scripts/try_missing_points.py                 # 全部十格
    python scripts/try_missing_points.py --case ivw-005  # 單一格
    python scripts/try_missing_points.py --dry-run       # 不打 API,只看候選
    python scripts/try_missing_points.py --jd-source extracted
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline.gap import GapComputer  # noqa: E402
from app.pipeline.missing_points import generate_missing_points  # noqa: E402
from app.pipeline.transcript import TranscriptAnalyzer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "fixtures" / "golden" / "interview"
VOCAB = ROOT / "fixtures" / "vocab" / "skills_v1.json"
VARIANTS = ROOT / "data" / "jd_variants_extracted.json"


def make_llm(model: str):
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return call


def load_vocab() -> list:
    raw = json.loads(VOCAB.read_text(encoding="utf-8"))
    return raw.get("skills", raw) if isinstance(raw, dict) else raw


def truth_gaps(case: dict) -> set[str]:
    """黃金集標記推出來的真漏講:has==2 且 wants==2 且 said==0。"""
    return {
        name
        for name, v in case["labels"]["skills"].items()
        if v.get("has") == 2 and v.get("wants") == 2 and v.get("said") == 0
    }


def hits(names: list[str], truth: set[str]) -> int:
    """粗略比對:display 含有標記名稱即算命中。

    正式的比對是成員 A 那邊經過正規化的,這裡只求人眼掃得出方向。
    """
    return sum(1 for n in names if any(t in n for t in truth))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="只跑單一格,例如 ivw-005")
    ap.add_argument("--dry-run", action="store_true", help="不打 API,只看候選")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--jd-source", choices=["catalog", "extracted"], default="catalog")
    args = ap.parse_args()

    if not args.dry_run:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("錯誤:找不到 OPENAI_API_KEY")
            return 1
        llm = make_llm(args.model)
        print(f"模型:{args.model}  JD 來源:{args.jd_source}\n")
    else:
        llm = None
        print(f"[dry-run] 只看候選,不打 API。JD 來源:{args.jd_source}\n")

    variants = {}
    if args.jd_source == "extracted":
        if not VARIANTS.exists():
            print(f"錯誤:找不到 {VARIANTS},請先跑 make_jd_extracted_variants.py")
            return 1
        variants = json.loads(VARIANTS.read_text(encoding="utf-8"))["variants"]

    analyzer = TranscriptAnalyzer(load_vocab())
    computer = GapComputer(analyzer)

    files = sorted(CASES.glob("ivw-*.json"))
    if args.case:
        files = [p for p in files if p.stem == args.case]
        if not files:
            print(f"錯誤:找不到 {args.case}")
            return 1

    tot_cand = tot_kept = tot_truth = 0
    tot_cand_hit = tot_kept_hit = 0

    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        cid = case["case_id"]
        truth = truth_gaps(case)
        transcript = "\n".join(case["transcript_segments"])

        if args.jd_source == "extracted":
            req = variants.get(cid, {}).get("required_skills", [])
            src = "extracted"
        else:
            req = case["jd"].get("required_skills") or case["jd"].get("requiredSkills") or []
            src = "catalog"
        jd = JDInput(
            required_skills=req,
            description=case["jd"].get("description", ""),
            source=src,
        )

        cands = computer.compute(case["resume"], jd, transcript)
        cand_names = [c.display for c in cands]

        print("=" * 74)
        print(f"{cid}   標記的真漏講 {len(truth)} 條:{'、'.join(sorted(truth)) or '(無)'}")
        print(f"        GapComputer 候選 {len(cands)} 條(命中 {hits(cand_names, truth)})")
        for c in cands:
            mark = "✓" if any(t in c.display for t in truth) else " "
            print(f"          {mark} {c.display}  kind={c.kind} w={c.weight}")

        tot_cand += len(cands)
        tot_truth += len(truth)
        tot_cand_hit += hits(cand_names, truth)

        if args.dry_run or llm is None:
            print()
            continue

        points, notices = generate_missing_points(
            cands, transcript, case["jd"].get("description", ""), llm,
        )
        kept_names = [p.point for p in points]
        tot_kept += len(points)
        tot_kept_hit += hits(kept_names, truth)

        print(f"        寫出理由後保留 {len(points)} 條(命中 {hits(kept_names, truth)})")
        for p in points:
            mark = "✓" if any(t in p.point for t in truth) else "✗"
            print(f"          {mark} {p.point}")
            print(f"              {p.why}")
        for n in notices:
            print(f"          · {n}")
        print()

    print("=" * 74)
    print(f"標記的真漏講合計          {tot_truth}")
    print(f"否決前候選 {tot_cand:>3} 條,命中 {tot_cand_hit:>2}   "
          f"precision {tot_cand_hit / tot_cand if tot_cand else 0:.3f}")
    if not args.dry_run:
        print(f"寫出理由後保留 {tot_kept:>3} 條,命中 {tot_kept_hit:>2}   "
              f"precision {tot_kept_hit / tot_kept if tot_kept else 0:.3f}")
        print()
        print("注意:這裡的比對是字面的,只求看出方向。正式數字以 interview_eval 為準。")
        print("這一層不做過濾,precision 應該與否決前接近;")
        print("差額來自敘事違規與空理由的丟棄,那是安全性而非正確性的理由。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
