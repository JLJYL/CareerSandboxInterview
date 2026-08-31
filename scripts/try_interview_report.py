"""A4 面試報告手跑驗收:用黃金集實跑,看四個區塊的產出品質。

【為什麼要手跑】
測試驗的是形狀——固定值域對不對、失敗會不會炸。實跑才看得出:

    verdict 是不是每格都同一個模子
    better 有沒有捏造履歷裡沒有的數字
    fromAnswer 實際對得上幾條(這是 A4 風險最高的一項)
    分數有沒有鑑別力,還是每個人都 70–80

A5 改了三輪才對,每一輪的問題都是實跑才看到的,測試全綠。

【模型路由】
fromAnswer 是跨行原文引用,gpt-4o-mini 做不到。
預設 starParts 用 gpt-4o、其餘用 gpt-4o-mini,可用 --same-model 對照差異。

執行:
    python scripts/try_interview_report.py --case ivw-001
    python scripts/try_interview_report.py                # 全部十格
    python scripts/try_interview_report.py --same-model    # 全部用 mini,看差多少
    python scripts/try_interview_report.py --dry-run       # 不打 API,只看輸入
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.pipeline.interview_report import (  # noqa: E402
    VERBATIM_FAIL_MARKER,
    format_stats,
    generate_report,
)
from app.pipeline.transcript import TranscriptAnalyzer  # noqa: E402
from app.schemas.interview import TurnDTO  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "fixtures" / "golden" / "interview"
VOCAB = ROOT / "fixtures" / "vocab" / "skills_v1.json"


def make_llm(model: str):
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return call


def load_vocab() -> list:
    raw = json.loads(VOCAB.read_text(encoding="utf-8"))
    return raw.get("skills", raw) if isinstance(raw, dict) else raw


def build_turns(case: dict) -> list[TurnDTO]:
    """黃金集的逐字稿是連續分段,沒有逐題切分。

    這裡把每一段當成一輪,question 留空——那是實際輸入形狀的近似。
    正式環境的 turns 由前端逐題送出,結構更乾淨。
    """
    segs = case["transcript_segments"]
    return [
        TurnDTO(question=f"(第 {i} 段)", answer=s, input_mode="voice")
        for i, s in enumerate(segs, 1)
    ]


def resume_text(case: dict) -> str:
    lines = []
    for e in case.get("resume", []):
        bits = [e.get("title", ""), e.get("role", ""), e.get("action", ""), e.get("result", "")]
        tags = "、".join(e.get("tags", []))
        lines.append("・" + " / ".join(b for b in bits if b) + (f"(標籤:{tags})" if tags else ""))
    return "\n".join(lines)


def verbatim_rate(report, raw_answers: str) -> tuple[int, int]:
    """回傳 (通過驗證的引用數, 模型嘗試引用的總數)。

    分母必須含被降級的那些,否則量不到模型的引用可靠度——
    管線已經把驗證失敗的段落降成 present=False,只數留下來的會永遠是 100%。

    實測踩到的坑:某次 gpt-4o 在引用時多加了一個「那」字,
    機械檢查抓到並降級,而診斷卻印出「28/28 全對」。
    失敗被自己的防護機制藏起來了。
    """
    passed = [p for p in report.star_parts if p.present and p.from_answer]
    # 再驗一次。這裡若發現不符,代表管線的驗證與這裡不一致,那本身是 bug。
    src = raw_answers.replace(" ", "")
    ok = sum(1 for p in passed if p.from_answer.replace(" ", "") in src)
    downgraded = sum(1 for n in report.notices if VERBATIM_FAIL_MARKER in n)
    return ok, len(passed) + downgraded


async def run_case(path: pathlib.Path, llm, llm_verbatim, analyzer) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    turns = build_turns(case)
    raw_answers = "\n".join(t.answer for t in turns)
    stats = analyzer.text_stats(raw_answers)
    rtext = resume_text(case)

    print("=" * 74)
    print(f"{case['case_id']}   {len(turns)} 段 / {stats.char_count} 字")
    stats_text, measurable = format_stats(stats, "voice")
    print("  " + stats_text.replace("\n", "\n  "))

    if llm is None:
        print()
        return {}

    report = await generate_report(
        mode="single", turns=turns, stats=stats, input_mode="voice",
        resume_text=rtext, resume_grounded=bool(rtext),
        llm=llm, llm_verbatim=llm_verbatim,
    )

    print("\n  【三維面向】")
    for d in report.face_dimensions:
        print(f"    {d.letter} {d.name} {d.score:>3}  {d.verdict}")
        for p in d.points:
            print(f"           · {p}")
        if d.prosody:
            print(f"           prosody: {[(x.label, x.value) for x in d.prosody]}")

    print("\n  【六項細分】", "  ".join(f"{s.name} {s.score}" for s in report.sub_scores))

    print("\n  【STAR】")
    for p in report.star_parts:
        if p.present:
            print(f"    {p.key} ✓ 「{p.from_answer}」")
        else:
            print(f"    {p.key} ✗ {p.hint}")

    print("\n  【逐題回顧】(只印第一題)")
    if report.question_feedbacks:
        q = report.question_feedbacks[0]
        print(f"    comment: {q.comment}")
        print(f"    better : {q.better}")

    print("\n  【improvements】")
    for x in report.improvements:
        print(f"    · {x}")

    if report.notices:
        print("\n  【notices】")
        for n in report.notices:
            print(f"    · {n}")

    ok, total = verbatim_rate(report, raw_answers)
    by_name = {s.name: s.score for s in report.sub_scores}
    print()
    return {
        "case": case["case_id"],
        "quote_ok": ok,
        "quote_total": total,
        "star_present": sum(1 for p in report.star_parts if p.present),
        "better_cleared": sum(1 for q in report.question_feedbacks if not q.better),
        "better_total": len(report.question_feedbacks),
        # 有錨的:內容深度(量化詞處數)、表達流暢度(填充詞率)
        "anchored": [by_name.get("內容深度", 0), by_name.get("表達流暢度", 0)],
        # 無錨的:純印象分,鑑別力要看這三項
        "unanchored": [by_name.get(n, 0) for n in ("互動能力", "應變能力", "自信程度")],
        "logic": by_name.get("邏輯清晰度", 0),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--verbatim-model", default=os.getenv("OPENAI_MODEL_VERBATIM", "gpt-4o"))
    ap.add_argument("--same-model", action="store_true", help="starParts 也用 --model,看差多少")
    args = ap.parse_args()

    llm = llm_verbatim = None
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
        vm = args.model if args.same_model else args.verbatim_model
        llm_verbatim = make_llm(vm)
        print(f"一般:{args.model}   STAR:{vm}\n")
    else:
        print("[dry-run] 只看輸入與統計\n")

    # engine_filler_policy="partial" 對應 STT 探測的實測結論:
    # Android 移除非詞彙填充音(嗯、呃),保留詞彙型(那個、就是、然後)。
    # 不帶這個參數的話 filler_reliability 一律退回 "unknown",
    # 流暢度就永遠走備援訊號,等於白白丟掉唯一乾淨的量化訊號。
    # 端點接線時同樣要帶,見 app/pipeline/interview_report.py 的說明。
    analyzer = TranscriptAnalyzer(load_vocab(), engine_filler_policy="partial")
    files = sorted(CASES.glob("ivw-*.json"))
    if args.case:
        files = [p for p in files if p.stem == args.case]
        if not files:
            print(f"錯誤:找不到 {args.case}")
            return 1

    rows: list[dict] = []
    for path in files:
        r = await run_case(path, llm, llm_verbatim, analyzer)
        if r:
            rows.append(r)

    if not rows:
        return 0

    print("=" * 74)
    q_ok = sum(r["quote_ok"] for r in rows)
    q_tot = sum(r["quote_total"] for r in rows)
    fails = q_tot - q_ok
    print(f"fromAnswer 引用可靠度  {q_ok}/{q_tot}  ({q_ok / q_tot:.1%})"
          if q_tot else "fromAnswer 引用可靠度  無引用")
    if fails:
        print(f"  失敗 {fails} 次,已被機械檢查降為未命中(分母含這些,否則量不到可靠度)")

    star = [r["star_present"] for r in rows]
    print(f"STAR 命中段數       每格 {star}   平均 {sum(star)/len(star):.1f}/4")

    b_cleared = sum(r["better_cleared"] for r in rows)
    b_tot = sum(r["better_total"] for r in rows)
    print(f"better 被清空       {b_cleared}/{b_tot}"
          f"   {'' if b_cleared == 0 else '★ 模型仍在加原回答沒有的數字'}")

    # 有錨與無錨分開看。全距一個數字看不出問題在哪——
    # 有錨的項目跟著統計走是預期的,無錨的擠在一起才是鑑別力不足。
    anchored = [v for r in rows for v in r["anchored"]]
    unanchored = [v for r in rows for v in r["unanchored"]]
    print()
    print(f"有錨兩項(內容深度/表達流暢度)  {min(anchored)}–{max(anchored)}  "
          f"全距 {max(anchored) - min(anchored)}")
    print(f"無錨三項(互動/應變/自信)       {min(unanchored)}–{max(unanchored)}  "
          f"全距 {max(unanchored) - min(unanchored)}")

    # 每一格內部,無錨三項彼此的差距。這才是「這個人強在哪弱在哪」分不分得出來。
    within = [max(r["unanchored"]) - min(r["unanchored"]) for r in rows]
    avg_within = sum(within) / len(within)
    print(f"無錨三項的格內差距             平均 {avg_within:.1f}  每格 {within}")
    if avg_within < 12:
        print("  ★ 格內差距偏小:同一個人的三項印象分擠在一起,分不出強弱")

    print()
    print("要人眼看的三件:")
    print("  verdict 是不是每格同一個模子")
    print("  better 有沒有出現逐字稿裡沒有的**事實**(數字以外的,機械檢查抓不到)")
    print("  STAR 的 hint 是不是具體到照著做得出來")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
