"""協作評分手跑驗收:四個 BARS 維度各跑一次,看等第分不分得開。

【為什麼這一支特別需要實跑】
其他三個模組都有量化錨點可以驗(填充詞率、量化詞處數、verbatim 子字串)。
協作是**零量化錨點**——rubric 明文禁用發言次數、字數、時間佔比、
首次發言早晚、發言長度,四個維度全靠語意判斷。

測試只驗得了形狀(四次呼叫、level 在 1-4、level 0 不進報告)。
等第分不分得開、evidence 是不是真的原文,只有實跑看得到。

【要驗的四件】
    halo    四個維度會不會全部黏在同一級。rubric 第八節第 2 點的實測:
            一次評四維 r̄ ≈ .86–.92,分開評掉到 .26–.35。
            我們已經分開評了,這裡驗那個分離是不是真的發生。
    原文    evidence 必須是逐字稿裡出現過的話。這是 starParts 的同一種檢查。
    誠實    沒有立場分歧時,協作姿態該回 level 0,不是給一個低分。
    對比    好的討論與壞的討論要拿到不同的等第,否則這個功能沒有意義。

執行:
    python scripts/try_collab_score.py                # 三種情境各跑一次
    python scripts/try_collab_score.py --case good    # 只跑一種
    python scripts/try_collab_score.py --dry-run      # 不打 API,只看切片與 prompt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import FakeCollabObserver, Utterance  # noqa: E402
from app.pipeline.collab_score import score_collab  # noqa: E402
from app.prompts.collab_rubric import LEVEL_TO_SCORE, compose_collab_prompt  # noqa: E402
from app.schemas.interview import COLLAB_DIM_NAMES  # noqa: E402

SCORE_TO_LEVEL = {v: k for k, v in LEVEL_TO_SCORE.items()}


# ---------------------------------------------------------------------------
# 三種情境。刻意設計成等第應該明顯不同——分不出來就是這個功能沒用。
# ---------------------------------------------------------------------------

CASES: dict[str, tuple[str, list[Utterance]]] = {
    "good": (
        "會主動定錨、會接住別人、有理由有證據、分歧時先確認再提整合方案",
        [
            Utterance("主考官", "今天的題目是公司要推新的會員制度,第一步該做什麼?"),
            Utterance("user", "我覺得第一步是先看現有客群的消費頻率分佈 因為會員制度的門檻要訂在哪 取決於多數人落在哪個區間 訂太高沒人升級 訂太低沒有差異化"),
            Utterance("AI-強勢", "我覺得直接先上線一個版本比較快 邊做邊修"),
            Utterance("user", "先上線這點我同意 速度重要 但我會先跑一週的資料再訂門檻 因為門檻訂錯之後改 已經升級的人會有反彈 這個成本比晚一週高"),
            Utterance("AI-邏輯", "你說的消費頻率分佈 母數多少才有代表性?"),
            Utterance("user", "你這個問題對 我之前做電商報表的時候 樣本低於三十筆的分群我們都不採用 這次如果活躍客戶不到那個量 那就先用消費金額分級 不要硬切頻率"),
        ],
    ),
    "poor": (
        "只回應提問、沒有理由、分歧時直接否定、沒有主動方向",
        [
            Utterance("主考官", "今天的題目是公司要推新的會員制度,第一步該做什麼?"),
            Utterance("user", "我覺得先做市場調查比較好"),
            Utterance("AI-強勢", "我覺得直接先上線一個版本比較快 邊做邊修"),
            Utterance("user", "那樣不行啦 還是要先調查"),
            Utterance("AI-邏輯", "調查要調查什麼?"),
            Utterance("user", "就是調查一下大家的想法 這樣比較好"),
        ],
    ),
    "no_conflict": (
        "討論全程沒有立場對立——協作姿態應該回 level 0,不是低分",
        [
            Utterance("主考官", "今天的題目是公司要推新的會員制度,第一步該做什麼?"),
            Utterance("user", "我會先看現有客群的消費頻率分佈 門檻要訂在多數人落點附近"),
            Utterance("AI-親切", "我也覺得從資料開始比較穩"),
            Utterance("user", "對 而且分完群之後可以先試一個小範圍 看升級率再放大"),
        ],
    ),
}


def make_llm(model: str):
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,  # rubric 第八節:評分要可重現
        )
        return resp.choices[0].message.content or ""

    return call


def wrap(text: str, indent: int = 8, width: int = 58) -> str:
    pad = " " * indent
    return "\n".join(pad + text[i : i + width] for i in range(0, len(text), width))


def verbatim_ok(evidence: str, utterances: list[Utterance]) -> bool:
    """evidence 必須是逐字稿裡出現過的話。跟 starParts 同一種檢查。"""
    if not evidence:
        return True
    src = "".join(u.text for u in utterances).replace(" ", "")
    return evidence.strip("「」\"' ").replace(" ", "") in src


async def run_case(name: str, llm) -> dict:
    desc, utterances = CASES[name]
    print("=" * 72)
    print(f"{name}   {desc}")
    print("=" * 72)

    slices = FakeCollabObserver().observe(utterances)
    print("\n【切片】")
    for sl in slices:
        n = len(sl.excerpts)
        print(f"  {sl.name}  {n} 則{'  ' + sl.note[:40] if sl.note else ''}")

    if llm is None:
        print()
        return {}

    dims, notices = await score_collab(
        slices, llm, "\n".join(u.text for u in utterances)
    )
    by_name = {d.name: d for d in dims}

    print("\n【評分】")
    levels: dict[str, int | None] = {}
    for sl in slices:
        d = by_name.get(sl.name)
        if d is None:
            levels[sl.name] = None
            print(f"  {sl.name}  未評分")
            continue
        lv = SCORE_TO_LEVEL.get(d.score, 0)
        levels[sl.name] = lv
        print(f"  {sl.name}  等第 {lv}  ({d.score} 分)")
        print(wrap(d.hint))
        if d.evidence:
            print(wrap(f"依據:「{d.evidence}」", indent=8))

    for n in notices:
        print(f"  · {n}")
    print()
    return {"case": name, "levels": levels, "utterances": utterances, "dims": dims}


def report(rows: list[dict]) -> None:
    print("=" * 72)

    # 一、各場的等第分佈。同場全距為 0 不代表 halo——
    # good 的設計就是四項都做對,四維全 3 是合理的。
    # halo 的證據是「不同品質的場次拿到相同分數」,那要看跨場,見第二項。
    print("\n【等第分佈】")
    for r in rows:
        line = "  ".join(
            f"{n[:4]} {r['levels'][n] if r['levels'][n] is not None else '—'}"
            for n in COLLAB_DIM_NAMES
        )
        print(f"  {r['case']:12} {line}")

    # 二、對比:好與壞要拿到不同等第
    good = next((r for r in rows if r["case"] == "good"), None)
    poor = next((r for r in rows if r["case"] == "poor"), None)
    if good and poor:
        print("\n【區辨力檢查】good 應該高於 poor")
        print("  這才是 halo 的正式判準:同一維度在不同品質的場次拿到相同等第,")
        print("  代表那個維度讀不出行為差異。")
        blind = 0
        for n in COLLAB_DIM_NAMES:
            g, p = good["levels"].get(n), poor["levels"].get(n)
            if g is None or p is None:
                print(f"  {n}  有一邊未評分,無法比較")
                continue
            if g > p:
                mark = ""
            elif g == p:
                mark = "  ★ 讀不出差異"
                blind += 1
            else:
                mark = "  ★ 方向相反"
                blind += 1
            print(f"  {n}  good {g} vs poor {p}{mark}")
        if blind:
            print(f"\n  {blind}/4 個維度沒有區辨力。那幾維目前不該顯示在報告上——")
            print("  一個分不出好壞的分數,比沒有分數糟。")

    # 三、誠實:沒有分歧時協作姿態要回未評分
    nc = next((r for r in rows if r["case"] == "no_conflict"), None)
    if nc:
        lv = nc["levels"].get("協作姿態")
        print("\n【誠實檢查】沒有立場分歧時的協作姿態")
        if lv is None:
            print("  未評分 —— 正確。沒有觀察機會不該給低分")
        else:
            print(f"  ★ 給了等第 {lv} —— 這場沒有分歧,不該有觀察結果")

    # 四、原文
    print("\n【原文檢查】evidence 是否出現在逐字稿")
    bad = 0
    for r in rows:
        for d in r["dims"]:
            if not verbatim_ok(getattr(d, "evidence", ""), r["utterances"]):
                bad += 1
    total = sum(1 for r in rows for d in r["dims"] if d.evidence)
    print(f"  {total} 條引用,{'全部對得上' if bad == 0 else f'★ {bad} 條不存在於逐字稿'}")
    empty = sum(1 for r in rows for d in r["dims"] if not d.evidence)
    if empty:
        print(f"  {empty} 個維度沒有給依據(找不到對應句子時留空是對的,但太多代表判斷沒有根據)")

    print("\n要人眼看的兩件:")
    print("  hint 是不是照著做得出來,還是「建議加強協作能力」這種形容")
    print("  poor 那場的 hint 有沒有指出具體問題,而不是泛泛的鼓勵")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    if args.dry_run:
        print("[dry-run] 只看切片與 prompt\n")
        for name in [args.case] if args.case else list(CASES):
            await run_case(name, None)
        print("=" * 72)
        t = compose_collab_prompt("論點建構")
        print(f"\n論點建構的 prompt({len(t)} 字):\n")
        print(t)
        return 0

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤:找不到 OPENAI_API_KEY")
        return 1

    llm = make_llm(args.model)
    print(f"模型:{args.model}   temperature=0\n")

    names = [args.case] if args.case else list(CASES)
    rows = [r for r in [await run_case(n, llm) for n in names] if r]
    if len(rows) > 1:
        report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
