"""Live 引擎手跑驗收:模擬一整場面試,看追問品質與 persona 分辨度。

【為什麼要手跑】
測試驗的是形狀與降級路徑。實跑才看得出:

    追問是不是真的接著上一句,還是換了個說法問同一件事
    panel 三位主管的語氣分不分得出來
    群面的 AI 同儕像不像競爭者,還是變成友善的助教
    已問領域有沒有真的被避開

A4 與 A5 都是實跑之後才發現問題,測試全綠。

【怎麼模擬使用者】
用固定的回答腳本,不用 LLM 扮演使用者——
兩個 LLM 互相對話會讓問題來源變得無法歸因:追問變差是出題端的問題,
還是回答端剛好講了含糊的話?腳本固定,變因才只有一個。

執行:
    python scripts/try_interview_live.py --mode single
    python scripts/try_interview_live.py --mode panel --turns 6
    python scripts/try_interview_live.py --mode group
    python scripts/try_interview_live.py --all          # 三個模式各跑一次
    python scripts/try_interview_live.py --dry-run      # 不打 API,只看 prompt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.pipeline.interview_live import next_turn, start_interview  # noqa: E402
from app.prompts.interview_live import compose_turn_prompt  # noqa: E402
from app.prompts.probe_rules import MAX_TURNS_PER_SESSION  # noqa: E402
from app.schemas.interview import InterviewContext  # noqa: E402

CTX = InterviewContext(
    round="初試",
    language="中文",
    type="行為",
    difficulty="中等",
    custom_role="資料分析實習生",
    custom_company="某電商公司",
    custom_industry="電子商務",
    custom_seniority="新鮮人",
    custom_jd=(
        "協助建立銷售報表與資料清理,需具備 SQL 撰寫能力,"
        "能建立自動化報表流程,需與產品、工程團隊密切協作。"
    ),
)

# 固定的使用者回答腳本。刻意設計成有好有壞,看追問會不會跟著變。
#
#   1  含糊,沒有數字        → 應該被追問細節
#   2  有數字有方法          → 應該被追問判斷而不是重複問數字
#   3  提到團隊衝突          → panel 應該轉給 HR 主管
#   4  明確說不會            → 應該被接住,不是繼續逼問
#   5  提到搶快先做          → 群面的 AI-強勢 應該有反應
#   6  提到取捨與代價        → panel 應該轉給用人主管
ANSWERS = [
    "我之前在系學會做過行銷 然後也有去電商公司實習 做一些資料的東西",
    "實習的時候我用 SQL 重寫了週報的查詢 把產出時間從四小時縮短到一小時",
    "那時候行銷跟工程對需求的優先順序有分歧 我有去協調兩邊的排程",
    "這個我沒想過 我不太確定該怎麼回答",
    "我覺得與其一直討論 不如先做一個版本出來再修 這樣比較快",
    "後來我決定先做報表自動化 放掉了另一個視覺化的需求 因為時間只夠做一件事",
]


def make_llm(model: str):
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,  # 出題要有一點變化,全 0 會讓追問變得很制式
        )
        return resp.choices[0].message.content or ""

    return call


def wrap(text: str, indent: int = 6, width: int = 62) -> str:
    """中文不靠空白斷行,textwrap 會整段不換,所以按字數硬切。"""
    pad = " " * indent
    return "\n".join(pad + text[i : i + width] for i in range(0, len(text), width)) or pad


def _near_duplicate(question: str, asked: list[str]) -> str | None:
    """找出與先前問題高度重疊的一題。

    純字面重疊,不做語意——這只是診斷,目的是讓人看到「換個說法問同一件事」。
    門檻訂得比標籤比對嚴,因為問句本來就會共用很多虛詞。
    """
    q = set(question) - set("的了嗎呢你我他這那有沒是不在,。?、 ")
    for a in asked:
        av = set(a) - set("的了嗎呢你我他這那有沒是不在,。?、 ")
        if not q or not av:
            continue
        shared = len(q & av)
        if shared >= 4 and shared / min(len(q), len(av)) >= 0.7:
            return a
    return None


async def run_mode(mode: str, llm, n_turns: int) -> dict:
    print("=" * 74)
    print(f"模式:{mode}")
    print("=" * 74)

    start = await start_interview(mode=mode, context=CTX, llm=llm)
    speaker = f"[{start.opening_speaker}] " if start.opening_speaker else ""
    print(f"\n開場  {speaker}{start.opening_question}")
    print(f"      領域:{start.opening_topic}")
    if start.personas:
        print(f"      場上:{'、'.join(p.display_name for p in start.personas)}")
    if start.interrupt_lines:
        print("      搶話台詞:")
        for line in start.interrupt_lines:
            print(f"        · {line}")
    print("      備援追問:")
    for p in start.fallback_probes:
        print(f"        · {p}")
    for n in start.notices:
        print(f"      ⚠ {n}")

    asked_qs = [start.opening_question]   # 累積問題原文,不是領域標籤
    question = start.opening_question
    follow_up_idx = 0
    speakers_used: list[str] = []
    topics: list[str] = []
    repeats = 0
    capped = 0   # 被上限強制推進的次數。這幾輪看不出模型自己會不會換主題
    empty = 0    # 模型沒產出問題的次數

    for i in range(min(n_turns, len(ANSWERS))):
        if follow_up_idx > MAX_TURNS_PER_SESSION:
            break
        answer = ANSWERS[i]
        print(f"\n── 第 {i + 1} 輪  (followUpIdx={follow_up_idx})")
        print("  使用者:")
        print(wrap(answer))

        r = await next_turn(
            mode=mode, answer=answer, follow_up_idx=follow_up_idx,
            asked_questions=asked_qs, spoken_by=speakers_used,
            question=question, fallback=start.fallback_probes,
            llm=llm, context=CTX,
        )

        who = f"[{r.speaker}] " if r.speaker else ""
        tag = "追問" if r.is_follow_up else "新主題"
        print(f"  {who}({tag}) ")
        print(wrap(r.next_question))
        if r.reaction:
            print(f"      反應:{r.reaction}")
        print(f"      領域:{r.topic or '(未標)'}"
              f"   shouldAdvance={r.should_advance}")
        for n in r.notices:
            print(f"      ⚠ {n}")
            if "強制推進" in n:
                capped += 1
            if "未產出問題" in n or "生成失敗" in n:
                empty += 1

        if r.speaker:
            speakers_used.append(r.speaker)
        if r.topic:
            topics.append(r.topic)
        # 只有新主問題才進已問清單。追問是同一題的延伸,加進去會誤擋後續追問。
        if not r.is_follow_up:
            near = _near_duplicate(r.next_question, asked_qs)
            if near:
                repeats += 1
                print("      ★ 這個新主問題與先前問過的重複:")
                print(f"         先前:{near}")
            asked_qs.append(r.next_question)

        question = r.next_question
        # followUpIdx 是「整場問到第幾題」,一路遞增不歸零。
        # 早期版本寫成 shouldAdvance 時歸零,那是把它當成「這題追問幾次」的殘留——
        # 前端 followUpIdx >= 4 是進入反問環節,不是重來一輪。
        follow_up_idx += 1
        if r.should_advance:
            print("      ── 整場問答結束,前端此時進入反問環節 ──")
            break

    return {
        "mode": mode,
        "speakers": speakers_used,
        "topics": topics,
        "repeats": repeats,
        "capped": capped,
        "empty": empty,
        "personas": [p.display_name for p in start.personas],
    }


def report(rows: list[dict]) -> None:
    print()
    print("=" * 74)
    for r in rows:
        print(f"\n{r['mode']}")
        if r["personas"]:
            used = {s: r["speakers"].count(s) for s in r["personas"]}
            silent = [k for k, v in used.items() if v == 0]
            print(f"  發言分佈  {used}")
            if silent:
                print(f"    ★ 全程沒開口:{'、'.join(silent)}"
                      "(派發偏食,某個 persona 等於不存在)")
        print(f"  領域序列  {' → '.join(r['topics']) or '(無)'}")
        if r["repeats"]:
            print(f"    ★ 領域重複或近義 {r['repeats']} 次"
                  "(禁則沒生效,或標籤沒有沿用既有的)")
        if r["empty"]:
            print(f"    ★ 模型未產出問題 {r['empty']} 次,那幾輪走了備援池")
        if r["capped"]:
            print(f"  上限強制推進 {r['capped']} 次"
                  "(那幾輪看不出模型自己會不會換主題)")

    print()
    print("要人眼看的四件:")
    print("  追問是不是接著上一句,還是換個說法問同一件事")
    print("  說「我沒想過」那一輪有沒有換路,而不是同一件事換個說法再問一次")
    print("  panel:三位主管的語氣分不分得出來")
    print("  group:AI 同儕像不像競爭者,還是變成友善的助教")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "panel", "group"], default="single")
    ap.add_argument("--all", action="store_true", help="三個模式各跑一次")
    ap.add_argument("--turns", type=int, default=len(ANSWERS))
    ap.add_argument("--dry-run", action="store_true", help="不打 API,只印 prompt")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    if args.dry_run:
        t = compose_turn_prompt(args.mode, ["自我介紹"])
        print(f"[dry-run] {args.mode} 的每輪 system prompt({len(t)} 字):\n")
        print(t)
        print(f"\n整場上限:{MAX_TURNS_PER_SESSION} 題")
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
    print(f"模型:{args.model}   temperature=0.4\n")

    modes = ["single", "panel", "group"] if args.all else [args.mode]
    rows = [await run_mode(m, llm, args.turns) for m in modes]
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
