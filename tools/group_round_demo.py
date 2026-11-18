#!/usr/bin/env python3
"""在終端機跑一場群面一題一輪,用真的模型看 AI 應徵者和主持方講什麼。

    python tools/group_round_demo.py --fake                # 先確認管線,不打 API
    python tools/group_round_demo.py --role 理財專員 --company 某銀行 --questions 2

不開 App 也能看 prompt 的實際效果。主考官出題時點第一位,每位答完再點下一位。
每一則(點名句或 AI 的回答)出來之前會問你要不要搶先:直接按 Enter 讓它出來,
輸入文字就是你先答——那一則會丟掉,跟 App 做的一樣。主考官點到你時直接等你輸入。

印出的「等 x 秒」是 App 要等的時間(上一則講完加停頓),這裡不真的等。
加 --notices 會看到點名順序、每位分到的切入角度,和「太像、已重生成」這類紀錄。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import make_llm  # noqa: E402
from app.pipeline.interview_live import next_turn, start_interview  # noqa: E402
from app.schemas.interview import InterviewContext, UtteranceDTO  # noqa: E402

FAKE_BY_ANGLE = {
    "結論先行": "結論先講:我選先做小規模試辦,兩週就看得到結果。",
    "講第一步": "我明天第一件事是約三位資深同事,一小時內把現況盤點出來。",
    "看數字": "我會看第一週的轉換率,低於百分之三就停掉換方案。",
    "看風險": "最壞是上線後客訴爆量,所以我會先留一條人工處理的退路。",
    "看成本": "人力只有兩個人,我寧願砍掉報表功能,保住核心流程。",
    "從使用者看": "客戶真正在意的是等多久,所以我先把排隊時間砍半。",
    "從團隊看": "我會先讓業務和工程坐下來對一次目標,不然做出來沒人用。",
    "點出前提": "這題假設預算固定,如果可以追加,我會先爭取人力而不是砍功能。",
    "舉經驗": "我之前辦活動也遇過,當時先找十個人試,改完才全面推。",
    "唱反調": "我跟前面的想法相反,我會先停下來把需求問清楚再動。",
}
FAKE_QUESTIONS = iter([
    "下一題:如果團隊裡有人一直不配合,你會怎麼處理?",
    "下一題:上線前一天發現重大錯誤,你會怎麼做?",
    "下一題:主管和客戶的要求互相衝突,你會聽誰的?",
] + [f"下一題:第 {i} 個假題目,你會怎麼排優先順序?" for i in range(4, 50)])
ANGLE_RE = re.compile(r"【你這一題的切入角度】([^::\n]+)[::]")
QUESTION_RE = re.compile(r"問的這一題】(.+)")


def fake_llm(system: str, user: str) -> str:
    """不打 API 的假模型。應徵者照分到的切入角度回一句,看得出每個人不一樣。"""
    if "開場問題" in system:
        return json.dumps(
            {"openingQuestion": "假設新產品下個月上線,你會先確認哪一件事?",
             "openingTopic": "上線準備", "fallbackProbes": ["預算砍半你會先砍哪裡?"]},
            ensure_ascii=False,
        )
    if "【上一題】" in user:
        return json.dumps(
            {"speaker": "", "nextQuestion": next(FAKE_QUESTIONS), "topic": "下一題"},
            ensure_ascii=False,
        )
    m = ANGLE_RE.search(user)
    angle = m.group(1) if m else ""
    q = QUESTION_RE.search(user)
    about = q.group(1).strip()[:12] if q else ""
    line = f"({angle or '假'}){about}——{FAKE_BY_ANGLE.get(angle, '我會先把目標講清楚。')}"
    return json.dumps({"speaker": "", "nextQuestion": line, "topic": ""}, ensure_ascii=False)


async def main_async(args) -> int:
    if args.fake:
        llm = exact = fake_llm
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("找不到 OPENAI_API_KEY。放在專案根目錄的 .env 裡,或設成環境變數。")
            print("想先確認工具本身對不對,用 --fake(不打 API)。")
            return 1
        llm = make_llm(args.model, 0.4)
        exact = make_llm(args.model, 0.0)

    ctx = InterviewContext(
        custom_role=args.role, custom_company=args.company,
        group_size=args.size, group_flow="round",
    )
    start = await start_interview(mode="group", context=ctx, llm=llm, llm_exact=exact)
    print(f"\n{start.opening_intro}\n")
    print(f"[{start.opening_speaker}] {start.opening_question}\n")
    if args.notices:
        for n in start.notices:
            print(f"    · {n}")
    says = [UtteranceDTO(speaker=start.opening_speaker, content=start.opening_question)]
    hosts = {p.display_name for p in start.personas if p.role != "peer"}
    asked_count = 1
    answered = False

    while True:
        r = await next_turn(
            mode="group", answer="", follow_up_idx=len(says),
            asked_questions=[start.opening_question], fallback=start.fallback_probes,
            llm=llm, context=ctx, group_says=says,
            group_flow="round", session_id=start.session_id,
        )
        if args.notices:
            for n in r.notices:
                print(f"    · {n}")
        if r.await_user:
            text = input("  (主考官點到你了)你:").strip() or "(使用者沒有輸入)"
            says.append(UtteranceDTO(speaker="你", content=text, is_user=True))
            answered = True
            continue

        new_question = r.speaker in hosts and not r.cue
        if new_question and asked_count >= args.questions:
            print("\n(題數到了,結束)")
            return 0
        if not new_question and not answered and r.delay_ms:
            what = "點名" if r.cue else "開口"
            text = input(f"  ({r.speaker} 再 {r.delay_ms / 1000:.1f} 秒{what},Enter 繼續,"
                         "或直接輸入搶先回答)你:").strip()
            if text:
                says.append(UtteranceDTO(speaker="你", content=text, is_user=True))
                answered = True
                print(f"    ({r.speaker} 那句丟掉了)")
                continue
        print(f"[{r.speaker}] {r.next_question}\n")
        says.append(UtteranceDTO(speaker=r.speaker, content=r.next_question))
        if new_question:
            asked_count += 1
            answered = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="理財專員")
    ap.add_argument("--company", default="")
    ap.add_argument("--size", type=int, default=4, help="小組人數,含你,3–5。")
    ap.add_argument("--questions", type=int, default=2, help="跑幾題就停。")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--notices", action="store_true", help="印出每一輪的 notices。")
    ap.add_argument("--fake", action="store_true", help="不打 API,只確認管線。")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
