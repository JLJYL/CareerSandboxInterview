#!/usr/bin/env python3
"""在終端機跑一場群面一題一輪,用真的模型看 AI 應徵者和主持方講什麼。

    python tools/group_round_demo.py --fake                # 先確認管線,不打 API
    python tools/group_round_demo.py --role 理財專員 --company 某銀行 --questions 2

App 還沒接上一題一輪之前,用這支看 prompt 的實際效果。每一輪先生成下一位
AI 的話,再問你要不要搶先:直接按 Enter 讓他講,輸入文字就是你先答——
那一句 AI 的話會丟掉,跟 App 該做的一樣。輪到只剩你的時候會直接等你輸入。

印出的「等 x 秒」是 App 要等的時間,這裡不真的等。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import make_llm  # noqa: E402
from app.pipeline.interview_live import next_turn, start_interview  # noqa: E402
from app.schemas.interview import InterviewContext, UtteranceDTO  # noqa: E402

FAKE_LINES = iter(f"(假回答 {i})我會先把目標講清楚。" for i in range(1, 1000))


def fake_llm(system: str, user: str) -> str:
    """不打 API 的假模型:開場回一題,其餘每次回一句不同的話。"""
    if "開場問題" in system:
        return json.dumps(
            {"openingQuestion": "假設新產品下個月上線,你會先確認哪一件事?",
             "openingTopic": "上線準備", "fallbackProbes": ["預算砍半你會先砍哪裡?"]},
            ensure_ascii=False,
        )
    if "【上一題】" in user:
        return json.dumps(
            {"speaker": "", "nextQuestion": f"下一題:{next(FAKE_LINES)}要怎麼排優先順序?",
             "topic": "優先順序"},
            ensure_ascii=False,
        )
    return json.dumps({"speaker": "", "nextQuestion": next(FAKE_LINES), "topic": ""},
                      ensure_ascii=False)


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
    says = [UtteranceDTO(speaker=start.opening_speaker, content=start.opening_question)]
    hosts = {p.display_name for p in start.personas if p.role != "peer"}
    asked_count = 1

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
            text = input("  (AI 都答完了,輪到你)你:").strip() or "(使用者沒有輸入)"
            says.append(UtteranceDTO(speaker="你", content=text, is_user=True))
            continue

        is_host = r.speaker in hosts
        if is_host and asked_count >= args.questions:
            print("\n(題數到了,結束)")
            return 0
        if not is_host and r.delay_ms:
            text = input(f"  ({r.speaker} 再 {r.delay_ms / 1000:.1f} 秒開口,Enter 讓他講,"
                         "或直接輸入搶先回答)你:").strip()
            if text:
                says.append(UtteranceDTO(speaker="你", content=text, is_user=True))
                print(f"    ({r.speaker} 那句丟掉了)")
                continue
        print(f"[{r.speaker}] {r.next_question}\n")
        says.append(UtteranceDTO(speaker=r.speaker, content=r.next_question))
        if is_host:
            asked_count += 1


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
