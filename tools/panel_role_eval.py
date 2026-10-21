#!/usr/bin/env python3
"""量 panel 三位面試官的角色有沒有反映在題目上。

    python3 tools/panel_role_eval.py --fake            # 先確認管線
    python3 tools/panel_role_eval.py --rounds 3 --dump data/panel_v2.json

【第一版的缺陷,這一版修的就是它】
第一版讓派發決定誰開口,結果五場的派發順序完全一樣:

    HR 主管 → 技術主管 → 用人主管 → 技術主管

而回答是固定四段。於是 HR 永遠只看到第 0 段、用人主管永遠只看到第 2 段。
「同角色」變成「同一段回答」,「跨角色」變成「不同回答」,量到的
0.195 差距是**回答不同**造成的,不是角色不同。那份資料答不了原本的問題。

這一版把**同一段回答餵給三位**,用 `next_turn(force_speaker=...)` 繞過派發。

【三個數字,分別回答三件事】

    角色效果   同一段回答,三位之間的重疊
               低 = 三位問的東西真的不一樣

    腳本化     同一位,跨不同回答的重疊
               高 = 這一位不管聽到什麼都問同一句

    貼合度     問題與它所回應的那段回答之間的重疊
               低 = 這句問題換到別段回答後面也成立

【第三個數字是後來補的,因為前兩個漏掉了 HR】
第二版量到 HR 腳本化 0.241——三位裡最低,看起來最不罐頭。但把第一節
並排讀下來,它只有在「團隊衝突」那一段問到自己的東西,另外三段都是
「有沒有遇到什麼挑戰」這一類誰都問得出來的話。

它每次換詞,所以跨回答的重疊低;而那些詞沒有一個來自回答本身。
前兩個數字都看不到這件事:角色效果看的是三位之間,腳本化看的是同一位
跨回答之間,兩個都沒拿回答當比較對象。

腳本化低而貼合度也低,就是這個樣子。單看貼合度不夠——問題本來就比
回答短,而且可以換詞咬住同一件事,所以這個數字只在**同一段回答上
比較三位**時可讀,不要跨回答比,也不要當絕對門檻。

第一版量到 HR 自己跟自己 0.709,而它跟任何人的重疊都只有 0.26–0.34。
那不是「HR 跟用人主管撞了」,是 HR 只會問一句話。但第一版分不出那是
人格描述沒生效,還是派發把它配到一段沒東西可問的回答上——
這一版兩個都量得到。

【回答刻意涵蓋四種觸發】
三位的 `routes_when` 各自吃不同的關鍵字。回答只有一種類型的話,
另外兩位本來就沒東西可問,那樣量到的「角色沒差別」是假的。
所以 ANSWERS 四段分別踩團隊衝突、數據分析、取捨決定、以及一段
三個關鍵字都不踩的泛泛而談。

【重疊是字面不是語意】
用的是 `is_near_duplicate` 那一套。它看不出「你怎麼跟工程師溝通」與
「跨部門合作的經驗」講的是同一件事,所以這個數字只是**下界**。
判斷要看第一節那份並排,那是人的工作。

【呼叫數】
每個 round:1 次開場 + 4 段 × (3 次事實判讀 + 3 位各一次生成)= 25 次。
預設 3 rounds 約 75 次。

`--force-next` 不跑事實判讀(決定是指定的),所以每個 round 13 次,
3 rounds 約 39 次。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline.interview_live import (  # noqa: E402
    _STOPWORDS,
    next_turn,
    start_interview,
)
from app.pipeline.interviewer_spec import derive_specs  # noqa: E402
from app.pipeline.turn_decision import (  # noqa: E402
    CODE,
    NEXT,
    TurnDecision,
    decide_turn,
)
from app.prompts.interview_personas import speaker_names  # noqa: E402
from app.schemas.interview import InterviewContext  # noqa: E402

CTX = InterviewContext(
    custom_role="資料分析實習生",
    custom_company="某電商",
    custom_industry="電子商務",
    custom_jd="需具備 SQL 撰寫能力,熟悉報表自動化,能與跨部門溝通需求。",
    difficulty="中等",
    transcription_engine="api",
)

JD = JDInput(required_skills=["SQL", "報表自動化", "跨部門溝通"], source="catalog")

#: 四段回答,刻意踩不同的 routes_when。
#: 每一段都會餵給三位,所以每一位都會遇到「自己的題材」與「別人的題材」。
ANSWERS: list[tuple[str, str]] = [
    (
        "團隊衝突",
        "有一次行銷想要的欄位跟工程師能給的對不上,兩邊在會議上吵了起來。"
        "我把兩邊的需求列成一張表,找出重疊的那三個欄位先做,"
        "剩下的排到下一期。後來那個做法變成我們提需求的固定流程。",
    ),
    (
        "數據分析",
        "實習的時候我用 SQL 重寫了每週報表的查詢。原本人工整理要三小時,"
        "改完之後剩二十分鐘。我有先比對一週的舊報表確認數字一致才上線。",
    ),
    (
        "取捨決定",
        "那一季我同時接到兩個需求,但只有兩週。我選了退貨分析先做,"
        "因為那一塊每週要人工看兩小時,另一個是一次性的。"
        "代價是行銷那邊的活動成效晚了三週才拿到。",
    ),
    (
        "泛泛而談",
        "我覺得做分析最難的是知道該問什麼問題。數字本身不會告訴你哪裡有問題,"
        "要先知道想驗證什麼。這件事我還在學。",
    ),
]


def make_llm(model: str, temp: float):
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
        )
        return resp.choices[0].message.content or ""

    return call


def fake_llm(system: str, user: str) -> str:
    """不打 API 的假模型。只證明管線接得起來,**不是量測**。

    三位回同一句,所以角色效果必然是 1.000 —— 那正是「分不開」的樣子。
    拿這個數字去宣稱角色沒有區分是錯的。
    """
    if "開場問題" in system:
        return json.dumps(
            {"openingQuestion": "x", "openingTopic": "自我介紹",
             "fallbackProbes": ["備援一", "備援二"]}, ensure_ascii=False)
    if "這一次只判斷一件事" in system:
        return json.dumps({"present": True, "quote": "我把兩邊的需求列成一張表"},
                          ensure_ascii=False)
    if "把每一句分派到一個標籤" in system:
        return json.dumps({"1": "A"}, ensure_ascii=False)
    if "接話" in system:
        return json.dumps({"nextQuestion": "那一段可以再多講一點嗎?",
                           "topic": "團隊合作"}, ensure_ascii=False)
    return "{}"


def overlap(a: str, b: str) -> float:
    """兩句問題的字面重疊比例。跟 `is_near_duplicate` 同一套。"""
    sa, sb = set(a) - _STOPWORDS, set(b) - _STOPWORDS
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def run_round(llm, exact, rnd: int, force_next: bool = False) -> list[dict]:
    """跑一輪。每一段回答都問過三位。

    `force_next` 把決定改寫成 NEXT,不問事實判讀。

    【為什麼需要這個開關】
    NEXT 那條路是另一種行為:它要模型換話題。v4 與 v5 兩次各 36 筆,
    NEXT 觸發 0 次——這四段回答都夠完整,決定一律是 FOLLOW_UP。
    也就是說平常跑這支工具**量不到 NEXT**。

    v3 偶然量到六筆(數據分析那段剛好三句,坐在 `min_star_parts` 門檻上
    會翻),而那六筆就是三位一起塌成「有沒有遇到什麼挑戰」的那六筆。
    靠門檻翻動來觀察一條路徑不是量測,是運氣,所以改成能指定。

    ANSWERS 不動,因為 v2–v5 的角色效果是拿同一組回答比出來的。
    加一段會觸發 NEXT 的短回答會讓那條時間線斷掉。
    """
    start = await start_interview(
        mode="panel", context=CTX, llm=llm, llm_exact=exact, jd=JD
    )
    spec = derive_specs("panel", jd=JD, difficulty=CTX.difficulty)[0]
    roles = list(speaker_names("panel"))

    out: list[dict] = []
    for idx, (kind, answer) in enumerate(ANSWERS):
        # 決定只看回答,跟誰問無關,所以三位共用同一個——
        # 每位各算一次的話會多燒兩倍的事實判讀,而結果一樣。
        if force_next:
            decision = TurnDecision(NEXT, "工具指定,不是判讀出來的", source=CODE)
        else:
            decision = await decide_turn(spec=spec, answer=answer, llm=exact, depth=0)
        for role in roles:
            resp = await next_turn(
                mode="panel",
                answer=answer,
                follow_up_idx=idx,
                asked_questions=[start.opening_question],
                spoken_by=[],
                question=start.opening_question,
                prev_topic=start.opening_topic,
                fallback=list(start.fallback_probes),
                llm=llm,
                context=CTX,
                decision=decision,
                jd=JD,
                force_speaker=role,
            )
            out.append({
                "round": rnd,
                "answer_kind": kind,
                "role": role,
                "spoke": resp.speaker,
                "question": resp.next_question,
                "is_follow_up": resp.is_follow_up,
                # NEXT 的問題會引用上一段回答,所以光看問句分不出它到底換題
                # 了沒有。topic 換了才是換題;沒換就表示模型其實在追問,
                # 那時 NEXT 這個標籤是錯的,而 is_follow_up 跟著錯。
                "topic": resp.topic,
                "prev_topic": start.opening_topic,
                "action": decision.action,
            })
    return out


def report(rows: list[dict]) -> None:
    roles = sorted({r["role"] for r in rows})
    kinds = [k for k, _ in ANSWERS]

    print("\n" + "=" * 74)
    print("一、同一段回答,三位分別問了什麼")
    print("=" * 74)
    for kind in kinds:
        print(f"\n【回答類型:{kind}】")
        for role in roles:
            qs = [r["question"] for r in rows
                  if r["answer_kind"] == kind and r["role"] == role]
            for i, q in enumerate(qs):
                print(f"  {role if i == 0 else '':10} {q}")

    wrong = [r for r in rows if r["spoke"] != r["role"]]
    if wrong:
        print(f"\n⚠ {len(wrong)} 筆指定了說話者但回的是別人,那幾筆不可信。")

    print("\n" + "=" * 74)
    print("二、角色效果:同一段回答,三位之間有多像")
    print("=" * 74)
    print("   低 = 角色有反映在題目上\n")
    per_kind = {}
    for kind in kinds:
        vals = []
        for r1, r2 in combinations(roles, 2):
            a = [r["question"] for r in rows
                 if r["answer_kind"] == kind and r["role"] == r1]
            b = [r["question"] for r in rows
                 if r["answer_kind"] == kind and r["role"] == r2]
            vals += [overlap(x, y) for x in a for y in b]
        per_kind[kind] = mean(vals)
        print(f"  {kind:10} {per_kind[kind]:.3f}")
    role_effect = mean(list(per_kind.values()))
    print(f"\n  平均 {role_effect:.3f}")

    print("\n" + "=" * 74)
    print("三、腳本化:同一位,跨不同回答有多像")
    print("=" * 74)
    print("   高 = 這一位不管聽到什麼都問同一句\n")
    for role in roles:
        qs = [r["question"] for r in rows if r["role"] == role]
        same = len(qs) - len(set(qs))
        v = mean([overlap(a, b) for a, b in combinations(qs, 2)])
        tag = f"   一字不差重複 {same} 次" if same else ""
        print(f"  {role:10} {v:.3f}{tag}")

    print("\n" + "=" * 74)
    print("四、貼合度:問題有沒有咬住那段回答")
    print("=" * 74)
    print("   低 = 這句問題換到別段回答後面也成立,也就是沒在問眼前這個人\n")
    anchor: dict[str, float] = {}
    for role in roles:
        per = []
        for kind, answer in ANSWERS:
            qs = [r["question"] for r in rows
                  if r["role"] == role and r["answer_kind"] == kind]
            per.append((kind, mean([overlap(q, answer) for q in qs])))
        anchor[role] = mean([v for _, v in per])
        cells = "  ".join(f"{k} {v:.3f}" for k, v in per)
        print(f"  {role:10} {anchor[role]:.3f}   {cells}")
    loose = min(roles, key=lambda r: anchor[r])
    print(f"\n  平均最低的是 {loose}。")

    nexts = [r for r in rows if r.get("action") == "NEXT"]
    if nexts:
        print("\n" + "=" * 74)
        print("五、換題:NEXT 有沒有真的換掉話題")
        print("=" * 74)
        print("   新版的 NEXT 會引用上一段回答,所以光看問句分不出它換題了沒有。")
        print("   topic 跟開場那題一樣 = 其實在追問,那時 NEXT 這個標籤是錯的。\n")
        same = [r for r in nexts if r["topic"] == r["prev_topic"]]
        print(f"  {len(nexts)} 筆 NEXT,其中 {len(same)} 筆 topic 沒換")
        for r in nexts:
            flag = "  ← 沒換" if r["topic"] == r["prev_topic"] else ""
            print(f'    {r["role"]:10} {r["answer_kind"]:5} topic={r["topic"]}{flag}')
        print("\n   第四節的貼合度在這一輪特別重要:NEXT 的問題如果不咬回答,")
        print("   就是舊版那個「三位一起問有沒有遇到什麼挑戰」還在。")

    print("\n" + "=" * 74)
    print("怎麼讀")
    print("=" * 74)
    if role_effect < 0.35:
        print("  角色效果低 → 三位問的東西不一樣,角色有生效。")
    elif role_effect < 0.5:
        print("  角色效果中等 → 有差但不明顯,看第一節。")
    else:
        print("  角色效果高 → 三位在問同一件事,角色沒生效。")

    worst = max(roles, key=lambda r: mean([
        overlap(a, b) for a, b in
        combinations([x["question"] for x in rows if x["role"] == r], 2)
    ]))
    print(f"  腳本化最嚴重的是 {worst}。第三節那個數字如果比它跟別人的")
    print("  重疊還高,代表它只會問一句話,那跟角色區分是兩個不同的問題。")
    print(f"\n  {loose} 的貼合度最低。腳本化低而貼合度也低是最壞的組合:")
    print("  每次換詞所以第三節好看,但每一句都沒咬住回答——那是通用問句,")
    print("  不是這個角色的問題。第二版的 HR 就是這樣漏過去的。")
    print("\n  三個數字都是字面比對。問題可以用不同的詞問同一件事,也可以")
    print("  用不同的詞咬住同一段回答,那樣數字都會騙人。最終要看第一節。")


async def main_async(args) -> int:
    if args.fake:
        print("⚠ --fake:這不是量測。三位會回同一句,所以角色效果必然是 1.000,")
        print("  那只證明管線跑得起來。要真的量要打 API。\n")
        llm = exact = fake_llm
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ 找不到 OPENAI_API_KEY。放在專案根目錄的 .env 裡,或設成環境變數。")
            print("  想先確認工具本身對不對,用 --fake(不打 API)。")
            return 1
        llm = make_llm(args.model, args.temp)
        exact = make_llm(args.model, 0.0)

    rows: list[dict] = []
    for r in range(args.rounds):
        print(f"  第 {r + 1}/{args.rounds} 輪…", flush=True)
        try:
            rows += await run_round(llm, exact, r, force_next=args.force_next)
        except Exception as exc:  # noqa: BLE001
            print(f"    這一輪失敗,跳過:{exc}")

    if not rows:
        print("一輪都沒跑成功。")
        return 1

    report(rows)
    if args.dump:
        args.dump.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始輸出寫到 {args.dump}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3,
                    help="每輪會把四段回答各問過三位。")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0.4, help="對齊 get_llm_live。")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--fake", action="store_true",
                    help="不打 API,只確認管線跑得起來。不是量測。")
    ap.add_argument("--force-next", action="store_true",
                    help="把決定改寫成 NEXT。平常這四段回答一律判 FOLLOW_UP,"
                         "不指定就量不到換題那條路。這一輪的數字不要跟"
                         "平常的拿來比,兩者在量不同的行為。")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
