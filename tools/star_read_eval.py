"""STAR 事實判讀的準確率,以及一次問六題對上六次各問一題。

## 這支要回答的問題

1d 量到的結論:三種難度的追問率都是 0.500,而「有沒有可查證的數字」一個 bit
就能預測 180/180 次決定。放行條件寫進 prompt 兩次都沒生效。

所以決定本身要改成程式算:`accepts()` 已經能從結構化的 STAR 事實算出放行與否,
缺的只是誰來供應那些事實。這支驗模型供不供得出來。

## 三種條件

    split     六次,各問一題
    bundled   一次,問六題
    assign    句子分派(S/T/A/R)加兩次單題(量化、取捨)

前兩種是為了驗「綁成一包就塌」——那個現象在這個專案出現過三次(派發拆兩步、
決策不帶 persona、gap 的選項說明把追問率從 0.500 推到 0.567)。
實測的答案是:這種純事實判讀不受拆不拆影響,兩者差 0.00,所以用 bundled 省呼叫。

assign 是為了第三件事:**任務段讀不出來**。情境、行動、結果、量化、取捨
五項在兩種條件下都是 1.00,任務是 0.29–0.70。原因是格式——yes/no 加抄原文
是存在性宣稱,找到一句就能說「有」,而任何一段話裡總找得到一句可以被讀成
任務的。改問法沒用(兩種問法都試過),所以改格式。

## 為什麼每一題都要抄原文

封閉集合那一輪的教訓:給選單又附說明,模型會挑一個然後把說明背出來。
`缺情境` 在 A2 與 A3 上出現 20 次,而那兩筆的第一句就是情境——
`why` 幾乎逐字抄 prompt 裡的選項註解,沒有回去對答案。而且越嚴的難度編得越兇
(新手 0 次、中等 4 次、困難 16 次)。

要求抄原文把這件事堵住兩層:

    一   抄不出原文就是沒有,模型沒辦法只靠語氣宣稱「有交代」
    二   抄出來的句子,程式可以直接驗是不是真的在答案裡

第二層是重點。自述沒辦法驗,引文可以——「能用程式算的不要問 LLM」
在這裡的形式就是不問它有沒有說謊,直接查。

## 標準答案

`fixtures/golden/decision/answers_v1.json` 的 `star_parts`、`quantified`、
`tradeoff`。那組標註本來是給 1d 當量尺的,正好是這裡的 ground truth,
不用另外標。

## prompt 不住在這裡

判讀的 prompt 與解析在 `app/prompts/star_read.py` 與 `app/pipeline/star_read.py`,
這支只 import。`assign` 這個條件直接呼叫 `read_facts()`——也就是上線跑的那一支,
不是它的複製品。

這件事不是潔癖:工具量到的設定跟上線跑的設定只要差一點,量到的數字就不是
對線上行為的描述。1d 那輪就踩過一次——`parse_obj` 在工具裡解析失敗回 None、
在 app 裡回空 dict,同一段「說沒有」的輸出在兩邊的落地率不一樣。

這支還留著 `split` 與 `bundled` 兩個條件,那兩個是實驗用的對照組,
不會上線,所以留在工具裡。

用法:
    python tools/star_read_eval.py --fake               # 不打 API,驗工具本身
    python tools/star_read_eval.py --dry-run            # 只印三種 prompt
    python tools/star_read_eval.py --runs 5             # 三種條件都跑(約 350 次呼叫)
    python tools/star_read_eval.py --condition assign   # 只跑分派(約 105 次呼叫)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.interview_live import parse_obj  # noqa: E402
from app.pipeline.interviewer_spec import ACCEPT_RULES, STRUGGLE_FACT  # noqa: E402
from app.pipeline.star_read import (  # noqa: E402
    AnswerFacts,
    FactRead,
    read_facts,
    read_one,
)
from app.pipeline.transcript import segment_text  # noqa: E402
from app.prompts.star_read import (  # noqa: E402
    ASSIGN_SYSTEM,
    FACT_NAME,
    FACT_QUESTION,
    FACT_RULES,
    FACTS,
    assign_user_message,
    compose_fact_prompt,
    fact_user_message,
)
from tools.followup_rate import accepts, load_answers, pad  # noqa: E402

ANSWERS = Path("fixtures/golden/decision/answers_v1.json")
DIFFICULTIES = ("新手", "中等", "困難")

# ---------------------------------------------------------------------------
# 標準答案
# ---------------------------------------------------------------------------


def truth(label: dict, fact: str) -> bool:
    if fact in ("quantified", "tradeoff"):
        return bool(label[fact])
    if fact == STRUGGLE_FACT:
        # 沒標的時候是 False,但那不是答案——呼叫端要先問 `has_truth`。
        # 這一行以前會落到下面的 star_parts 判斷,靜默回 False,
        # 於是「沒有人標過」被量成「標記為沒有」,準確率看起來是真的。
        return bool(label.get(STRUGGLE_FACT))
    return fact in label["star_parts"]


def has_truth(label: dict, fact: str) -> bool:
    """這一筆標註有沒有這一項的人工答案。

    自述困難是後加的第七項,`answers_v1.json` 目前沒有這一欄。沒標的筆
    不計分,不當成「沒有」——否則兩種完全不同的事會量成同一個數字。
    """
    return fact != STRUGGLE_FACT or STRUGGLE_FACT in label


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

def compose_bundled() -> str:
    """一次問七題。只有這支留在工具裡——它是對照組,不會上線。

    問句與抄原文規則都從 `app.prompts.star_read` 來,所以對照組與上線那組
    問的是同一件事,差別只在包成一次還是拆成七次。
    """
    asked = "\n".join(f"    {f}　{FACT_QUESTION[f]}" for f in FACTS)
    shape = ",\n ".join(f'"{f}": {{"present": false, "quote": ""}}' for f in FACTS)
    return "\n\n".join(
        [
            FACT_RULES,
            f"【要判斷的七件事】\n\n{asked}",
            f"""【輸出格式】

只回這個 JSON,七個鍵都要在,不要加任何說明文字:

{{{shape}}}""",
        ]
    )


# ---------------------------------------------------------------------------
# 解析與查證
# ---------------------------------------------------------------------------

def read_bundle(raw: str, answer: str) -> dict[str, FactRead]:
    """拆開一次問六題的輸出。每一格丟回 `read_one` 驗引文,規則與單題一致。"""
    data = parse_obj(raw) or {}
    out: dict[str, FactRead] = {}
    for f in FACTS:
        cell = data.get(f)
        if not isinstance(cell, dict):
            out[f] = FactRead(False, "", False)
            continue
        out[f] = read_one(json.dumps(cell, ensure_ascii=False), answer)
    return out


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


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


def make_fake_llm(answers: list[dict]):
    """照標註完美回答,引文抄答案的開頭。

    驗的是管線:事實準確率應該 1.00、引文落地率應該 1.00。
    不是量測——真模型會不會這樣答就是這支要問的事。
    """
    by_text = sorted(answers, key=lambda a: len(a["text"]), reverse=True)

    def pick(user: str) -> dict:
        for a in by_text:
            if a["text"] in user:
                return a
        raise AssertionError("假模型在 prompt 裡找不到答案——答案沒被帶進去。")

    def cell(a: dict, fact: str) -> dict:
        has = truth(a["label"], fact)
        return {"present": has, "quote": a["text"][:12] if has else ""}

    def assigned(a: dict) -> dict[str, str]:
        """照標註把句子分派回去:S 第一句,T / A / R 依序,多的掛其他。"""
        sentences, _ = segment_text(a["text"], level="sentence")
        order = [f for f in ("S", "T", "A", "R") if truth(a["label"], f)]
        if truth(a["label"], "tradeoff"):
            order = order + ["取捨"]
        out = {str(i + 1): (order[i] if i < len(order) else "其他")
               for i in range(len(sentences))}
        return out

    def call(system: str, user: str) -> str:
        if system is ASSIGN_SYSTEM or system.startswith("你在讀一段面試回答,把每一句"):
            for a in by_text:
                if all(s in user for s in segment_text(a["text"], level="sentence")[0]):
                    return json.dumps(assigned(a), ensure_ascii=False)
            raise AssertionError("假模型在分派 prompt 裡找不到答案的句子。")
        a = pick(user)
        asked = [f for f in FACTS if FACT_QUESTION[f] in system]
        if len(asked) == 1:
            return json.dumps(cell(a, asked[0]), ensure_ascii=False)
        return json.dumps({f: cell(a, f) for f in FACTS}, ensure_ascii=False)

    return call


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------


def throttled(llm, limit: int):
    """限制同時在飛的呼叫數。

    用 threading 的 semaphore 不用 asyncio 的,因為 `read_facts()` 自己
    用 `asyncio.to_thread` 包同步的 llm,外面拿不到它的 await 點。
    限流包在 llm 上三個條件就共用同一個上限,也不必為了限流改 app 的介面。
    """
    gate = threading.BoundedSemaphore(max(1, limit))

    def call(system: str, user: str) -> str:
        with gate:
            return llm(system, user)

    return call


async def run_split(answer: dict, llm) -> dict[str, FactRead]:
    """七次,各問一題。對照組。"""

    async def one(fact: str):
        raw = await asyncio.to_thread(
            llm, compose_fact_prompt(fact), fact_user_message(answer["text"])
        )
        return fact, read_one(raw, answer["text"])

    return dict(await asyncio.gather(*(one(f) for f in FACTS)))


async def run_bundled(answer: dict, llm) -> dict[str, FactRead]:
    """一次,問七題。對照組。"""
    raw = await asyncio.to_thread(
        llm, compose_bundled(), fact_user_message(answer["text"])
    )
    return read_bundle(raw, answer["text"])


async def run_assign(answer: dict, llm) -> dict[str, FactRead]:
    """句子分派加三次單題——直接呼叫上線那一支。

    這裡刻意不自己組那三次呼叫:組法只要跟 `read_facts()` 差一點,
    量到的數字就不是對線上行為的描述。要改判讀就去改 app,這支會跟著動。
    """
    facts = await read_facts(answer=answer["text"], llm=llm)
    return dict(facts.reads)


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------


def print_facts(name: str, reads: dict[str, list[dict]], answers) -> dict[str, float]:
    print(f"\n【{name}:每一項事實的準確率】\n")
    print(pad("答案", 6, False) + "".join(pad(FACT_NAME[f], 9) for f in FACTS) + pad("整列", 7))
    per_fact: dict[str, list[float]] = {f: [] for f in FACTS}
    no_struggle: list[float] = []  # 人工標記為「沒有自述困難」的那些筆
    for a in answers:
        row = pad(a["id"], 6, False)
        scored = [f for f in FACTS if has_truth(a["label"], f)]
        for f in FACTS:
            if f not in scored:
                row += pad("—", 9)
                continue
            want = truth(a["label"], f)
            hits = [1.0 if r[f][0] == want else 0.0 for r in reads[a["id"]]]
            per_fact[f] += hits
            if f == STRUGGLE_FACT and not want:
                no_struggle += hits
            row += pad(f"{statistics.mean(hits):.2f}", 9)
        whole = [
            1.0 if all(r[f][0] == truth(a["label"], f) for f in scored) else 0.0
            for r in reads[a["id"]]
        ]
        row += pad(f"{statistics.mean(whole):.2f}", 7)
        print(row)

    print("\n" + pad("合計", 6, False) + "".join(
        pad(f"{statistics.mean(per_fact[f]):.2f}" if per_fact[f] else "—", 9)
        for f in FACTS
    ))
    if per_fact[STRUGGLE_FACT]:
        n_labeled = sum(1 for a in answers if has_truth(a["label"], STRUGGLE_FACT))
        print(f"\n  {FACT_NAME[STRUGGLE_FACT]}:有人工標記 {n_labeled}/{len(answers)} 筆"
              f"(判讀 {len(per_fact[STRUGGLE_FACT])} 次)")
        if no_struggle:
            # 派發只在「沒有」的時候觸發:誤判成「有」讓 HR 拿不到回合(退回現況),
            # 誤判成「沒有」讓 HR 拿到不該拿的回合。兩種錯的代價不對稱,所以分開看。
            print(f"  └ 標記為沒有自述困難的那些筆上的準確率:"
                  f"{statistics.mean(no_struggle):.2f}(判讀 {len(no_struggle)} 次)")
        else:
            print("  └ 沒有任何一筆標記為「沒有自述困難」,這個數字量不出來。")
    else:
        print(f"\n  {FACT_NAME[STRUGGLE_FACT]}:標準答案裡沒有這一欄,不計分。"
              f"引文落地率照量(見下)。")
    return {f: statistics.mean(per_fact[f]) for f in FACTS if per_fact[f]}


def print_quotes(name: str, reads: dict[str, list[dict]], answers) -> float:
    claims = 0
    landed = 0
    invented: list[tuple[str, str, str]] = []
    for a in answers:
        for r in reads[a["id"]]:
            for f in FACTS:
                ok, quote, g = r[f]
                if quote:
                    claims += 1
                    landed += 1 if g else 0
                    if not g:
                        invented.append((a["id"], FACT_NAME[f], quote))
    rate = landed / claims if claims else 1.0
    print(f"\n【{name}:引文落地率】 {landed}/{claims} = {rate:.1%}")
    if invented:
        print("  抄不到原文的前幾筆(宣稱有、引文不在答案裡):")
        for aid, fact, quote in invented[:5]:
            print(f"    {aid} {fact}:{quote[:40]}")
    return rate


def print_verdicts(name: str, reads: dict[str, list[dict]], answers) -> float:
    """把判讀餵進 accepts(),看放行判斷會不會因此對。

    這是真正要的答案:事實錯一項不一定改變判斷,而判斷才是追問率。
    """
    print(f"\n【{name}:把判讀餵進 accepts() 之後】\n")
    print(
        pad("難度", 6, False)
        + pad("判斷一致率", 12)
        + pad("推出的追問率", 14)
        + pad("標準答案", 10)
    )
    overall: list[float] = []
    for diff in DIFFICULTIES:
        rule = ACCEPT_RULES[diff]
        agree: list[float] = []
        got_fu: list[float] = []
        for a in answers:
            want = accepts(rule, a["label"])
            for r in reads[a["id"]]:
                mine = AnswerFacts(reads=r).accepted_by(rule)
                agree.append(1.0 if mine == want else 0.0)
                got_fu.append(0.0 if mine else 1.0)
        exp = statistics.mean(
            0.0 if accepts(rule, a["label"]) else 1.0 for a in answers
        )
        print(
            pad(diff, 6, False)
            + pad(f"{statistics.mean(agree):.2f}", 12)
            + pad(f"{statistics.mean(got_fu):.3f}", 14)
            + pad(f"{exp:.3f}", 10)
        )
        overall += agree
    return statistics.mean(overall)


def conclude(facts: dict[str, dict[str, float]], verdicts: dict[str, float]) -> None:
    print("\n" + "=" * 74)
    if len(facts) < 2:
        print("只跑了一種條件,沒辦法比。")
        return

    conds = list(facts)
    print(pad("事實", 6, False) + "".join(pad(c, 11) for c in conds))
    for f in FACTS:
        print(pad(FACT_NAME[f], 6, False)
              + "".join(pad(f"{facts[c][f]:.2f}", 11) for c in conds))
    print(pad("平均", 6, False)
          + "".join(pad(f"{statistics.mean(facts[c].values()):.2f}", 11) for c in conds))
    print("\n" + pad("判斷一致率", 6, False)
          + "".join(pad(f"{verdicts[c]:.2f}", 11) for c in conds))

    # 任務是唯一卡住的一項,所以結論看它,不看平均。
    best = max(conds, key=lambda c: facts[c]["T"])
    print()
    print(f"任務最高的是 {best}:{facts[best]['T']:.2f}")
    if facts[best]["T"] >= 0.9:
        print("★ 任務判讀通了。六項事實都夠準,決定可以整個交給 accepts() 算。")
        print("  判讀已經在 app/pipeline/star_read.py,這個條件跑的就是那一支。")
        print("  還差兩步:accepts() 搬進 interviewer_spec,decide_turn 改成吃事實。")
    elif "assign" in facts and facts["assign"]["T"] - max(
        facts[c]["T"] for c in conds if c != "assign"
    ) >= 0.2:
        print("分派有改善但還不夠。互斥約束是對的方向,再加強約束值得試一次。")
    else:
        print("✗ 換格式也沒用。任務段讀不出來這件事,三種格式、兩種問法都一樣。")
        print("  那就是結論:放行條件不能建立在讀不出來的事實上。")
        print("  下一步是把 T 從段數裡拿掉,只數 S/A/R——其餘五項都已經夠準。")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, default=ANSWERS)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="預設 0:判讀是事實抽取不是生成,要可重現。")
    ap.add_argument("--condition", choices=("split", "bundled", "assign", "all"), default="all")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()

    if args.runs < 1:
        raise SystemExit("✗ --runs 至少要 1。")
    if args.dump:
        try:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            args.dump.touch()
        except OSError as exc:
            raise SystemExit(f"✗ 寫不到 {args.dump}:{exc}") from exc

    answers = load_answers(args.answers, "api")

    if args.dry_run:
        print("=" * 74)
        print(compose_fact_prompt("T"))
        print("\n" + "=" * 74)
        print(compose_fact_prompt(STRUGGLE_FACT))
        print("\n" + "=" * 74)
        print(compose_bundled())
        print("\n" + "=" * 74)
        print(ASSIGN_SYSTEM)
        print("\n" + "-" * 74)
        print(assign_user_message(segment_text(answers[-1]["text"], level="sentence")[0]))
        return 0

    if not args.fake:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ 找不到 OPENAI_API_KEY。放在專案根目錄的 .env 裡,或設成環境變數。")
            print("  想先確認工具本身對不對,用 --fake(不打 API)。")
            return 1
    else:
        print("⚠ --fake:這不是量測。假模型照標註完美回答,")
        print("  所以準確率與引文落地率都該是 1.00——只證明管線對。\n")

    runs = 1 if args.fake else args.runs
    conditions = ("split", "bundled", "assign") if args.condition == "all" else (args.condition,)
    _calls = {"split": len(FACTS), "bundled": 1, "assign": 4}
    n = sum(_calls[c] for c in conditions) * runs * len(answers)
    print(f"條件 {'+'.join(conditions)} × 回合 {runs} × 答案 {len(answers)} "
          f"= {n} 次呼叫(temp={args.temp})")

    reads: dict[str, dict[str, list[dict]]] = {}
    log: list[dict] = []

    async def go():
        llm = throttled(
            make_fake_llm(answers) if args.fake else make_llm(args.model, args.temp),
            args.concurrency,
        )
        for cond in conditions:
            runner = {"split": run_split, "bundled": run_bundled, "assign": run_assign}[cond]
            got: dict[str, list[dict]] = {a["id"]: [] for a in answers}
            jobs = [(a, runner(a, llm)) for _ in range(runs) for a in answers]
            for a, job in jobs:
                got[a["id"]].append(await job)
            reads[cond] = got
            for a in answers:
                for i, r in enumerate(got[a["id"]]):
                    for f in FACTS:
                        ok, quote, g = r[f]
                        log.append(
                            {
                                "條件": cond, "回合": i, "答案": a["id"],
                                "事實": FACT_NAME[f],
                                "判讀": int(ok),
                                "標準": (int(truth(a["label"], f))
                                         if has_truth(a["label"], f) else ""),
                                "引文落地": int(g), "引文": quote,
                            }
                        )

    asyncio.run(go())

    facts: dict[str, dict[str, float]] = {}
    verdicts: dict[str, float] = {}
    for cond in conditions:
        facts[cond] = print_facts(cond, reads[cond], answers)
        print_quotes(cond, reads[cond], answers)
        if cond == "assign":
            print("  (分派模式的 S/T/A/R 引文由程式供給,落地必然成立,不是量測)")
        verdicts[cond] = print_verdicts(cond, reads[cond], answers)

    if args.dump:
        cols = ("條件", "回合", "答案", "事實", "判讀", "標準", "引文落地", "引文")
        lines = ["\t".join(cols)]
        lines += ["\t".join(str(r[c]).replace("\t", " ") for c in cols) for r in log]
        try:
            args.dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\n{len(log)} 筆判讀 → {args.dump}")
        except OSError as exc:
            print(f"\n⚠ 寫不到 {args.dump}({exc}),下面的結論仍然有效。")

    conclude(facts, verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
