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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.interviewer_spec import ACCEPT_RULES  # noqa: E402
from tools.followup_rate import accepts, load_answers, pad  # noqa: E402

ANSWERS = Path("fixtures/golden/decision/answers_v1.json")
DIFFICULTIES = ("新手", "中等", "困難")

FACTS: tuple[str, ...] = ("S", "T", "A", "R", "quantified", "tradeoff")

FACT_NAME: dict[str, str] = {
    "S": "情境",
    "T": "任務",
    "A": "行動",
    "R": "結果",
    "quantified": "量化",
    "tradeoff": "取捨",
}

FACT_QUESTION: dict[str, str] = {
    "S": "這段回答有沒有交代當時的背景或處境——事情是在什麼情況下發生的?",
    # 原本問「有沒有交代他要達成什麼」,split 模式在 T 不存在的四筆上 0.00——
    # 每一次都答「有」。一題單獨問,任何回答都能被讀成隱含一個目標:
    # 「我把訂單資料接進資料庫」可以解釋成「目標是自動化報表」,
    # 然後抄行動那句話當證據。引文落地率 100% 配上準確率 0.33,
    # 說明抄原文消滅的是編造證據,不是張冠李戴。所以改成點名這個失敗模式。
    "T": (
        "他有沒有**說出**別人要他做什麼,或者他當時的目標是什麼?"
        "只算他明講的那一句。"
        "不要從他做了什麼回頭推論目標——「我做了 X」不等於「我的任務是 X」。"
    ),
    "A": "這段回答有沒有交代他自己實際做了什麼?",
    "R": "這段回答有沒有交代最後怎麼樣——這件事的結果?",
    "quantified": "這段回答講的結果裡,有沒有可查證的數字?",
    "tradeoff": "這段回答有沒有交代他放棄了什麼,以及當時的判斷依據?",
}

_missing = set(FACTS) ^ set(FACT_QUESTION) | set(FACTS) ^ set(FACT_NAME)
if _missing:
    raise RuntimeError(f"FACTS 與問句/名稱對不上:{sorted(_missing)}")


# ---------------------------------------------------------------------------
# 標準答案
# ---------------------------------------------------------------------------


def truth(label: dict, fact: str) -> bool:
    if fact in ("quantified", "tradeoff"):
        return bool(label[fact])
    return fact in label["star_parts"]


def as_label(facts: dict[str, bool]) -> dict:
    """把判讀出來的事實擺成 accepts() 吃的形狀。"""
    return {
        "star_parts": [f for f in ("S", "T", "A", "R") if facts.get(f)],
        "quantified": bool(facts.get("quantified")),
        "tradeoff": bool(facts.get("tradeoff")),
    }


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

_RULES = """你在讀一段面試回答,只回報事實,不評價好壞,不給建議。

【抄原文的規則】

說「有」的時候,一定要從回答裡抄一句原文當證據,一字不改。
抄不出原文就是「沒有」。不要改寫、不要摘要、不要自己補上他沒說的話。"""

_ONE_OUTPUT = """【輸出格式】

只回這個 JSON,不要加任何說明文字:

{"present": true, "quote": "從回答裡抄的一句原文"}

    present  true 或 false
    quote    present 為 true 時必填,一字不改地抄。false 填空字串。"""


def compose_split(fact: str) -> str:
    return "\n\n".join(
        [_RULES, f"【這一次只判斷一件事】\n\n{FACT_QUESTION[fact]}", _ONE_OUTPUT]
    )


def compose_bundled() -> str:
    asked = "\n".join(f"    {f}　{FACT_QUESTION[f]}" for f in FACTS)
    shape = ",\n ".join(f'"{f}": {{"present": false, "quote": ""}}' for f in FACTS)
    return "\n\n".join(
        [
            _RULES,
            f"【要判斷的六件事】\n\n{asked}",
            f"""【輸出格式】

只回這個 JSON,六個鍵都要在,不要加任何說明文字:

{{{shape}}}""",
        ]
    )


def user_message(answer: str) -> str:
    return f"【他剛剛說的】\n{answer}"


# ---------------------------------------------------------------------------
# 句子分派
# ---------------------------------------------------------------------------

STAR_LABELS: tuple[str, ...] = ("S", "T", "A", "R", "取捨", "其他")
"""分派用的標籤。

【為什麼取捨要有自己的標籤】
第一輪分派把任務從 0.29 拉到 0.86,只剩 A7 失敗,而且 5/5 都把取捨那一句
掛成 T:「我判斷兩個月的時間只夠做一件事」讀起來就是一個目標。

同一份記錄裡,單題呼叫對同一句話的判讀是「取捨」而且正確。模型認得它,
是分派的標籤集合裡沒有它的位置——只能掛「其他」,而垃圾桶比不過
看起來很貼的 T。

取捨這個事實還是從單題問法來(那一項一直是 1.00),這個標籤的唯一作用
是讓那句話有地方去,T 就沒東西可搶。
"""

ASSIGN_SYSTEM = f"""你在讀一段面試回答,把每一句分派到一個標籤。只分派,不評價。

【標籤】

    S     當時的背景或處境
    T     他要達成什麼——別人交付給他的任務,或他自己的目標
    A     他實際做了什麼
    R     這件事最後怎麼樣
    取捨  他放棄了什麼,以及當時的判斷依據
    其他  以上都不是

【規則】

**一句只能掛一個標籤。** 一句話同時像行動又像任務的時候,挑比較貼的那一個,
不要兩個都給。掛不上去的掛「其他」。

一句話在講「我當時決定放掉什麼」的時候掛「取捨」,不要掛 T——
那是他自己做的取捨,不是別人交付給他的任務。

標籤不必用完。一段回答裡沒有任務段是很常見的——那就沒有任何一句掛 T,
不要為了把標籤用完而硬塞。

【輸出格式】

只回這個 JSON,鍵是句子編號的字串,值是 {" / ".join(STAR_LABELS)} 其中之一:

{{"1": "S", "2": "A", "3": "R"}}

每一句都要出現,一句都不能漏。"""


def split_sentences(text: str) -> list[str]:
    """按句號斷句。階梯裡的回答都用句號收尾,不需要更複雜的斷句。"""
    return [s.strip() for s in (text or "").split("。") if s.strip()]


def assign_user_message(sentences: list[str]) -> str:
    listed = "\n".join(f"{i}. {s}。" for i, s in enumerate(sentences, 1))
    return f"【他剛剛說的,一句一行】\n{listed}"


def parse_assign(raw: str, sentences: list[str]) -> dict[str, tuple[bool, str, bool]]:
    """從分派結果推出 S/T/A/R 在不在。

    【為什麼這個格式能治「一律答有」】
    yes/no 加抄原文是存在性宣稱:找到一句就能說「有」,而任何一段話裡
    總找得到一句可以被讀成任務的。實測兩種問法、兩種條件,T 不存在的那幾筆
    全部 0.00——不是偶爾錯,是每次都答「有」。

    分派把它變成互斥:一句只能掛一個標籤,所以要說「有任務」就得從「行動」
    那邊搶一句過來,而行動的判讀是 1.00。搶錯會在行動那一欄顯示出來,
    不會像現在這樣悄悄多出一個任務段。

    引文由程式供給(就是被分派的那一句),所以落地必然成立——
    這一欄在分派模式下不是量測。
    """
    data = parse_obj(raw) or {}
    chosen: dict[str, str] = {}
    for key, label in data.items():
        if not isinstance(label, str) or label.strip() not in STAR_LABELS:
            continue
        try:
            idx = int(str(key).strip())
        except ValueError:
            continue
        if 1 <= idx <= len(sentences):
            chosen[str(idx)] = label.strip()

    out: dict[str, tuple[bool, str, bool]] = {}
    for fact in ("S", "T", "A", "R"):
        hit = [sentences[int(i) - 1] for i, lab in chosen.items() if lab == fact]
        out[fact] = (bool(hit), hit[0] if hit else "", True)
    return out


# ---------------------------------------------------------------------------
# 解析與查證
# ---------------------------------------------------------------------------

_PUNCT = "，,。.、;；:：「」『』()（）?？!！…—　 \n\t"
_STRIP = str.maketrans("", "", _PUNCT)

MIN_QUOTE_CHARS = 4
"""短於這個長度的引文不算證據。「報表」在任何一筆裡都找得到。"""


def _norm(text: str) -> str:
    return (text or "").translate(_STRIP)


def grounded(quote: str, answer: str) -> bool:
    """引文是不是真的在答案裡。標點與空白正規化後做子字串比對。

    這是整支工具唯一不靠模型的判斷:它說「有交代」可以是編的,
    但它抄的那句話在不在原文裡,程式查得出來。
    """
    q = _norm(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _norm(answer)


def parse_obj(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        got = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, dict) else None


def read_one(raw: str, answer: str) -> tuple[bool, str, bool]:
    """回 (判讀, 引文, 引文有落地)。解析不出來當成「沒有」。

    `present` 說有但引文沒落地 → 判讀改成「沒有」。
    那是抄原文這條規則的執行點:宣稱無法查證就不採用。
    """
    data = parse_obj(raw)
    if data is None:
        return False, "", False
    quote = str(data.get("quote", "") or "")
    claimed = bool(data.get("present"))
    if not claimed:
        return False, quote, True
    ok = grounded(quote, answer)
    return ok, quote, ok


def read_bundle(raw: str, answer: str) -> dict[str, tuple[bool, str, bool]]:
    data = parse_obj(raw) or {}
    out: dict[str, tuple[bool, str, bool]] = {}
    for f in FACTS:
        cell = data.get(f)
        if not isinstance(cell, dict):
            out[f] = (False, "", False)
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
        sentences = split_sentences(a["text"])
        order = [f for f in ("S", "T", "A", "R") if truth(a["label"], f)]
        if truth(a["label"], "tradeoff"):
            order = order + ["取捨"]
        out = {str(i + 1): (order[i] if i < len(order) else "其他")
               for i in range(len(sentences))}
        return out

    def call(system: str, user: str) -> str:
        if system is ASSIGN_SYSTEM or system.startswith("你在讀一段面試回答,把每一句"):
            for a in by_text:
                if all(s in user for s in split_sentences(a["text"])):
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


async def run_split(answer: dict, llm, sem) -> dict[str, tuple[bool, str, bool]]:
    async def one(fact: str):
        async with sem:
            raw = await asyncio.to_thread(
                llm, compose_split(fact), user_message(answer["text"])
            )
        return fact, read_one(raw, answer["text"])

    return dict(await asyncio.gather(*(one(f) for f in FACTS)))


async def run_bundled(answer: dict, llm, sem) -> dict[str, tuple[bool, str, bool]]:
    async with sem:
        raw = await asyncio.to_thread(
            llm, compose_bundled(), user_message(answer["text"])
        )
    return read_bundle(raw, answer["text"])


async def run_assign(answer: dict, llm, sem) -> dict[str, tuple[bool, str, bool]]:
    """S/T/A/R 用句子分派,量化與取捨照原本的單題問法。

    量化與取捨不是 STAR 段落,是結果的屬性,沒辦法用分派表達。
    那兩項在前兩輪都是 1.00(量化只有一次 0.97),沿用就好。
    """
    sentences = split_sentences(answer["text"])

    async def one_fact(fact: str):
        async with sem:
            raw = await asyncio.to_thread(
                llm, compose_split(fact), user_message(answer["text"])
            )
        return fact, read_one(raw, answer["text"])

    async def assignment():
        async with sem:
            raw = await asyncio.to_thread(
                llm, ASSIGN_SYSTEM, assign_user_message(sentences)
            )
        return parse_assign(raw, sentences)

    parts, rest = await asyncio.gather(
        assignment(), asyncio.gather(*(one_fact(f) for f in ("quantified", "tradeoff")))
    )
    return {**parts, **dict(rest)}


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------


def print_facts(name: str, reads: dict[str, list[dict]], answers) -> dict[str, float]:
    print(f"\n【{name}:每一項事實的準確率】\n")
    print(pad("答案", 6, False) + "".join(pad(FACT_NAME[f], 7) for f in FACTS) + pad("整列", 7))
    per_fact: dict[str, list[float]] = {f: [] for f in FACTS}
    for a in answers:
        row = pad(a["id"], 6, False)
        for f in FACTS:
            want = truth(a["label"], f)
            hits = [1.0 if r[f][0] == want else 0.0 for r in reads[a["id"]]]
            per_fact[f] += hits
            row += pad(f"{statistics.mean(hits):.2f}", 7)
        whole = [
            1.0 if all(r[f][0] == truth(a["label"], f) for f in FACTS) else 0.0
            for r in reads[a["id"]]
        ]
        row += pad(f"{statistics.mean(whole):.2f}", 7)
        print(row)

    print("\n" + pad("合計", 6, False) + "".join(
        pad(f"{statistics.mean(per_fact[f]):.2f}", 7) for f in FACTS
    ))
    return {f: statistics.mean(per_fact[f]) for f in FACTS}


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
                mine = accepts(rule, as_label({f: r[f][0] for f in FACTS}))
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
        print("  下一步是把判讀與 accepts() 搬進 app/,決策那一支改成吃事實不吃判斷。")
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
        print(compose_split("T"))
        print("\n" + "=" * 74)
        print(compose_bundled())
        print("\n" + "=" * 74)
        print(ASSIGN_SYSTEM)
        print("\n" + "-" * 74)
        print(assign_user_message(split_sentences(answers[-1]["text"])))
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
    _calls = {"split": len(FACTS), "bundled": 1, "assign": 3}
    n = sum(_calls[c] for c in conditions) * runs * len(answers)
    print(f"條件 {'+'.join(conditions)} × 回合 {runs} × 答案 {len(answers)} "
          f"= {n} 次呼叫(temp={args.temp})")

    sem = asyncio.Semaphore(args.concurrency)
    reads: dict[str, dict[str, list[dict]]] = {}
    log: list[dict] = []

    async def go():
        llm = make_fake_llm(answers) if args.fake else make_llm(args.model, args.temp)
        for cond in conditions:
            runner = {"split": run_split, "bundled": run_bundled, "assign": run_assign}[cond]
            got: dict[str, list[dict]] = {a["id"]: [] for a in answers}
            jobs = [(a, runner(a, llm, sem)) for _ in range(runs) for a in answers]
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
                                "判讀": int(ok), "標準": int(truth(a["label"], f)),
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