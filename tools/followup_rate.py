"""追問率量測:難度到底有沒有生效。

## 這支要回答的問題

「三種難度產出的問句幾乎逐字相同」這件事抱怨了很久,但一直沒有數字。
1a 把放行條件變成 `AcceptRule`(可判定),1b 把決定變成 `TurnDecision`(可數),
到這裡才第一次量得出來。

量的是**決定**,不是問句。問句好不好是 `scripts/try_interview_live.py` 的事,
那支要人看;這支只數 FOLLOW_UP 出現幾次,不需要人判斷。

## 為什麼不能直接跟 Panfilova 的 19.4 / 45.7 比

那兩個數字是**每場面試的追問次數**,而他們的面試沒有輪次上限。
我們 `MAX_TURNS_PER_SESSION = 4`,一對一一場最多 4 輪,
結構上就到不了 19.4,比絕對值沒有意義。

可以比的是**兩種設定之間的倍率**:他們 45.7 / 19.4 = 2.36 倍。
我們對應的是 困難 / 新手。另外他們的受訪者是打字、我們是語音,
所以 CLARIFY 單獨數,不併進追問率跟他們比。

## 決定是算的

一次決策問三次「這段話裡有什麼」(句子分派 + 量化 + 取捨),
再用 `AcceptRule.accepts()` 算要不要追問。所以同樣的 --runs 比 1d 貴三倍。

2d 之前還有一條對照組(放行條件寫進 prompt 問模型挑動作),量完就刪了:

    1d  條件寫進 prompt   0.500 / 0.500 / 0.500,19/180 次降級
    2d  程式用事實算      0.143 / 0.429 / 0.857,一致率 1.00,0/630 次降級

## 兩種深度,因為上線的那種跟設計的那種不一樣

`forced_decision()` 有一條「追到 `max_depth` 就換題」,三種難度分別是 1 / 2 / 3。
但 `routes.py` 的 `_decide()` 傳的是 `depth=0`——後端沒有跨輪狀態,
而 `followUpIdx` 的語意是「整場第幾題」不是「同一個缺口追幾次」,不能拿來代入。

`depth` 永遠是 0,所以**那條深度上限在線上從來沒有觸發過**。
線上唯一還在作用的難度槓桿,是放行條件被寫進 prompt 的那段文字。

所以兩種都要量,而且要分開報:

    --regime wired     depth 固定 0。這是上線實際的行為,量的是 prompt 那段文字有沒有用。
    --regime tracked   工具自己數 depth。量的是深度上限如果接起來會買到什麼。
    --regime both      兩種都跑(預設)

如果 wired 分不出難度而 tracked 分得出來,結論就很明確:去把 depth 接起來。
那是一個具體的下一項,不是一句「難度沒用」。

## 為什麼一定要看 source

1c 出過一次:`test_api.py` 的假 LLM 少了決策分支,回 `"{}"`,解析失敗,
每一輪都降級成 NEXT,`is_follow_up` 從 True 翻成 False——
而 602 個測試全綠。

降級的決定長得跟「模型判斷講完了」一模一樣。所以這支一定要印
`code / llm / fallback` 的分佈,而且 fallback 超過門檻就不給結論。
沒有這一欄,一次 API key 設錯就會讓人得到「難度沒用」的錯結論。

用法:
    python tools/followup_rate.py --fake              # 不打 API,驗工具本身
    python tools/followup_rate.py --dry-run           # 只印一份 prompt
    python tools/followup_rate.py --runs 10           # 正式量測(facts:10 × 3 × 7 × 3 呼叫)
    python tools/followup_rate.py --runs 10 --temp 0  # 看噪音有多少是溫度來的
    python tools/followup_rate.py --engine device     # 看 device 判準會不會催出假 CLARIFY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.interview_live import _ADMITS_UNKNOWN  # noqa: E402
from app.pipeline.interviewer_spec import (  # noqa: E402
    ACCEPT_RULES,
    AcceptRule,
    derive_specs,
)
from app.pipeline.star_read import AnswerFacts  # noqa: E402
from app.pipeline.transcript import segment_text  # noqa: E402
from app.pipeline.turn_decision import (  # noqa: E402
    CLARIFY,
    DECISION_SOURCES,
    FALLBACK,
    FOLLOW_UP,
    NEXT,
    TurnDecision,
    decide_turn,
)
from app.prompts.star_read import (  # noqa: E402
    ASSIGN_SYSTEM,
    FACT_QUESTION,
    FACTS,
    assign_user_message,
    compose_fact_prompt,
)

ANSWERS = Path("fixtures/golden/decision/answers_v1.json")

CALLS_PER_DECISION = 3
"""一次決策要幾次模型呼叫:句子分派 + 量化 + 取捨,並行。

跟 `star_read.read_facts()` 綁在一起,那邊改成兩次或四次的話這裡要跟著改——
這個數字只用在估成本與印延遲,算錯不會讓量測失真,但會讓人照錯的數字決定 --runs。
"""

DIFFICULTIES = ("新手", "中等", "困難")

QUESTION = "可以講一個你最近做過的專案,還有你在裡面負責什麼嗎?"
"""當作上一題的脈絡丟進去。固定不動——這支只有難度一個變因。"""

#: Panfilova et al.(2026)兩種設定的每場追問次數。只拿倍率來比,不比絕對值。
PAPER_LOW, PAPER_HIGH = 19.4, 45.7

FALLBACK_GATE = 0.10
"""降級比例超過這個就不給結論。

訂 10% 不是統計上的講究,是「夠高到會翻轉結論」的量級:
fallback 一律回 NEXT,所以降級越多,追問率越被壓低、難度越看不出差別——
失敗的方向剛好就是我們想證明的方向,所以門檻要訂得保守。
"""


# ---------------------------------------------------------------------------
# 放行條件:標註與 AcceptRule 之間的接頭
# ---------------------------------------------------------------------------

def accepts(rule: AcceptRule, label: dict) -> bool:
    """照放行條件判斷這段回答算不算講完了。

    判斷本身在 `AcceptRule.accepts()`,這裡只是把標註擺成它吃的形狀。
    原本這支自己實作一份,外加一段欄位檢查防止兩份漂開——
    2b 之後只有一份,那段檢查跟著搬過去了。

    標註與線上判讀共用同一支函式,所以量到的期望值跟線上的行為
    不可能因為規則改動而分叉。
    """
    return AnswerFacts.from_label(label).accepted_by(rule)


def expected_action(rule: AcceptRule, label: dict) -> str:
    """期望的動作。不在 fixture 裡手寫,就是為了不要有第三份會漂的規則。"""
    return NEXT if accepts(rule, label) else FOLLOW_UP


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

_PUNCT = ",。、?!;:「」()…—　 "


def strip_punct(text: str) -> str:
    """模擬裝置端即時辨識:去標點。

    同一組答案不存兩份,存兩份就要同步兩份。
    """
    return "".join(c for c in text if c not in _PUNCT)


def load_answers(path: Path, engine: str) -> list[dict]:
    """讀答案階梯,順便擋掉會讓量測失去意義的內容。"""
    if not path.exists():
        raise SystemExit(f"✗ 找不到 {path}。這支要跟 fixture 一起用。")
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for a in data["answers"]:
        text = a["text"]
        hit = [k for k in _ADMITS_UNKNOWN if k in text]
        if hit:
            raise SystemExit(
                f"✗ {a['answer_id']} 含有 {hit}。forced_decision() 會直接回 NEXT,"
                "量到的是程式的決定不是模型的判斷。"
            )
        if len(text.strip()) < 2:
            raise SystemExit(f"✗ {a['answer_id']} 太短,會被當成空輪。")
        out.append(
            {
                "id": a["answer_id"],
                "text": strip_punct(text) if engine == "device" else text,
                "label": a["label"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


def make_llm(model: str, temp: float):
    """解析失敗的原文不在這裡側錄。

    原本這裡會另外收一份,結果一份算 wired 一份算全部,出現過 56 筆對 22 次
    那種對不上的數字。現在 `TurnDecision.superseded` 自己帶著被取代的原文,
    跟決定綁在一起,所以記錄只有一個來源。
    """
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


def fake_facts_reply(label: dict, answer: str, system: str) -> str | None:
    """事實判讀那三次呼叫的完美回答。不是判讀 prompt 就回 None。

    假模型少了這幾個分支的話每一輪的事實都會讀成什麼都沒有,
    於是每一輪都追問——而表格照樣印得出來。1c 出過一次一樣的事。
    """
    if "把每一句分派到一個標籤" in system:
        sentences, _ = segment_text(answer, level="sentence")
        order = [f for f in ("S", "T", "A", "R") if f in label["star_parts"]]
        if label["tradeoff"]:
            order = order + ["取捨"]
        return json.dumps(
            {str(i + 1): (order[i] if i < len(order) else "其他")
             for i in range(len(sentences))},
            ensure_ascii=False,
        )
    for fact in FACTS:
        if FACT_QUESTION[fact] in system:
            has = (
                fact in label["star_parts"]
                if fact in ("S", "T", "A", "R")
                else bool(label[fact])
            )
            return json.dumps(
                {"present": has, "quote": answer[:12] if has else ""}, ensure_ascii=False
            )
    return None


def make_fake_llm(rule: AcceptRule, answers: list[dict]):
    """完全照放行條件判斷的假模型。驗工具本身,不是量測。

    會從 user 訊息裡把答案認回來,而不是照呼叫順序回答——
    1c 的教訓是假模型如果不看實際內容,管線壞掉的時候它照樣綠。

    **由長到短比對**:A6 的全文是 A5 的前綴,照 fixture 順序找會把 A5 認成 A6。
    """
    ordered = sorted(answers, key=lambda a: len(a["text"]), reverse=True)

    def shows_up(answer: dict, user: str) -> bool:
        """這一筆答案在不在這段 user 訊息裡。

        兩種形狀都要認:單題問法帶的是整段原文,而句子分派把它拆成
        編號的一行一句,整段原文就不再是子字串了。
        只認前者的話分派那次會找不到答案 → 丟 AssertionError → read_facts
        記成讀取失敗 → 每一輪都降級,而表格照樣印得出來。
        """
        if answer["text"] in user:
            return True
        sentences, _ = segment_text(answer["text"], level="sentence")
        return bool(sentences) and all(x in user for x in sentences)

    def call(system: str, user: str) -> str:
        for a in ordered:
            if shows_up(a, user):
                facts = fake_facts_reply(a["label"], a["text"], system)
                if facts is not None:
                    return facts
                raise AssertionError(
                    f"假模型認不得這個 prompt:{system[:40]}。"
                    "2d 之後決策只問事實判讀那三種,多出來的呼叫是接錯了。"
                )
        raise AssertionError("假模型在 prompt 裡找不到任何一筆答案——答案沒被帶進去。")

    return call


# ---------------------------------------------------------------------------
# 兩種深度
# ---------------------------------------------------------------------------


async def decide(
    spec, answer: dict, llm, engine: str, depth: int
) -> TurnDecision:
    return await decide_turn(
        spec=spec,
        answer=answer["text"],
        llm=llm,
        question=QUESTION,
        depth=depth,
        ended_by="manual",
        transcription_engine=engine,
    )


async def run_wired(
    spec, answers, llm, engine, runs, sem, difficulty="", log=None,
    secs: list[float] | None = None,
) -> dict[str, list[TurnDecision]]:
    """depth 固定 0。上線實際的行為。

    `log` 不是 None 時記下每一個決定。追問率只說結果,不說模型拿什麼當理由,
    而覆寫是照 `gap` 的文字判斷的——同一筆答案 `gap` 寫「結果段」或「量化數據」
    會落到不同的結論。沒有這份記錄,數字跟預測不合就只能再猜一輪。
    """

    async def one(answer, _run):
        async with sem:
            t0 = time.perf_counter()
            d = await decide(spec, answer, llm, engine, 0)
            if secs is not None:
                secs.append(time.perf_counter() - t0)
            return answer["id"], d

    jobs = [one(a, r) for r in range(runs) for a in answers]
    got: dict[str, list[TurnDecision]] = {a["id"]: [] for a in answers}
    for aid, d in await asyncio.gather(*jobs):
        got[aid].append(d)
        if log is not None:
            log.append(
                {
                    "難度": difficulty,
                    "答案": aid,
                    "action": d.action,
                    "source": d.source,
                    "gap": d.gap,
                    "why": d.why,
                    "superseded": d.superseded,
                }
            )
    return got


async def run_tracked(
    spec, answers, llm, engine, runs, sem
) -> dict[str, list[int]]:
    """工具自己數 depth,追到決定換題或撞上限才停。

    回答不會變——固定腳本沒辦法讓他補內容,而讓 LLM 扮演使用者會讓
    「追問變差」無法歸因(`try_interview_live.py` 的開頭已經寫過這個理由)。

    所以這個數字是**上界**:他一直補不出來的時候,這個難度會在同一個點上
    耗幾輪才放他走。那正好就是深度上限存在的目的。
    """
    hard_cap = spec.accept.max_depth + 2  # forced_decision 應該先停,這只是保險

    async def one(answer):
        depth = 0
        while depth < hard_cap:
            async with sem:
                d = await decide(spec, answer, llm, engine, depth)
            if not d.is_follow_up:
                return answer["id"], depth
            depth += 1
        return answer["id"], depth

    jobs = [one(a) for _ in range(runs) for a in answers]
    got: dict[str, list[int]] = {a["id"]: [] for a in answers}
    for aid, n in await asyncio.gather(*jobs):
        got[aid].append(n)
    return got


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------


def pad(text: str, width: int, right: bool = True) -> str:
    """中文字在終端機是兩格寬,f-string 的 `>8` 只數字元,所以表格會歪。

    `unicodedata.east_asian_width` 的 W/F 算兩格,其餘算一格。
    """
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    space = " " * max(0, width - w)
    return space + text if right else text + space


def rate(ds: list[TurnDecision]) -> float:
    return sum(1 for d in ds if d.is_follow_up) / len(ds) if ds else 0.0


def sources(ds: list[TurnDecision]) -> dict[str, int]:
    """值域從 DECISION_SOURCES 來,不在這裡自己列一份。

    自己列的話,turn_decision 多一個 source 值而這裡沒跟著加,
    那一類會被整個漏掉而且報表照樣印得出來。
    """
    out = dict.fromkeys(DECISION_SOURCES, 0)
    for d in ds:
        out[d.source] = out.get(d.source, 0) + 1
    return out


def print_wired(by_difficulty: dict[str, dict[str, list[TurnDecision]]], answers) -> None:
    print("\n【wired:depth 固定 0,上線實際的行為】\n")
    print("每格是「期望動作 實測追問率」。✗ = 實測站錯邊。\n")
    head = pad("答案", 4, False) + pad("段數", 6) + pad("數字", 6) + pad("取捨", 6) + " "
    head += "".join(pad(d, 15) for d in DIFFICULTIES)
    print(head)
    for a in answers:
        lab = a["label"]
        row = (
            pad(a["id"], 4, False)
            + pad(str(len(lab["star_parts"])), 6)
            + pad("✓" if lab["quantified"] else "·", 6)
            + pad("✓" if lab["tradeoff"] else "·", 6)
            + " "
        )
        for diff in DIFFICULTIES:
            obs = rate(by_difficulty[diff][a["id"]])
            want_fu = expected_action(ACCEPT_RULES[diff], lab) == FOLLOW_UP
            ok = "✓" if (obs >= 0.5) == want_fu else "✗"
            row += pad("追問" if want_fu else "放行", 8) + f" {obs:>4.2f} {ok}"
        print(row)


def print_summary(by_difficulty, answers) -> dict[str, dict]:
    print("\n【每個難度的總計】\n")
    print(
        pad("難度", 4, False)
        + pad("實測追問率", 12)
        + pad("期望追問率", 12)
        + pad("一致率", 9)
        + "".join(f"{s:>10}" for s in DECISION_SOURCES)
        + f"{'CLARIFY':>9}"
        + pad("合格gap", 10)
    )
    stats = {}
    for diff in DIFFICULTIES:
        rule = ACCEPT_RULES[diff]
        flat = [d for a in answers for d in by_difficulty[diff][a["id"]]]
        exp_rate = statistics.mean(
            1.0 if expected_action(rule, a["label"]) == FOLLOW_UP else 0.0 for a in answers
        )
        agree = statistics.mean(
            1.0 if (rate(by_difficulty[diff][a["id"]]) >= 0.5)
            == (expected_action(rule, a["label"]) == FOLLOW_UP)
            else 0.0
            for a in answers
        )
        src = sources(flat)
        clar = sum(1 for d in flat if d.action == CLARIFY)
        obs = rate(flat)
        # 追問裡有幾個 gap 填得出來。2d 之後 gap 是 first_gap() 算的,
        # 所以這一欄理應永遠滿格——不滿格表示有決定走了降級那條路
        # (降級回 NEXT,不會進這個分母),或者 accepts() 與 first_gap() 不一致。
        fus = [d for d in flat if d.action == FOLLOW_UP]
        ok_gap = f"{sum(1 for d in fus if d.gap)}/{len(fus)}" if fus else "—"
        print(
            pad(diff, 4, False)
            + f"{obs:>12.3f}{exp_rate:>12.3f}{agree:>9.2f}"
            + "".join(f"{src[s]:>10}" for s in DECISION_SOURCES)
            + f"{clar:>9}"
            + pad(ok_gap, 10)
        )
        stats[diff] = {
            "observed": obs,
            "expected": exp_rate,
            "agree": agree,
            "src": src,
            "clarify": clar,
            "n": len(flat),
        }
    return stats


def print_tracked(tracked: dict[str, dict[str, list[int]]], answers) -> dict[str, float]:
    print("\n【tracked:工具自己數 depth,量深度上限會買到什麼】\n")
    print("每格是「在這一筆上耗掉的追問輪數」平均。上限 = 這個難度的 max_depth。\n")
    print(pad("答案", 6, False) + "".join(pad(d, 10) for d in DIFFICULTIES))
    for a in answers:
        row = pad(a["id"], 6, False)
        for diff in DIFFICULTIES:
            row += f"{statistics.mean(tracked[diff][a['id']]):>10.2f}"
        print(row)

    totals: dict[str, float] = {}
    row = pad("每場合計", 6, False)
    for diff in DIFFICULTIES:
        per_run = [
            sum(tracked[diff][a["id"]][i] for a in answers)
            for i in range(len(tracked[diff][answers[0]["id"]]))
        ]
        totals[diff] = statistics.mean(per_run)
        row += f"{totals[diff]:>10.2f}"
    print("\n" + row)

    row = pad("上限", 6, False)
    for diff in DIFFICULTIES:
        row += f"{ACCEPT_RULES[diff].max_depth * len(answers):>10}"
    print(row)
    return totals


def report_latency(secs: list[float], concurrency: int) -> None:
    """一次決定要等多久。2d 順手量,因為它不多花一毛 API 錢。

    **這不是單獨一次請求的延遲。** 計時從拿到並行額度那一刻開始算,
    但同時有 `--concurrency` 個請求在飛,所以這個數字含伺服器端的排隊。
    上線一次只服務一個人,實際會比這裡好;當成上界看。

    facts 路徑一次決定要三次呼叫、並行發出,所以理想情況下
    它應該接近 prompt 路徑的單次延遲,而不是三倍。
    那是這一欄要回答的事:並行有沒有真的買到東西。
    """
    if not secs:
        return
    ordered = sorted(secs)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    print(f"\n一次決定的耗時(每次 {CALLS_PER_DECISION} 次呼叫,concurrency={concurrency}):")
    print(f"  中位數 {p50:.2f}s  p90 {p90:.2f}s  最慢 {ordered[-1]:.2f}s  n={len(ordered)}")
    print("  三次呼叫是並行的,所以這個數字該接近單次呼叫而不是三倍。")
    print("  明顯接近三倍表示並行沒生效(檢查 --concurrency 與速率限制)。")


def conclude(
    stats: dict[str, dict],
    tracked_totals: dict[str, float] | None,
    secs: list[float] | None = None,
    concurrency: int = 0,
) -> int:
    print("\n" + "=" * 74)

    total = sum(s["n"] for s in stats.values())
    fb = sum(s["src"][FALLBACK] for s in stats.values())
    if total and fb / total > FALLBACK_GATE:
        print(f"✗ 降級 {fb}/{total} = {fb / total:.1%},超過 {FALLBACK_GATE:.0%},這輪不給結論。")
        print("  fallback 一律回 NEXT,會把追問率壓低——失敗的方向剛好跟想證明的方向相同。")
        print("  先查 API key、模型名稱,以及決策輸出是不是被擋掉了。")
        return 1
    if fb:
        print(f"⚠ 有 {fb}/{total} 次降級({fb / total:.1%}),在門檻內但不是零。\n")

    obs = [stats[d]["observed"] for d in DIFFICULTIES]
    print("追問率:  " + "  ".join(f"{d} {o:.3f}" for d, o in zip(DIFFICULTIES, obs)))
    print("期望:    " + "  ".join(f"{d} {stats[d]['expected']:.3f}" for d in DIFFICULTIES))

    mono = obs[0] < obs[1] < obs[2]
    print(f"\n{'★' if mono else '✗'} 單調性 新手 < 中等 < 困難:{'成立' if mono else '不成立'}")
    if not mono:
        print("  判斷在程式裡,照理不可能不單調——放行條件本身是單調的。")
        print("  所以不單調表示事實判讀壞了,不是判斷壞了。先跑 tools/star_read_eval.py,")
        print("  再看上面的 source 欄有多少 fallback。")

    if obs[0] > 0:
        ratio = obs[2] / obs[0]
        print(f"\n倍率 困難/新手 = {ratio:.2f}×(Panfilova 兩種設定之間是 "
              f"{PAPER_HIGH / PAPER_LOW:.2f}×)")
        print(f"  絕對值不可比:他們每場 {PAPER_LOW} / {PAPER_HIGH} 次且無輪次上限,")
        print("  我們一對一一場上限 4 輪。")
    else:
        print("\n新手追問率 0,倍率算不出來。")

    agree = statistics.mean(stats[d]["agree"] for d in DIFFICULTIES)
    print(f"\n與放行條件一致率:{agree:.2f}")
    if agree < 0.7:
        print("  一致率低而單調性成立也不算過關——那表示難度有差,但差在別的地方,")
        print("  不是差在我們寫的那幾條。逐格看上面的 ✗ 是哪幾條門檻。")

    clar = sum(stats[d]["clarify"] for d in DIFFICULTIES)
    if clar:
        print(f"\n⚠ 乾淨逐字稿上出現 {clar} 次 CLARIFY。")
        print("  這組答案沒有辨識錯誤,所以每一次都是誤判——")
        print("  模型把「講得含糊」當成「沒聽清楚」,而 CLARIFY_BY_ENGINE 已經明寫不該這樣。")
    else:
        print("\n乾淨逐字稿上沒有 CLARIFY。判準有被遵守。")

    if tracked_totals:
        print("\n深度上限的效果(tracked 每場合計):")
        print("  " + "  ".join(f"{d} {tracked_totals[d]:.2f}" for d in DIFFICULTIES))
        t = [tracked_totals[d] for d in DIFFICULTIES]
        if t[0] < t[1] < t[2] and not mono:
            print("\n★ 這就是結論:prompt 那段文字分不出難度,深度上限分得出來,")
            print("  而深度上限在線上沒有接(routes.py 的 _decide 傳 depth=0)。")
            print("  下一項是把 depth 接起來,不是繼續調措辭。")

    report_latency(secs or [], concurrency)
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, default=ANSWERS)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0.4,
                    help="預設 0.4,跟 get_llm_live() 一樣——量的是上線的設定。")
    ap.add_argument("--engine", choices=("api", "device"), default="api")
    ap.add_argument("--regime", choices=("wired", "tracked", "both"), default="both")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--fake", action="store_true", help="不打 API,用完全照條件判斷的假模型驗工具")
    ap.add_argument("--dry-run", action="store_true", help="只印一份 prompt")
    ap.add_argument("--dump", type=Path, default=None,
                    help="把每一個決定(含 gap 與 source)與解析不過的原文寫到這個檔")
    args = ap.parse_args()

    if args.runs < 1:
        raise SystemExit("✗ --runs 至少要 1。")

    # 輸出路徑在**打任何 API 之前**就驗。寫檔原本排在表格前面,資料夾不存在時
    # 180 次呼叫跑完才炸,表格一行都沒印,整輪白跑。
    if args.dump:
        try:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            args.dump.touch()
        except OSError as exc:
            raise SystemExit(f"✗ 寫不到 {args.dump}:{exc}") from exc

    answers = load_answers(args.answers, args.engine)
    specs = {d: derive_specs("single", jd=None, difficulty=d)[0] for d in DIFFICULTIES}

    if args.dry_run:
        # 決策不再有自己的 prompt:它問的是 star_read 那三個。
        print(ASSIGN_SYSTEM)
        print("\n" + "-" * 74)
        sentences, _ = segment_text(answers[3]["text"], level="sentence")
        print(assign_user_message(sentences))
        print("\n" + "=" * 74)
        print(compose_fact_prompt("quantified"))
        return 0

    if args.fake:
        print("⚠ --fake:這不是量測。假模型完全照 accepts() 回答,")
        print("  所以單調性必然成立——這只證明管線有把答案帶進去、數字有算對,")
        print("  以及 override 那一欄是 0(模型跟規則一致時覆寫不該觸發)。\n")
    else:
        # repo 裡每一支要打 API 的都先讀 .env,照同一個做法。
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ 找不到 OPENAI_API_KEY。放在專案根目錄的 .env 裡,或設成環境變數。")
            print("  想先確認工具本身對不對,用 --fake(不打 API)。")
            return 1

    runs = 1 if args.fake else args.runs
    per = CALLS_PER_DECISION
    n_wired = len(DIFFICULTIES) * runs * len(answers)

    # tracked 會對同一筆答案重複決定到換題或撞上限,所以它的成本不是
    # 「答案數」而是「每一筆會耗掉幾輪」。
    #
    # 2d 跑完才發現原本的估計只算了 wired:印 630,實際花了 1710。
    # 量測工具低報成本比高報糟——人會照著印出來的數字決定 --runs。
    n_tracked = 0
    if args.regime in ("tracked", "both"):
        for diff in DIFFICULTIES:
            rule = specs[diff].accept
            for a in answers:
                # 追到上限那一輪由 forced_decision 接手,不呼叫模型,所以不算。
                passes = accepts(rule, a["label"])
                n_tracked += (1 if passes else rule.max_depth) * runs

    total = (n_wired + n_tracked) * per
    print(f"難度 {len(DIFFICULTIES)} × 回合 {runs} × 答案 {len(answers)}")
    print(f"  wired   {n_wired:>5} 次決策"
          + (f" × {per} = {n_wired * per} 次呼叫" if per > 1 else " 次呼叫"))
    if n_tracked:
        print(f"  tracked {n_tracked:>5} 次決策"
              + (f" × {per} = {n_tracked * per} 次呼叫" if per > 1 else " 次呼叫"))
    print(f"  合計    {total:>5} 次呼叫")
    print(f"  regime={args.regime} engine={args.engine} temp={args.temp}")

    sem = asyncio.Semaphore(args.concurrency)
    wired: dict[str, dict] = {}
    tracked: dict[str, dict] = {}
    log: list[dict] | None = [] if args.dump else None
    secs: list[float] = []

    async def go():
        for diff in DIFFICULTIES:
            spec = specs[diff]
            llm = (
                make_fake_llm(spec.accept, answers)
                if args.fake
                else make_llm(args.model, args.temp)
            )
            if args.regime in ("wired", "both"):
                wired[diff] = await run_wired(
                    spec, answers, llm, args.engine, runs, sem, diff, log,
                    secs,
                )
            if args.regime in ("tracked", "both"):
                tracked[diff] = await run_tracked(
                    spec, answers, llm, args.engine, runs, sem
                )

    asyncio.run(go())

    if log is not None:
        cols = ("難度", "答案", "action", "source", "gap", "why", "superseded")
        lines = ["\t".join(cols)]
        lines += ["\t".join(str(r[c]).replace("\t", " ") for c in cols) for r in log]

        # 普查**不過濾**。上一版只收 action == FOLLOW_UP,結果被程式改掉的那些
        # (action 變 NEXT)整批消失——而那正是要看的。涵蓋全部,一筆都不漏。
        census: dict[tuple[str, str, str, str], int] = {}
        for r in log:
            key = (r["答案"], r["action"], r["source"], r["gap"])
            census[key] = census.get(key, 0) + 1
        lines += ["", "# 每一種決定(答案 / action / source / gap / 次數)"]
        lines += [f"{a}\t{act}\t{s}\t{g or '—'}\t{n}" for (a, act, s, g), n in sorted(census.items())]

        dropped = [r for r in log if r["superseded"]]
        if dropped:
            lines += ["", f"# 被取代的原文 {len(dropped)} 筆"]
            lines += [f"{r['難度']}\t{r['答案']}\t{r['superseded']}" for r in dropped]
        # 寫檔失敗不能吃掉整輪。API 錢已經花了,表格一定要印出來。
        try:
            args.dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\n{len(log)} 筆決定、{len(dropped)} 筆有被取代的原文 → {args.dump}")
        except OSError as exc:
            print(f"\n⚠ 寫不到 {args.dump}({exc}),記錄沒存下來,下面的表格仍然有效。")

    if wired:
        print_wired(wired, answers)
        stats = print_summary(wired, answers)
    else:
        stats = {}
    totals = print_tracked(tracked, answers) if tracked else None
    if not stats:
        print("\n(只跑 tracked,沒有 wired 的追問率可以下結論。)")
        return 0
    return conclude(stats, totals, secs, args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main())
