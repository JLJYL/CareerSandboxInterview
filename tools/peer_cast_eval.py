"""群面競爭者生成的實跑。十個黃金職缺,每個兩次呼叫。

## 這支要回答的問題

3b 的 29 條測試用假 LLM 驗了形狀與降級:編號超範圍會丟、撞 JD 會數、
id 對人不是順序對人、欄位過長退回預設、誇大掃得到、兩次都掛還開得起來。
那些跟真模型無關,所以已經是終局。

測不到的只有兩件事,而兩件都要真的呼叫:

    一   真模型挑的相鄰技能像不像話
    二   生成的人物讀起來像不像真的求職者

第二件只能用眼睛看,所以這支把競爭者整組印出來。第一件有數字可看。

## 四個數字,各自在驗一條 prompt 指示

    source 分佈    llm / topped_up / fallback —— 相鄰技能那次呼叫有沒有在工作
    out_of_range   模型回了清單外的編號 —— 「用編號回答」這條有沒有生效
    on_jd          模型挑到職缺已經要求的 —— 規則 4 有沒有生效
    repaired       欄位不合格退回預設 —— 長度規則有沒有生效

`on_jd` 長期偏高的話那條規則該換做法,不是寫得更長。1d 那輪的教訓:
prompt 改兩次都沒生效就不要寫第三次。

## 有一個數字是有正確答案的

三個指標(同質度、單人壓迫度、聯集壓迫度)**應該精確等於設定值**,
因為它們是構造出來的不是量出來的。對不上就是程式錯了,不是模型錯了。

所以這支把「指標對不上」當失敗印出來,而不是當量測結果。
這一欄是整支工具唯一的硬檢查——其餘都是給人看的。

## 成本

每個職缺兩次呼叫,十個職缺 = 20 次。對照 2d 那輪的 1710 次,
這支便宜到不需要先做小規模試探。`--runs` 開大才需要看一下乘出來的數字。

## prompt 不住在這裡

在 `app/prompts/peer_profile.py`,這支只 import。走的是
`build_peers()`——也就是 `create_interview` 真的會走的那一條,
不是這支自己串的流程。這支串自己的流程就只驗到自己。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.peer_generate import names_for, peer_count, verify  # noqa: E402
from app.pipeline.peer_profile import (  # noqa: E402
    FALLBACK,
    TOPPED_UP,
    build_peers,
)
from app.pipeline.skill_pool import audit, default_pool  # noqa: E402
from app.prompts.peer_profile import ADJACENT_MARK, PROFILE_MARK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "fixtures/golden/interview"
CALLS_PER_CASE = 2
"""挑相鄰技能一次、寫人物一次。改這個數字之前先看 `build_peers` 真的呼幾次——
2d 那輪我把估計值寫成 630,實跑是 1710,因為漏算了一整個分支。"""


def load_cases(path: Path = GOLDEN) -> list[dict]:
    """黃金案例的職缺。`requiredSkills` 是型錄的結構化欄位,順序即優先序。"""
    out = []
    for p in sorted(glob.glob(str(path / "ivw-*.json"))):
        d = json.loads(Path(p).read_text("utf-8"))
        jd = d.get("jd") or {}
        out.append(
            {
                "id": d.get("case_id") or Path(p).stem,
                "title": jd.get("title") or "",
                "required": list(jd.get("requiredSkills") or []),
            }
        )
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


def make_fake_llm(pool):
    """挑前 30 個編號,人物照格式填。

    驗的是管線:source 應該是 llm、on_jd 與 out_of_range 應該是 0。
    **不是量測**——真模型會不會這樣答就是這支要問的事。
    """

    def call(system: str, user: str) -> str:
        if ADJACENT_MARK in system:
            return json.dumps(list(range(1, 31)))
        if PROFILE_MARK in system:
            ids = [
                line.split("應徵者 ")[1].rstrip("】")
                for line in user.splitlines()
                if "【應徵者" in line
            ]
            return json.dumps(
                [
                    {
                        "id": pid,
                        "name": f"假{'一二三四'[i]}號",
                        "background": "政大資管畢業,做過一年的系統維護。",
                        "style": "先講結論",
                    }
                    for i, pid in enumerate(ids)
                ],
                ensure_ascii=False,
            )
        raise AssertionError(f"沒有分支吃這個 prompt:{system[:40]!r}")

    return call


def throttled(llm, limit: int):
    """限制同時在飛的呼叫數。

    用 threading 的 semaphore 不用 asyncio 的,因為 `build_peers` 自己
    用 `asyncio.to_thread` 包同步的 llm,外面拿不到它的 await 點。
    """
    gate = threading.BoundedSemaphore(max(1, limit))

    def call(system: str, user: str) -> str:
        with gate:
            return llm(system, user)

    return call


def timed(llm, bucket: list[float]):
    def call(system: str, user: str) -> str:
        t0 = time.perf_counter()
        try:
            return llm(system, user)
        finally:
            bucket.append(time.perf_counter() - t0)

    return call


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------


def row_for(case: dict, cast, pool) -> dict:
    """一個職缺跑完的結果攤平成一列。

    `drift` 是這支工具唯一有正確答案的欄位:三個指標是構造出來的,
    拿挑好的集合重算要一模一樣。對不上就是程式錯了,不是模型錯了。
    """
    plan = cast.plan
    recomputed = verify(plan.peers, plan.denominator)
    drift = max(
        abs(a - b)
        for a, b in zip(
            recomputed, (plan.homogeneity, plan.mean_pressure, plan.union_pressure)
        )
    )
    return {
        "case": case["id"],
        "title": case["title"],
        "denominator": len(plan.denominator),
        "homogeneity": plan.homogeneity,
        "mean_pressure": plan.mean_pressure,
        "union_pressure": plan.union_pressure,
        "drift": drift,
        "source": cast.adjacent.source,
        "kept": cast.adjacent.kept,
        "returned": cast.adjacent.returned,
        "out_of_range": cast.adjacent.out_of_range,
        "on_jd": cast.adjacent.on_jd,
        "repaired": sum(len(p.repaired) for p in cast.profiles),
        "overclaims": sum(len(p.overclaims) for p in cast.profiles),
        "fell_back": sum(1 for p in cast.profiles if p.source == "fallback"),
        "peers": [
            {
                "id": p.persona_id,
                "name": p.name,
                "skills": list(names_for(p.skills, pool)),
                "background": p.background,
                "style": p.style,
                "overclaims": list(names_for(p.overclaims, pool)),
            }
            for p in cast.profiles
        ],
        "notices": list(cast.notices),
    }


# ---------------------------------------------------------------------------
# 印
# ---------------------------------------------------------------------------


def report(
    rows: list[dict], *, homogeneity: float, pressure: float, latency: list[float]
) -> None:
    print("\n" + "=" * 78)
    print(f"目標 同質度 {homogeneity:.3f} / 單人壓迫度 {pressure:.3f}")
    print("=" * 78)

    print(
        f"\n{'案例':12} {'分母':>4} {'同質':>7} {'單人':>7} {'聯集':>7} "
        f"{'來源':>10} {'留':>4} {'超範圍':>6} {'撞JD':>5} {'修欄位':>6} {'誇大':>5}"
    )
    for r in rows:
        print(
            f"{r['case']:12} {r['denominator']:>4} {r['homogeneity']:>7.3f} "
            f"{r['mean_pressure']:>7.3f} {r['union_pressure']:>7.3f} "
            f"{r['source']:>10} {r['kept']:>4} {r['out_of_range']:>6} "
            f"{r['on_jd']:>5} {r['repaired']:>6} {r['overclaims']:>5}"
        )

    worst = max((r["drift"] for r in rows), default=0.0)
    print(f"\n指標重算最大偏差 {worst:.2e}", end="  ")
    print(
        "✓ 構造與指標一致"
        if worst < 1e-9
        else "✗ 指標對不上構造——這是程式錯誤,不是模型問題"
    )

    by_source = collections.Counter(r["source"] for r in rows)
    print("\n相鄰技能那一次呼叫")
    for name in ("llm", TOPPED_UP, FALLBACK):
        n = by_source.get(name, 0)
        print(f"  {name:11} {n:>3}/{len(rows)}")
    tot_ret = sum(r["returned"] for r in rows)
    print(
        f"  模型回了 {tot_ret} 個編號,其中清單外 "
        f"{sum(r['out_of_range'] for r in rows)}、"
        f"撞職缺要求 {sum(r['on_jd'] for r in rows)}"
    )

    print("\n人物那一次呼叫")
    print(
        f"  退回固定人設  {sum(r['fell_back'] for r in rows)}/"
        f"{len(rows) * peer_count()} 位"
    )
    print(f"  欄位不合格    {sum(r['repaired'] for r in rows)} 次")
    print(f"  背景誇大      {sum(r['overclaims'] for r in rows)} 次")

    if latency:
        s = sorted(latency)
        print(
            f"\n單次呼叫延遲 中位 {statistics.median(s):.2f}s  "
            f"p90 {s[int(len(s) * 0.9)]:.2f}s  max {s[-1]:.2f}s  (n={len(s)})"
        )

    print("\n" + "-" * 78)
    print("競爭者(這一段只能用眼睛看)")
    print("-" * 78)
    for r in rows:
        print(f"\n【{r['case']}】{r['title']}")
        for p in r["peers"]:
            print(f"  {p['name']}({p['id']})")
            print(f"    技能 {'、'.join(p['skills'])}")
            print(f"    背景 {p['background']}")
            if p["style"]:
                print(f"    風格 {p['style']}")
            if p["overclaims"]:
                print(f"    ⚠ 宣稱了沒有的職缺技能:{'、'.join(p['overclaims'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--temp", type=float, default=0.0, help="預設 0:挑技能是分類不是生成,要可重現。"
    )
    ap.add_argument(
        "--write-temp",
        type=float,
        default=0.4,
        help="寫人物那一次的溫度。對齊 get_llm_live。",
    )
    ap.add_argument("--homogeneity", type=float, default=0.5)
    ap.add_argument("--pressure", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument(
        "--pool-audit",
        action="store_true",
        help="印技能池的分層組成。L3_market 佔多數是已知風險。",
    )
    args = ap.parse_args()

    if args.dump:
        try:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            args.dump.touch()
        except OSError as exc:
            raise SystemExit(f"✗ 寫不到 {args.dump}:{exc}") from exc

    pool = list(default_pool())
    cases = load_cases(args.golden)
    if not cases:
        raise SystemExit(f"✗ {args.golden} 裡沒有 ivw-*.json。")

    if args.pool_audit:
        a = audit(pool)
        print("技能池分層組成")
        for key in sorted(k for k in a if not k.startswith("_")):
            print(f"  {key:24} {a[key]:>4}")
        print(f"  {'合計':24} {a['_total']:>4}   最長名稱 {a['_longest_name']} 字")

    if args.dry_run:
        from app.prompts.peer_profile import (
            adjacent_user_message,
            compose_adjacent_prompt,
            compose_profile_prompt,
        )

        print("=" * 78)
        print(compose_adjacent_prompt())
        print("\n" + "-" * 78)
        print(
            adjacent_user_message(
                cases[0]["title"],
                cases[0]["required"][:4],
                [s.display for s in pool[:12]],
            )
        )
        print("\n" + "=" * 78)
        print(compose_profile_prompt())
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
        print("⚠ --fake:這不是量測。假模型照格式完美回答,")
        print("  所以 source 應該全是 llm、超範圍與撞 JD 應該是 0——只證明管線對。\n")

    print(
        f"職缺 {len(cases)} 個 × 每個 {CALLS_PER_CASE} 次呼叫 "
        f"= {len(cases) * CALLS_PER_CASE} 次(挑技能 temp={args.temp}、"
        f"寫人物 temp={args.write_temp})"
    )

    latency: list[float] = []

    async def go() -> list[dict]:
        if args.fake:
            base = make_fake_llm(pool)
            pick_llm = write_llm = throttled(timed(base, latency), args.concurrency)
        else:
            pick_llm = throttled(
                timed(make_llm(args.model, args.temp), latency), args.concurrency
            )
            write_llm = throttled(
                timed(make_llm(args.model, args.write_temp), latency), args.concurrency
            )

        async def one(case: dict) -> dict:
            cast = await build_peers(
                title=case["title"],
                required=case["required"],
                llm=pick_llm,
                write_llm=write_llm,
                group_size=4,
                homogeneity=args.homogeneity,
                pressure=args.pressure,
                seed=args.seed,
                pool=pool,
            )
            return row_for(case, cast, pool)

        return list(await asyncio.gather(*(one(c) for c in cases)))

    rows = asyncio.run(go())
    report(rows, homogeneity=args.homogeneity, pressure=args.pressure, latency=latency)

    if args.dump:
        args.dump.write_text(
            json.dumps(
                {
                    "config": vars(args)
                    | {"golden": str(args.golden), "dump": str(args.dump)},
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\n明細寫到 {args.dump}")

    return 0 if all(r["drift"] < 1e-9 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
