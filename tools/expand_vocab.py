"""詞彙表擴充（成員 A）：把 jobs_all 的高頻 requiredSkills 併進 skills_v1.json。

## 為什麼要做

面試黃金集實測：`skills_v1.json` 只認得 87 個真實技能字串中的 28 個（32%）。
對不上的會被排除計分，而它們正好是最難的那批（AJAX / ASP.NET / Angular /
C# / MES 全部沒有條目），於是指標會**偏樂觀**——看起來變好，其實是
難題被移出考卷。

實測涵蓋率（依 jobs_all 出現次數加權）：

    前 100 條 → 76.3%      前 200 條 → 87.5%
    前 150 條 → 82.9%      前 300 條 → 93.5%

前 150–200 條是投資報酬率的甜蜜點，572 條全補是完美主義。

## ★ 這個動作會影響面試以外的模組

`skills_v1.json` 是 A1 檢索、B1 計分、面試漏講三邊共用的。加 200 條會改變
正規化行為，**W2 那份 golden_pairs.v1 校準的 0.62 門檻可能會漂**。

所以流程必須是：
    1. 先跑本工具 --dry-run 看要加什麼
    2. --apply 產生新檔（不覆蓋原檔）
    3. **拿新檔重跑 run_golden_pairs.py**，確認 15 格沒有退步
    4. 沒退步才換上去

跳過第 3 步就是拿面試模組的方便去換掉 A1/B1 的正確性。

## 三件不能省的

**命名空間一致** —— sk: / skm: 分錯會讓差集靜默對不上，不報錯只回全 0。
本工具從既有條目**推導**分派規則，不自己發明。

**詞頭別名** —— `報表彙整與管理` 要同時登記 `報表彙整`，因為人講話講詞頭。
實測全語料中文技能 54% 抽得出詞頭，不補等於少掉一半命中。

**規格型條目排除** —— `中文打字20~50` 在 jobs_all 排第 105 名、
`中文打字50~75` 排第 45 名。它們不是技能是任職條件，人不會在面試裡講出
這串字，補進去會變成**永久的假指控**——每一次都出現。

用法:
    python tools/expand_vocab.py --vocab data/skills_v1.json ^
        --jobs data/jobs_all.jsonl --top 200 --dry-run
    python tools/expand_vocab.py --vocab data/skills_v1.json ^
        --jobs data/jobs_all.jsonl --top 200 --apply --out data/skills_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.gap import is_spec_entry  # noqa: E402
from app.pipeline.transcript import extract_head  # noqa: E402

_ASCII = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------- 慣例推導


def detect_id_scheme(vocab: list[dict]) -> tuple[str, int] | None:
    """從既有條目反推 skill_id 是怎麼產生的。

    ★ 不要自己發明編號規則。既有 id 長 sk:5384b25ce9 這樣,若新條目用別種
      方式編,兩批 id 混在同一個檔裡,之後沒有人分得出哪些是後來加的,
      也無法重現。所以先試著重現既有規則,重現不出來就明講、不硬猜。
    """
    for algo in ("md5", "sha1", "sha256"):
        for length in (8, 10, 12, 16):
            ok = 0
            for e in vocab[:20]:
                sid = str(e.get("skill_id") or e.get("id") or "")
                name = str(e.get("name_zh") or e.get("name") or "")
                if ":" not in sid or not name:
                    continue
                digest = getattr(hashlib, algo)(name.encode("utf-8")).hexdigest()
                if sid.split(":", 1)[1] == digest[:length]:
                    ok += 1
            if ok >= 3:
                return algo, length
    return None


def detect_namespace_rule(vocab: list[dict]) -> str:
    """推導 sk: / skm: 的分派規則。

    回傳 "ascii"（含拉丁字母走 sk:，其餘 skm:）或 "unknown"。
    推不出來就回 unknown，由使用者明確指定——猜錯會讓差集靜默失效。
    """
    hits = total = 0
    for e in vocab:
        sid = str(e.get("skill_id") or e.get("id") or "")
        name = str(e.get("name_zh") or e.get("name") or "")
        if ":" not in sid or not name:
            continue
        total += 1
        ns = sid.split(":", 1)[0]
        if (ns == "sk") == bool(_ASCII.search(name)):
            hits += 1
    return "ascii" if total and hits / total >= 0.85 else "unknown"


def make_id(name: str, scheme: tuple[str, int] | None, ns: str) -> str:
    if scheme:
        algo, length = scheme
        return f"{ns}:{getattr(hashlib, algo)(name.encode('utf-8')).hexdigest()[:length]}"
    return f"{ns}:{hashlib.md5(name.encode('utf-8')).hexdigest()[:10]}"


def inspect(vocab: list[dict]) -> None:
    """把既有詞彙表的命名空間慣例攤開來看。

    推導不出規則的時候不要猜——猜錯的後果是差集靜默對不上,不報錯、只回全 0。
    這個模式把資料攤開,規則通常一眼就看得出來。
    """
    from collections import Counter, defaultdict
    ns_of, names_of = Counter(), defaultdict(list)
    other_fields = Counter()
    for e in vocab:
        sid = str(e.get("skill_id") or e.get("id") or "")
        name = str(e.get("name_zh") or e.get("name") or "")
        ns = sid.split(":", 1)[0] if ":" in sid else "(無前綴)"
        ns_of[ns] += 1
        names_of[ns].append(name)
        for k in e:
            if k not in ("skill_id", "id", "name_zh", "name_en", "name", "aliases"):
                other_fields[k] += 1

    print(f"詞彙表 {len(vocab)} 條\n")
    print("【命名空間分佈】")
    for ns, n in ns_of.most_common():
        names = names_of[ns]
        ascii_n = sum(1 for x in names if _ASCII.search(x))
        print(f"  {ns:6s} {n:4d} 條   含拉丁字母 {ascii_n} 條（{ascii_n/n:.0%}）")
        print(f"           樣本：{'、'.join(names[:8])}")
    print()

    if other_fields:
        print("【其他欄位】命名空間可能跟其中某個欄位對齊：")
        for k, n in other_fields.most_common():
            vals = Counter(str(e.get(k)) for e in vocab if e.get(k) is not None)
            print(f"  {k:14s} 出現 {n} 次，取值：{dict(list(vals.most_common(6)))}")
        print()
        # 檢查有沒有欄位跟命名空間完全對齊。
        # ★ 常數欄位要排除:全表同一個值時「每個命名空間只對應一個值」自動成立,
        #   會報出毫無資訊的假對齊(實測 ucan_code 全空,就被誤報過)。
        for k in other_fields:
            distinct = {str(e.get(k)) for e in vocab if e.get(k) is not None}
            if len(distinct) <= 1:
                continue
            pairs = Counter()
            for e in vocab:
                sid = str(e.get("skill_id") or e.get("id") or "")
                ns = sid.split(":", 1)[0] if ":" in sid else "?"
                pairs[(ns, str(e.get(k)))] += 1
            by_ns = defaultdict(set)
            for (ns, v), _ in pairs.items():
                by_ns[ns].add(v)
            if all(len(v) == 1 for v in by_ns.values()) and len(by_ns) > 1:
                print(f"  ★ 欄位 `{k}` 與命名空間完全對齊："
                      f"{ {ns: list(v)[0] for ns, v in by_ns.items()} }")
                print("    → 新條目照這個欄位分派即可。")

    # 非空的分類欄位在各命名空間怎麼分佈——這通常才是前綴的真正意義
    for k in other_fields:
        vals = [(str(e.get(k) or ""), str(e.get("skill_id") or "")) for e in vocab]
        nonempty = [(v, sid) for v, sid in vals if v]
        if not nonempty or len(nonempty) == len(vals):
            continue
        by_ns = Counter(sid.split(":", 1)[0] for _, sid in nonempty)
        total_ns = Counter(str(e.get("skill_id") or "").split(":", 1)[0] for e in vocab)
        print(f"【欄位 `{k}` 有值的 {len(nonempty)} 條落在哪個命名空間】")
        for ns, n in by_ns.most_common():
            print(f"  {ns:6s} {n:3d} / {total_ns[ns]} 條有值（{n/total_ns[ns]:.0%}）")
        print("  → 若集中在單一命名空間,前綴編的多半是**資料來源**,不是技能性質。\n")

    print("看完之後用 --namespace-rule 指定，或改 make_id 的分派邏輯。")
    print("目前內建的規則只有 ascii（含拉丁字母走 sk:）與 all-skm（全部走 skm:）。")


# ---------------------------------------------------------------- 主流程


def surfaces_of(entry: dict) -> set[str]:
    out = set()
    for key in ("name_zh", "name_en", "name"):
        if entry.get(key):
            out.add(str(entry[key]).strip().lower())
    for a in entry.get("aliases") or []:
        if a:
            out.add(str(a).strip().lower())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--golden", type=Path,
                    help="面試黃金集目錄，用來回報補完之後覆蓋率變多少")
    ap.add_argument("--inspect", action="store_true",
                    help="只檢視既有詞彙表的命名空間慣例，不做任何修改")
    ap.add_argument("--include-golden", action="store_true",
                    help="額外補齊黃金集用到的全部技能（見下方兩種覆蓋率的說明）")
    ap.add_argument("--namespace-rule", choices=["ascii", "all-skm"], default=None,
                    help="推導不出來時手動指定")
    args = ap.parse_args()

    raw = json.loads(args.vocab.read_text(encoding="utf-8"))
    vocab = raw if isinstance(raw, list) else (
        raw.get("skills") or raw.get("entries") or list(raw.values()))

    if args.inspect:
        inspect(vocab)
        return 0

    scheme = detect_id_scheme(vocab)
    rule = args.namespace_rule or detect_namespace_rule(vocab)
    print(f"既有詞彙表 {len(vocab)} 條")
    print(f"  skill_id 規則  : {scheme[0] + '[:' + str(scheme[1]) + ']' if scheme else '推導不出來'}")
    print(f"  命名空間規則   : {rule}")
    if not scheme:
        print("  ⚠ 無法重現既有 id 規則，新條目會用 md5[:10]。"
              "兩批 id 混在同一檔裡會分不出來源——建議先確認原本怎麼編的。")
    if rule == "unknown":
        print("  ✗ 推導不出 sk: / skm: 的分派規則。分錯會讓差集靜默對不上，"
              "不報錯、只回全 0。請用 --namespace-rule 明確指定。")
        return 1

    known = set()
    for e in vocab:
        known |= surfaces_of(e)

    freq = Counter()
    for line in args.jobs.open(encoding="utf-8"):
        for s in (json.loads(line).get("requiredSkills") or []):
            if s:
                freq[s.strip()] += 1

    added, skipped_spec, already = [], [], 0
    for name, count in freq.most_common(args.top):
        if is_spec_entry(name):
            skipped_spec.append(name)
            continue
        if name.lower() in known:
            already += 1
            continue
        ns = "sk" if (rule == "ascii" and _ASCII.search(name)) else "skm"
        head = extract_head(name)
        aliases = [head] if head and head.lower() not in known else []
        added.append({
            "skill_id": make_id(name, scheme, ns),
            "name_zh": name,
            "name_en": "",
            "aliases": aliases,
            "source": "jobs_all.requiredSkills",
            "jd_frequency": count,
        })

    print(f"\njobs_all 前 {args.top} 名中：")
    print(f"  已存在        {already} 條")
    print(f"  規格型跳過     {len(skipped_spec)} 條  {skipped_spec[:6]}")
    print(f"  **新增        {len(added)} 條**  其中 {sum(1 for a in added if a['aliases'])} 條帶詞頭別名")

    # ★ 兩種覆蓋率是不同的東西，不要混為一談：
    #
    #   頻率加權覆蓋率  補到前 N 條能接到多少比例的**真實 JD 需求**
    #                  → 這是「產品好不好用」,補越多越好,前 150–200 條是甜蜜點
    #
    #   黃金集覆蓋率    黃金集用到的技能有多少在詞彙表裡
    #                  → 這是「量得準不準」,必須接近 100%,不然指標是偏的
    #                    (對不上的被排除計分,而它們往往是最難的那批)
    #
    # 後者不是「補越多越好」的問題,是**先決條件**:低於 80% 時上面所有
    # 數字都不可解讀。所以 --include-golden 是補齊測量前提,不是灌水——
    # 黃金集的技能本來就全部來自 13 份真實 JD 與 3 份真實履歷。
    if args.golden and args.golden.exists():
        gold = set()
        for f in sorted(args.golden.glob("ivw-*.json")):
            gold |= set(json.loads(f.read_text(encoding="utf-8"))["labels"]["skills"])
        after = known | {a["name_zh"].lower() for a in added} | {
            x.lower() for a in added for x in a["aliases"]}
        before_n = sum(1 for g in gold if g.lower() in known)
        after_n = sum(1 for g in gold if g.lower() in after)
        missing = sorted(g for g in gold if g.lower() not in after)

        if args.include_golden and missing:
            for name in missing:
                if is_spec_entry(name):
                    skipped_spec.append(name)
                    continue
                ns = "sk" if (rule == "ascii" and _ASCII.search(name)) else "skm"
                head = extract_head(name)
                added.append({
                    "skill_id": make_id(name, scheme, ns),
                    "name_zh": name, "name_en": "",
                    "aliases": [head] if head else [],
                    "source": "interview_golden_set",
                    "jd_frequency": freq.get(name, 0),
                })
            after_n = len(gold) - sum(1 for g in missing if is_spec_entry(g))
            print(f"\n黃金集覆蓋率  {before_n}/{len(gold)} ({before_n/len(gold):.1%})"
                  f"  →  {after_n}/{len(gold)} ({after_n/len(gold):.1%})"
                  f"   （--include-golden 另補了 {len(missing)} 條）")
        else:
            print(f"\n黃金集覆蓋率  {before_n}/{len(gold)} ({before_n/len(gold):.1%})"
                  f"  →  {after_n}/{len(gold)} ({after_n/len(gold):.1%})")
            if after_n / len(gold) < 0.80 and missing:
                print("  ⚠ 仍低於 80% 閘門，這些是長尾、頻率排不進前 N 名：")
                print(f"      {', '.join(missing)[:200]}")
                print("  → 加 --include-golden 一併補齊。這是補測量前提,不是灌水：")
                print("     黃金集的技能全部來自 13 份真實 JD 與 3 份真實履歷。")

    print("\n新增條目前 12 筆：")
    for a in added[:12]:
        alias = f"  別名 {a['aliases']}" if a["aliases"] else ""
        print(f"  [{a['jd_frequency']:4d}次] {a['skill_id']}  {a['name_zh']}{alias}")

    if not args.apply:
        print("\n（--dry-run 模式，未寫檔。確認無誤後加 --apply --out data/skills_v2.json）")
        return 0

    out = args.out or args.vocab.with_name(args.vocab.stem + "_expanded.json")
    merged = list(vocab) + added
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已寫入 {out}（{len(merged)} 條）")
    print("\n★ 下一步不是換上去，是先驗證沒有弄壞別人的東西：")
    print(f"    python tools/run_golden_pairs.py --vocab {out}")
    print("  golden_pairs.v1 那 15 格不能退步。skills_v1.json 是 A1 檢索、")
    print("  B1 計分、面試漏講三邊共用的，加 200 條會改變正規化行為，")
    print("  W2 校準的 0.62 門檻可能會漂。確認沒退步才換上去。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
