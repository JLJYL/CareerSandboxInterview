"""面試模組黃金測試集跑分器（W1 D2–D3）。

同步點 1（D5）要求 `interview_eval` exit 0。本檔是那個裁判。

## 為什麼分開量三個 collector

漏講 = 履歷集 ∩ JD 集 − 提及集。只量最後那個集合的話,錯了有三個嫌犯在場,
不知道該修誰。所以黃金集標的是每個技能的三格（has / wants / said）,本檔
分別量三個 collector,再量端到端。壞掉的時候報表直接指出是誰。

## 為什麼提及集的兩種錯誤要分開報

    提及漏抓（人標 said=2,機器沒抓到）→ gap 多一條 → 對著講過的人說「你漏講了」
    提及誤抓（人標 said=0,機器抓到了）→ gap 少一條 → 少給一條建議

前者當場摧毀信任,後者只是可惜。合成一個 F1 會把這個量級差抹平,所以本檔
**不報 F1**,兩個方向分開報,門檻也分開設。

## 不計分區

任一格標 1 的技能,分子分母都不計。它處理的是真實存在的模糊地帶,
不排除的話,你調參數其實是在調對模糊案例的猜測。

用法:

    python tools/interview_eval.py --cases fixtures/golden/interview --vocab data/skills_v1.json
    python tools/interview_eval.py --cases ... --vocab ... --baseline baseline.json --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.interview_protocols import JDInput  # noqa: E402
from app.pipeline.gap import GapComputer  # noqa: E402
from app.pipeline import transcript as T_MOD  # noqa: E402
from app.pipeline.transcript import TranscriptAnalyzer  # noqa: E402

#: 目前的執行模式，供報表與 baseline 命名使用。用 list 是為了讓
#: print_report 這種模組層函式讀得到，不必到處傳參數。
MODE = ["surface"]
JD_SRC = ["catalog"]

# ---------------------------------------------------------------- 驗收門檻
# 【待校準】D5 之前這些是報表用的參考線,不當閘門。D5 凍結一份 baseline 之後,
# 閘門改成「不得比 baseline 退步」,而不是「不得低於絕對值」——絕對值訂太高會
# 讓同步點卡在沒意義的地方,訂太低等於沒有閘門。

#: 詞彙表覆蓋率下限。低於此值時整份評測不可解讀,拒絕寫 baseline。
#:
#: ★ 為什麼要硬擋:對不上的標記鍵會被排除計分,而它們往往正是最難的那批
#:   (實測 skills_v1.json 對 AJAX/ASP.NET/Angular/C#/MES 全部沒有條目)。
#:   於是指標會「上升」——不是變好了,是難題被移出考卷。這種偏誤沒有任何
#:   跡象可循,報表照樣印出漂亮的小數點,所以必須由工具擋住。
MIN_VOCAB_COVERAGE = 0.80

MIN_SAID_RECALL = 0.85  # 人標 said=2 的,機器要抓到幾成（漏抓＝會誤指控,從嚴）
MIN_SAID_PRECISION = 0.60  # 機器說講到的,人標也說講到的比例（誤抓＝少給建議,從寬）
MIN_GAP_RECALL = 0.70
MIN_GAP_PRECISION = 0.70
REGRESSION_TOLERANCE = 0.02  # baseline 比對容許的浮動


# ---------------------------------------------------------------- 指標


@dataclass
class SetMetric:
    """一個集合比對的計數。刻意不算 F1——見檔頭。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    skipped: int = 0     # 落在不計分區的（人標 1）
    unlabeled: int = 0   # 機器找到、但候選池裡從來沒這一條 → 不是錯，是沒標過
    unlabeled_items: list[str] = field(default_factory=list)
    fp_items: list[str] = field(default_factory=list)
    fn_items: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    def merge(self, other: "SetMetric") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.skipped += other.skipped
        self.unlabeled += other.unlabeled
        self.unlabeled_items.extend(other.unlabeled_items)
        self.fp_items.extend(other.fp_items)
        self.fn_items.extend(other.fn_items)


def compare(truth: dict[str, int], predicted: set[str], target: int = 2) -> SetMetric:
    """人標 == target 算正,== 0 算負,== 1 跳過（不計分區）。"""
    metric = SetMetric()
    for skill_id, label in truth.items():
        if label == 1:
            metric.skipped += 1
            continue
        is_true = label == target
        is_pred = skill_id in predicted
        if is_true and is_pred:
            metric.tp += 1
        elif is_true and not is_pred:
            metric.fn += 1
            metric.fn_items.append(skill_id)
        elif not is_true and is_pred:
            metric.fp += 1
            metric.fp_items.append(skill_id)
    # ★ 池外預測「不算錯」,單獨列。
    #
    #   原本把它算成 FP,理由是「否則自由補充那步就白做了」。但實測發現那是
    #   反向誘因:詞彙表補了 176 條之後,JD 散文掃描找到更多技能,那些不在
    #   標記池裡(池是用舊詞彙表建的),於是 FP 從 3 跳到 27,precision
    #   0.975→0.811。**詞彙表變好，分數變差。**
    #
    #   池外只代表「沒人標過」,不代表機器錯。要判定對錯必須有人去標它。
    #   所以改成單獨計數並提醒補標,不併進 FP。
    for skill_id in predicted - set(truth):
        metric.unlabeled += 1
        metric.unlabeled_items.append(skill_id)
    return metric


# ---------------------------------------------------------------- 單一 case


#: skill_id → name_zh。報表用。
#: ★ 沒有這個的話報表印的是 skm:4803e6b9e6 這種雜湊,人完全無法診斷——
#:   看到「假指控 8 筆」卻不知道是哪 8 個技能,那個數字就只能拿來焦慮,
#:   不能拿來修東西。
ID_TO_NAME: dict[str, str] = {}


def build_id_names(vocab: list[Any]) -> None:
    for entry in vocab:
        get = entry.get if isinstance(entry, dict) else lambda k, d=None: getattr(entry, k, d)
        sid = get("skill_id") or get("id")
        nm = get("name_zh") or get("name") or get("name_en")
        if sid:
            ID_TO_NAME[str(sid)] = str(nm or sid)


def show(skill_id: str) -> str:
    """skill_id → 「名稱」,查不到就回原 id。"""
    return ID_TO_NAME.get(skill_id, skill_id)


def build_name_index(vocab: list[Any]) -> dict[str, str]:
    """技能字串 → skill_id。

    ★ 黃金集的標記是用**技能字串**當鍵(人標的時候看到的是「JavaScript」,
      不是「sk:004」),但 mentioned_skills 依合約回 skill_id。
      兩邊命名空間不同,不翻譯就永遠零命中。
      這正是 D1 異議第二條要定死回傳型別的原因——差集算錯不會報錯,
      只會安靜地回全 0。
    """
    out: dict[str, str] = {}
    for entry in vocab:
        get = entry.get if isinstance(entry, dict) else lambda k, d=None: getattr(entry, k, d)
        sid = get("skill_id") or get("id")
        if not sid:
            continue
        for form in [get("name_zh"), get("name_en")] + list(get("aliases") or []):
            if form:
                out.setdefault(str(form), str(sid))
    return out


def load_embedding():
    """載入 bge-m3。載不起來就丟例外,**不得靜默退回表面掃描**。

    ★ 這是最難查的一種錯:退回之後數字看起來完全合理(0.471 是個正常數字),
      但它被貼上 semantic 的標籤寫進 baseline。之後每次回歸都在跟一個
      標錯模式的基準線比,而且沒有任何跡象。寧可炸掉。
    """
    import os
    if os.environ.get("INTERVIEW_EVAL_FAKE_EMBED"):
        # 只給測試用:驗流程接不接得起來,分數沒有語意。
        import numpy as np

        class _Fake:
            def embed(self, texts):
                out = []
                for t in texts:
                    v = np.zeros(64, dtype="float32")
                    for i, ch in enumerate(t):
                        v[(ord(ch) * 7 + i) % 64] += 1.0
                    n = np.linalg.norm(v)
                    out.append((v / n if n else v).tolist())
                return out

        print("⚠ INTERVIEW_EVAL_FAKE_EMBED 已設,使用假向量——數字無意義,不得寫 baseline。")
        return _Fake()
    from app.providers.embeddings import BgeM3Embedding
    return BgeM3Embedding()


def run_case(case: dict[str, Any], vocab: list[Any],
             embedding: Any = None,
             stt_aliases: dict[str, str] | None = None,
             jd_variant: dict[str, Any] | None = None) -> dict[str, Any]:
    analyzer = TranscriptAnalyzer(
        vocab=vocab, embedding=embedding, enable_semantic=embedding is not None,
        stt_aliases=stt_aliases,
    )
    name2id = build_name_index(vocab)
    # emit_soft 跟語意段連動:語意段沒開時軟技能的漏講判定全部低信心,
    # 不發出去。開了才發。見 gap.GapComputer 的說明。
    computer = GapComputer(analyzer)

    resume = case["resume"]
    raw_jd = case["jd"]
    # 黃金集存的是型錄原形;compute() 吃 JDInput。source 照實標,
    # 之後接 B 的抽取器輸出時會有 extracted 變體,兩者要分開統計。
    # ★ 兩種變體共用同一份人工標記。
    #   wants 是「這份 JD 到底要不要這個技能」——那是人對 JD 的判斷,
    #   跟 required_skills 怎麼產生的無關。所以同一份標記可以當兩種輸入的答案,
    #   兩邊跑完的差額就是**抽取損失的直接測量**,零額外標記成本。
    if jd_variant:
        jd = JDInput(
            required_skills=list(jd_variant.get("required_skills") or []),
            description=raw_jd.get("description") or "",
            source="extracted",
        )
    else:
        jd = JDInput(
            required_skills=list(raw_jd.get("requiredSkills") or []),
            description=raw_jd.get("description") or "",
            source="catalog",
        )
    # 多段 → 用換行接。換行本來就是 _PUNCT_RE 的斷句符號,所以 STT 的
    # 自動送出邊界會直接變成句界,sentence_count 從估算值變成實測值。
    segs = case.get("transcript_segments") or []
    transcript = "\n".join(s for s in segs if s.strip()) or case.get("transcript", "")
    machine = computer.explain(resume, jd, transcript)

    # 標記鍵翻成 skill_id;翻不到的丟進 unmapped 另外報,不要靜默吞掉
    raw_labels = case["labels"]["skills"]
    labels, unmapped = {}, []
    for name, lab in raw_labels.items():
        sid = name2id.get(name)
        if not sid:
            unmapped.append(name)
        elif sid in labels:
            # 兩個標記鍵指到同一個 skill_id（詞彙表把它們合併了）。
            # 直接覆寫會靜默丟掉一筆標記,所以取較保守的:has/wants 取小,
            # said 取大（寧可算他講了,少給建議勝過假指控）。
            prev = labels[sid]
            labels[sid] = {
                "has": min(prev.get("has", 0), lab.get("has", 0)),
                "wants": min(prev.get("wants", 0), lab.get("wants", 0)),
                "said": max(prev.get("said") or 0, lab.get("said") or 0),
            }
        else:
            labels[sid] = lab

    has_truth = {k: v.get("has", 0) for k, v in labels.items()}
    wants_truth = {k: v.get("wants", 0) for k, v in labels.items()}
    said_truth = {k: (v.get("said") if v.get("said") is not None else 0)
                  for k, v in labels.items()}

    # 漏講正解:三格全部人標,機器不參與 → 不循環
    gap_truth: dict[str, int] = {}
    for skill_id, lab in labels.items():
        h = lab.get("has", 0)
        w = lab.get("wants", 0)
        s = lab.get("said") if lab.get("said") is not None else 0
        if 1 in (h, w, s):
            gap_truth[skill_id] = 1  # 任一格模糊 → 整條進不計分區
        elif h == 2 and w == 2 and s == 0:
            gap_truth[skill_id] = 2
        else:
            gap_truth[skill_id] = 0

    machine_gap = set(machine["gap"])
    ranked = [c.skill_id for c in computer.compute(resume, jd, transcript)]
    human_rank = [name2id.get(x, x) for x in case["labels"].get("gap_ranking", [])]

    return {
        "case_id": case["case_id"],
        "source": case.get("meta", {}).get("transcript_source", "unknown"),
        "resume": compare(has_truth, set(machine["resume"])),
        "jd": compare(wants_truth, set(machine["jd"])),
        "said": compare(said_truth, set(machine["mentioned"])),
        "gap": compare(gap_truth, machine_gap),
        "rank_top1_hit": bool(human_rank and human_rank[0] in ranked[:3]),
        "rank_has_truth": bool(human_rank),
        "machine_rank": ranked,
        "human_rank": human_rank,
        "free_add_done": case["labels"].get("free_add_done", False),
        "unmapped": unmapped,
        "mapped": sorted(set(labels)),
    }


# ---------------------------------------------------------------- 報表


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {k: SetMetric() for k in ("resume", "jd", "said", "gap")}
    by_source: dict[str, SetMetric] = {}
    rank_hits = rank_total = 0

    for res in results:
        for key in totals:
            totals[key].merge(res[key])
        bucket = by_source.setdefault(res["source"], SetMetric())
        bucket.merge(res["said"])
        if res["rank_has_truth"]:
            rank_total += 1
            rank_hits += int(res["rank_top1_hit"])

    return {
        "totals": totals,
        "by_source": by_source,
        "rank_top1_in_top3": (rank_hits / rank_total) if rank_total else None,
        "rank_total": rank_total,
    }


def print_report(results: list[dict[str, Any]], agg: dict[str, Any]) -> float:
    print(f"\n{'=' * 68}")
    print(f"面試黃金測試集  {len(results)} 個 case   "
          f"模式：{MODE[0]}／JD 來源：{JD_SRC[0]}")
    if MODE[0] == "surface":
        print("  （表面掃描＋詞頭匹配。語意段未開，加 --semantic 才是完整管線）")
    else:
        print(f"  （語意段開啟：視窗 {T_MOD.SEMANTIC_WINDOW}／"
              f"一般門檻 {T_MOD.MENTION_THRESHOLD}／"
              f"履歷候選門檻 {T_MOD.CANDIDATE_THRESHOLD}）")
    print("=" * 68)

    print("\n【三個 collector 各自的準確率】")
    print(f"  {'集合':<10}{'precision':>11}{'recall':>9}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'不計分':>7}{'未標記':>7}")
    labels = {"resume": "履歷集", "jd": "JD 集", "said": "提及集", "gap": "漏講(端到端)"}
    for key, name in labels.items():
        m = agg["totals"][key]
        print(f"  {name:<10}{m.precision:>11.3f}{m.recall:>9.3f}"
              f"{m.tp:>5}{m.fp:>5}{m.fn:>5}{m.skipped:>7}{m.unlabeled:>7}")

    said = agg["totals"]["said"]
    print("\n【提及集的兩種錯誤——量級不一樣,分開看】")
    print(f"  漏抓 {said.fn} 個（人說講了、機器沒抓到 → 會對使用者誤指控,從嚴）")
    if said.fn_items:
        print(f"       {'、'.join(sorted({show(x) for x in said.fn_items})[:14])}")
    print(f"  誤抓 {said.fp} 個（人說沒講、機器認為講了 → 少給一條建議,從寬）")
    if said.fp_items:
        print(f"       {'、'.join(sorted({show(x) for x in said.fp_items})[:14])}")

    tot_unlab = sum(agg["totals"][k].unlabeled for k in ("resume", "jd", "said"))
    if tot_unlab:
        items = sorted({show(x) for k in ("resume", "jd", "said")
                        for x in agg["totals"][k].unlabeled_items})
        print(f"\n【未標記 {tot_unlab} 筆】機器找到但候選池沒有的技能——"
              "不是錯，是沒人標過")
        print(f"  {'、'.join(items[:14])}")
        print("  → 這些多半是補詞彙表之後 JD 散文掃描新找到的。要判定對錯，")
        print("     得把它們加進候選池再標一次 wants。在那之前它們不計分。")

    gap = agg["totals"]["gap"]
    if gap.fp_items:
        print(f"\n【漏講的 {gap.fp} 筆假指控】使用者其實講了，系統卻說他沒講")
        print(f"  {'、'.join(sorted({show(x) for x in gap.fp_items}))}")
        print("  → 這是這個產品最傷的錯誤。每一筆都要能說出成因："
              "STT 轉爛、還是展演式陳述。")

    if len(agg["by_source"]) > 1:
        print("\n【STT vs 手打——兩群差很多的話,這就是本次最重要的發現】")
        for source, m in sorted(agg["by_source"].items()):
            print(f"  {source:<14} 提及集 precision {m.precision:.3f}  recall {m.recall:.3f}")

    if agg["rank_top1_in_top3"] is not None:
        print(f"\n【排序】人工首選落在機器前三名: "
              f"{agg['rank_top1_in_top3']:.3f}（{agg['rank_total']} 個 case 有排序標記）")

    unmapped = sorted({n for r in results for n in r.get("unmapped", [])})
    mapped = sorted({n for r in results for n in r.get("mapped", [])})
    total = len(unmapped) + len(mapped)
    coverage = (len(mapped) / total) if total else 1.0
    if unmapped:
        print(f"\n  ⚠ 詞彙表覆蓋率 {coverage:.1%}"
              f"（{len(mapped)}/{total}），{len(unmapped)} 個標記鍵對不上,已排除計分")
        print(f"      {', '.join(unmapped[:14])}")
        if coverage < MIN_VOCAB_COVERAGE:
            print(f"\n  ✗ 覆蓋率低於 {MIN_VOCAB_COVERAGE:.0%}，上面所有數字都不可解讀。")
            print("      被排除的往往是最難的技能，指標會因此偏樂觀——"
                  "看起來變好，其實是難題被移出考卷。")
            print("      先補詞彙表，不要調參數。")
    return coverage

    if not unmapped:
        print(f"\n  詞彙表覆蓋率 100%（{total}/{total}）")

    missing = [r["case_id"] for r in results if not r["free_add_done"]]
    if missing:
        print(f"\n  ⚠ 未完成自由補充的 case: {', '.join(missing)}")
        print("    候選池只有機器輸出 → 這些 case 的 recall 數字偏樂觀,不可信")


def check_gates(agg: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    said, gap = agg["totals"]["said"], agg["totals"]["gap"]

    if baseline:
        for key, name in (("said", "提及集"), ("gap", "漏講")):
            for metric in ("precision", "recall"):
                now = getattr(agg["totals"][key], metric)
                was = baseline.get(key, {}).get(metric)
                if was is not None and now < was - REGRESSION_TOLERANCE:
                    failures.append(f"{name} {metric} 退步: {was:.3f} → {now:.3f}")
    else:
        checks = (
            (said.recall, MIN_SAID_RECALL, "提及集 recall"),
            (said.precision, MIN_SAID_PRECISION, "提及集 precision"),
            (gap.recall, MIN_GAP_RECALL, "漏講 recall"),
            (gap.precision, MIN_GAP_PRECISION, "漏講 precision"),
        )
        for value, floor, name in checks:
            if value < floor:
                failures.append(f"{name} {value:.3f} < 門檻 {floor:.3f}")
    return failures


# ---------------------------------------------------------------- 入口


def load_cases(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases = []
    for f in files:
        if f.name.startswith("_") or "template" in f.name:
            continue
        cases.append(json.loads(f.read_text(encoding="utf-8")))
    return cases


# ---------------------------------------------------------------- 就緒閘門
# ★ 這一節是結構性防呆,不是提醒。
#
# 未完成的格子被靜默納入計分,是這種專案最容易發生也最難察覺的事故:數字看起來
# 很好,但它其實只量了一半的資料。所以「不採計」由工具強制,不靠人記得。
#
# 一格要「就緒」必須四件事都成立:
#   1. transcript 非空
#   2. transcript_source 是 stt 或 handwritten（不是 PENDING、不是 synthetic）
#   3. labels_verified — has/wants 的機器預填已被人覆核
#   4. said_labeled    — said 那格已人工標完
#
# 合成逐字稿要納入必須明寫 --allow-synthetic,而且報表會整份標記為不可校準。

READY_SOURCES = {"stt", "handwritten"}


def readiness(case: dict[str, Any]) -> tuple[bool, list[str]]:
    meta = case.get("meta", {})
    reasons = []
    segs = case.get("transcript_segments") or []
    if not (case.get("transcript") or "").strip() and not any(x.strip() for x in segs):
        reasons.append("逐字稿空白")
    src = meta.get("transcript_source")
    if src == "PENDING":
        reasons.append("尚未錄音")
    elif src == "synthetic":
        reasons.append("合成逐字稿（不可用於校準）")
    elif src not in READY_SOURCES:
        reasons.append(f"transcript_source 不明: {src!r}")
    if not meta.get("labels_verified"):
        reasons.append("has/wants 未經人工覆核")
    if not meta.get("said_labeled"):
        reasons.append("said 未標記")
    return (not reasons), reasons


def partition(cases: list[dict[str, Any]], allow_synthetic: bool
              ) -> tuple[list[dict[str, Any]], list[tuple[str, list[str]]]]:
    ready, blocked = [], []
    for case in cases:
        ok, reasons = readiness(case)
        if not ok and allow_synthetic:
            reasons = [r for r in reasons if "合成逐字稿" not in r]
            ok = not reasons
        (ready.append(case) if ok else blocked.append((case["case_id"], reasons)))
    return ready, blocked


def load_vocab(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("skills", "entries", "items", "data"):
            if key in data:
                return data[key]
        return list(data.values())
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="面試黃金測試集跑分器")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="與凍結的 baseline 比對,只看退步")
    parser.add_argument("--gate", action="store_true", help="不合格時 exit 1（D5 同步點用）")
    parser.add_argument("--write-baseline", type=Path, help="把本次結果寫成 baseline")
    parser.add_argument("--jd-source", choices=["catalog", "extracted"],
                        default="catalog",
                        help="JD 需求的來源。catalog=104 型錄的 requiredSkills；"
                             "extracted=B 的抽取器跑同一份 description 的產出。"
                             "★ 正式環境是 extracted，型錄資料只有黃金集有。")
    parser.add_argument("--jd-variants", type=Path,
                        default=Path("data/jd_variants_extracted.json"),
                        help="抽取器輸出（--jd-source extracted 時使用）")
    parser.add_argument("--stt-aliases", type=Path,
                        help="STT 轉寫對照表（stt_confusions.v1.json）。"
                             "safe_aliases 會併進表面掃描,是確定性的一道防線。")
    parser.add_argument("--semantic", action="store_true",
                        help="開語意段（載 bge-m3 約 2.3GB）。預設關閉,秒跑完。")
    parser.add_argument("--by-case", action="store_true",
                        help="逐格分解。「哪一格壞掉」比「整體平均」有用得多。")
    parser.add_argument("--draft", action="store_true",
                        help="納入尚未人工覆核的草稿標記。輸出全程標示為草稿,不可寫 baseline。")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="納入合成逐字稿。結果一律標為不可校準,只能拿來驗管線通不通。")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        print("找不到任何 case", file=sys.stderr)
        return 1
    vocab = load_vocab(args.vocab)
    build_id_names(vocab)

    # ★ STT 對照表原本只是躺在 data/ 的一份文件——做好了但沒接上管線,
    #   而且不會報錯。實測「C Sharp」(C# 的口語形)一直在漏抓清單裡,
    #   就是因為 safe_aliases 從來沒被載進去。
    jd_variants = None
    if args.jd_source == "extracted":
        if not args.jd_variants.exists():
            print(f"✗ 找不到 {args.jd_variants}。extracted 模式需要抽取器輸出。")
            return 1
        jd_variants = json.loads(args.jd_variants.read_text(encoding="utf-8"))["variants"]
        print(f"JD 來源：extracted（{len(jd_variants)} 格，來自 B 的抽取器）")

    stt_aliases = None
    if args.stt_aliases and args.stt_aliases.exists():
        conf = json.loads(args.stt_aliases.read_text(encoding="utf-8"))
        stt_aliases = conf.get("safe_aliases") or {}
        print(f"載入 STT 安全別名 {len(stt_aliases)} 條"
              f"（碰撞別名不載,那要靠履歷條件判定）")

    embedding = None
    if args.semantic:
        print("載入 bge-m3（約 2.3GB），第一次會很久⋯")
        embedding = load_embedding()   # 載不起來直接炸,不退回表面掃描
    MODE[0] = "semantic" if args.semantic else "surface"
    JD_SRC[0] = args.jd_source

    if args.draft:
        for c in cases:
            c["meta"]["labels_verified"] = True
            c["meta"]["said_labeled"] = True
    ready, blocked = partition(cases, args.allow_synthetic)

    if blocked:
        print(f"\n【未就緒 {len(blocked)}/{len(cases)} 格,不列入計分】")
        for case_id, reasons in blocked:
            print(f"  {case_id}: {'、'.join(reasons)}")

    if not ready:
        print("\n沒有任何就緒的格子。這是預期狀態——逐字稿還沒錄。")
        print("完成順序: 錄音 → 填 transcript 與 stt_engine → 覆核 has/wants "
              "→ 標 said → 三個旗標改 true")
        return 0 if not args.gate else 1

    results = [
        run_case(c, vocab, embedding, stt_aliases,
                 jd_variants.get(c["case_id"]) if jd_variants else None)
        for c in ready
    ]
    agg = aggregate(results)
    coverage = print_report(results, agg)

    if args.by_case:
        # ★ 整體平均會藏住分布。實測 precision 0.467 這個數字，拆開來看是
        #   某一個 persona 幾乎全錯、其他兩個幾乎全對——那是完全不同的處置。
        print("\n【逐格分解】")
        print(f"  {'case':<10}{'錄音':<6}{'漏講TP':>7}{'假指控':>7}{'precision':>11}   假指控的是")
        for r, c in zip(results, ready):
            g = r["gap"]
            p = g.tp / (g.tp + g.fp) if (g.tp + g.fp) else 1.0
            names = "、".join(sorted({show(x) for x in g.fp_items})) or "—"
            print(f"  {r['case_id']:<10}{c['meta'].get('recording_id', '?'):<6}"
                  f"{g.tp:>7}{g.fp:>7}{p:>11.3f}   {names}")

    # 有效樣本數 ≠ 格數。同一段逐字稿被多格共用時,提及集的誤差是相關的。
    transcripts = {r["case_id"][:0] or c["meta"].get("recording_id") or c["case_id"]
                   for r, c in zip(results, ready)}
    if len(transcripts) < len(ready):
        print(f"\n  ⚠ {len(ready)} 格只由 {len(transcripts)} 段逐字稿支撐。"
              f"提及集指標的有效樣本數是 {len(transcripts)},不是 {len(ready)}——"
              f"同段逐字稿的誤差相關,不要當成獨立觀測。")

    if args.allow_synthetic:
        print("\n  ⚠⚠ 本次含合成逐字稿。這些數字只證明管線跑得通,"
              "不得用於調參數,也不得寫成 baseline。")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if baseline and baseline.get("_jd_source", "catalog") != JD_SRC[0]:
        print(f"\n  ✗ baseline 的 JD 來源是 {baseline.get('_jd_source')}，"
              f"本次是 {JD_SRC[0]}。兩者量的是不同輸入，比對沒有意義。")
        return 1
    if baseline and baseline.get("_mode") and baseline["_mode"] != MODE[0]:
        print(f"\n  ✗ baseline 是 {baseline['_mode']} 模式，本次是 {MODE[0]} 模式。"
              "\n      兩者量的是不同管線，比對沒有意義。請用對應模式的 baseline。")
        return 1
    failures = check_gates(agg, baseline)

    if args.write_baseline:
        if coverage < MIN_VOCAB_COVERAGE:
            print(f"\n  ✗ 拒絕寫入 baseline：詞彙表覆蓋率僅 {coverage:.1%}。"
                  "凍一份建立在殘缺詞彙表上的基準線,之後每次回歸都在跟它比。")
            return 1
        if args.draft:
            print("\n  ✗ 拒絕寫入 baseline：本次用的是未覆核的草稿標記。")
            return 1
        if args.allow_synthetic:
            print("\n  ✗ 拒絕寫入 baseline：本次含合成逐字稿。"
                  "拿合成資料當基準線,之後每次回歸都在跟一個假的過去比。")
            return 1
        # ★ 兩種模式的數字不能互比。表面掃描 0.471 與語意段 0.727 量的是
        #   不同的管線,混在同一份 baseline 裡會讓之後的回歸完全失去意義——
        #   看到「退步」時分不出是程式壞了還是模式不同。所以檔名帶模式,
        #   而且 baseline 內部也記模式,載入時對不上會擋。
        out = args.write_baseline
        import os
        if os.environ.get("INTERVIEW_EVAL_FAKE_EMBED"):
            print("\n  ✗ 拒絕寫入 baseline：本次使用假向量。")
            return 1
        tag = f"{MODE[0]}_{JD_SRC[0]}"
        if tag not in out.stem:
            out = out.with_name(f"{out.stem}_{tag}{out.suffix}")
        snapshot = {
            "_mode": MODE[0],
            "_jd_source": JD_SRC[0],
            # ★ 參數要跟數字一起凍。沒有這些欄位的話,回歸時看到差異
            #   分不出是程式壞了還是參數被改過。
            "_semantic": ({"window": T_MOD.SEMANTIC_WINDOW,
                           "threshold": T_MOD.MENTION_THRESHOLD,
                           "candidate_threshold": T_MOD.CANDIDATE_THRESHOLD}
                          if MODE[0] == "semantic" else None),
            "_stt_aliases": len(stt_aliases) if stt_aliases else 0,
            "_cases": len(ready),
            **{key: {"precision": m.precision, "recall": m.recall}
               for key, m in agg["totals"].items()},
        }
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n  baseline 已寫入 {out}（{len(ready)} 格，模式 {MODE[0]}）")

    print()
    if failures:
        print("  ✗ 未通過:")
        for line in failures:
            print(f"      {line}")
        if args.gate:
            return 1
        print("      （未帶 --gate,僅報告）")
    else:
        print("  ✓ 全部通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
