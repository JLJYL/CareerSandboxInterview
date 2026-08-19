"""面試黃金測試集產生器（成員 A，W1）。

從 persona_150 + jobs_all 生出 10 格的**標記工作台**。沿用 golden_pairs.v1 的規矩：
素材出處僅 persona_150 + jobs_all，**零生成內容**。

## 這支腳本生什麼、不生什麼

  生   10 格的 (履歷 × JD) 配對、候選技能池、has/wants 的機器預填、
       字串合併待裁決清單、每格的分層標籤與錄音指派
  不生 **逐字稿**。逐字稿必須是真人錄音跑真 STT 的原始輸出。

  理由寫在 RECORDING_PLAN.md。簡短版：手打或生成的逐字稿沒有填充詞、沒有
  STT 錯字、句子是文章不是話。用它校準出來的參數，W3 接上真 STT 那天會全崩，
  而且測試一路是綠的，你找不到原因。

## 三格標記裡，機器只能碰兩格

  has   履歷真的有嗎 → owned 判 2、learning 判 1（學習中＝真實的模糊地帶）
  wants JD 真的要嗎  → requiredSkills 判 2、只在散文出現判 1（要人裁）
  said  逐字稿講到嗎 → **一律 null，機器不得填**。沒有逐字稿就沒有這一格。

has/wants 的機器預填只是**省打字**，不是答案。人要逐條覆核，覆核完把
meta.labels_verified 改 true。沒改 true 的格子，interview_eval 不採計。

用法:
    python tools/build_interview_golden.py \
        --personas persona_150.jsonl --jobs jobs_all.jsonl \
        --out fixtures/golden/interview
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 選材
# 三份 persona：理工／企劃PM／財金。挑選理由寫在各自的 rationale。

PERSONAS = {
    "resume-013": "理工：碩二資工，8 個技能全是專有名詞硬技能（JavaScript/C#/Git/Angular），"
                  "表面掃描的舒適區，當基準線。",
    "resume-140": "企劃PM：大四行流，核心技能是『專案管理』『專案時間』——抽象名詞，"
                  "面試時只會被展演不會被指名。這是軟技能分層的主力。有 3 段經歷，錄音有話講。",
    "resume-079": "財金：大四會計，8 個技能是密集專業術語（帳務處理／會計或審計／稅務業務處理），"
                  "且多為 104 技能表的前綴截斷形，正規化壓力最大。",
}

#: 六段錄音。10 格由 6 段逐字稿支撐——同一段配不同 JD 可把 JD 變因隔離出來，
#: 跟 golden_pairs.v1「5 履歷×3 JD」同一個道理。
#: ★ 但這代表提及集指標的有效樣本數是 6 不是 10（誤差相關），報表要照實講。
RECORDINGS = {
    "T1": {"persona": "resume-013", "brief": "應徵前端工程師，正常發揮",
           "stratum": "baseline"},
    "T2": {"persona": "resume-013", "brief": "應徵前端工程師，請盡量把履歷上會的東西都講到；"
                                             "英文縮寫請用唸字母的方式講（H T M L、A J A X）",
           "stratum": "near-perfect-mention + stt-letter-split"},
    "T3": {"persona": "resume-140", "brief": "應徵專案管理職，正常發揮",
           "stratum": "soft-skill-demonstrated"},
    "T4": {"persona": "resume-140", "brief": "應徵企劃職，請多講故事和感受，"
                                             "**不要**刻意提到任何技能或工具名稱",
           "stratum": "near-zero-mention"},
    "T5": {"persona": "resume-079", "brief": "應徵財會助理，正常發揮",
           "stratum": "baseline"},
    "T6": {"persona": "resume-079", "brief": "應徵財務主管職，請講快一點、口語一點，不要停頓修飾",
           "stratum": "dense-jargon + heavy-disfluency"},
}

#: 10 格。每格＝(履歷, JD, 逐字稿)。
#: ★ 分層有兩軸，不要混為一談：
#:     配對軸  由 (履歷 × JD) 決定——硬/軟技能為主、JD 散文埋需求、明顯不匹配
#:     逐字稿軸 由**錄音指示**決定——幾乎全講到、幾乎沒講到、STT 拆字母
#:   「幾乎全講到」不是配對的性質，是對受試者下的指令。這點想清楚才不會白錄。
CELLS = [
    ("ivw-001", "resume-013", "754pk", "T1", ["hard-skill-dominant"],
     "基準線。全對照池最大交集（6 項），兩側都是專有名詞。這格若不漂亮，"
     "是實作壞了不是題目難。"),
    ("ivw-002", "resume-013", "935gh", "T1", ["hard-skill-dominant", "jd-prose-heavy"],
     "同一段逐字稿換 JD，隔離 JD 變因。desc 555 字，測散文掃描。"),
    ("ivw-003", "resume-013", "8fu7z", "T1", ["mismatched-pair"],
     "資工履歷配純專案管理 JD，交集應為空。測退化路徑：gap 該是空的，"
     "不該硬湊建議出來。"),
    ("ivw-004", "resume-013", "4cd0f", "T2", ["hard-skill-dominant", "near-perfect-mention",
                                              "stt-letter-split"],
     "受試者被要求講滿＋唸字母。gap 該接近空；若不空，多半是 normalize_for_scan "
     "的字母黏合沒生效。"),
    ("ivw-005", "resume-140", "8u1w8", "T3", ["soft-skill-dominant"],
     "★核心格。專業交集只有『專案管理』『專案時間』兩項且都是截斷形，"
     "另有三項通用文書技能。表面掃描在這格該近乎全滅——語意段非選配的證據。"
     "註：交集 2 是 persona 真實天花板（它只有兩個專案類技能），不是選材失誤。"),
    ("ivw-006", "resume-140", "8wz3o", "T3", ["soft-skill-dominant", "jd-prose-heavy",
                                              "huge-required-list"],
     "同段逐字稿換 JD。該 JD 有 31 項 requiredSkills、872 字散文——"
     "測 JD 位置衰減在長清單上會不會把尾巴壓到失去意義。"),
    ("ivw-007", "resume-140", "92o3z", "T4", ["mixed-skill", "near-zero-mention"],
     "受試者被要求不提任何技能名稱。gap 該接近交集全集——測天花板。"),
    ("ivw-008", "resume-079", "7crf2", "T5", ["dense-jargon", "truncated-surface"],
     "財會術語密集，四項專業交集有兩項是前綴截斷形"
     "（會計核算和帳務 vs 會計核算和帳務處理）。正規化壓力最大的一格。"),
    ("ivw-009", "resume-079", "92ho1", "T6", ["dense-jargon", "heavy-disfluency"],
     "口語快講版，填充詞會很多。同時驗 text_stats 的填充詞計數與 "
     "segmentation 標記是否誠實。"),
    ("ivw-010", "resume-079", "92u21", "T5", ["cross-domain-overlap"],
     "會計履歷配行政助理 JD，交集落在 Excel/Word/報表這類通用技能。"
     "測通用技能會不會在排序上壓過專業技能。"),
]


# ---------------------------------------------------------------- 載入


def load_personas(path: Path) -> dict[str, Any]:
    return {json.loads(l)["id"]: json.loads(l) for l in path.open(encoding="utf-8")}


def load_jobs(path: Path) -> dict[str, Any]:
    out = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        out[row.get("sourceId")] = row
    return out


# ---------------------------------------------------------------- 形狀轉換


def to_resume(persona: dict[str, Any]) -> list[dict[str, Any]]:
    """persona → **ExperienceDTO 清單**（B 的 compute 簽章要的形狀）。

    ★ 這裡照 golden_pairs.v1 的實際慣例，不是我原本猜的形狀。實測 5 份履歷：
      ExperienceDTO = {id, title, category, timeRange, description, tags}，
      **沒有頂層技能欄位**，persona.skills.owned 全部原樣進 experiences[0].tags。

      直接後果：履歷側的每一個技能都是 raw_tag 來源。SOURCE_WEIGHT 的
      structured/raw_tag 區分在這個資料形狀下不會發生，見 D1 異議第四條。

    learning 不進 tags——學習中≠會，進去等於憑空生 gap。它在 labels 預填 has=1
    （不計分區），那才是它的正確位置。
    """
    exps = persona.get("experiences") or []
    out: list[dict[str, Any]] = []
    for i, e in enumerate(exps):
        out.append({
            "id": f"e{i + 1}",
            "title": e.get("title", ""),
            "category": {"volunteer": "社團", "project": "學業", "campus": "社團",
                         "part_time": "工作", "internship": "工作"}.get(e.get("type", ""), "其他"),
            "timeRange": e.get("period", ""),
            "description": e.get("description", ""),
            "tags": [],
        })
    if not out:
        # golden_pairs.v1 對無經歷的 persona 就是補一筆空殼來掛 tags（見 r_resume-006）
        out.append({"id": "e1", "title": "主要經歷", "category": "工作",
                    "timeRange": "", "description": "", "tags": []})
    out[0]["tags"] = list(persona["skills"]["owned"])
    return out


def to_jd(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "_job_id": job.get("sourceId"),
        "_source": job.get("source"),
        "_url": job.get("url"),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "requiredSkills": list(job.get("requiredSkills") or []),
        "description": job.get("description") or "",
    }


# ---------------------------------------------------------------- 候選池


def propose_merges(resume_skills: list[str], jd_skills: list[str]) -> list[dict[str, Any]]:
    """提出「這兩個字串可能是同一個技能」的待裁決清單。

    ★ 只提出，不裁決。前綴啟發式會誤判——實測到的例子：
        專案管理 → 專案管理架構及專案說明   ← 這兩個不是同一件事
    所以每一條都標 decision=null，人來填 true/false。這一格就是
    golden_pairs.v1 裡「Photoshop=Adobe Photoshop 全名簡稱」那種裁決。
    """
    out = []
    for r in resume_skills:
        for j in jd_skills:
            if r == j:
                continue
            reason = None
            if j.startswith(r) and len(r) >= 3:
                reason = "JD 為履歷字串的擴充形（前綴截斷）"
            elif r.startswith(j) and len(j) >= 3:
                reason = "履歷為 JD 字串的擴充形"
            elif len(r) >= 3 and len(j) >= 3 and (r in j or j in r):
                reason = "字串包含"
            if reason:
                out.append({
                    "resume_surface": r,
                    "jd_surface": j,
                    "heuristic": reason,
                    "decision": None,  # ← 人填：true=同一技能 false=不同技能
                    "note": "",
                })
    return out


def scan_prose(description: str, surfaces: list[str]) -> dict[str, str]:
    """JD 散文裡出現、但 requiredSkills 沒列的技能 → {surface: 所在整句}。"""
    hits = {}
    lowered = description.lower()
    for s in surfaces:
        if not s or len(s) < 2:
            continue
        pos = lowered.find(s.lower())
        if pos < 0:
            continue
        lo, hi = pos, pos + len(s)
        while lo > 0 and description[lo - 1] not in "。！？；\n.!?;":
            lo -= 1
        while hi < len(description) and description[hi] not in "。！？；\n.!?;":
            hi += 1
        hits[s] = description[lo:hi].strip()[:120]
    return hits


def build_labels(resume: list[dict[str, Any]], jd: dict[str, Any],
                 persona: dict[str, Any]) -> dict[str, Any]:
    """候選池 ＋ has/wants 機器預填。said 一律 null。"""
    skills: dict[str, Any] = {}

    for exp in resume:
        for s in exp.get("tags") or []:
            skills.setdefault(s, {"has": 2, "wants": 0, "said": None,
                                  "_src": f"experiences[{exp['id']}].tags（raw_tag）",
                                  "note": ""})

    for s in persona["skills"].get("learning") or []:
        # 學習中 ≠ 會。預填 1 = 不計分區，這是它的正確位置。
        skills.setdefault(s, {"has": 1, "wants": 0, "said": None,
                              "_src": "resume.learning（學習中，預設不計分）", "note": ""})

    for s in jd["requiredSkills"]:
        entry = skills.setdefault(s, {"has": 0, "wants": 0, "said": None,
                                      "_src": "jd.requiredSkills", "note": ""})
        entry["wants"] = 2

    prose = scan_prose(jd["description"], list(skills.keys()))
    for s, sentence in prose.items():
        entry = skills[s]
        if entry["wants"] == 0:
            entry["wants"] = 1  # 只在散文出現 → 要人裁 1 或 2
            entry["_src"] += " + jd.description"
            entry["note"] = f"JD 散文提及：「{sentence}」→ 請判定 nice-to-have(1) 或明確要求(2)"

    resume_surfaces = [t for e in resume for t in (e.get("tags") or [])]
    return {
        "skills": skills,
        "alias_decisions": propose_merges(resume_surfaces, jd["requiredSkills"]),
        "gap_ranking": [],
        "free_add_done": False,
    }


# ---------------------------------------------------------------- 主流程


def build_case(cell, personas, jobs) -> dict[str, Any]:
    case_id, pid, jid, rid, strata, rationale = cell
    persona = personas[pid]
    job = jobs[jid]
    resume = to_resume(persona)
    jd = to_jd(job)
    rec = RECORDINGS[rid]

    return {
        "case_id": case_id,
        "meta": {
            "persona_id": pid,
            "job_source_id": jid,
            "job_url": jd["_url"],
            "recording_id": rid,
            "recording_brief": rec["brief"],
            "strata": strata,
            "rationale": rationale,
            # ── 三道閘門，全部預設 false ──────────────────────────
            "transcript_source": "PENDING",   # 錄完改 "stt"
            # 已由 STT_TERM_PROBE 實測確定,錄音時照填即可
            "stt_engine": "Android SpeechRecognizer (zh-TW) / Galaxy S24 / Android 16 "
                          "/ Google App 17.44.15.ve.arm64",
            "engine_filler_policy": "partial",
            "engine_notes": [
                "非詞彙填充音(嗯/呃)全數移除;詞彙型(那個/就是/然後)保留",
                "停頓約 1–2 秒即自動送出,長回答必然被切成多段",
                "已知語意反轉:「蠻有興趣」→「沒有興趣」,兩次獨立測試皆出現",
            ],
            "labels_verified": False,          # has/wants 人工覆核完改 true
            "said_labeled": False,             # said 標完改 true
            "labeled_by": [],
            "double_labeled": False,
        },
        "resume": resume,          # ← ExperienceDTO 清單，對齊 compute() 簽章
        "jd": jd,
        # ★ 多段。實測停頓 1–2 秒就自動送出,一次回答必然被切成數段。
        #   每段一個字串,原樣貼入,不要自己併成一段也不要加標點。
        #   下游用 "\n".join(segments) 餵給 compute()——換行本來就是斷句符號,
        #   所以段界會被 text_stats 當成句界,sentence_count 反而變得有意義。
        "transcript_segments": [],
        "labels": build_labels(resume, jd, persona),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True, type=Path)
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    personas = load_personas(args.personas)
    jobs = load_jobs(args.jobs)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for cell in CELLS:
        case = build_case(cell, personas, jobs)
        path = args.out / f"{case['case_id']}.json"
        path.write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
        labs = case["labels"]["skills"]
        merges = case["labels"]["alias_decisions"]
        # 字面交集：兩側同一個字串
        literal = [k for k, v in labs.items() if v["has"] == 2 and v["wants"] >= 1]
        # 潛在交集：再加上待裁決的合併（人裁 true 才會真的成立）
        potential = len(literal) + len({m["resume_surface"] for m in merges})
        summary.append({
            "case": case["case_id"],
            "persona": cell[1],
            "job": case["jd"]["title"][:20],
            "rec": cell[3],
            "pool": len(labs),
            "literal": len(literal),
            "potential": potential,
            "merges": len(merges),
        })

    print(f"{'case':<10}{'persona':<13}{'job':<22}{'rec':<5}"
          f"{'池':>4}{'字面交集':>7}{'潛在交集':>7}{'待裁決':>6}")
    for s in summary:
        print(f"{s['case']:<10}{s['persona']:<13}{s['job']:<22}{s['rec']:<5}"
              f"{s['pool']:>4}{s['literal']:>7}{s['potential']:>7}{s['merges']:>6}")

    zero = [s["case"] for s in summary if s["potential"] == 0]
    print(f"\n{len(summary)} 格已產生於 {args.out}")
    print(f"逐字稿 {len(set(c[3] for c in CELLS))} 段待錄音——見 RECORDING_PLAN.md")
    print("\n★ 字面交集 ≠ 潛在交集：差額全靠 alias_decisions 的人工裁決。")
    print("  裁決前這份黃金集算不出 gap，因為兩側根本對不上字串。")
    if zero:
        print(f"  以下格潛在交集為 0（若非刻意的 mismatched 分層，就是選材要換）：{', '.join(zero)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
