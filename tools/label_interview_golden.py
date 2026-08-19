"""黃金測試集標記草稿（成員 A，W1）。

把 said 標上去、把 alias_decisions 裁決掉，每一筆附證據與理由。

## said 的定義：問「這個人有沒有講」，不是問「文字裡有沒有」

兩種定義在 STT 出錯的地方會分岔，而分岔處正是整個產品的風險所在：

  定義甲  這個人有沒有講到  → 「加巴screen」算講了 JavaScript
  定義乙  文字裡看不看得出  → 「加巴screen」不算

採甲。理由：使用者的傷害不管是 STT 還是比對器造成的都一樣——
他講了 Angular，系統說他沒講，他就不信任這個系統。用乙會把 STT 的損害
排除在指標外，那等於量了一個跟使用者體驗無關的東西。

## 三態怎麼用

  2  確定講了：字面出現，或出現可辨識的走樣形（gats←Git、ajex←AJAX），
     或具體行為描述（展演）
  0  確定沒講：該講的情境出現了但沒有任何痕跡，且要點卡本來就要求不提
  1  判不出來：要點卡要求提、但逐字稿完全沒有痕跡——分不出是她跳過了
     還是引擎吃掉了。這種進不計分區，不讓它主導指標

★ 第三態是這份資料最需要的一格。T1 的 Angular 整個消失，我們無法從逐字稿
  判斷她到底講了沒有，硬標 2 或 0 都是在編造。

## 展演算 2 的邊界

「我跟組員說設一個 deadline、追進度、處理延遲」＝ 專案時間控管，標 2，
即使她一次都沒說出那五個字。這條規則是 T3 整段的意義所在。

反面：「我在餐廳打工過」推不出「抗壓性」，標 0。差別在於前者描述了
**具體的、可指認的行為**，後者只是提供了一個可以想像的情境。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.transcript import extract_head, normalize_for_scan  # noqa: E402

CASES = Path("fixtures/golden/interview")
REC_OF = {"ivw-001": "T1", "ivw-002": "T1", "ivw-003": "T1", "ivw-004": "T2",
          "ivw-005": "T3", "ivw-006": "T3", "ivw-007": "T4", "ivw-008": "T5",
          "ivw-009": "T6", "ivw-010": "T5"}

# ---------------------------------------------------------------- 人工裁決
# 格式：(錄音, 技能) -> (said, 理由)
# 只列判斷過的；字面找得到的由程式自動標 2，完全沒痕跡的自動標 0。

ADJUDICATED: dict[tuple[str, str], tuple[int, str]] = {
    # ── T1 ────────────────────────────────────────────────────────────
    ("T1", "JavaScript"): (2, "「前端的部分我是用加巴screen」——加巴≈java、screen≈script，"
                              "位置與語境都對得上，是可辨識的走樣形"),
    ("T1", "Git"): (2, "「版本控制的部分我們使用gats」——gats≈git，且前面就是「版本控制」"),
    ("T1", "Angular"): (1, "★不計分。「我是用加巴screen那時候我們也用了這個框架」——"
                           "「這個框架」是懸空指涉，暗示有個框架名被吃掉了，但也可能是"
                           "指前面那個。判不出她講了沒有，硬標 2 或 0 都是編造"),
    ("T1", "軟體程式設計"): (2, "展演：「用Collin去開發這些頁面跳轉的邏輯還有按鈕點擊的回饋」"
                             "是具體的程式開發行為描述"),
    ("T1", "ASP.NET"): (0, "要點卡要求不提，逐字稿無痕跡"),
    ("T1", "C#"): (0, "要點卡要求不提，逐字稿無痕跡"),
    ("T1", "AJAX"): (0, "要點卡要求不提，逐字稿無痕跡"),

    # ── T2 ────────────────────────────────────────────────────────────
    ("T2", "Angular"): (2, "「那時候還用了Andrew了這個框架」——有獨立的詞元佔著框架名的位置，"
                           "Andrew≈Angular 首音節相同，且 Angular 本來就是框架。"
                           "比 T1 明確，因為 T1 那個位置是空的"),
    ("T2", "Git"): (2, "「版本控制的部分呢我們是用Gate」"),
    ("T2", "ASP.NET"): (2, "「有用到ASP點net」——「點」是 . 的口語形"),
    ("T2", "C#"): (2, "「搭配C Sharp」——C Sharp 是 C# 的口語形，不是辨識錯誤"),
    ("T2", "AJAX"): (2, "「有做過ajex費同步處理」——ajex≈ajax，且「非同步」確認了語境"),
    ("T2", "軟體程式設計"): (2, "展演：描述前後端開發分工與實作"),

    # ── T3（★核心格）────────────────────────────────────────────────
    ("T3", "專案時間"): (2, "★展演。「我負責的部分是約訪談的時間」「有人在拖延就直接傳訊息"
                           "並且給他設定一個daylight[deadline]」「希望他在那個期限之前完成」"
                           "——排程、設期限、追延遲，三件都做到了，一次都沒說出那四個字"),
    ("T3", "專案管理"): (2, "★展演。分工（總共四個人、我負責約訪談與統整資料）＋"
                           "進度介入（傳訊息、設期限）＋交付（產出企劃書）"),
    ("T3", "Line"): (0, "⚠假陽性測資。「會想要設一個在LINE」實際是"
                        "「設一個 deadline」被切成「在LINE」。表面掃描會抓到 line 並"
                        "記成技能 Line——這是 STT 造成的假陽性，正解是 0"),
    ("T3", "簡報製作"): (2, "「我同時也首[熟]word還有Excel以及簡報製作」"),
    ("T3", "中文打字"): (0, "無痕跡"),

    # ── T4（要求什麼都不提）──────────────────────────────────────────
    ("T4", "專案時間"): (2, "★要點卡要求不提任何技能名，她確實沒提——但她"
                           "「跟組員說明現在專案的時辰[時程]的安排」「希望他在某個期限之前完成」，"
                           "展演成立。這格證明了『不指名』與『不展演』是兩件事，"
                           "near-zero-mention 分層只擋得住前者"),
    ("T4", "專案管理"): (2, "同上，展演成立"),
    ("T4", "Excel"): (0, "要求不提，確實沒提"),
    ("T4", "Word"): (0, "要求不提，確實沒提"),

    # ── T5 ────────────────────────────────────────────────────────────
    ("T5", "財務報表製作"): (2, "「我也有做未滿熟悉做財務報表類的工作」，"
                              "第四段再次出現「做財務報表的以及Excel工具上的應用」"),
    ("T5", "專案時間"): (2, "展演：「每週幫他們規劃複習的進度」"
                           "「安排他們每一週學習的章節的內容跟程度」——排程行為明確"),
    ("T5", "稅務業務處理"): (0, "要點卡要求不提，逐字稿無痕跡"),
    ("T5", "帳務處理"): (0, "無痕跡"),
    ("T5", "會計核算和帳務"): (0, "無痕跡"),
    ("T5", "PowerPoint"): (0, "無痕跡"),

    # ── T6 ────────────────────────────────────────────────────────────
    ("T6", "財務報表製作"): (2, "「然後我有在做財務報表」，第四段再次出現"),
    ("T6", "PowerPoint"): (0, "★要點卡要求提到，但她**實際沒講**——不是轉寫錯誤。"
                              "這是「要點卡≠said」的實例，照實標 0"),
    ("T6", "帳務處理"): (0, "要點卡要求不提，逐字稿無痕跡"),
}

# ---------------------------------------------------------------- 合併裁決

ALIAS_RULINGS: dict[tuple[str, str], tuple[bool, str]] = {
    ("HTML/CSS", "HTML"): (True, "HTML/CSS 是把兩個技能寫在一格，涵蓋 HTML"),
    ("HTML/CSS", "CSS"): (True, "同上，涵蓋 CSS"),
    ("專案時間", "專案時間╱進度控管"): (
        True, "履歷是 104 技能表的前綴截斷形，同一個技能"),
    ("專案管理", "專案管理架構及專案說明"): (
        False, "★不同技能。前者是做專案的能力，後者是「能說明專案管理架構」的"
               "知識性要求（面試會被問方法論）。前綴相同但職能不同，"
               "比照 golden_pairs.v1「財務報表製作≠報表彙整與管理」的保守判法"),
    ("會計核算和帳務", "會計核算和帳務處理"): (
        True, "前綴截斷形，同一個技能"),
    ("會計或審計", "會計或審計專業知識"): (
        True, "前綴截斷形，同一個技能"),
    ("帳務處理", "會計核算和帳務處理"): (
        True, "帳務處理是後者的核心動作，實務上同一件事"),
    ("帳務處理", "結帳作業與帳務處理"): (
        False, "★不同技能。結帳作業是期末關帳的特定流程，帳務處理是日常記帳。"
               "字串包含不代表職能相同，保守判"),
}


# ---------------------------------------------------------------- 執行


def label() -> None:
    transcripts = json.loads(
        (CASES / "raw/transcripts_v1.json").read_text(encoding="utf-8"))
    stats = {"auto2": 0, "auto0": 0, "manual": 0, "uncertain": 0}

    for path in sorted(CASES.glob("ivw-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        rid = REC_OF[case["case_id"]]
        text, _ = normalize_for_scan("\n".join(transcripts[rid]))

        for skill, lab in case["labels"]["skills"].items():
            key = (rid, skill)
            if key in ADJUDICATED:
                said, why = ADJUDICATED[key]
                lab["said"] = said
                lab["said_note"] = why
                stats["manual"] += 1
                if said == 1:
                    stats["uncertain"] += 1
                continue
            head = extract_head(skill)
            if skill.lower() in text or (head and head.lower() in text):
                lab["said"] = 2
                lab["said_note"] = "字面出現於逐字稿"
                stats["auto2"] += 1
            else:
                lab["said"] = 0
                lab["said_note"] = "逐字稿無任何痕跡"
                stats["auto0"] += 1

        for m in case["labels"]["alias_decisions"]:
            k = (m["resume_surface"], m["jd_surface"])
            if k in ALIAS_RULINGS:
                m["decision"], m["note"] = ALIAS_RULINGS[k]

        # 真漏講 = has2 & wants2 & said0，照「最想被告知」排序：
        # JD 明確要求 > 散文提及；同級時專業技能 > 通用文書
        GENERIC = {"Word", "Excel", "PowerPoint", "Outlook", "中文打字"}
        gaps = [s for s, v in case["labels"]["skills"].items()
                if v["has"] == 2 and v["wants"] == 2 and v["said"] == 0]
        case["labels"]["gap_ranking"] = sorted(
            gaps, key=lambda s: (s in GENERIC, s))
        case["labels"]["free_add_done"] = True
        case["meta"]["labeled_by"] = ["A-draft"]
        case["meta"]["said_labeled"] = False   # ← 人工覆核後才改 true
        case["meta"]["labels_verified"] = False
        path.write_text(json.dumps(case, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print(f"自動標 2（字面出現）      {stats['auto2']}")
    print(f"自動標 0（完全無痕跡）    {stats['auto0']}")
    print(f"人工裁決                {stats['manual']}  其中不計分 {stats['uncertain']}")


if __name__ == "__main__":
    label()
