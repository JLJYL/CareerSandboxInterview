"""JD 抽取結果的機械正規化。

【為什麼要有這一層】
兩個問題用 prompt 修過兩次都失敗:

    v1 → v2  加了「輸出一律使用台灣用語」的原則性說明
             結果:「用戶界面實現」只變成「用戶界面」,詞根沒改
    v2 → v3  改成 16 組對照表加「這不是選項」
             結果:「代碼測試」仍在,而且多出「數據可視化」「用戶體驗優化」

同樣地,「C# 不可加後綴」寫進禁則兩次,輸出仍然是「C#程式開發」。

這兩件事本質上是**確定性字串替換**,不是判斷。用 prompt 解決確定性問題
必然有殘留率,而且 prompt 越長殘留越多——v3 已經 2544 字、六條禁則。

沿用既有的三層防禦:規則 → 示範 → 機械修復。前兩層做過了,這裡是第三層。

【這一層只做兩件事】
1. 用語正規化:非台灣用語換成台灣用語
2. 專有名詞去後綴:C#程式開發 → C#

不做語意判斷。判斷仍然留在 prompt。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 用語對照
# ---------------------------------------------------------------------------
# 只收**明確**的對照。有歧義的一律不收,寧可漏改也不要改錯:
#   數據 → 資料   不收。「數據分析」在台灣通用,但「數據庫」不通用,
#                 所以只收「數據庫」這個組合,不收單獨的「數據」。
#   質量 → 品質   不收。台灣「質量」在物理脈絡下是另一個意思。
#   優化 → 最佳化 不收。台灣兩者皆用。
#
# 長詞排前面,因為替換是依序做的,先換長的才不會被短的截斷。

TERM_MAP: tuple[tuple[str, str], ...] = (
    # 組合詞優先
    ("用戶體驗", "使用者體驗"),
    ("用戶界面", "使用者介面"),
    ("數據可視化", "資料視覺化"),
    ("數據庫", "資料庫"),
    ("移動端", "行動裝置"),
    ("服務器", "伺服器"),
    ("可視化", "視覺化"),
    # 單詞
    ("用戶", "使用者"),
    ("界面", "介面"),
    ("交互", "互動"),
    ("代碼", "程式碼"),
    ("軟件", "軟體"),
    ("網絡", "網路"),
    ("視頻", "影片"),
    ("信息", "資訊"),
    ("調試", "偵錯"),
    ("項目管理", "專案管理"),
)


# ---------------------------------------------------------------------------
# 專有名詞去後綴
# ---------------------------------------------------------------------------
# 模型常在專有名詞後面補一個通用中文後綴(C#程式開發、Excel操作)。
# 詞彙表的條目是裸的專有名詞,加了後綴就對不上,那條技能永遠抓不到。

GENERIC_SUFFIXES: tuple[str, ...] = (
    "程式設計",
    "程式開發",
    "程式撰寫",
    "開發能力",
    "操作能力",
    "使用能力",
    "應用能力",
    "相關技術",
    "開發",
    "操作",
    "使用",
    "應用",
    "撰寫",
    "技術",
    "能力",
    "工具",
)

_PROPER_NOUN_HEAD = re.compile(r"^([A-Za-z][A-Za-z0-9加.\+#/\-]*)(?=[\u4e00-\u9fff])")
"""開頭是拉丁字母、後面接中文字元 —— 判定為「專有名詞 + 中文後綴」的形狀。

刻意用形狀判斷而非白名單:白名單永遠列不完,而且新技術一直出現。
形狀判斷的誤傷面積很小,因為以拉丁字母開頭又緊接中文的組合,
在技能標籤裡幾乎只有這一種情況。
"""


def normalize_locale(term: str) -> str:
    """非台灣用語換成台灣用語。"""
    out = term
    for src, dst in TERM_MAP:
        out = out.replace(src, dst)
    return out


def strip_proper_noun_suffix(term: str) -> str:
    """拉丁字母開頭的專有名詞,去掉後面的通用中文後綴。

        C#程式開發   → C#
        Excel操作    → Excel
        TypeScript開發 → TypeScript

    不動的情況:

        RESTful API      後面不是中文
        前端開發          開頭不是拉丁字母
        Google雲端試算表   後綴不在通用清單裡(那是產品名的一部分)
    """
    m = _PROPER_NOUN_HEAD.match(term)
    if not m:
        return term
    head = m.group(1)
    tail = term[len(head) :]
    if tail in GENERIC_SUFFIXES:
        return head
    return term


def normalize_skill(term: str) -> str:
    """單一技能字串的完整正規化。順序:先用語、後去後綴。"""
    return strip_proper_noun_suffix(normalize_locale(term))


def normalize_skills(items: list[str]) -> tuple[list[str], list[str]]:
    """整批正規化,回傳 (結果, notices)。

    正規化可能製造重複(用戶界面與使用者介面),所以要重新去重。
    **順序一律保留** —— 順序即重要性順序,成員 A 的 JD_POSITION_DECAY 依賴它。
    """
    notices: list[str] = []
    changed: list[str] = []
    out: list[str] = []
    seen: set[str] = set()

    for item in items:
        fixed = normalize_skill(item)
        if fixed != item:
            changed.append(f"{item}→{fixed}")
        key = fixed.casefold().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(fixed)

    if changed:
        notices.append("機械正規化:" + "、".join(changed))
    if len(out) < len(items):
        notices.append(f"正規化後去重,由 {len(items)} 項縮為 {len(out)} 項")
    return out, notices
