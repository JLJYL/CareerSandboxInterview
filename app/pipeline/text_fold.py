"""繁簡折疊:只用於比對,不用於顯示。

【為什麼需要】
Whisper 對中文的繁簡輸出沒有一致保證。成員 A 引怡君的實測:
同一位講者、同一支後端,五段 App 錄音輸出繁體、一段電腦錄音輸出簡體。

而這個模組有三處拿 LLM 的輸出跟逐字稿做字串比對:

    verify_star_verbatim   fromAnswer 必須是逐字稿子字串
    verify_evidence        協作維度的 evidence 必須是逐字稿子字串
    ungrounded_numbers     better 裡的數量必須出現在原回答

逐字稿是簡體而 prompt 是繁體時,LLM 很可能把引用「順手」轉成繁體
——那是它讀到的語境。比對就會失敗,而失敗的方式特別糟:
**正確的引用被當成編造**,starParts 降成未命中、協作維度標成依據不可信。

實測確認兩處會誤判(ungrounded_numbers 因為只比對數字所以沒事)。

【為什麼是折疊不是轉換】
折疊只求「兩邊化到同一形」,不保證化出來的是正確的繁體或簡體。
比對用夠了,而且不需要完整的映射表——完整轉換要處理一對多
(發→發/髮、乾→乾/幹),那需要詞彙層的判斷,是另一件事。

顯示給使用者的文字一律用原文,不要經過這裡。

【方向:繁 → 簡,不是簡 → 繁】
早期版本折向繁體,結果一對多的字(发→發/髮、干→乾/幹)只能整批不收,
表因此只能手寫、只收 157 筆。實測涵蓋不足:

    軟體程式設計 vs 软体程式设计   ✗ 對不上
    報表製作     vs 报表制作       ✗ 對不上

那兩個都是黃金測試集裡真實存在的技能——也就是說遇到簡體輸入時,
verify_evidence 仍然會把正確的引用判成編造。

折向簡體是多對一:髮→发、發→发,確定性,表可以做完整(4105 筆)。
「頭髮」折成「头发」是正確的簡體,比對仍然正確。

【★ 折過的字串不可拿去顯示】
乾/幹/干 全折成 干,所以「乾淨」跟「幹部」折疊後共用字首。
比對沒問題(兩邊都折,一致),但顯示會變成錯字。

四處呼叫端目前都只拿來比對,回傳給使用者的是原文:

    verify_star_verbatim  比對後回問題描述,引用本身用原文
    verify_evidence       同上
    ungrounded_numbers    **從原文抽 token,只用折過的版本比對**
                          ——早期版本從折過的字串抽,「三個」會顯示成「三个」

新增呼叫端時要檢查這一條。
"""

from __future__ import annotations

import json
import pathlib

# 簡 → 繁。只收面試逐字稿與技能詞會出現的字。
# 一對多的情況(發/髮、乾/幹、餘/余)一律不收——那要詞彙層判斷,
# 收了會把「頭髮」折成「頭發」,製造新的比對失敗。
#: 繁體→簡體的字元層對照,從 data/t2s_chars.v1.json 載入(4105 筆)。
#:
#: 來源是 OpenCC 的 TSCharacters.txt(Apache-2.0),只收 1 字元對 1 字元
#: ——fold_chars 依賴等長(gap.py 用位移切原文當證據)。
_T2S_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "t2s_chars.v1.json"


def _load_t2s() -> dict[str, str]:
    """讀繁簡對照。讀不到就回空 dict——降級但不擋啟動。"""
    try:
        raw = json.loads(_T2S_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in raw.get("chars", {}).items() if len(k) == 1 == len(v)}


_T2S: dict[str, str] = _load_t2s()




def fold_variants(text: str) -> str:
    """把繁簡折疊到同一形,只用於比對。

    折向簡體。表裡沒有的字原樣保留。
    """
    return "".join(_T2S.get(c, c) for c in text or "")


def same_text(a: str, b: str) -> bool:
    """兩段文字在忽略繁簡與空白之後是否相同。"""
    return _norm(a) == _norm(b)


def contains(haystack: str, needle: str) -> bool:
    """needle 是否出現在 haystack 裡,忽略繁簡、空白與換行。

    這是三個 verbatim 檢查共用的比對函式。直接用 `in` 會在
    逐字稿與引用的繁簡不一致時把正確引用判成編造。
    """
    n = _norm(needle)
    return bool(n) and n in _norm(haystack)


def _norm(text: str) -> str:
    return fold_variants(text).replace(" ", "").replace("\n", "").replace("\u3000", "")


# ---------------------------------------------------------------- 比對鍵折疊

#: 外觀相同但碼位不同的字元 → 正規形。
#:
#: 逐字稿端本來就折疊全形,但詞彙表端只做 strip().lower()。兩端不對稱的
#: 後果是**含全形字元的詞彙表條目完全抓不到**,而且不報錯。
#:
#:   U+2571 ╱  製表符對角線,外觀像斜線。104 與政府開放資料大量使用,
#:             而詞彙表**內部就不一致**——「專案成本╱品質╱風險管理」用它,
#:             「專案時間/進度控管」用半形。
#:   U+FF0F ／  全形斜線。
#:   U+200B     零寬空格,看不見。★ 映射成半形空格而非移除:等長是硬約束。
_LOOKALIKE_MAP: dict[str, str] = {
    "\u2571": "/",
    "\uff0f": "/",
    "\u200b": " ",
}


def fold_chars(text: str) -> str:
    """比對鍵的字元折疊。**輸出與輸入等長。**

    跟 fold_variants 的分工:

        fold_variants  只折繁簡,給 verbatim 比對用(same_text / contains)
        fold_chars     繁簡 ＋ 全形半形 ＋ 相似字元 ＋ 小寫,
                       給索引鍵與掃描用(SurfaceIndex、詞彙表、黃金集標記鍵)

    ★ 等長是硬性約束:gap.py 用 SurfaceIndex.scan() 回傳的位移去切**原始**
      文字當證據,長度一變證據就切到錯的位置,而且不報錯。
      新增對照條目一律 1 字元對 1 字元。

    ☆ 跟 fold_variants 一樣**不可拿去顯示**——見模組 docstring 那條警告。
    """
    out: list[str] = []
    for ch in text:
        mapped = _LOOKALIKE_MAP.get(ch)
        if mapped is not None:
            out.append(mapped)
            continue
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        elif o == 0x3000:
            ch = " "
        out.append(_T2S.get(ch, ch).lower())
    return "".join(out)
