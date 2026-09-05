"""逐字稿分析（成員 A，面試 W1 D2–D3）。

TranscriptAnalyzer 的真實作。責任兩件：

  1. mentioned_skills()  逐字稿 → 使用者實際講出來的技能集合（已正規化）
  2. text_stats()        填充詞、量化詞、句長——當 LLM 評分的錨點

## 為什麼不能直接把逐字稿丟給 VocabNormalizer

W1/W2 的正規化器吃的是**離散技能字串**（JD 的 requiredSkills、履歷的 tags），
一次一條、每條本來就打算是個技能。逐字稿是**連續口語**：

    「那個我那時候就是有用 python 去把資料抓下來然後畫成圖表給主管看」

正規化器對這種輸入無事可做——它沒有「這串話裡哪幾個片段是技能」的概念。
中間少的那一層叫**候選抽取**，本檔補的就是它。抽出候選之後才輪到正規化。

## 三段式，形狀刻意對齊 VocabNormalizer

  第一段 表面掃描   詞彙表約 500 個表面形做最長匹配掃描。確定性、零模型、
                    毫秒級。這一段就撐得住 D5 交付。
  第二段 語意補抓   滑動視窗 × 餘弦最近鄰，抓改述（「把數字畫成圖表」→ 資料視覺化）。
                    ★ 預設關閉，等 W2 D3–D5 校準完再開。理由見 MENTION_THRESHOLD。
  第三段 殘留區     residuals()，交給 B 的 LLM 批次覆核，跟 normalizer.residuals()
                    同一個用法。

## 一個決定整個門檻方向的不對稱

漏講 = 履歷 ∩ JD − 提及。所以：

  提及集**太小**（該抓到卻沒抓到）→ gap 太大 → 我們對著使用者說「你漏講了 Python」
  但他明明講了 → 使用者當場不信任整個系統。

  提及集**太大**（不該算卻算了）→ gap 太小 → 少給一條建議。可惜，但不傷人。

兩種錯誤的代價差一個量級，所以**這裡的門檻要比正規化器的 0.62 更鬆**（偏召回）。
這不是隨手調的，是這個模組的產品性質決定的。
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Iterable, Protocol, Sequence

from app.contracts.interview_protocols import TextStats

# MentionEvidence 是 A 側內部型別,不在合約裡——合約的 GapCandidate 只吃
# 扁平字串,所以證據的定位資訊在邊界被壓平。內部保留完整結構是為了校準
# 與 debug 能回溯「這一條是哪一段抓到的」。
from dataclasses import dataclass


@dataclass(frozen=True)
class MentionResult:
    """一次 mentions() 呼叫的完整輸出。

    ★ 為什麼要有這個容器:合約明訂 TranscriptAnalyzer 必須是 app 啟動時載入的
      singleton(bge-m3 約 2.3GB,不能每個 request new 一個)。而先前 near_misses
      與 residuals 是寫在**實例**上的每次呼叫狀態,兩個後果:

        near_misses  併發時互相污染。A 的請求可能讀到 B 的結果,
                     而 B 側拿它寫 why——錯得很具體、很有說服力。
        residuals    跨請求累積不清空,長時間執行的伺服器會持續長大。

      單執行緒測試永遠看不出來。改成回傳值之後,實例上不再有可變狀態。
    """

    mentions: dict[str, list["MentionEvidence"]]
    near_misses: dict[str, "MentionEvidence"]
    residuals: list[dict[str, Any]]


@dataclass(frozen=True)
class MentionEvidence:
    """逐字稿裡「講到某技能」的一次命中(A 側內部)。

    start/end 是**原始逐字稿**的字元位移,可直接拿去前端高亮。
    stage 記錄是哪一段抓到的,校準時要分開看兩段的品質。
    """

    skill_id: str
    surface: str
    start: int
    end: int
    quote: str
    stage: str      # "surface" | "semantic"
    score: float    # surface 恆為 1.0;semantic 為餘弦相似度

# ---------------------------------------------------------------- 參數
# 沿用 W2 的規矩：參數是準合約，改參數＝改測試，同一個 commit。

#: 語意段採納門檻。**W2 D3–D5 已校準**（先前是 0.55，憑常識填的）。
#:
#: 掃描 4 視窗 × 7 門檻 × 10 格，實測結果：
#:
#:     語意段關閉        提及 R 0.488   漏講 P 0.471   ← 假指控 9 筆
#:     視窗16/門檻0.45   提及 R 0.805   漏講 P 0.727   ← 假指控 3 筆
#:
#: 漏講 recall 全程維持 1.000——真漏講一筆都沒被吃掉。
#:
#: ★ 為什麼選 0.45 而不是掃描器推薦的 0.40（recall 略高）：
#:   視窗 16 在 0.40／0.45／0.50 三個門檻的四項指標完全相同,是一片**平原**;
#:   0.40 在掃描範圍的邊緣,不知道再低會不會崩。參數落在平原中間比落在
#:   邊緣穩健。同樣 recall 下 0.45 的提及 precision 也較好（0.717 vs 0.660）。
#:
#: ★ 提及 precision 從 0.952 掉到 0.717 是刻意接受的:語意段會把一些沒講到的
#:   算成講到,方向上安全（少給建議 vs 假指控）。但這個保證建立在 6 段逐字稿上,
#:   樣本一大可能不成立。
MENTION_THRESHOLD = 0.45

#: 【待校準】差一點的下界。落在 [NEAR_MISS_FLOOR, MENTION_THRESHOLD) 的視窗
#: 不算「講到」，但會存進 near_misses——它是 B 寫 why 時最有用的一種料。
NEAR_MISS_FLOOR = 0.42

#: 【待校準】**履歷候選的放寬門檻**。合約 candidates 參數的實際用途。
#:
#: 為什麼可以放寬:漏講 = 履歷∩JD − 提及。一筆假指控只會發生在
#: **履歷上有、JD 也要**的技能上。所以對「這個人履歷上的那幾個技能」
#: 放寬偵測,正好對準傷害發生的地方,而範圍只有 8 個詞,誤傷面積很小。
#:
#: 為什麼安全:放寬只會讓 mentioned 變大,而 gap = 交集 − mentioned,
#: 所以它**只能減少 gap,不可能新增假指控**。代價在另一邊——
#: 放太寬會把真漏講也吃掉,gap recall(目前 1.000)會掉。那才是要量的。
#:
#: **W2 已校準 = 0.40。** 掃描 0.45→0.20 的實測結果：
#:
#:     0.45(不放寬)   提及 R 0.805   漏講 P 0.727 / R 1.000   假指控 3
#:     0.40 以下      提及 R 0.829   漏講 P 0.727 / R 1.000   假指控 3
#:
#: 取 0.40:提及 recall +0.024,代價為零(漏講兩項都沒動)。再低沒有額外好處。
#:
#: ★ 但它**沒有解決目標問題**。剩下那 3 筆假指控(專案管理×1、軟體程式設計×2)
#:   門檻降到 0.20 都抓不到——不是「差一點」,是根本不在附近。
#:
#:   原因:「專案管理」是 4 字抽象標籤,而使用者講的是
#:   「我負責約訪談時間、有人拖延就傳訊息設 deadline」。
#:   **具體行為的向量跟抽象標籤的向量,在 bge-m3 的空間裡本來就遠**——
#:   這不是門檻問題,是兩者屬於不同的語意層級。
#:
#:   真要解決得把技能的**行為描述**放進詞彙表當別名
#:   (「排時程、追進度、設期限」→ 專案管理),拿行為對行為比。
#:   那是詞彙表工程,不是參數調整。
CANDIDATE_THRESHOLD = 0.40

#: 語意段的滑動視窗（字元）。無標點逐字稿沒有句子，只能用固定視窗。
#:
#: **W2 已校準。視窗比門檻更關鍵**——實測 recall：
#:     視窗 12 → 0.805    視窗 20 → 0.707
#:     視窗 16 → 0.805    視窗 28 → 0.561
#:
#: 原因:長視窗把一整段話拿去跟四個字的技能名比相似度,中間的「我那時候」
#: 「然後」全是雜訊,餘弦分數被稀釋。28 字幾乎沒用。
#: 12 與 16 的 recall 相同,取 16 因為提及 precision 較好。
SEMANTIC_WINDOW = 16
SEMANTIC_STRIDE = 8

#: 引用切片的前後文長度，給 B 的 prompt 用。
QUOTE_PAD = 18

#: 詞頭抽取:中文技能名多半是「核心名詞 + 動作後綴」或「A╱B」複合形。
#: 實測全語料 511 條純中文技能,54% 抽得出詞頭。
#:     財務報表製作 → 財務報表     人會這樣講
#:     報表彙整與管理 → 報表彙整     人會這樣講
#:     供應商原物料異常分析處理 → 供應商原物料異常分析   人不會這樣講
#: 所以詞頭要夠短才算數——長詞頭只是短了一截的長片語,一樣沒人講。
HEAD_SEP = re.compile(r"[╱/、與及]")
HEAD_SUFFIX = ("處理", "管理", "製作", "控管", "作業", "規劃", "執行", "維護",
               "分析", "應用", "操作", "撰寫", "設計", "評估", "稽核", "查核",
               "安排", "建檔", "事項", "知識", "能力")
HEAD_MAX_LEN = 6      # 超過就不是自然口語詞組
#: ★ 下限從 2 提到 4。實測補詞彙表時,2–3 字的詞頭會誤中無關句子:
#:     「規劃」 ← 規劃、組織、指導及協調組織內部行政作業
#:              → 「我負責活動規劃」「做財務規劃」全部誤中
#:     「理貨」 ← 理貨╱揀貨╱補貨作業  →「整理貨架」誤中
#:   中文表面形不做邊界檢查（詞與詞之間沒有空白可依），所以短詞頭必然
#:   以子字串形式亂命中。四字以上才有足夠的辨識度。
#:   代價是少掉幾條可匹配的詞頭,但誤中的代價是假指控,量級不同。
HEAD_MIN_LEN = 4

#: 例外:確實會被單獨講出來、且夠獨特的短詞。實測逐字稿出現過的才放進來,
#: 不要憑想像加——每加一條就多一分子字串誤中的風險。
HEAD_SHORT_ALLOW = ("零用金", "新產品")


def extract_head(surface: str) -> str | None:
    """中文技能名 → 口語會講的詞頭。抽不出或太長回 None。

    這是表面掃描與語意段之間的第三條路:確定性、零模型、毫秒級,
    專門處理中文流程名詞。它跟 JD 側的截斷問題是同一個形狀,方向相反——
    那邊是履歷用截斷形而 JD 用完整形,這邊是說話用截斷形而詞彙表用完整形。
    """
    if not surface or _ASCII_ONLY.search(surface):
        return None
    head = None
    if HEAD_SEP.search(surface):
        cand = HEAD_SEP.split(surface)[0].strip()
        if HEAD_MIN_LEN <= len(cand) < len(surface):
            head = cand
    if head is None:
        for suf in HEAD_SUFFIX:
            if surface.endswith(suf) and len(surface) - len(suf) >= 3:
                head = surface[: -len(suf)]
                break
    if head is None or len(head) > HEAD_MAX_LEN:
        return None
    if len(head) < HEAD_MIN_LEN and head not in HEAD_SHORT_ALLOW:
        return None
    return head


_ASCII_ONLY = re.compile(r"[A-Za-z]{2,}")

#: 非詞彙填充音。
#: ★ 實測結論(Galaxy S24 / Android 16):Android 會**全數移除**這一類——
#:   原稿「嗯」5 次→轉寫 0 次,「呃」5 次→0 次,部分還被轉成「而」。
#:   所以它們在正式環境**永遠不會出現**,只剩兩個用途:
#:     1. 若某天出現了,代表換了引擎或裝置,是引擎變更的偵測器
#:     2. 打字輸入(input_mode)時會出現,那時 filler 計數才是完整的
#:   它們**不計入 filler_count**,見 FILLERS。
NON_LEXICAL_FILLERS: tuple[str, ...] = ("呃", "嗯", "欸")

#: 詞彙型填充詞。它們是真的詞,引擎通常保留,所以「有這些但沒有上面那些」
#: 正好是**部分抑制**的特徵。
LEXICAL_FILLERS: tuple[str, ...] = (
    "基本上", "怎麼說", "之類的", "反正", "其實", "那個", "這個",
    "就是", "然後", "對啊", "的話",
)

#: ★ filler_count 只計詞彙型。
#:   實測:「那個」2→2、「就是」7→8、「然後」1→3,詞彙型完整保留(甚至因斷詞超計)。
#:   非詞彙音一律被引擎吃掉,把它們算進去只會讓計數看起來完整、實際恆缺一類。
#:   縮小定義讓 filler_count 變成**誠實的測量**:它測的是詞彙型填充詞,
#:   而那本身就是有效的流暢度訊號(每句都講「就是」的人確實不流暢)。
#:   所以流暢度維度沒有死,只是變窄。B 的 prompt 不要宣稱它涵蓋全部口語習慣。
#: 註:「然後」同時是填充詞也是語段分界,兩邊都用,這是刻意的。
FILLERS: tuple[str, ...] = LEXICAL_FILLERS

#: 判定「部分抑制」需要的最短長度。太短的回答本來就可能一個填充音都沒有,
#: 不足以構成證據。150 字約當 45–60 秒的口語。
PARTIAL_SUPPRESSION_MIN_CHARS = 150

#: 因果連接詞。B 要「邏輯清晰度」的錨（見 REQUEST_connector_rate.md）。
#:
#: ★ 只做因果一類。他原本列三類，實測六段逐字稿（2,477 字）之後兩類不成立：
#:
#:     轉折（但是/不過/然而/雖然）  出現 **0 次**。不是稀少，是完全沒有。
#:     序列（首先/接著/最後）       扣掉「然後」只剩 10 次，而「然後」39 次
#:                                 且已在 FILLERS 裡——同時算連接詞與填充詞的話，
#:                                 講話越不流暢的人邏輯分數越高，兩個指標會打架。
#:
#:   因果類不含「然後」，所以那個打架不會發生，不必做上下文判斷。
#:   做了永遠是 0 的欄位比沒有欄位更容易被誤用，所以另外兩類不做。
CAUSAL_CONNECTORS: tuple[str, ...] = (
    "因為", "所以", "因此", "由於", "導致", "造成", "使得", "才能", "為了",
)

#: 無標點時的語段分界詞。
DISCOURSE_MARKERS: tuple[str, ...] = (
    "然後", "接下來", "後來", "所以", "再來", "另外", "最後",
    "因為", "但是", "不過", "那時候", "第一", "第二", "第三",
)

_PUNCT_RE = re.compile(r"[。！？；，、\n\.!\?;]+")
#: 真標點,不含換行。換行是我們自己接多段時加的,不是使用者說話的產物。
_REAL_PUNCT_RE = re.compile(r"[。！？；，、\.!\?;]")
_ARABIC_QUANT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|倍|成|萬|千|百|人|個|次|天|週|周|月|年|小時|分鐘|元|k|m)",
    re.IGNORECASE,
)
_CHINESE_QUANT_RE = re.compile(
    r"[一二兩三四五六七八九十百千萬]+\s*(?:%|％|倍|成|人|個|次|天|週|周|月|年|小時|分鐘|倍數)"
)
_PURE_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％)")


# ---------------------------------------------------------------- 外部接口
# ★ 唯一要對你 repo 的地方就是這一節。下面兩個 adapter 是全檔僅有的耦合點。


class _NormalizerLike(Protocol):
    """VocabNormalizer 的鴨子型別。本檔只用得到「字串 → skill_id」這一件事。"""

    def normalize(self, text: str) -> Any: ...


def resolve_skill_id(normalizer: Any, surface: str) -> str | None:
    """把一個表面形交給正規化器換 skill_id。

    ★ 這是全檔唯一的 repo 耦合點。你的 VocabNormalizer 方法名若不是 normalize，
      改這個函式就好，其他地方零改動。目前依序嘗試 normalize / lookup / __call__，
      回傳值支援 .skill_id 屬性、dict["skill_id"]、或直接是字串。
    """
    if normalizer is None:
        return None
    for attr in ("normalize", "lookup"):
        fn = getattr(normalizer, attr, None)
        if callable(fn):
            return _unwrap_skill_id(fn(surface))
    if callable(normalizer):
        return _unwrap_skill_id(normalizer(surface))
    return None


def _unwrap_skill_id(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result or None
    sid = getattr(result, "skill_id", None)
    if isinstance(sid, str):
        return sid or None
    if isinstance(result, dict):
        sid = result.get("skill_id")
        return sid or None
    if isinstance(result, (list, tuple)) and result:
        return _unwrap_skill_id(result[0])
    return None


#: 詞彙表向量的全域快取。key = (embedding 物件 id, 表面形 tuple)。
#:
#: ★ 沒有這個的話調參會跑到天亮。實測：4 視窗 × 7 門檻 × 10 格 = 280 次
#:   重算整份詞彙表的向量,共 14.6 萬次向量化,跑了 **2.7 小時**。
#:   詞彙表在整輪掃描中完全沒變,那 279 次全是白算的。
_VOCAB_VEC_CACHE: dict[tuple, Any] = {}


def embed_vocab_cached(embedding: Any, surfaces: list[str]) -> Any:
    """詞彙表向量,跨實例快取。逐字稿視窗不快取（每次都不一樣）。"""
    key = (id(embedding), tuple(surfaces))
    if key not in _VOCAB_VEC_CACHE:
        _VOCAB_VEC_CACHE[key] = embed_texts(embedding, surfaces)
    return _VOCAB_VEC_CACHE[key]


def embed_texts(embedding: Any, texts: list[str]) -> Any:
    """呼叫 embedding provider。

    ★ 你們既有的介面是 `EmbeddingProvider.embed(texts) -> list[list[float]]`
      （app/providers/embeddings.py）。我原本寫死 `.encode()`（sentence-transformers
      的慣例），跟自家 Protocol 對不上,語意段一開就 AttributeError。
      這裡兩個都吃,`embed` 優先——那才是這個專案的正式介面。
    """
    import numpy as np
    for name in ("embed", "encode"):
        fn = getattr(embedding, name, None)
        if callable(fn):
            return np.asarray(fn(list(texts)), dtype="float32")
    raise AttributeError(
        f"{type(embedding).__name__} 沒有 embed() 也沒有 encode()——"
        "請對齊 app/providers/embeddings.py 的 EmbeddingProvider"
    )


def _entry_fields(entry: Any) -> tuple[str, str, list[str]]:
    """CanonicalSkill → (skill_id, name_zh, 全部表面形)。dict 與物件都吃。"""
    get = entry.get if isinstance(entry, dict) else lambda k, d=None: getattr(entry, k, d)
    skill_id = get("skill_id") or get("id") or ""
    name_zh = get("name_zh") or get("name") or ""
    name_en = get("name_en") or ""
    aliases = list(get("aliases") or [])
    surfaces = [s for s in ([name_zh, name_en] + aliases) if s]
    return str(skill_id), str(name_zh), surfaces


# ---------------------------------------------------------------- 文字正規化


#: 外觀相同但碼位不同的字元 → 正規形。
#:
#: 【為什麼要有這張表】
#: 逐字稿端本來就折疊全形(U+FF01–FF5E 整段減 0xFEE0),但詞彙表端只做
#: strip().lower()。兩端不對稱的後果是**含全形字元的詞彙表條目完全抓不到**,
#: 而且不報錯——那個技能只是永遠偵測不到。目前詞彙表剛好是乾淨的(實測 0 筆),
#: 但 build_vocab.py 是從 104 與政府開放資料重建的,而那些來源確實含這些字元。
#:
#: 【只收有唯一正確答案的】
#: 跟 jd_normalize 的判準一致:確定性字串替換才進機械層,有歧義的不收。
#:
#:   U+2571 ╱  製表符對角線。外觀像斜線,實際是 BOX DRAWINGS。104 與政府
#:             資料大量使用(實測黃金集 26 次、詞彙表 4 次),而詞彙表**內部
#:             就不一致**——「專案成本╱品質╱風險管理」用它,
#:             「專案時間/進度控管」用半形。同樣的意思因為一個看不見的字元
#:             差異而得到不同比對結果。
#:   U+FF0F ／  全形斜線。已在 FF01–FF5E 範圍內,列在這裡是為了詞彙表端也蓋到。
#:   U+200B     零寬空格。完全看不見,落在技能名稱裡會造成無法解釋的比對失敗。
#:             實測出現在 104 職缺的散文裝飾裡,不在技能名稱。
#:             ★ 映射成半形空格而非移除——gap.py 用 scan() 回傳的位移去切
#:               **原始** JD 文字當證據,折疊若改變字串長度,證據就會錯位。
#:               嚴格等長是這張表的硬性約束,新增條目一律 1 字元對 1 字元。
#: 繁體→簡體的字元層對照表路徑。
#:
#: 【為什麼需要】
#: Whisper 對中文繁簡輸出**沒有一致保證**。怡君 2026-09-05 的實測報告:
#: 同一位講者、同一支後端、同樣的呼叫方式,五段 App 錄音輸出繁體,
#: 一段電腦錄音輸出簡體。原因未定(懷疑是收音鏈路差異),但不影響結論——
#: 我們不能假設逐字稿一定是繁體。
#:
#: 【不修的後果:靜默的假指控】
#: 詞彙表是繁體。逐字稿回簡體時,中文技能**全部**比對不到——實測 7 個測 7 個失敗。
#: 而 gap = 履歷∩JD − 提及,提及變小,交集裡的技能就變成假指控。
#: 拿黃金集實測(只把逐字稿轉簡體、其餘不動):假指控 7 → 10 筆。
#: 不報錯,只是告訴使用者「你什麼都沒講到」。
#:
#: 【為什麼是繁→簡,不是簡→繁】
#: 繁→簡是多對一,確定性的;簡→繁是一對多,要消歧義、會猜錯。
#: 比對只需要兩邊落到同一個形,取確定性的那個方向。
#: 產出給人看的文字一律用**原始**逐字稿(evidence 靠位移切原文),
#: 所以這裡轉成簡體不會讓使用者看到簡體。
T2S_CHARS_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "t2s_chars.v1.json"


def _load_t2s() -> dict[str, str]:
    """讀繁簡字元對照。讀不到就回空 dict——降級但不擋啟動。

    ★ 只收 1 字元對 1 字元的項目(建表時已過濾),因為 fold_chars 必須等長:
      gap.py 用 scan() 的位移去切**原始**文字當證據,長度一變證據就錯位。
    """
    try:
        raw = json.loads(T2S_CHARS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in raw.get("chars", {}).items() if len(k) == 1 == len(v)}


_T2S_CHARS: dict[str, str] = _load_t2s()

_LOOKALIKE_MAP: dict[str, str] = {
    "\u2571": "/",   # ╱ BOX DRAWINGS LIGHT DIAGONAL
    "\uff0f": "/",   # ／ FULLWIDTH SOLIDUS
    "\u200b": " ",   # 零寬空格 → 半形空格(等長)
}


def fold_chars(text: str) -> str:
    """字元層折疊。**所有把字串當成比對鍵或掃描對象的地方都要用這一支**。

    做四件事:

        1. 外觀相同碼位不同的字元 → 正規形(見 _LOOKALIKE_MAP)
        2. 全形 ASCII → 半形(U+FF01–FF5E 減 0xFEE0)
        3. 全形空格 → 半形空格
        4. 英文轉小寫

    ★ 嚴格等長:輸出與輸入的字元數一定相同。gap.py 依賴這個性質——
      它用 scan() 回傳的位移去切原始 JD 文字當證據,長度一變證據就錯位。

    ★★ 哪裡漏用就會產生不對稱,而不對稱的症狀是**靜默失敗**:
       比對不到,不報錯,那個技能只是永遠抓不到。目前的使用點:

           SurfaceIndex.from_vocab      詞彙表表面形與詞頭
           TranscriptAnalyzer.__init__  stt_aliases 的兩側
           normalize_for_scan           逐字稿(規則同步,但自己跑迴圈以維護位移表)
           gap.py 的三處 index.scan     JD 與履歷的散文
           interview_eval.build_name_index  黃金集標記鍵
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
        # 繁→簡。放在最後:全形轉半形之後才輪到中文字元。
        # 已經是簡體的字不在表裡,原樣通過(冪等)。
        out.append(_T2S_CHARS.get(ch, ch).lower())
    return "".join(out)


def normalize_for_scan(raw: str) -> tuple[str, list[int]]:
    """逐字稿 → (可掃描字串, 位移對照表)。

    對照表讓證據的 start/end 能指回**原始**逐字稿，前端高亮與引用才對得上。

    做三件事：全形轉半形、英文轉小寫、把 STT 逐字母拆開的縮寫黏回去
    （「s q l」→「sql」、「p h p」→「php」）。第三件是中文 STT 的家常便飯，
    不處理的話 SQL 這種三字母技能永遠掃不到。

    只有「每個 run 都是單字母」才黏——「data analysis」不會被黏成一團。
    """
    # ★ 先移除中日韓字元之間的空白。實測逐字稿出現「做財務報 表的工作」,
    #   中文詞被空白切開,表面掃描直接失效。空白在中文詞內部沒有語意,
    #   移除是安全的;英文之間的空白必須保留（"data analysis" 不能黏成一團）。
    raw = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", raw)

    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(raw):
        mapped = _LOOKALIKE_MAP.get(ch)
        if mapped is not None:
            chars.append(mapped)
            idx.append(i)
            continue
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        elif o == 0x3000:
            ch = " "
        # ★ 這裡刻意**不做繁簡轉換**,那是 fold_chars / SurfaceIndex.scan 的事。
        #
        #   這支的輸出同時餵給兩條路。表面掃描走 scan(),它自己會折疊(含繁簡),
        #   對得上折疊過的索引鍵。但**語意段拿的是這支的原始輸出**,而語意段的
        #   詞彙表向量用的是未折疊的表面形——這裡若轉成簡體,視窗是簡體、
        #   詞彙表是繁體,相似度就被字形差異污染了。
        #
        #   ☆ 語意段兩側都折疊會更一致,但那會改變送進 bge-m3 的文字,
        #     相似度分布跟著變,而 MENTION_THRESHOLD=0.45 是在現有向量上
        #     校準出來的。等下一輪重新校準時再一起處理,不要現在動。
        chars.append(ch.lower())
        idx.append(i)

    out_c: list[str] = []
    out_i: list[int] = []
    n = len(chars)
    k = 0
    while k < n:
        if (
            _is_ascii_alpha(chars[k])
            and (k == 0 or not _is_ascii_alnum(chars[k - 1]))
        ):
            letters: list[str] = []
            positions: list[int] = []
            j = k
            while True:
                letters.append(chars[j])
                positions.append(idx[j])
                nxt_is_single = (
                    j + 2 < n
                    and chars[j + 1] == " "
                    and _is_ascii_alpha(chars[j + 2])
                    and not (j + 3 < n and _is_ascii_alpha(chars[j + 3]))
                )
                if nxt_is_single:
                    j += 2
                    continue
                break
            if len(letters) >= 2:
                out_c.extend(letters)
                out_i.extend(positions)
                k = j + 1
                continue
        out_c.append(chars[k])
        out_i.append(idx[k])
        k += 1
    return "".join(out_c), out_i


def _is_ascii_alpha(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _is_ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


# ---------------------------------------------------------------- 表面形索引


class SurfaceIndex:
    """詞彙表全部表面形的最長匹配掃描器。

    不用 regex 也不用第三方 AC 自動機——詞彙表約 500 條、逐字稿數千字，
    首字分桶＋長度倒序的暴力掃描是毫秒級，而且零依賴、行為完全可預測。
    這個取捨跟 W2 自幹 VectorRetriever 是同一套理由。

    兩條規則：
      最長者勝 —— 「javascript」不會被拆成「java」+「script」
      不重疊   —— 命中後直接跳到尾端，同一段字不會算兩次
      ASCII 邊界 —— 全 ASCII 的表面形要求前後不是英數字，
                    否則「r」會在「react」裡面命中
    """

    def __init__(self, surface_to_skill: dict[str, str]) -> None:
        self._heads: set[str] = set()
        self._stt_forms: set[str] = set()
        self._buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for surface, skill_id in surface_to_skill.items():
            if not surface:
                continue
            self._buckets[surface[0]].append((surface, skill_id))
        for bucket in self._buckets.values():
            bucket.sort(key=lambda pair: -len(pair[0]))

    @classmethod
    def from_vocab(cls, vocab: Iterable[Any], *, with_heads: bool = True,
                   extra_aliases: dict[str, str] | None = None) -> "SurfaceIndex":
        """建索引。三趟,順序就是優先序:

            1. 完整表面形（name_zh / name_en / aliases）
            2. STT 轉寫別名（extra_aliases，{轉寫形: 正式名}）
            3. 詞頭（只補洞）

        ★ 三趟必須全域分開,不能逐條目做完三件事。
          先前把 STT 別名合併寫在外面、逐條目「完整形→詞頭」,結果第一條的
          詞頭會贏過第二條的完整形——「報表彙整與管理」的詞頭「報表彙整」
          蓋掉了「報表彙整」這個獨立條目,同一句話解析到不同技能,而且不報錯。
          實測履歷集 recall 因此從 1.000 掉到 0.958。
        """
        vocab = list(vocab)
        mapping: dict[str, str] = {}
        heads: set[str] = set()
        for entry in vocab:
            skill_id, _name, surfaces = _entry_fields(entry)
            if not skill_id:
                continue
            for surface in surfaces:
                # ★ fold_chars 而不是只 strip().lower()——逐字稿端會折疊全形
                #   與相似字元,詞彙表端不折疊的話兩邊永遠對不上,而且不報錯。
                key = fold_chars(surface.strip())
                if not key:
                    continue
                # 單字元的中文表面形太吵（「圖」「數」），只留單字元的英文（R、C）
                if len(key) == 1 and not _is_ascii_alpha(key):
                    continue
                mapping.setdefault(key, skill_id)
        stt_forms: set[str] = set()
        if extra_aliases:
            # 第二趟：STT 轉寫別名。正式名要先能對到 skill_id 才登記。
            name_to_id: dict[str, str] = {}
            for entry in vocab:
                sid, _zh, forms = _entry_fields(entry)
                for f in forms:
                    name_to_id.setdefault(fold_chars(f.strip()), sid)
            for stt_form, canonical in extra_aliases.items():
                sid = name_to_id.get(fold_chars(str(canonical).strip()))
                key = fold_chars(str(stt_form).strip())
                if sid and key and key not in mapping:
                    mapping[key] = sid
                    stt_forms.add(key)

        if with_heads:
            # 第三趟：詞頭只補洞,不搶。setdefault 保證前兩趟優先。
            for entry in vocab:
                skill_id, _name, surfaces = _entry_fields(entry)
                if not skill_id:
                    continue
                for surface in surfaces:
                    head = extract_head(surface.strip())
                    folded_head = fold_chars(head) if head else ""
                    if folded_head and folded_head not in mapping:
                        mapping[folded_head] = skill_id
                        heads.add(folded_head)
        index = cls(mapping)
        index._heads = heads
        index._stt_forms = stt_forms
        return index

    def scan(self, text: str) -> list[tuple[int, int, str, str]]:
        """→ [(start, end, surface, skill_id), ...]，位移是 text 上的。

        ★ 折疊在這裡做,不在呼叫端。索引鍵是折疊過的(全形→半形、
          相似字元正規化、繁→簡),掃描文字若沒折疊就永遠對不上,
          而且不報錯——只是什麼都掃不到。

          之前把折疊放在呼叫端,結果 tests/test_interview_w1.py 直接
          呼叫 scan() 時漏掉,整個索引靜默失效。責任放在這裡,
          呼叫端就不可能忘。

        ★★ fold_chars 嚴格等長,所以回傳的位移在**原始** text 上也成立,
          呼叫端可以直接拿去切原文當證據。等長是這個設計的前提,
          _LOOKALIKE_MAP 與繁簡表新增條目時一律 1 字元對 1 字元。
        """
        text = fold_chars(text)
        hits: list[tuple[int, int, str, str]] = []
        i, n = 0, len(text)
        while i < n:
            matched = None
            for surface, skill_id in self._buckets.get(text[i], ()):
                end = i + len(surface)
                if end <= n and text[i:end] == surface and self._boundary_ok(text, i, end, surface):
                    matched = (i, end, surface, skill_id)
                    break
            if matched:
                hits.append(matched)
                i = matched[1]
            else:
                i += 1
        return hits

    @staticmethod
    def _boundary_ok(text: str, start: int, end: int, surface: str) -> bool:
        if not all(_is_ascii_alnum(c) for c in surface):
            return True
        if start > 0 and _is_ascii_alnum(text[start - 1]):
            return False
        if end < len(text) and _is_ascii_alnum(text[end]):
            return False
        return True


# ---------------------------------------------------------------- 主類別


class TranscriptAnalyzer:
    """TranscriptAnalyzer Protocol 的真實作。

    用法（面試模組上游只會這樣用）：

        analyzer = TranscriptAnalyzer(vocab=load_vocab(), normalizer=normalizer)
        said = analyzer.mentioned_skills(transcript)      # set[skill_id]
        stats = analyzer.text_stats(transcript)           # 給 B 的錨點

    要開語意段（W2 校準後）：

        analyzer = TranscriptAnalyzer(vocab=..., normalizer=..., 
                                      embedding=BgeM3Embedding(), enable_semantic=True)
    """

    def __init__(
        self,
        vocab: Iterable[Any],
        normalizer: Any | None = None,
        embedding: Any | None = None,
        *,
        enable_semantic: bool = False,
        threshold: float = MENTION_THRESHOLD,
        near_miss_floor: float = NEAR_MISS_FLOOR,
        candidate_threshold: float = CANDIDATE_THRESHOLD,
        engine_filler_policy: str = "unknown",
        stt_aliases: dict[str, str] | None = None,
    ) -> None:
        # stt_aliases: {STT 轉寫形: 正式技能名}。來自 STT_TERM_PROBE 實測。
        # ★ 只放**安全**別名——轉寫形本身不是任何真實技能的那些
        #   （gate→Git、Jerry→jQuery）。碰撞的（Android→Angular）不放這裡,
        #   它們要靠履歷做條件判定,見 gap.resolve_confusions。
        self._stt_aliases = dict(stt_aliases or {})
        # engine_filler_policy: "keeps" | "suppressed" | "unknown"
        # ★ 這是**引擎的**性質,不是逐字稿的性質。單一逐字稿證明不了抑制
        #   （沒有填充詞也可能只是講得很順）,所以它由 STT_TERM_PROBE 的結果
        #   一次設定,不做逐次推論。探測回來之前維持 "unknown"。
        self._engine_filler_policy = engine_filler_policy
        self._vocab = list(vocab)
        self._index = SurfaceIndex.from_vocab(
            self._vocab, extra_aliases=self._stt_aliases or None
        )
        self._normalizer = normalizer
        self._embedding = embedding
        self._enable_semantic = bool(enable_semantic and embedding is not None)
        self._threshold = threshold
        self._near_miss_floor = near_miss_floor
        self._candidate_threshold = candidate_threshold
        self._display: dict[str, str] = {}
        for entry in self._vocab:
            skill_id, name_zh, _ = _entry_fields(entry)
            if skill_id:
                self._display[skill_id] = f"{name_zh}({skill_id})" if name_zh else skill_id
        self._vec_cache: Any = None

    # -------------------------------------------------- 契約方法

    def mentioned_skills(self, transcript: str,
                         candidates: set[str] | None = None) -> set[str]:
        """逐字稿 → 使用者實際講出來的技能集合（已正規化）。

        回傳 skill_id（sk: / skm:），不是 name_zh。要顯示字串走 display()。

        candidates=None 時掃全詞彙表（W1 行為）。給定時只回落在該集合內的。

        ★ candidates 不是事後過濾,是**放寬門檻**。落在 candidates 裡的技能
          用 CANDIDATE_THRESHOLD(較低),其餘用 MENTION_THRESHOLD。
          範圍限縮之後才敢放寬,這才是這個參數的價值所在。
        """
        return set(self.mentions(transcript, candidates).keys())

    def text_stats(self, transcript: str) -> TextStats:
        """填充詞、量化詞、語段長度。全部確定性，不經模型。"""
        text = transcript or ""
        char_count = len(re.sub(r"\s+", "", text))
        lowered = text.lower()

        breakdown: dict[str, int] = {}
        consumed = [False] * len(lowered)
        for filler in sorted(FILLERS, key=len, reverse=True):
            count = 0
            start = 0
            while True:
                pos = lowered.find(filler, start)
                if pos < 0:
                    break
                if not any(consumed[pos : pos + len(filler)]):
                    for k in range(pos, pos + len(filler)):
                        consumed[k] = True
                    count += 1
                start = pos + 1
            if count:
                breakdown[filler] = count
        filler_count = sum(breakdown.values())

        causal = {}
        for w in CAUSAL_CONNECTORS:
            n = lowered.count(w)
            if n:
                causal[w] = n
        causal_n = sum(causal.values())

        quant_spans: set[tuple[int, int]] = set()
        for pattern in (_ARABIC_QUANT_RE, _CHINESE_QUANT_RE, _PURE_PERCENT_RE):
            for m in pattern.finditer(text):
                quant_spans.add((m.start(), m.end()))
        quantifier_count = len(_merge_spans(quant_spans))

        segments, mode = self._segment(text)
        seg_lengths = [len(re.sub(r"\s+", "", s)) for s in segments if s.strip()]
        avg_len = (sum(seg_lengths) / len(seg_lengths)) if seg_lengths else 0.0

        return TextStats(
            char_count=char_count,
            filler_count=filler_count,
            filler_detail=breakdown,
            quantifier_count=quantifier_count,
            sentence_count=len(seg_lengths),
            avg_sentence_len=round(avg_len, 1),
            segmentation=mode,
            filler_reliability=self._filler_reliability(breakdown, char_count),
            connector_rate=round(causal_n / char_count * 100, 2) if char_count else 0.0,
            connector_detail={"causal": causal_n} if causal_n else {},
        )

    def _filler_reliability(self, breakdown: dict[str, int], char_count: int) -> str:
        """filler_count 可不可以當流暢度的錨點。

        判定順序,由確定到不確定:

          1. 引擎已知會抑制（探測確認過）        → "suppressed"
          2. 逐字稿裡有非詞彙填充音（呃/嗯/欸）  → "measured"
             引擎顯然沒在濾,計數反映使用者的口語習慣
          3. 只有詞彙型填充詞、沒有非詞彙填充音,
             而且逐字稿夠長                      → "unknown"
             這是**部分抑制**的特徵:引擎濾掉非詞彙音、保留真詞。
             此時 filler_count 有值但系統性偏低,回 "measured" 會過度宣稱
          4. 其餘                                → "unknown"

        ★ 這裡刻意不回 "suppressed" 做推論。單一逐字稿沒有填充詞,可能是
          引擎濾掉,也可能是講者真的很順——分不出來。錯判成 suppressed 會讓
          B 放棄一個其實可用的錨點;維持 unknown 只是讓他改用備援訊號。
          兩種錯誤代價不同,取代價小的。
        """
        if self._engine_filler_policy == "suppressed":
            return "suppressed"
        if self._engine_filler_policy == "partial":
            # 實測確認的狀態:非詞彙音被吃掉,詞彙型保留。FILLERS 已縮小為
            # 只含詞彙型,所以計數對它自己的定義而言是完整的 → measured。
            # 合約不需要第五個值,因為改變的是「測什麼」不是「測得準不準」。
            return "measured" if breakdown else "unknown"
        if any(breakdown.get(f) for f in NON_LEXICAL_FILLERS):
            return "measured"
        if breakdown and char_count >= PARTIAL_SUPPRESSION_MIN_CHARS:
            return "unknown"   # 疑似部分抑制,保守
        if breakdown:
            return "measured"
        return "unknown"

    # -------------------------------------------------- 擴充方法（A 側自用）

    def mentions(self, transcript: str,
                 candidates: set[str] | None = None) -> dict[str, list[MentionEvidence]]:
        """→ {skill_id: 證據}。要 near_misses / residuals 請用 analyse()。"""
        return self.analyse(transcript, candidates).mentions

    def analyse(self, transcript: str,
                candidates: set[str] | None = None) -> MentionResult:
        """帶證據的提及集。GapComputer 用這個，不用 mentioned_skills()。

        契約規定 mentioned_skills 回 set[str]，但 gap 要求「附證據」，
        set 裡沒有證據可帶。所以內部走這條，對外仍守契約。
        """
        near_misses: dict[str, MentionEvidence] = {}
        residuals: list[dict[str, Any]] = []
        text, offsets = normalize_for_scan(transcript or "")
        found: dict[str, list[MentionEvidence]] = defaultdict(list)

        for start, end, surface, skill_id in self._index.scan(text):
            resolved = resolve_skill_id(self._normalizer, surface) or skill_id
            src_start = offsets[start] if start < len(offsets) else start
            src_end = offsets[end - 1] + 1 if end - 1 < len(offsets) else end
            found[resolved].append(
                MentionEvidence(
                    skill_id=resolved,
                    surface=surface,
                    start=src_start,
                    end=src_end,
                    quote=_quote(transcript or "", src_start, src_end),
                    stage=("stt_alias" if surface in getattr(self._index, "_stt_forms", ()) else
                           "head" if surface in getattr(self._index, "_heads", ()) else "surface"),
                    score=(0.85 if surface in getattr(self._index, "_stt_forms", ()) else
                           0.9 if surface in getattr(self._index, "_heads", ()) else 1.0),
                )
            )

        if self._enable_semantic:
            cands = candidates or set()
            for ev in self._semantic_pass(text, offsets, transcript or "", residuals):
                # 履歷上有的技能用放寬門檻——傷害只發生在這些技能上
                bar = (self._candidate_threshold if ev.skill_id in cands
                       else self._threshold)
                if ev.score >= bar:
                    found[ev.skill_id].append(ev)
                elif ev.score >= self._near_miss_floor:
                    prev = near_misses.get(ev.skill_id)
                    if prev is None or ev.score > prev.score:
                        near_misses[ev.skill_id] = ev

        return MentionResult(dict(found), near_misses, residuals)

    def near_misses(self) -> dict[str, MentionEvidence]:
        """★ 已移除。用 analyse(transcript).near_misses。

        這是 singleton 下的併發陷阱:先前它讀實例狀態,兩個請求同時進來時
        後者會覆蓋前者,而呼叫端讀到的是別人的結果。
        """
        raise AttributeError(
            "near_misses() 已移除（singleton 併發不安全）。"
            "改用 analyse(transcript).near_misses"
        )

    def residuals(self) -> list[dict[str, Any]]:
        """殘留區——語意段也搆不到門檻、但看起來像技能的片段。

        ★ 已移除，理由同 near_misses()——它會跨請求累積不清空。
        改用 analyse(transcript).residuals。
        """
        raise AttributeError(
            "residuals() 已移除（會跨請求累積）。改用 analyse(transcript).residuals"
        )

    def display(self, skill_id: str) -> str:
        """skill_id → 「name_zh(skill_id)」。沿用 W1 決議的 log 可讀性折衷。"""
        return self._display.get(skill_id, skill_id)

    # -------------------------------------------------- 內部

    def _segment(self, text: str) -> tuple[list[str], str]:
        """四態,對齊合約的 segmentation 欄位。

          punctuation       有真標點,精確值
          stt_segment       無標點,但邊界來自 STT 自動送出點,是實測值
          discourse_marker  無標點,以語氣詞估算——是估算值
          unavailable       都沒有,兩個欄位回 0

        ★ punctuation 與 stt_segment 一定要分開,這是我原本寫錯的地方。
          _PUNCT_RE 含 \n,所以多段用換行接起來之後會命中「有標點」——
          但實測逐字稿裡一個真標點都沒有,回 punctuation 等於宣稱那 76 字
          是精確句長,實際上它是「她停頓前講了多長」。
          過度宣稱比不宣稱糟:B 的 prompt 會照著把它講成句子長度。
        """
        if not text.strip():
            return [], "unavailable"
        real_punct = _REAL_PUNCT_RE.search(text)
        if real_punct:
            return [x for x in _PUNCT_RE.split(text) if x.strip()], "punctuation"
        if "\n" in text:
            # 段界＝STT 自動送出點。量的不是句長,是語流連續性。
            return [x for x in text.split("\n") if x.strip()], "stt_segment"
        pattern = "|".join(re.escape(m) for m in DISCOURSE_MARKERS)
        parts = [p for p in re.split(f"(?={pattern})", text) if p.strip()] if pattern else []
        if len(parts) <= 1:
            # 切不出東西就誠實說切不出來,不要拿整段長度冒充平均句長
            return [], "unavailable"
        return parts, "discourse_marker"

    def _semantic_pass(
        self, text: str, offsets: list[int], raw: str,
        residuals: list[dict[str, Any]],
    ) -> list[MentionEvidence]:
        """滑動視窗 × 餘弦最近鄰。無標點逐字稿沒有句子，只能用固定視窗。"""
        import numpy as np

        windows: list[tuple[int, int, str]] = []
        for start in range(0, max(len(text) - 1, 1), SEMANTIC_STRIDE):
            chunk = text[start : start + SEMANTIC_WINDOW]
            if len(chunk.strip()) >= 6:
                windows.append((start, start + len(chunk), chunk))
        if not windows:
            return []

        if self._vec_cache is None:
            surfaces: list[str] = []
            owners: list[str] = []
            for entry in self._vocab:
                skill_id, name_zh, forms = _entry_fields(entry)
                for form in forms:
                    surfaces.append(form)
                    owners.append(skill_id)
            self._vec_cache = (embed_vocab_cached(self._embedding, surfaces), owners)
        vocab_vecs, owners = self._vec_cache

        win_vecs = embed_texts(self._embedding, [w[2] for w in windows])
        sims = win_vecs @ vocab_vecs.T

        # ★ 這裡**不做門檻判斷**,回傳全部候選讓 mentions() 決定。
        #   兩個理由:一是候選技能要用不同門檻(見 CANDIDATE_THRESHOLD);
        #   二是校準掃描門檻時不必重算向量——實測全域快取加上這個改動,
        #   一輪掃描從 2.7 小時降到幾分鐘。
        FLOOR = min(self._near_miss_floor, self._candidate_threshold) * 0.85
        out: list[MentionEvidence] = []
        for wi, (start, end, chunk) in enumerate(windows):
            best = int(sims[wi].argmax())
            score = float(sims[wi][best])
            if score < FLOOR:
                # ★ residual 的下界必須跟著 FLOOR 走,不能寫死用 near_miss_floor。
                #   先前寫成「低於 near_miss_floor*0.85 就丟進 residuals 並 continue」,
                #   於是候選門檻放寬到 0.05 時,低分的候選在比門檻之前就被攔掉——
                #   放寬完全失效,而且看起來像「語意段沒抓到」,查不出原因。
                residuals.append(
                    {"window": chunk, "best": owners[best], "sim": round(score, 4)}
                )
                continue
            src_start = offsets[start] if start < len(offsets) else start
            src_end = offsets[min(end, len(offsets)) - 1] + 1 if offsets else end
            out.append(
                MentionEvidence(
                    skill_id=owners[best],
                    surface=chunk,
                    start=src_start,
                    end=src_end,
                    quote=_quote(raw, src_start, src_end),
                    stage="semantic",
                    score=round(score, 4),
                )
            )
        return out


# ---------------------------------------------------------------- 小工具


def _quote(text: str, start: int, end: int, pad: int = QUOTE_PAD) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    body = text[lo:hi].strip()
    return f"{'…' if lo > 0 else ''}{body}{'…' if hi < len(text) else ''}"


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
