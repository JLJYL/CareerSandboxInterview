"""漏講計算（成員 A，面試 W1 D3–D5）。對齊 D1 凍結合約。

    履歷技能集 ∩ JD 需求集 − 逐字稿提及集 = 漏講候選

## 兩側門檻刻意不對稱

  履歷側 **寧缺勿濫**。多算一個技能 → 憑空生出一條 gap → 對使用者說
  「你漏講了 Kubernetes」而他履歷根本沒有。當場失去信任。
  預設只吃 experiences[].tags，不掃 description 散文。

  JD 側 **寧濫勿缺**。漏掉一條需求 → 少給一條建議。可惜，不傷人。
  required_skills 之外也掃 description。

同一模組兩個相反的預設，因為兩種錯誤代價不同。這條寫在這裡免得被當成手滑。

## 履歷側的兩個參數目前不可識別（D1 異議第五條定案）

實測 ExperienceDTO 沒有頂層技能欄位，技能全部從 experiences[].tags 進來，
於是履歷側證據來源恆為 raw_tag、權重恆為常數：

    weight = W_JD × jd_weight + W_RESUME × RESUME_CONST
                                └──────────────────────┘ 不隨候選變動

所以 RAW_TAG_DISCOUNT 與 W_RESUME 沒有任何黃金集能校準它們。合併成一個常數，
不假裝校準。待 ExperienceDTO 帶出處欄位後再拆開（B 已定案面試路徑不加
provenance，故此事延後至 B1 擷取器接入時再議）。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.contracts.interview_protocols import GapCandidate, JDInput
from app.pipeline.transcript import (
    SurfaceIndex, TranscriptAnalyzer, extract_head, resolve_skill_id,
)

# ---------------------------------------------------------------- 參數

#: JD 側來源權重。
SOURCE_WEIGHT: dict[str, float] = {"structured": 1.0, "prose": 0.8}

#: 履歷側常數。見檔頭：目前不可識別，不要當成可調參數。
RESUME_CONST = 0.6

#: 【待校準】兩側配重。JD 側較重：「JD 要不要」比「履歷有多硬」更能決定該不該講。
W_JD = 0.6
W_RESUME = 0.4

#: 【待校準】JD 列表位置衰減。required_skills 排前面的通常較關鍵。
#: ★ 合約 docstring 明訂 required_skills 不得重排，此參數依賴該保證。
JD_POSITION_DECAY = 0.3

#: 回傳上限。產品決定，非技術參數。
TOP_N = 5

#: 規格型條目：不是技能，是任職條件。人不會在面試中講出這串字，
#: 所以它永遠會被算成「漏講」——那是假指控。實測全語料 7 條，全是打字速度。
SPEC_PATTERN = re.compile(
    r"\d+\s*[~～-]\s*\d+|\d+\s*(字|級|年以上|分鐘)|證照|檢定|執照|駕照"
)

#: 中文專有名詞（ERP／系統品牌）。這些是硬技能，但 ASCII 判別法會誤判成軟技能。
#: 實測語料出現：鼎新、正航、天心資訊、文中系統。
CJK_PROPER_NOUNS = ("鼎新", "正航", "天心", "文中", "趨勢科技", "用友", "金蝶")

#: 單一拉丁字母就算數——C / R / C# / C++ / Go 都是硬技能，
#: 用 [A-Za-z]{2,} 會把它們漏掉（實測 C# 被誤判成 soft）。
#: 中文技能名不含拉丁字母，所以不會誤傷。
_ASCII_RUN = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------- kind 判定


#: 【實測推翻了原本的假設】STT_TERM_PROBE 單次實測(Galaxy S24 / Android 16):
#:     英文/ASCII 技能詞  管線抓到 15/25 (60%)
#:     中文技能詞        管線抓到  7/9  (78%)
#: 原本的判準是「ASCII → hard(可靠)、中文抽象詞 → soft(濾掉)」,**方向是反的**。
#: 被我濾掉的「專案時間╱進度控管」「帳務處理」實測都完美存活;
#: 被我判成可靠的 Angular / Git / jQuery / AJAX / MS SQL / ASP.NET 全滅。
#:
#: 所以 kind 不能從字串推導,只能查表。表由實測產生,查不到的走保守預設。
MEASURED_KIND: dict[str, str] = {}


def load_measured_kind(confusions: dict) -> None:
    """把 stt_confusions.v1.json 的實測結果灌進 MEASURED_KIND。

    verified_ok      → hard(實測抓得到,「沒抓到」的判斷可信)
    safe_aliases     → hard(補進 aliases 之後抓得到)
    colliding_aliases→ soft(要靠履歷條件判定,信心不足)
    """
    for term in confusions.get("verified_ok", []):
        MEASURED_KIND[term] = "hard"
    for term in confusions.get("safe_aliases", {}).values():
        MEASURED_KIND[term] = "hard"
    for info in confusions.get("colliding_aliases", {}).values():
        MEASURED_KIND[info["means"]] = "soft"


def classify_kind(name_zh: str, aliases: Iterable[str] = ()) -> str:
    """→ "hard" | "soft"。決定 B 的 why 走哪一路。

    判別的真正依據不是「硬技能／軟技能」這個分類學問題，而是
    **這個詞在口語中會不會被原字講出來**。實測 657 條真實技能字串分三群：

      ASCII 專有名詞   146 條(22%)  Python / SQL / ASP.NET / Excel   → 必定被指名
      中文短詞          89 條(14%)  灰帶。同時混著兩種東西：
                                     專有名詞 鼎新／天心資訊     → 會被指名
                                     抽象能力 專案時間／專案管理  → 只會被展演
      中文長片語       422 條(64%)  供應商原物料異常分析處理     → 沒有人這樣講話

    ★ 目前這是啟發式，不是量出來的。正確做法是等黃金測試集標完，
      對每個 said=2 的技能查「表面掃描有沒有抓到」，那才是 surface-findable
      的實測值。W2 D3–D5 用測得的結果取代本函式。在那之前保守走。
    """
    zh = (name_zh or "").strip()
    if not zh:
        return "soft"

    # ① 實測值優先。啟發式只在沒量過的詞上生效。
    if zh in MEASURED_KIND:
        return MEASURED_KIND[zh]

    # ★ 只看中文正式名,不看 name_en。詞彙表是雙語的,每個技能都有英文名——
    #   「溝通協調」的 name_en 是 Communication,但沒有人會在中文面試裡講
    #   Communication。把 name_en 納入判別會讓每個軟技能都變 hard,
    #   判別法整個失效。這是實測踩到的。
    if _ASCII_RUN.search(zh):
        return "hard"                      # Python / SQL / C# / ISO 9000
    if any(brand in zh for brand in CJK_PROPER_NOUNS):
        return "hard"                      # 鼎新 / 正航 / 天心資訊
    if extract_head(zh):
        return "hard"                      # 財務報表製作 → 財務報表,人會這樣講

    # 中文別名裡有夠短、口語講得出來的形式,也算搆得到
    for alias in aliases or ():
        a = (alias or "").strip()
        if a and not _ASCII_RUN.search(a) and (extract_head(a) or 2 <= len(a) <= 4):
            if a != zh:
                return "hard"
    return "soft"


def is_spec_entry(surface: str) -> bool:
    """規格型條目（中文打字20~50）→ 不該進候選池。"""
    return bool(SPEC_PATTERN.search(surface))


# ---------------------------------------------------------------- 集合建構


class _Ev:
    """A 側內部證據。合約的 GapCandidate 只吃扁平字串，這裡保留完整定位
    是為了校準與 debug 能回溯「這條是哪裡來的」，在邊界才壓平。"""

    __slots__ = ("skill_id", "source", "locator", "quote", "weight", "experience_ids")

    def __init__(self, skill_id, source, locator, quote, weight, experience_ids=()):
        self.skill_id = skill_id
        self.source = source
        self.locator = locator
        self.quote = quote
        self.weight = round(weight, 4)
        self.experience_ids = list(experience_ids)


def collect_resume_skills(resume: list[dict], normalizer, index: SurfaceIndex,
                          *, scan_prose: bool = False) -> dict[str, _Ev]:
    """ExperienceDTO 清單 → {skill_id: 證據}。

    ★ 同一技能出現在多筆經歷時，experience_ids 累積全部，不是取第一筆。
      合約的 resume_experience_ids 是複數就是為了這件事。
    """
    out: dict[str, _Ev] = {}
    for exp in resume or []:
        eid = str(exp.get("id", ""))
        for j, tag in enumerate(exp.get("tags") or []):
            text = _as_text(tag)
            if not text or is_spec_entry(text):
                continue
            sid = _resolve(normalizer, index, text)
            if not sid:
                continue
            prev = out.get(sid)
            if prev is None:
                out[sid] = _Ev(sid, "raw_tag", f"experiences[{eid}].tags[{j}]",
                               text, RESUME_CONST, [eid])
            elif eid not in prev.experience_ids:
                prev.experience_ids.append(eid)
        if scan_prose:
            desc = _as_text(exp.get("description"))
            for start, end, surface, sid_hint in index.scan(desc.lower()):
                sid = _resolve(normalizer, index, surface) or sid_hint
                if sid and sid not in out:
                    out[sid] = _Ev(sid, "prose", f"experiences[{eid}].description@{start}:{end}",
                                   desc[max(0, start - 12):end + 12], RESUME_CONST, [eid])
    return out


def collect_jd_skills(jd: JDInput, normalizer, index: SurfaceIndex) -> dict[str, _Ev]:
    """JDInput → {skill_id: 證據}。結構化帶位置衰減，散文權重打八折。"""
    out: dict[str, _Ev] = {}
    required = list(jd.required_skills or [])
    total = max(len(required), 1)
    for i, raw in enumerate(required):
        text = _as_text(raw)
        if not text or is_spec_entry(text):
            continue
        sid = _resolve(normalizer, index, text)
        if not sid:
            continue
        w = SOURCE_WEIGHT["structured"] * (1.0 - JD_POSITION_DECAY * (i / total))
        prev = out.get(sid)
        if prev is None or w > prev.weight:
            out[sid] = _Ev(sid, "structured", f"required_skills[{i}]", text, w)

    desc = jd.description or ""
    for start, end, surface, sid_hint in index.scan(desc.lower()):
        sid = _resolve(normalizer, index, surface) or sid_hint
        if not sid or sid in out:
            continue
        out[sid] = _Ev(sid, "prose", f"description@{start}:{end}",
                       _sentence_around(desc, start, end), SOURCE_WEIGHT["prose"])
    return out


def resolve_confusions(mentioned: set[str], resume_ids: set[str],
                       confusions: dict, name_to_id) -> dict[str, str]:
    """碰撞別名的履歷條件判定 → {補進去的 skill_id: 理由}。

    問題:STT 把 Angular 轉成 **Android**,而 Android 本身是常見技能。
    無條件加別名會讓真正的 Android 永遠被誤判成 Angular。

    解法用不對稱性:使用者講了 Angular、系統說他沒講 → 假指控,當場失去信任。
    使用者其實講 Android、系統算成 Angular → 少一條建議,可惜不傷人。
    所以**只在履歷有 Y 而沒有 X 時**才把 X 的命中補記為 Y。
    這個條件把誤判限縮在「履歷本來就有那個技能」的情況,範圍很窄。

    是加記不是改記:Android 若不在 履歷∩JD 裡,加記它是空操作。
    """
    added: dict[str, str] = {}
    for stt_form, info in (confusions.get("colliding_aliases") or {}).items():
        wrong_id = name_to_id(stt_form)
        right_id = name_to_id(info["means"])
        if not right_id or wrong_id not in mentioned:
            continue
        if right_id in resume_ids and wrong_id not in resume_ids:
            added[right_id] = f"STT 把「{info['means']}」轉成「{stt_form}」,履歷有前者無後者"
    return added


# ---------------------------------------------------------------- 主類別


class GapComputer:
    """GapComputer Protocol 的真實作。

    ## emit_soft：不要交出自己不敢背書的東西

    B 的過濾規則是「kind == soft 且信心不足時直接濾掉」。但語意段沒開的時候，
    軟技能的漏講判定**全部**是低信心——表面掃描本來就抓不到「我排了每週進度表」
    這種展演式陳述，所以每一條軟技能都會被誤判成漏講。

    與其把一堆假指控交出去再請 B 濾，不如我這邊就不發。所以 emit_soft 預設
    跟語意段連動：語意段沒開 → 不發 soft。語意段開了（W2 校準後）→ 發，
    並帶 kind="soft" 讓 B 的 prompt 有分寸，B 的過濾成為第二道防線。

    這樣沒有任何一個時間窗會讓假指控到使用者面前，也不會讓過濾器永久開著
    把軟技能整類砍掉——那會讓整個功能退化成關鍵字比對。
    """

    def __init__(self, analyzer: TranscriptAnalyzer, normalizer: Any | None = None, *,
                 top_n: int = TOP_N, scan_resume_prose: bool = False,
                 emit_soft: bool | None = None,
                 confusions: dict | None = None) -> None:
        self._confusions = confusions or {}
        self._analyzer = analyzer
        self._normalizer = normalizer
        self._index = analyzer._index
        self._top_n = top_n
        self._scan_resume_prose = scan_resume_prose
        self._emit_soft = (analyzer._enable_semantic if emit_soft is None else emit_soft)
        # (name_zh, aliases)——刻意不存 name_en,見 classify_kind 的說明
        self._names: dict[str, tuple[str, list[str]]] = {}
        for entry in analyzer._vocab:
            get = entry.get if isinstance(entry, dict) else lambda k, d=None: getattr(entry, k, d)
            sid = get("skill_id") or get("id")
            if sid:
                self._names[str(sid)] = (
                    str(get("name_zh") or get("name") or ""), list(get("aliases") or [])
                )

    def _name_to_id(self, name: str) -> str | None:
        for sid, (zh, aliases) in self._names.items():
            if zh == name or name in aliases:
                return sid
        return None

    def compute(self, resume: list[dict], jd: JDInput, transcript: str) -> list[GapCandidate]:
        """回傳漏講候選，依 weight 由高到低排序。履歷為空回空清單，不拋例外。"""
        if not resume:
            return []
        if isinstance(jd, str):  # 過渡期防呆：合約已改，但舊呼叫端可能還沒改
            jd = JDInput(required_skills=[], description=jd, source="extracted")

        resume_skills = collect_resume_skills(resume, self._normalizer, self._index,
                                              scan_prose=self._scan_resume_prose)
        jd_skills = collect_jd_skills(jd, self._normalizer, self._index)
        mentioned = self._analyzer.mentions(transcript or "")
        if self._confusions:
            extra = resolve_confusions(set(mentioned), set(resume_skills),
                                       self._confusions, self._name_to_id)
            for sid in extra:
                mentioned.setdefault(sid, [])

        out: list[GapCandidate] = []
        for sid in (set(resume_skills) & set(jd_skills)) - set(mentioned):
            r, j = resume_skills[sid], jd_skills[sid]
            zh, aliases = self._names.get(sid, (sid, []))
            kind = classify_kind(zh, aliases)
            if kind == "soft" and not self._emit_soft:
                continue
            out.append(GapCandidate(
                skill_id=sid,
                display=self._analyzer.display(sid),
                kind=kind,
                resume_experience_ids=list(r.experience_ids),
                resume_evidence=r.quote,
                jd_evidence=j.quote,
                weight=round(W_JD * j.weight + W_RESUME * r.weight, 4),
            ))
        # skill_id 當末位 tie-breaker：同分時順序也要確定，否則同輸入跑兩次
        # 前五名不一樣（W2 那個 FakeEmbedding 用 hash() 導致分數漂移的教訓）
        out.sort(key=lambda c: (-c.weight, c.skill_id))
        return out[: self._top_n]

    def explain(self, resume: list[dict], jd: JDInput, transcript: str) -> dict[str, Any]:
        """三集合中間值，給校準與 debug（不在合約裡，A 側自用）。"""
        if isinstance(jd, str):
            jd = JDInput(required_skills=[], description=jd)
        r = collect_resume_skills(resume, self._normalizer, self._index,
                                  scan_prose=self._scan_resume_prose)
        j = collect_jd_skills(jd, self._normalizer, self._index)
        m = self._analyzer.mentions(transcript or "")
        overlap = set(r) & set(j)
        gap = overlap - set(m)
        return {
            "resume": sorted(r), "jd": sorted(j), "mentioned": sorted(m),
            "overlap": sorted(overlap), "gap": sorted(gap),
            "gap_soft_suppressed": sorted(
                s for s in gap
                if classify_kind(*self._names.get(s, (s, []))) == "soft" and not self._emit_soft
            ),
        }


# ---------------------------------------------------------------- 小工具


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name_zh", "name", "skill", "text", "value"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def _resolve(normalizer: Any, index: SurfaceIndex, text: str) -> str | None:
    """字串 → skill_id。先問正規化器，問不到退回表面形精確匹配。

    退路存在的理由：正規化器第二段要載模型，gap 的單元測試不該綁模型。
    真實環境永遠是正規化器先答話。
    """
    sid = resolve_skill_id(normalizer, text)
    if sid:
        return sid
    hits = index.scan(text.lower())
    return hits[0][3] if hits else None


def _sentence_around(text: str, start: int, end: int) -> str:
    """JD 散文命中 → 回整句。合約要求 jd_evidence 是「JD 原文片段」，
    而 why 只能引用它，所以要給得起論證的完整句子，不是一個詞。"""
    puncts = "。！？；\n.!?;"
    lo, hi = start, end
    while lo > 0 and text[lo - 1] not in puncts:
        lo -= 1
    while hi < len(text) and text[hi] not in puncts:
        hi += 1
    return text[lo:hi].strip()
