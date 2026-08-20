"""面試模組 W1 驗收測試（成員 A）。

比照 W1/W2 的做法：每個 sprint 交付都配驗收測試，測試數當交付指標。
本檔全部不依賴模型——語意段預設關閉，這些測試在 CI 上秒過。
"""

from __future__ import annotations

import pytest

from app.contracts.interview_protocols import (
    CollabObserver as CollabObserverProto,
    GapComputer as GapComputerProto,
    JDInput,
    TranscriptAnalyzer as TranscriptAnalyzerProto,
)
from app.pipeline.gap import (
    GapComputer,
    classify_kind,
    collect_jd_skills,
    collect_resume_skills,
    is_spec_entry,
)
from app.pipeline.transcript import (
    SurfaceIndex,
    TranscriptAnalyzer,
    extract_head,
    normalize_for_scan,
)


# ---------------------------------------------------------------- fixtures


VOCAB = [
    {"skill_id": "sk:python", "name_zh": "Python", "name_en": "Python",
     "aliases": ["python3", "py"]},
    {"skill_id": "sk:sql", "name_zh": "SQL", "name_en": "SQL",
     "aliases": ["資料庫查詢", "mysql"]},
    {"skill_id": "skm:dataviz", "name_zh": "資料視覺化", "name_en": "Data Visualization",
     "aliases": ["圖表製作", "tableau", "power bi"]},
    {"skill_id": "sk:java", "name_zh": "Java", "name_en": "Java", "aliases": []},
    {"skill_id": "sk:js", "name_zh": "JavaScript", "name_en": "JavaScript",
     "aliases": ["javascript", "js"]},
    {"skill_id": "skm:comm", "name_zh": "溝通協調", "name_en": "Communication",
     "aliases": ["跨部門溝通", "溝通"]},
    {"skill_id": "skm:pm", "name_zh": "專案管理", "name_en": "Project Management",
     "aliases": ["專案時間/進度控管"]},
    {"skill_id": "sk:r", "name_zh": "R", "name_en": "R", "aliases": []},
]


@pytest.fixture
def analyzer() -> TranscriptAnalyzer:
    return TranscriptAnalyzer(vocab=VOCAB)


@pytest.fixture
def index() -> SurfaceIndex:
    return SurfaceIndex.from_vocab(VOCAB)


# ExperienceDTO 清單。實測 golden_pairs.v1 的慣例:技能全在 tags,
# 且集中在第一筆經歷——所以 resume_experience_ids 必須是複數。
RESUME = [
    {"id": "e1", "title": "資料分析實習", "category": "工作",
     "timeRange": "2024.07", "description": "用 Java 寫過小工具",
     "tags": ["Python", "SQL", "資料視覺化", "跨部門溝通", "中文打字20~50"]},
    {"id": "e2", "title": "課程專題", "category": "學業",
     "timeRange": "2025.02", "description": "", "tags": ["Python"]},
]

JD = JDInput(
    required_skills=["Python", "SQL", "資料視覺化", "中文打字20~50"],
    description="需要能跟不同部門溝通協調的人。熟悉專案管理尤佳。",
    source="catalog",
)


# ---------------------------------------------------------------- 文字正規化


def test_fullwidth_and_case_normalized():
    text, _ = normalize_for_scan("我會用 ＰＹＴＨＯＮ 做分析")
    assert "python" in text


def test_stt_spaced_letters_are_joined():
    """中文 STT 常把縮寫逐字母吐出來——不黏回去的話 SQL 永遠掃不到。"""
    text, _ = normalize_for_scan("那時候是用 s q l 撈資料")
    assert "sql" in text


def test_normal_english_phrases_are_not_glued():
    """只黏「每個 run 都是單字母」的，一般英文片語不能受影響。"""
    text, _ = normalize_for_scan("我做 data analysis")
    assert "data analysis" in text


def test_offsets_point_back_to_original(analyzer):
    raw = "前面墊一些字然後我用ＰＹＴＨＯＮ做的"
    ev = analyzer.mentions(raw)["sk:python"][0]
    assert raw[ev.start : ev.end].lower().replace("　", "") != ""
    assert "ＰＹＴＨＯＮ" in raw[ev.start : ev.end] or "PYTHON" in raw[ev.start : ev.end].upper()


# ---------------------------------------------------------------- 表面掃描


def test_longest_match_wins(index):
    """JavaScript 不能被拆成 Java + Script。"""
    hits = index.scan("我主要寫 javascript")
    assert [h[3] for h in hits] == ["sk:js"]


def test_ascii_boundary_blocks_substring_hits(index):
    """單字母技能 R 不能在 react / error 裡面命中。"""
    assert index.scan("我用 react 做前端") == [] or all(
        h[3] != "sk:r" for h in index.scan("我用 react 做前端")
    )
    assert any(h[3] == "sk:r" for h in index.scan("統計都用 r 跑"))


def test_alias_hits_map_to_canonical_id(analyzer):
    said = analyzer.mentioned_skills("那個報表我都是用 power bi 做圖表製作")
    assert said == {"skm:dataviz"}


# ---------------------------------------------------------------- 提及集


def test_mentioned_skills_returns_skill_ids(analyzer):
    said = analyzer.mentioned_skills("我用 python 接 mysql 然後做圖表製作")
    assert said == {"sk:python", "sk:sql", "skm:dataviz"}


def test_mentions_carry_quote_evidence(analyzer):
    evidences = analyzer.mentions("那時候我就是用 python 把資料抓下來")
    ev = evidences["sk:python"][0]
    assert ev.stage == "surface" and ev.score == 1.0
    assert "python" in ev.quote.lower()


def test_empty_transcript_yields_empty_set(analyzer):
    assert analyzer.mentioned_skills("") == set()


def test_analysis_is_deterministic(analyzer):
    text = "我用 python 跟 sql 做了報表製作然後跟業務溝通"
    assert analyzer.mentioned_skills(text) == analyzer.mentioned_skills(text)


# ---------------------------------------------------------------- 文字統計


def test_text_stats_counts_fillers():
    a = TranscriptAnalyzer(vocab=VOCAB)
    stats = a.text_stats("那個我就是覺得然後那個部分基本上還行")
    assert stats.filler_count >= 4
    assert "那個" in stats.filler_detail
    assert stats.filler_count / stats.char_count > 0


def test_text_stats_counts_quantifiers():
    a = TranscriptAnalyzer(vocab=VOCAB)
    stats = a.text_stats("我把時間縮短了 30%，團隊有 12 人，效率是原本的三倍。")
    assert stats.quantifier_count >= 3


def test_filler_reliability_measured_when_non_lexical_present():
    """有「嗯/呃/欸」代表引擎沒在濾,計數反映使用者的口語習慣。"""
    a = TranscriptAnalyzer(vocab=VOCAB)
    assert a.text_stats("嗯我那時候就是負責這塊呃印象很深").filler_reliability == "measured"


def test_filler_reliability_unknown_on_partial_suppression_signature():
    """長逐字稿只有詞彙型填充詞、一個非詞彙填充音都沒有——
    這是引擎濾掉「嗯/呃」但保留「那個/就是」的特徵。此時 filler_count
    有值但系統性偏低,回 measured 會過度宣稱,保守回 unknown。"""
    a = TranscriptAnalyzer(vocab=VOCAB)
    long_text = ("那個我那時候就是負責整個流程然後我是用排程把資料抓下來"
                 "然後就是存進資料庫後來主管想看趨勢所以我就做了儀表板"
                 "那個大概讓他們每週省下三個小時其實這件事讓我學到很多") * 2
    stats = a.text_stats(long_text)
    assert stats.char_count >= 150 and stats.filler_count > 0
    assert stats.filler_reliability == "unknown"


def test_filler_reliability_suppressed_is_engine_config_not_inference():
    """suppressed 是引擎的性質,不是逐字稿的性質——單一逐字稿沒有填充詞,
    可能是引擎濾掉也可能是講者很順,分不出來。所以只由探測結果一次設定。"""
    a = TranscriptAnalyzer(vocab=VOCAB)
    assert a.text_stats("我負責資料處理。").filler_reliability == "unknown"
    b = TranscriptAnalyzer(vocab=VOCAB, engine_filler_policy="suppressed")
    assert b.text_stats("嗯那個就是").filler_reliability == "suppressed"


def test_stt_segment_is_distinct_from_punctuation():
    """★ 這條守著我原本寫錯的地方。

    _PUNCT_RE 含 \n，所以多段用換行接起來會命中「有標點」。但實測逐字稿
    一個真標點都沒有——回 punctuation 等於宣稱那 76 字是精確句長，
    實際上它是「她停頓前講了多長」。過度宣稱比不宣稱糟：
    B 的 prompt 會照著把它講成句子長度。
    """
    a = TranscriptAnalyzer(vocab=VOCAB)
    stt = a.text_stats("我那時候用了排程\n後來主管想看趨勢\n所以我做了儀表板")
    assert stt.segmentation == "stt_segment"
    assert stt.sentence_count == 3          # 段界＝STT 自動送出點
    assert a.text_stats("我先做分析。然後報告。").segmentation == "punctuation"


def test_mentioned_skills_accepts_candidates():
    """合約 W2 增補：candidates=None 維持 W1 行為，給定時只回集合內的。"""
    a = TranscriptAnalyzer(vocab=VOCAB)
    text = "我用 python 接 mysql"
    assert a.mentioned_skills(text) == {"sk:python", "sk:sql"}
    assert a.mentioned_skills(text, candidates={"sk:python"}) == {"sk:python"}
    assert a.mentioned_skills(text, candidates=set()) == set()


def test_segmentation_flag_is_honest_without_punctuation():
    """無標點時 avg_segment_length 是估算，必須誠實標記——W3 D1 要驗的就是這格。"""
    a = TranscriptAnalyzer(vocab=VOCAB)
    with_punct = a.text_stats("我先做了分析。然後跟主管報告。")
    without = a.text_stats("我先做了分析然後跟主管報告後來又改了一版")
    assert with_punct.segmentation == "punctuation"
    assert without.segmentation == "discourse_marker"
    assert without.sentence_count >= 2


# ---------------------------------------------------------------- 集合建構


def test_resume_side_weight_is_constant(index):
    """履歷側權重恆為常數——技能全從 tags 進來,structured/raw_tag 區分不發生。
    這代表 RAW_TAG_DISCOUNT 與 W_RESUME 不可校準,見 D1 異議第五條。"""
    skills = collect_resume_skills(RESUME, None, index)
    assert len({ev.weight for ev in skills.values()}) == 1


def test_resume_experience_ids_accumulate(index):
    """同一技能出現在多筆經歷時要累積,不是取第一筆——合約用複數就是為了這個。"""
    skills = collect_resume_skills(RESUME, None, index)
    assert skills["sk:python"].experience_ids == ["e1", "e2"]
    assert skills["sk:sql"].experience_ids == ["e1"]


def test_resume_prose_is_off_by_default(index):
    """履歷寧缺勿濫：description 裡的 Java 預設不算他會 Java。"""
    assert "sk:java" not in collect_resume_skills(RESUME, None, index)
    assert "sk:java" in collect_resume_skills(RESUME, None, index, scan_prose=True)


def test_jd_prose_is_scanned(index):
    """JD 寧濫勿缺：散文裡的溝通協調與專案管理要抓到。"""
    skills = collect_jd_skills(JD, None, index)
    assert "skm:comm" in skills and "skm:pm" in skills
    assert skills["skm:comm"].source == "prose"


def test_jd_evidence_quotes_a_whole_sentence(index):
    """why 只能引用 jd_evidence,所以它要給得起論證的整句,不是一個詞。"""
    skills = collect_jd_skills(JD, None, index)
    assert skills["skm:comm"].quote.endswith("的人")


def test_jd_position_decay(index):
    skills = collect_jd_skills(JD, None, index)
    assert skills["sk:python"].weight > skills["skm:dataviz"].weight


def test_spec_entries_never_become_candidates(index):
    """中文打字20~50 是任職條件不是技能,人不會講出這串字,
    留著它就會永遠被算成漏講——那是假指控。"""
    assert is_spec_entry("中文打字20~50")
    assert not is_spec_entry("中文打字")
    assert all("打字" not in ev.locator for ev in collect_resume_skills(RESUME, None, index).values())
    assert "中文打字20~50" not in {ev.quote for ev in collect_jd_skills(JD, None, index).values()}


# ---------------------------------------------------------------- 差集


def test_gap_is_intersection_minus_mentioned(analyzer):
    computer = GapComputer(analyzer, emit_soft=True)
    transcript = "我這邊主要是用 python 做分析"
    gaps = {c.skill_id for c in computer.compute(RESUME, JD, transcript)}
    # 履歷∩JD 四個:python/sql 兩側結構化、dataviz 履歷 tag ∩ JD 結構化、
    # comm 履歷 tag「跨部門溝通」∩ JD 散文「跟不同部門溝通協調」——
    # 最後這條就是 JD 散文掃描的價值:需求埋在句子裡,required_skills 沒列。
    assert gaps == {"sk:sql", "skm:dataviz", "skm:comm"}


def test_soft_gaps_suppressed_while_semantic_off():
    """語意段沒開時不發 soft——表面掃描抓不到展演式陳述,每一條都會誤判。
    不要交出自己不敢背書的東西,再讓 B 去濾。

    ★ kind 的正確語意不是「這是不是軟技能」,而是
      **「如果使用者講了,我抓不抓得到」**。抓得到 → 沒抓到就代表真的沒講,
      這個否定判斷有意義 → hard。抓不到 → 沒抓到什麼都不代表 → soft。
      所以它其實是詞彙表別名覆蓋度的函數,不是分類學屬性。
    """
    vocab = [
        {"skill_id": "sk:python", "name_zh": "Python", "name_en": "Python", "aliases": []},
        # 只有正式名、沒有口語別名 → 使用者說「我排了每週進度表」抓不到 → soft
        {"skill_id": "skm:pm_time", "name_zh": "專案時間", "name_en": "Project Scheduling",
         "aliases": []},
    ]
    resume = [{"id": "e1", "title": "專題", "tags": ["Python", "專案時間"], "description": ""}]
    jd = JDInput(required_skills=["Python", "專案時間"], source="catalog")
    analyzer = TranscriptAnalyzer(vocab=vocab)

    off = GapComputer(analyzer).compute(resume, jd, "")
    on = GapComputer(analyzer, emit_soft=True).compute(resume, jd, "")
    assert {c.skill_id for c in off} == {"sk:python"}
    assert {c.skill_id for c in on} == {"sk:python", "skm:pm_time"}
    assert {c.kind for c in off} == {"hard"}


def test_gap_accepts_plain_string_jd_defensively(analyzer):
    """合約已改 JDInput,但舊呼叫端可能還沒改。不要炸,降級處理。"""
    computer = GapComputer(analyzer, emit_soft=True)
    assert computer.compute(RESUME, "需要 SQL 的人", "") is not None


def test_gap_candidates_carry_both_sides_of_evidence(analyzer):
    computer = GapComputer(analyzer, emit_soft=True)
    for c in computer.compute(RESUME, JD, "我就是講一些沒有內容的話"):
        assert c.resume_evidence and c.jd_evidence
        assert c.resume_experience_ids
        assert c.display.endswith(f"({c.skill_id})")
        assert c.kind in ("hard", "soft")


def test_skill_only_in_jd_is_not_a_gap(analyzer):
    """JD 要但履歷沒有，那是能力落差不是漏講，不該進這條管線。"""
    computer = GapComputer(analyzer)
    gaps = {c.skill_id for c in computer.compute(RESUME, JD, "")}
    assert "skm:pm" not in gaps


def test_weight_ranks_structured_jd_hits_higher(analyzer):
    computer = GapComputer(analyzer, emit_soft=True)
    gaps = computer.compute(RESUME, JD, "")
    weights = [c.weight for c in gaps]
    assert weights == sorted(weights, reverse=True)
    assert gaps[0].skill_id == "sk:python"


def test_equal_weight_breaks_ties_by_skill_id(analyzer):
    """同分時順序也必須確定,否則同一份輸入跑兩次會給出不同的前五名——
    W2 那個 FakeEmbedding 用 hash() 導致分數漂移的教訓。"""
    computer = GapComputer(analyzer, emit_soft=True)
    a = [c.skill_id for c in computer.compute(RESUME, JD, "")]
    b = [c.skill_id for c in computer.compute(RESUME, JD, "")]
    assert a == b == sorted(a, key=lambda s: (
        -{c.skill_id: c.weight for c in computer.compute(RESUME, JD, "")}[s], s))


def test_top_n_caps_output(analyzer):
    computer = GapComputer(analyzer, top_n=1, emit_soft=True)
    assert len(computer.compute(RESUME, JD, "")) == 1


def test_empty_resume_returns_empty_not_exception(analyzer):
    """沒建經歷是正常情況,端點要能優雅降級——合約明訂。"""
    assert GapComputer(analyzer).compute([], JD, "abc") == []


def test_gap_output_is_deterministic(analyzer):
    computer = GapComputer(analyzer, emit_soft=True)
    first = [c.skill_id for c in computer.compute(RESUME, JD, "")]
    second = [c.skill_id for c in computer.compute(RESUME, JD, "")]
    assert first == second


def test_explain_exposes_all_three_sets(analyzer):
    computer = GapComputer(analyzer)
    detail = computer.explain(RESUME, JD, "我用 python")
    assert {"resume", "jd", "mentioned", "overlap", "gap"} <= set(detail)
    assert "sk:python" in detail["mentioned"]


# ---------------------------------------------------------------- 合約符合性


def test_implementations_satisfy_frozen_protocols(analyzer):
    assert isinstance(analyzer, TranscriptAnalyzerProto)
    assert isinstance(GapComputer(analyzer), GapComputerProto)


def test_text_stats_matches_frozen_field_names(analyzer):
    stats = analyzer.text_stats("那個我就是然後基本上")
    for field in ("char_count", "filler_count", "filler_detail", "quantifier_count",
                  "sentence_count", "avg_sentence_len", "segmentation"):
        assert hasattr(stats, field)


# ---------------------------------------------------------------- 詞頭匹配


def test_head_extraction_keeps_only_sayable_heads():
    """財務報表製作 → 財務報表(人會這樣講);
    供應商原物料異常分析處理 抽不出夠短的詞頭(人不會這樣講)。"""
    assert extract_head("財務報表製作") == "財務報表"
    assert extract_head("報表彙整與管理") == "報表彙整"
    assert extract_head("供應商原物料異常分析處理") is None
    assert extract_head("Python") is None


def test_head_match_finds_spoken_short_form():
    a = TranscriptAnalyzer(vocab=[{"skill_id": "sk:fin", "name_zh": "財務報表製作",
                                   "aliases": []}])
    assert a.mentioned_skills("我每個月都要做財務報表給主管看") == {"sk:fin"}
    ev = a.mentions("我每個月都要做財務報表")["sk:fin"][0]
    assert ev.stage == "head" and ev.score < 1.0


def test_full_surface_beats_head(index):
    """詞頭只補洞,不搶——完整表面形優先。"""
    hits = index.scan("我做資料視覺化")
    assert hits and hits[0][3] == "skm:dataviz"


def test_kind_classification():
    assert classify_kind("Python") == "hard"
    assert classify_kind("C#") == "hard"            # 單字母語言,曾誤判
    assert classify_kind("鼎新") == "hard"           # 中文品牌
    assert classify_kind("財務報表製作") == "hard"     # 抽得出詞頭
    assert classify_kind("專案管理") == "soft"        # 真抽象能力
    assert classify_kind("帳務處理") == "soft"


def test_english_name_must_not_make_soft_skills_hard():
    """詞彙表是雙語的,每個技能都有 name_en。若把 name_en 納入判別,
    「溝通協調」會因為 Communication 而變成 hard,判別法整個失效。"""
    assert classify_kind("溝通協調", ["Communication"]) == "soft"
    assert classify_kind("專案時間", ["Project Time Management"]) == "soft"
