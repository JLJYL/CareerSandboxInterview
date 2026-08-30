"""A4 面試報告管線測試。不打真 LLM。

重點在**機械保證**:固定值域、verbatim、輸入方式對評分的影響、
以及任一區塊失敗時其餘仍然可用。
"""

from __future__ import annotations

import json

import pytest

from app.contracts.interview_protocols import TextStats
from app.pipeline.interview_report import (
    derive_improvements,
    format_stats,
    generate_report,
    parse_array,
    VERBATIM_FAIL_MARKER,
    score_depth,
    score_fluency,
    ungrounded_numbers,
)
from app.schemas.interview import (
    SUB_SCORE_NAMES,
    FaceDimensionDTO,
    TurnDTO,
)

TURNS = [
    TurnDTO(question="請自我介紹", answer="我是資管系大三的學生 過去在系學會做行銷", input_mode="voice"),
    TurnDTO(question="講一個解決問題的經驗", answer="我們辦過一場聯名活動 我沒先測試就直接全推", input_mode="voice"),
]

VOICE_STATS = TextStats(
    char_count=200, filler_count=12, filler_detail={"那個": 5, "就是": 7},
    quantifier_count=3, sentence_count=4, avg_sentence_len=78.0,
    filler_reliability="measured", segmentation="stt_segment",
)
TYPED_STATS = TextStats(
    char_count=200, filler_count=0, filler_detail={},
    quantifier_count=3, sentence_count=6, avg_sentence_len=33.0,
    filler_reliability="not_applicable", segmentation="punctuation",
)


def fake_llm(face=None, subs=None, questions=None, star=None):
    """依 system prompt 的開頭判斷是哪個區塊,回對應的假輸出。"""
    def call(system: str, user: str) -> str:
        if "三個面向" in system:
            return json.dumps(face if face is not None else [
                {"letter": "內", "verdict": "有講到重點但缺數字", "points": ["自我介紹加一個量化成果"]},
                {"letter": "構", "verdict": "邏輯清楚但缺結果段", "points": ["講經歷補上結果那一段"]},
                {"letter": "達", "verdict": "用詞直接但填充詞偏多", "points": ["重點句講完停半秒"]},
            ], ensure_ascii=False)
        if "細分分數" in system:
            names = [n for n in SUB_SCORE_NAMES if n not in ("表達流暢度", "內容深度")]
            return json.dumps(subs if subs is not None else
                              [{"name": n, "score": v} for n, v in zip(names, [82, 68, 64, 80])],
                              ensure_ascii=False)
        if "逐題" in system:
            return json.dumps(questions if questions is not None else [
                {"id": 1, "comment": "有交代背景但沒帶出成果", "better": "我是資管系大三,系學會行銷把追蹤數從零做到一千二"},
                {"id": 2, "comment": "情境清楚但沒講結果", "better": "那次全推成效不如預期,後來改成先小規模測試"},
            ], ensure_ascii=False)
        return json.dumps(star if star is not None else [
            {"key": "S", "present": True, "fromAnswer": "我們辦過一場聯名活動", "hint": ""},
            {"key": "T", "present": False, "fromAnswer": "", "hint": "缺任務:你負責哪一塊"},
            {"key": "A", "present": True, "fromAnswer": "我沒先測試就直接全推", "hint": ""},
            {"key": "R", "present": False, "fromAnswer": "", "hint": "缺結果:最後怎麼了"},
        ], ensure_ascii=False)
    return call


async def run(llm=None, **kw):
    return await generate_report(
        mode=kw.pop("mode", "single"), turns=kw.pop("turns", TURNS),
        stats=kw.pop("stats", VOICE_STATS), input_mode=kw.pop("input_mode", "voice"),
        resume_text=kw.pop("resume_text", ""), resume_grounded=kw.pop("resume_grounded", False),
        llm=llm or fake_llm(), **kw,
    )


# ---------------------------------------------------------------------------
# 輸入方式對評分的影響
# ---------------------------------------------------------------------------


def test_typed_input_disables_fluency_measurement() -> None:
    """打字的答案沒有填充詞、標點乾淨,不可以因此拿到高流暢度。

    那會在面試模擬器裡獎勵錯誤的行為——認真用講的反而低分。
    """
    text, measurable = format_stats(TYPED_STATS, "typed")
    assert measurable is False
    assert "打字輸入" in text


def test_unknown_input_treated_as_typed() -> None:
    _, measurable = format_stats(VOICE_STATS, "unknown")
    assert measurable is False


def test_suppressed_fillers_disable_measurement() -> None:
    """引擎移除填充詞時,計數反映的是後處理不是使用者。"""
    s = TextStats(char_count=200, filler_count=0, filler_reliability="suppressed")
    _, measurable = format_stats(s, "voice")
    assert measurable is False


def test_voice_with_measured_fillers_is_measurable() -> None:
    text, measurable = format_stats(VOICE_STATS, "voice")
    assert measurable is True
    assert "每百字 6.0" in text


def test_segment_length_is_not_called_sentence_length() -> None:
    """段界來自語音辨識的送出點,量的是語流連續性,不是句子長度。"""
    text, _ = format_stats(VOICE_STATS, "voice")
    assert "語流連續性" in text


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixed_enumerations_are_exact() -> None:
    r = await run()
    assert [s.name for s in r.sub_scores] == list(SUB_SCORE_NAMES)
    assert [p.key for p in r.star_parts] == ["S", "T", "A", "R"]
    assert [d.letter for d in r.face_dimensions] == ["內", "構", "達"]


@pytest.mark.asyncio
async def test_face_score_matches_sub_score() -> None:
    """同一件事不可以出現兩個不一致的數字。"""
    r = await run()
    by_name = {s.name: s.score for s in r.sub_scores}
    by_letter = {d.letter: d.score for d in r.face_dimensions}
    assert by_letter["內"] == by_name["內容深度"]
    assert by_letter["達"] == by_name["表達流暢度"]


@pytest.mark.asyncio
async def test_fluency_comes_from_formula_not_llm() -> None:
    """能用公式算的不要問 LLM。

    實測填充詞每百字 3.0(正常區間)時模型給 71,分數沒有跟著錨走。
    """
    r = await run()
    fluency = next(s.score for s in r.sub_scores if s.name == "表達流暢度")
    expected, _ = score_fluency(VOICE_STATS, True)
    assert fluency == expected
    assert any("由系統計算" in n for n in r.notices)


@pytest.mark.asyncio
async def test_llm_fluency_value_is_ignored() -> None:
    """模型就算硬回流暢度,也不採用。"""
    llm = fake_llm(subs=[{"name": "表達流暢度", "score": 20}, {"name": "內容深度", "score": 78}])
    r = await run(llm)
    fluency = next(s.score for s in r.sub_scores if s.name == "表達流暢度")
    assert fluency != 20


def test_depth_comes_from_formula_not_llm() -> None:
    """實測十格,LLM 給的內容深度有 7 格是完全相同的 68——
    它把 prompt 的區間邊界當成答案,不是在分辨。"""
    assert score_depth(TextStats(quantifier_count=2))[0] < score_depth(
        TextStats(quantifier_count=12)
    )[0]


def test_depth_is_monotonic() -> None:
    scores = [score_depth(TextStats(quantifier_count=n))[0] for n in (0, 3, 6, 12, 30)]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores), "每個輸入要有不同輸出,否則沒有鑑別力"


def test_llm_depth_value_is_ignored() -> None:
    """模型硬回內容深度也不採用。"""
    import asyncio

    llm = fake_llm(subs=[{"name": "內容深度", "score": 5}, {"name": "邏輯清晰度", "score": 78}])
    r = asyncio.run(run(llm))
    assert next(s.score for s in r.sub_scores if s.name == "內容深度") != 5


def test_fluency_follows_measured_anchors() -> None:
    """成員 A 的實測錨點:正常 3.0、多講故事 6.4、刻意講亂 9.8。"""
    def at(rate: float) -> int:
        st = TextStats(char_count=1000, filler_count=int(rate * 10), filler_reliability="measured")
        return score_fluency(st, True)[0]

    assert at(3.0) > at(6.4) > at(9.8)
    assert at(3.0) >= 80
    assert at(9.8) <= 62


def test_fluency_defaults_to_mid_when_unmeasurable() -> None:
    """不可測時不給高分(打字的人會佔便宜),也不給低分(那是懲罰未知)。"""
    score, basis = score_fluency(TYPED_STATS, False)
    assert 68 <= score <= 76
    assert "不可測" in basis


@pytest.mark.asyncio
async def test_better_with_fabricated_number_is_cleared() -> None:
    """實測失敗案例:回答只說「約訪談的時間」,模型寫成「訪談了三位消費者」。

    使用者會把 better 背下來拿去真正的面試講,那個數字他答不出後續追問。
    """
    llm = fake_llm(questions=[
        {"id": 1, "comment": "背景清楚", "better": "我訪談了三位消費者並產出企劃書"},
        {"id": 2, "comment": "情境清楚", "better": "我們辦過一場聯名活動,我沒先測試就全推"},
    ])
    r = await run(llm)
    assert r.question_feedbacks[0].better == ""
    assert r.question_feedbacks[1].better
    assert any("原回答沒有的數字" in n for n in r.notices)


def test_grounded_numbers_pass() -> None:
    assert ungrounded_numbers("我用 SQL 花了 4 小時", "那時候花了 4 小時寫 SQL") == []


def test_chinese_numerals_need_a_measure_word() -> None:
    """只抓帶量詞的中文數字,避免把「一直」「十分清楚」誤判成數量。"""
    assert ungrounded_numbers("我一直在做這件事", "他說了一些話") == []
    assert ungrounded_numbers("這件事我十分清楚", "我知道這件事") == []
    assert ungrounded_numbers("我帶過十人團隊", "我做過專案") == ["十人"]


def test_number_compared_with_measure_not_alone() -> None:
    """實測踩到的坑:來源有「資管系大三」,只比對「三」會誤判成有依據。"""
    assert ungrounded_numbers("訪談了三位消費者", "我是資管系大三的學生") == ["三位"]


def test_arabic_number_mismatch_caught() -> None:
    assert ungrounded_numbers("追蹤數成長到 5000", "追蹤數從零成長到 1200") == ["5000"]


@pytest.mark.asyncio
async def test_fabricated_quote_is_downgraded_not_kept() -> None:
    """fromAnswer 必須是逐字稿原文。對不上就降成未命中,不是改寫。"""
    llm = fake_llm(star=[
        {"key": "S", "present": True, "fromAnswer": "我在台積電帶過十人團隊", "hint": ""},
        {"key": "T", "present": False, "fromAnswer": "", "hint": "缺任務"},
        {"key": "A", "present": False, "fromAnswer": "", "hint": "缺行動"},
        {"key": "R", "present": False, "fromAnswer": "", "hint": "缺結果"},
    ])
    r = await run(llm)
    s = next(p for p in r.star_parts if p.key == "S")
    assert s.present is False
    assert s.from_answer == ""
    assert any(VERBATIM_FAIL_MARKER in n for n in r.notices)


@pytest.mark.asyncio
async def test_real_quote_survives() -> None:
    r = await run()
    s = next(p for p in r.star_parts if p.key == "S")
    assert s.present is True
    assert s.from_answer == "我們辦過一場聯名活動"


@pytest.mark.asyncio
async def test_one_block_failure_keeps_the_rest() -> None:
    """任一區塊失敗不該讓整份報告消失。"""
    def flaky(system: str, user: str) -> str:
        if "細分分數" in system:
            raise RuntimeError("timeout")
        return fake_llm()(system, user)

    r = await run(flaky)
    assert r.question_feedbacks, "其他區塊應該還在"
    # 流暢度由公式算,不受 LLM 區塊失敗影響——這正是把它移出 LLM 的好處之一
    by_name = {s.name: s.score for s in r.sub_scores}
    # 公式算的兩項不受 LLM 區塊失敗影響——這是把它們移出 LLM 的附帶好處
    assert by_name["表達流暢度"] > 0
    assert by_name["內容深度"] > 0
    # 交給 LLM 的那幾項才會補 0
    assert by_name["邏輯清晰度"] == 0, "失敗的區塊補 0 並記入 notices"
    assert any("subs" in n for n in r.notices)


@pytest.mark.asyncio
async def test_locale_normalized_in_generated_text() -> None:
    llm = fake_llm(face=[
        {"letter": "內", "verdict": "你的用戶界面說明不夠具體", "points": ["補上交互功能的細節"]},
        {"letter": "構", "verdict": "結構清楚", "points": []},
        {"letter": "達", "verdict": "表達直接", "points": []},
    ])
    r = await run(llm)
    d = r.face_dimensions[0]
    assert "使用者介面" in d.verdict
    assert "互動功能" in d.points[0]


@pytest.mark.asyncio
async def test_prosody_only_when_measurable() -> None:
    r = await run()
    assert next(d for d in r.face_dimensions if d.letter == "達").prosody
    r2 = await run(input_mode="typed", stats=TYPED_STATS)
    assert next(d for d in r2.face_dimensions if d.letter == "達").prosody is None


@pytest.mark.asyncio
async def test_group_without_rubric_shows_no_collab() -> None:
    r = await run(mode="group")
    assert r.collab_dims == []
    assert any("協作評分尚未啟用" in n for n in r.notices)


# ---------------------------------------------------------------------------
# improvements
# ---------------------------------------------------------------------------


def test_improvements_rotate_across_dimensions() -> None:
    """各面向輪流取,不是把某一個面向取完——否則建議會集中在同一面向。"""
    dims = [
        FaceDimensionDTO(letter="內", name="內容", score=70, verdict="x", points=["內1", "內2", "內3"]),
        FaceDimensionDTO(letter="構", name="結構", score=70, verdict="x", points=["構1"]),
        FaceDimensionDTO(letter="達", name="表達", score=70, verdict="x", points=["達1", "達2"]),
    ]
    assert derive_improvements(dims, limit=4) == ["內1", "構1", "達1", "內2"]


def test_improvements_dedup() -> None:
    dims = [
        FaceDimensionDTO(letter="內", name="內容", score=70, verdict="x", points=["同一句"]),
        FaceDimensionDTO(letter="構", name="結構", score=70, verdict="x", points=["同一句"]),
    ]
    assert derive_improvements(dims) == ["同一句"]


def test_improvements_empty_input() -> None:
    assert derive_improvements([]) == []


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "抱歉我無法處理", "{}", "null"])
def test_bad_output_returns_empty_not_exception(bad: str) -> None:
    assert parse_array(bad) == []


def test_strips_markdown_fence() -> None:
    assert parse_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_verbatim_failure_is_countable_by_marker() -> None:
    """失敗次數要能被下游可靠地數,不能靠比對整句措辭。

    措辭改一次,靠字串比對的統計就靜默歸零,而且沒有人會發現——
    它看起來只是「這次沒有失敗」。
    """
    assert VERBATIM_FAIL_MARKER
    assert "引用失敗" in VERBATIM_FAIL_MARKER


@pytest.mark.asyncio
async def test_downgrade_leaves_a_countable_notice() -> None:
    """實測踩到的坑:gpt-4o 引用時多加一個「那」字,機械檢查抓到並降級,
    診斷卻印出「28/28 全對」——失敗被自己的防護機制藏起來了。
    """
    llm = fake_llm(star=[
        {"key": "S", "present": True, "fromAnswer": "這句話逐字稿裡沒有", "hint": ""},
        {"key": "T", "present": False, "fromAnswer": "", "hint": "缺任務"},
        {"key": "A", "present": True, "fromAnswer": "我們辦過一場聯名活動", "hint": ""},
        {"key": "R", "present": False, "fromAnswer": "", "hint": "缺結果"},
    ])
    r = await run(llm)
    hits = [n for n in r.notices if VERBATIM_FAIL_MARKER in n]
    assert len(hits) == 1, "一次失敗要留下剛好一筆可數的紀錄"
    assert "S" in hits[0], "要看得出是哪一段失敗"


# ---------------------------------------------------------------------------
# ended_by 在報告端的兩個消費點
# ---------------------------------------------------------------------------


def test_truncated_turns_are_marked_in_transcript() -> None:
    """不標的話 starParts 會因為缺 R 段而寫「你沒有交代結果」,
    那是責備一個被打斷的人。"""
    from app.pipeline.interview_report import build_transcript_text

    text = build_transcript_text([
        TurnDTO(question="q1", answer="講到一半", ended_by="timeout"),
        TurnDTO(question="q2", answer="講完了", ended_by="user"),
    ])
    assert "[回答](被切斷) 講到一半" in text
    assert "[回答] 講完了" in text


@pytest.mark.asyncio
async def test_report_notes_truncated_turns() -> None:
    r = await run(turns=[
        TurnDTO(question="q1", answer="我們辦過一場聯名活動", ended_by="timeout", input_mode="voice"),
        TurnDTO(question="q2", answer="我沒先測試就直接全推", ended_by="user", input_mode="voice"),
    ])
    assert any("被語音辨識切斷" in n for n in r.notices)
    assert any("不代表使用者沒講" in n for n in r.notices)


@pytest.mark.asyncio
async def test_no_note_when_nothing_truncated() -> None:
    r = await run()
    assert not any("被語音辨識切斷" in n for n in r.notices)
