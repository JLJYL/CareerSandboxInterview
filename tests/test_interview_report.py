"""A4 面試報告管線測試。不打真 LLM。

重點在**機械保證**:固定值域、verbatim、輸入方式對評分的影響、
以及任一區塊失敗時其餘仍然可用。
"""

from __future__ import annotations

import json

import pytest

from app.contracts.interview_protocols import TextStats
from app.prompts.turn_emotion import EMOTION_MARK
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


def fake_llm(face=None, subs=None, questions=None, star=None, emotion=None):
    """依 system prompt 的開頭判斷是哪個區塊,回對應的假輸出。

    情緒那一塊的辨識字串由 `EMOTION_MARK` 宣告,不在這裡抄一段:
    抄的那一段跟 prompt 漂開的時候,漂掉的方向永遠是「悄悄沒接上」。
    """
    def call(system: str, user: str) -> str:
        if EMOTION_MARK in system:
            return json.dumps(emotion if emotion is not None else {}, ensure_ascii=False)
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
        engine=kw.pop("engine", "device"),
        mode=kw.pop("mode", "single"), turns=kw.pop("turns", TURNS),
        stats=kw.pop("stats", VOICE_STATS), input_mode=kw.pop("input_mode", "voice"),
        resume_text=kw.pop("resume_text", ""), resume_grounded=kw.pop("resume_grounded", False),
        llm=llm or fake_llm(), **kw,
    )


# ---------------------------------------------------------------------------
# 逐題情緒
# ---------------------------------------------------------------------------


async def test_emotion_block_is_wired_without_asking_the_model() -> None:
    """明確說不知道那一題要被標成卡住,而且不靠模型回答。

    這一條是接線證據。情緒那一塊的「沒有要標的」是合法結果,所以假 LLM
    沒接上的時候,結果跟「整場都平穩」長得一模一樣——測試全綠而一整塊沒接。
    程式判的那一項不依賴模型,所以它能證明這塊真的被呼叫到了。
    """
    r = await run(
        turns=[
            TurnDTO(question="q1", answer="我做過一個報表自動化的專案", input_mode="voice"),
            TurnDTO(question="q2", answer="這個我沒想過耶", input_mode="voice"),
        ],
        llm=fake_llm(emotion={}),
    )
    assert [(e.turn_index, e.label) for e in r.turn_emotions] == [(2, "卡住")]


async def test_the_emotion_read_never_sees_another_speaker() -> None:
    """情緒只判使用者自己的回答,不碰群面裡其他人的發言。

    ERC 的文獻(CoMPM, NAACL 2022)把「不要混淆說話者」當成核心問題之一,
    而我們這邊是結構上就分開的:`turns[].answer` 是使用者的回答,
    競爭者與面試官的發言在 `group_says`,情緒那一塊只吃前者。

    釘住它是因為現在是「剛好對」:以後有人為了給情緒更多脈絡而把
    group_says 餵進來,就會變成拿 agent 講的話去判使用者的情緒,
    而那種錯不會壞掉,只會悄悄給錯的標註。
    """
    seen: list = []

    def spy(system: str, user: str) -> str:
        if EMOTION_MARK in system:
            seen.append(user)
            return "{}"
        return fake_llm()(system, user)

    await run(
        mode="group",
        turns=[TurnDTO(question="q1", answer="我負責把資料接起來", input_mode="voice")],
        llm=spy,
    )
    assert seen
    assert "我負責把資料接起來" in seen[0]
    assert "我覺得應該先做市場調查" not in seen[0], "競爭者的發言不該進情緒判讀"


async def test_the_pause_measurement_reaches_the_emotion_read() -> None:
    """前端送了段落結束時點之後,「長停頓」從推論變成實測。

    這一條驗的是接線:`segment_ends_ms` 進來,停頓算出來,
    那一行出現在情緒判讀的 user 訊息裡。
    """
    seen: list = []

    def spy(system: str, user: str) -> str:
        if EMOTION_MARK in system:
            seen.append(user)
            return "{}"
        return fake_llm()(system, user)

    await run(
        turns=[
            TurnDTO(
                question="講一個失敗的經驗",
                answer="那個、那個我想一下 就是我當時 嗯 我不太確定要怎麼講",
                answer_segments=["那個、那個我想一下", "就是我當時 嗯", "我不太確定要怎麼講"],
                segment_starts_ms=[0, 5200, 12700],
                segment_ends_ms=[4000, 8200, 15000],
                input_mode="voice",
            )
        ],
        llm=spy,
    )
    assert seen, "情緒那一塊沒被呼叫到"
    assert "最長一次 4.5 秒" in seen[0]
    assert "實測值" in seen[0]


async def test_without_end_times_the_pause_line_is_absent() -> None:
    """前端還沒送那一欄的時候,整塊安靜退回今天的行為,不是印一個 0。"""
    seen: list = []

    def spy(system: str, user: str) -> str:
        if EMOTION_MARK in system:
            seen.append(user)
            return "{}"
        return fake_llm()(system, user)

    await run(
        turns=[
            TurnDTO(
                question="q1",
                answer="我做過一個報表自動化的專案",
                answer_segments=["我做過一個報表自動化的專案"],
                segment_starts_ms=[0],
                input_mode="voice",
            )
        ],
        llm=spy,
    )
    assert seen and "停頓" not in seen[0]


async def test_emotion_does_not_enter_any_score() -> None:
    """情緒不進分數也不進 improvements。

    情緒進分數等於「緊張的人分數比較低」,而產品定位是練習回饋不是能力評估。
    """
    calm = await run(llm=fake_llm(emotion={}))
    tense = await run(
        llm=fake_llm(
            emotion={"2": {"label": "焦慮", "quote": "我沒先測試就直接全推"}}
        )
    )
    assert [e.turn_index for e in tense.turn_emotions] == [2]
    assert calm.turn_emotions == []
    assert [(s.name, s.score) for s in calm.sub_scores] == [
        (s.name, s.score) for s in tense.sub_scores
    ]
    assert calm.improvements == tense.improvements
    assert [(d.letter, d.score) for d in calm.face_dimensions] == [
        (d.letter, d.score) for d in tense.face_dimensions
    ]


async def test_an_ungrounded_emotion_quote_is_not_reported() -> None:
    """抄不出那一題的原文就不標註。報告裡不會出現沒有證據的情緒。"""
    r = await run(llm=fake_llm(emotion={"1": {"label": "焦慮", "quote": "我整個人都慌了"}}))
    assert r.turn_emotions == []


async def test_self_blame_reaches_the_report_without_its_quote() -> None:
    """過度自責要標出哪一題,但不把那句話引回去。"""
    r = await run(
        turns=[TurnDTO(question="q1", answer="我覺得我真的很爛,每次都做不好", input_mode="typed")],
        llm=fake_llm(emotion={"1": {"label": "過度自責", "quote": "我覺得我真的很爛"}}),
    )
    assert [(e.label, e.quote) for e in r.turn_emotions] == [("過度自責", "")]


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
    """量化詞越多分數不可以往下掉,而且在真實落點上要分得開。

    原本這條要求 (0, 3, 6, 12, 30) 五個輸入五個不同分數。那個要求
    把刻度攤到 30 個量化詞才頂天,結果是真實回答落的 2–4 這一段
    只剩 6 分可用——黃金集十份有八份擠在 64–70。

    現在量化詞到 `QUANT_FULL` 就推到該帶頂端,超過不再加分。
    鑑別力要求改成落在真實分布上,那才是它該有的地方。
    """
    scores = [score_depth(TextStats(quantifier_count=n))[0] for n in (0, 3, 6, 12, 30)]
    assert scores == sorted(scores), "不可以往下掉"

    real = [score_depth(TextStats(quantifier_count=n))[0] for n in range(0, 7)]
    assert len(set(real)) >= 5, "真實回答落在 0–6 處,這一段要分得開"


def test_star_completeness_is_the_main_axis() -> None:
    """同樣的量化詞處數,STAR 講滿與沒講要差得出來。

    這是 D 的核心:量化詞單獨撐不起這個維度,因為多數回答都是 2–4 處。
    """
    at = lambda hits: score_depth(TextStats(quantifier_count=3), hits)[0]
    by_star = [at(h) for h in range(5)]

    assert by_star == sorted(by_star), "STAR 段數越多分數不可以往下掉"
    assert by_star[4] - by_star[0] >= 30, "四段全中與一段都沒有,要拉得開"


def test_depth_spreads_wider_than_the_old_single_signal() -> None:
    """黃金集的量化詞分布是 2 / 3 / 4 / 12,舊公式讓八份擠在 64–70。

    新公式在同一組量化詞上、配不同的 STAR 段數,全距要明顯變寬。
    """
    golden_quantifiers = (2, 2, 2, 2, 3, 3, 3, 4, 12, 12)

    old_like = [score_depth(TextStats(quantifier_count=q), 2)[0] for q in golden_quantifiers]
    spread_one_star_level = max(old_like) - min(old_like)

    everything = [
        score_depth(TextStats(quantifier_count=q), h)[0]
        for q in golden_quantifiers for h in range(5)
    ]
    assert max(everything) - min(everything) > spread_one_star_level + 20


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


def test_fluency_is_unscored_when_unmeasurable() -> None:
    """不可測時回未評分,不是回一個中間值。

    原本回 72,理由是「不可以給高分(打字的人會佔便宜),也不可以給低分
    (那是懲罰未知)」。推理對,結論錯:兩邊都不可以,代表這一項不該有分數。
    72 一旦進了平均,就跟真的量到 72 完全一樣。
    """
    score, basis = score_fluency(TYPED_STATS, False)
    assert score is None
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
    # 交給 LLM 的那幾項標成未評分,不補 0——
    # 補 0 會讓「這一項的呼叫失敗了」跟「這一項表現最差」在報告上長得一樣。
    assert by_name["邏輯清晰度"] is None, "失敗的區塊標為未評分並記入 notices"
    # 而且總分只由真的有分數的那幾項算出來,不會被失敗的區塊拉低
    assert r.score_basis == 2, "只有兩個公式項活著"
    assert r.overall_score == pytest.approx(
        (by_name["表達流暢度"] + by_name["內容深度"]) / 2
    )
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


# ---------------------------------------------------------------------------
# 轉錄引擎:兩種失真方式相反,prompt 要跟著變
# ---------------------------------------------------------------------------


def test_caveat_differs_by_engine() -> None:
    """裝置端會漏字、保留填充詞;雲端不漏字、填充詞被清掉。

    用同一段說明會讓模型做錯事:告訴它「填充詞保留」而逐字稿一個都沒有,
    它會推論使用者講話很流暢——那是把「模型清掉了」誤讀成「使用者沒講」。
    """
    from app.prompts.interview_report import stt_caveat

    device, api = stt_caveat("device"), stt_caveat("api")
    assert "保留" in device
    assert "已經被移除" in api
    assert "不要評論填充詞的多寡" in api
    assert "標點是模型加的" in api, "雲端的標點是模型判斷的,不是使用者的節奏"


def test_api_engine_forbids_fluency_comment() -> None:
    """填充詞被清掉時,流暢度在這份資料上沒有證據。"""
    from app.prompts.interview_report import compose_sub_score_prompt

    assert "沒有證據" in compose_sub_score_prompt(engine="api")


@pytest.mark.asyncio
async def test_engine_reaches_the_prompt() -> None:
    """欄位加了就要有人讀。"""
    seen = {}

    def spy(system: str, user: str) -> str:
        if "細分分數" in system:
            seen["system"] = system
        return fake_llm()(system, user)

    await run(spy, engine="api")
    assert "已經被移除" in seen["system"]


def test_whisper_style_stats_degrade_honestly() -> None:
    """Whisper 的輸出:有標點、沒有填充詞、reliability=unknown。

    流暢度要回未評分,不是給高分——打字與雲端轉錄都沒有填充詞,
    給高分等於獎勵一件跟使用者無關的事。
    """
    from app.pipeline.interview_report import format_stats, score_fluency

    whisper = TextStats(
        char_count=200, filler_count=0, filler_reliability="unknown",
        segmentation="punctuation", avg_sentence_len=26.0, quantifier_count=3,
    )
    text, measurable = format_stats(whisper, "voice")
    assert measurable is False
    score, basis = score_fluency(whisper, measurable)
    assert score is None, "不給高分也不給低分,就是不給分"
    assert "不可測" in basis
    assert "有標點,精確值" in text, "雲端有標點,句長是精確的"


# ---------------------------------------------------------------------------
# 可評性閘門的接線證據(A / B / C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_session_with_nothing_in_it_scores_nothing() -> None:
    """C:沒講話的那一場,六項要全部未評分,不是拿地板分。

    這一條同時是**接線證據**。閘門單元測試全綠不代表 generate_report
    真的呼叫了它——`tests/test_scorability.py` 驗的是模組,這一條驗的是
    那個模組有被接進管線。拆掉 generate_report 裡的 assess() 這條會紅。
    """
    r = await run(
        turns=[
            TurnDTO(question="請自我介紹", answer="嗯。", input_mode="voice"),
            TurnDTO(question="講一個經驗", answer="喔。", input_mode="voice"),
        ],
        stats=TextStats(char_count=4, filler_reliability="measured"),
    )
    assert all(s.score is None for s in r.sub_scores), "一項都不該有分數"
    assert r.overall_score is None
    assert r.score_basis == 0
    assert any("無法給總分" in n for n in r.notices)


@pytest.mark.asyncio
async def test_saying_you_dont_know_scores_the_lowest_anchor_not_a_blank() -> None:
    """B:全場答不出來要拿最低錨點。

    不評分的話使用者會收到一份六項全空的報告,他什麼都得不到。
    他有回應——「我沒做過這件事」就是關於這幾項能力的觀察證據。
    """
    from app.pipeline.scorability import LOWEST_ANCHOR

    r = await run(
        turns=[
            TurnDTO(question="請自我介紹", answer="不知道。", input_mode="voice"),
            TurnDTO(question="講一個經驗", answer="這個我沒想過。", input_mode="voice"),
        ],
        stats=TextStats(char_count=12, filler_reliability="measured"),
    )
    by_name = {s.name: s.score for s in r.sub_scores}
    assert by_name["表達流暢度"] == LOWEST_ANCHOR
    assert by_name["內容深度"] == LOWEST_ANCHOR
    assert r.overall_score is not None, "答不出來是可評的,要有總分"
    assert any("答不出來" in s.basis for s in r.sub_scores)


@pytest.mark.asyncio
async def test_face_dimensions_do_not_invent_a_zero_for_an_unscored_sub() -> None:
    """三維面向的分數是從 subScores 抄的。來源未評分時這裡也要未評分。

    抄一個 0 過來,等於把 subScores 剛修好的問題在隔壁欄位重演一次,
    而且那一欄沒有人在看。
    """
    r = await run(
        turns=[TurnDTO(question="q", answer="嗯。", input_mode="voice")],
        stats=TextStats(char_count=2, filler_reliability="measured"),
    )
    assert r.face_dimensions, "面向本身還在,只是沒有分數"
    assert all(d.score is None for d in r.face_dimensions)


@pytest.mark.asyncio
async def test_a_normal_session_still_gets_a_total_from_all_six() -> None:
    """閘門不可以誤傷正常的一場。"""
    r = await run()
    assert r.score_basis == len(SUB_SCORE_NAMES)
    assert r.overall_score is not None
    scored = [s.score for s in r.sub_scores if s.score is not None]
    assert r.overall_score == pytest.approx(sum(scored) / len(scored))


@pytest.mark.asyncio
async def test_depth_uses_the_verified_star_not_what_the_model_claimed() -> None:
    """編一句不存在的引用,不可以把內容深度推上去。

    STAR 段數現在是內容深度的主軸,所以它變成一條攻擊路徑:模型只要宣稱
    四段全中、附一句編的原文,分數就從 44 跳到 85。`verify_star_verbatim`
    會把對不上的段落降成未命中,內容深度吃的必須是降級後的結果。

    這跟 `ungrounded_numbers` 是同一個理由——判斷型規則有殘留率,
    所以在它後面補一道機械檢查。
    """
    claimed_all_four = [
        {"key": k, "present": True,
         "fromAnswer": "這句話逐字稿裡沒有出現過", "hint": ""}
        for k in ("S", "T", "A", "R")
    ]
    r = await run(fake_llm(star=claimed_all_four))

    depth = next(s.score for s in r.sub_scores if s.name == "內容深度")
    honest_zero = score_depth(VOICE_STATS, 0)[0]
    assert depth == honest_zero, "四段全被驗掉,分數要等同一段都沒中"
    assert sum(1 for n in r.notices if VERBATIM_FAIL_MARKER in n) == 4


@pytest.mark.asyncio
async def test_depth_rises_when_the_star_quotes_are_real() -> None:
    """對照組:引用真的在逐字稿裡的時候,分數要跟著上去。

    沒有這一條,上面那一條用「內容深度恆為最低」也會過。
    """
    real = [
        {"key": "S", "present": True, "fromAnswer": "我們辦過一場聯名活動", "hint": ""},
        {"key": "T", "present": True, "fromAnswer": "我是資管系大三的學生", "hint": ""},
        {"key": "A", "present": True, "fromAnswer": "我沒先測試就直接全推", "hint": ""},
        {"key": "R", "present": True, "fromAnswer": "過去在系學會做行銷", "hint": ""},
    ]
    r = await run(fake_llm(star=real))

    depth = next(s.score for s in r.sub_scores if s.name == "內容深度")
    assert depth == score_depth(VOICE_STATS, 4)[0]
    assert depth > score_depth(VOICE_STATS, 0)[0] + 30


# ---------------------------------------------------------------------------
# better:不可以補回答裡沒有的技術名詞(前端 10/29 錄音)
# ---------------------------------------------------------------------------

N1_Q2 = (
    "我們在比賽當中,就是我們主要是在前期的時候會用一些分析下去進行股票的這個數據的分析, "
    "那我們會用到一些 Python 啊,然後去用 Model 去跑它的數據分析, "
    "然後跑一些圖表跟像是巨量資料之類的分析這樣子。"
)
N1_Q2_BETTER = (
    "在比賽中，我們前期使用 Python 進行股票數據分析，運用回歸模型來預測風險，"
    "並生成圖表來視覺化數據，這樣能更清楚地了解市場趨勢。"
)


@pytest.mark.asyncio
async def test_better_inventing_a_method_is_cleared() -> None:
    """normal-1 第 2 題:原回答只說「用 Model 去跑」,better 寫成「回歸模型」。

    使用者把那句背下來,被追問「為什麼用回歸」就答不出來。
    """
    from app.pipeline.interview_report import ungrounded_terms

    assert ungrounded_terms(N1_Q2_BETTER, N1_Q2) == ["回歸模型"]

    llm = fake_llm(questions=[{"id": 1, "comment": "有提到工具", "better": N1_Q2_BETTER}])
    r = await run(llm, turns=[TurnDTO(question="怎麼做風險控管", answer=N1_Q2, input_mode="voice")])
    assert r.question_feedbacks[0].better == ""
    assert any("原回答沒有的名詞" in n and "回歸模型" in n for n in r.notices)


def test_terms_the_answer_already_used_are_allowed() -> None:
    """Python 原回答就有;「模型」不是具體方法,換句話說不算捏造。"""
    from app.pipeline.interview_report import ungrounded_terms

    assert ungrounded_terms("我們用 Python 建立模型分析數據", N1_Q2) == []


def test_an_english_tool_name_not_in_the_answer_is_caught() -> None:
    from app.pipeline.interview_report import ungrounded_terms

    assert ungrounded_terms("我用 Tableau 做視覺化報表", "我用試算表做圖") == ["Tableau"]
    assert ungrounded_terms("我用 python 處理資料", "我用 Python 處理資料") == [], "大小寫不算不同"


def test_the_same_method_in_another_spelling_is_grounded() -> None:
    """迴歸/回歸、繁簡都是同一個詞。"""
    from app.pipeline.interview_report import ungrounded_terms

    assert ungrounded_terms("我用回歸分析找出因子", "我用迴歸分析找出因子") == []
    assert ungrounded_terms("我用回歸分析找出因子", "我用回归分析找出因子") == []


def test_returning_to_normal_is_not_a_regression() -> None:
    """「回歸」只在接模型/分析或前面有線性/邏輯/多元時算方法。"""
    from app.pipeline.interview_report import ungrounded_terms

    assert ungrounded_terms("讓出貨流程回歸正常", "出貨流程亂掉了") == []


def test_one_as_an_article_is_not_a_number() -> None:
    """normal-1 第 1 題的 better 因為「參加過一個比賽」被清空,那句沒有捏造數量。"""
    answer = "考官你好,我之前有參加過理財相關的這個商業比賽"
    assert ungrounded_numbers("我參加過一個理財相關的商業比賽", answer) == []
    assert ungrounded_numbers("我只試了一次就上線", "我直接上線") == []
    assert ungrounded_numbers("我花了一個月準備", answer) == ["一個"], "接時長的「一個」是數量"
    assert ungrounded_numbers("我們做了三個版本", answer) == ["三個"], "其他數字照查"


# ---------------------------------------------------------------------------
# 三維面向的評語要跟分數一致(前端 10/29:結構 80 分,評語卻說缺乏邏輯)
# ---------------------------------------------------------------------------


def _spy(base):
    seen: list = []

    def call(system: str, user: str) -> str:
        seen.append((system, user))
        return base(system, user)

    return call, seen


@pytest.mark.asyncio
async def test_face_verdicts_are_written_with_the_scores_in_hand() -> None:
    """評語原本跟分數並行產生,寫的時候不知道分數。"""
    call, seen = _spy(fake_llm())
    r = await run(call)
    face_users = [u for s, u in seen if "三個面向" in s]
    assert len(face_users) == 1
    assert "結構  82(正常水準)" in face_users[0], "結構的分數抄自邏輯清晰度"
    depth = next(x.score for x in r.sub_scores if x.name == "內容深度")
    assert f"內容  {depth}(" in face_users[0], "內容的分數是系統算的內容深度"


@pytest.mark.asyncio
async def test_an_unscored_dimension_is_shown_as_unscored() -> None:
    """打字作答量不到表達流暢度。評語不該被告知一個不存在的分數。"""
    call, seen = _spy(fake_llm())
    await run(call, stats=TYPED_STATS, input_mode="typed")
    face_user = next(u for s, u in seen if "三個面向" in s)
    assert "表達  未評分" in face_user


def test_score_bands_match_the_sub_score_meanings() -> None:
    from app.pipeline.interview_report import score_band

    got = [score_band(x) for x in (95, 85, 84, 70, 69, 55, 54, None)]
    assert got == [
        "強項", "強項", "正常水準", "正常水準",
        "有明顯缺口", "有明顯缺口", "幾乎沒有表現", "未評分",
    ]


def test_the_face_prompt_ties_verdicts_to_the_bands() -> None:
    from app.prompts.interview_report import compose_face_prompt

    prompt = compose_face_prompt("api")
    assert "verdict 的方向跟系統給的分數一致" in prompt
    assert "70–84" in prompt and "未評分" in prompt


@pytest.mark.asyncio
async def test_a_face_failure_still_leaves_the_rest_of_the_report() -> None:
    """三維面向改成第二批之後,失敗隔離照舊。"""
    base = fake_llm()

    def call(system: str, user: str) -> str:
        if "三個面向" in system:
            raise RuntimeError("timeout")
        return base(system, user)

    r = await run(call)
    assert any(s.score is not None for s in r.sub_scores), "細分分數不受影響"
    assert any("face" in n and "失敗" in n for n in r.notices)


@pytest.mark.asyncio
async def test_the_emotion_block_knows_the_transcription_engine() -> None:
    """報告的 engine 要傳到情緒那一塊,雲端辨識的語音有自己的判讀說明。"""
    call, seen = _spy(fake_llm())
    await run(call, engine="api")
    emotion_systems = [s for s, u in seen if EMOTION_MARK in s]
    assert emotion_systems and "比他實際講的順" in emotion_systems[0]


@pytest.mark.asyncio
async def test_a_session_that_switched_to_typing_labels_each_turn() -> None:
    """前端 10/21 起一對一與 panel 都能改用打字。混用時情緒那一塊逐題標作答方式。"""
    call, seen = _spy(fake_llm())
    await run(
        call,
        turns=[
            TurnDTO(question="請自我介紹", answer="我是資管系大三的學生", input_mode="voice"),
            TurnDTO(question="講一個經驗", answer="我們辦過一場聯名活動", input_mode="typed"),
        ],
    )
    emotion = [(s, u) for s, u in seen if EMOTION_MARK in s]
    assert emotion, "情緒那一塊沒被呼叫到"
    system, user = emotion[0]
    assert "有的題是講的、有的題是打的" in system
    assert "作答方式:講的" in user and "作答方式:打的" in user
