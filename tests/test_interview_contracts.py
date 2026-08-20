"""面試合約測試。

這些測試的作用是:任何人改動 schema 而沒有同步改前端,這裡會先紅燈。
不打真 LLM,不連網,全部是形狀與值域的檢查。

執行:
  pytest tests/test_interview_contracts.py -v
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from app.contracts.interview_protocols import (
    WHY_MUST_BE_JD_SIDE,
    CollabObserver,
    FakeCollabObserver,
    FakeGapComputer,
    FakeTranscriptAnalyzer,
    GapComputer,
    JDInput,
    TranscriptAnalyzer,
    Utterance,
)
from app.schemas.interview import (
    COLLAB_DIM_NAMES,
    FACE_DIMENSIONS,
    STAR_PARTS,
    SUB_SCORE_NAMES,
    CollabDimDTO,
    FaceDimensionDTO,
    ReportRequest,
    ReportResponse,
    StartInterviewResponse,
    StarPartDTO,
    SubScoreDTO,
    TurnDTO,
    TurnRequest,
    TurnResponse,
    UtteranceDTO,
)
from app.schemas.interview_repair import (
    repair_report,
    repair_star_parts,
    repair_sub_scores,
    verify_star_verbatim,
)

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "interview" / "golden"


def _load(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 一、golden JSON 必須能被 schema 接受
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,model",
    [
        ("res_start_single", StartInterviewResponse),
        ("res_start_panel", StartInterviewResponse),
        ("res_start_group", StartInterviewResponse),
        ("res_turn_single", TurnResponse),
        ("res_turn_panel", TurnResponse),
        ("res_turn_group", TurnResponse),
        ("res_report_single", ReportResponse),
        ("res_report_group", ReportResponse),
        ("res_report_no_resume", ReportResponse),
    ],
)
def test_golden_validates(name: str, model) -> None:
    model.model_validate(_load(name))


# ---------------------------------------------------------------------------
# 二、固定值域:名稱與順序零偏移
# ---------------------------------------------------------------------------


def test_sub_score_names_and_order() -> None:
    report = ReportResponse.model_validate(_load("res_report_single"))
    assert [s.name for s in report.sub_scores] == list(SUB_SCORE_NAMES)


def test_star_keys_and_order() -> None:
    report = ReportResponse.model_validate(_load("res_report_single"))
    assert [p.key for p in report.star_parts] == [k for k, _ in STAR_PARTS]
    assert [p.name for p in report.star_parts] == [n for _, n in STAR_PARTS]


def test_face_dimension_letters() -> None:
    report = ReportResponse.model_validate(_load("res_report_single"))
    assert [d.letter for d in report.face_dimensions] == [ltr for ltr, _ in FACE_DIMENSIONS]


def test_collab_names_only_in_group() -> None:
    grp = ReportResponse.model_validate(_load("res_report_group"))
    assert [c.name for c in grp.collab_dims] == list(COLLAB_DIM_NAMES)

    single = ReportResponse.model_validate(_load("res_report_single"))
    assert single.collab_dims == []


# ---------------------------------------------------------------------------
# 三、機械修復:壞資料不得造成例外
# ---------------------------------------------------------------------------


def test_repair_fills_missing_sub_scores() -> None:
    fixed, notices = repair_sub_scores([SubScoreDTO(name="內容深度", score=80)])
    assert [s.name for s in fixed] == list(SUB_SCORE_NAMES)
    assert fixed[0].score == 80
    assert notices  # 有補值就要有紀錄


def test_repair_renames_by_position() -> None:
    """LLM 寫成近義變體時,依位置歸位而不是丟掉。"""
    bad = [SubScoreDTO(name=n, score=50) for n in SUB_SCORE_NAMES]
    bad[1] = SubScoreDTO(name="邏輯清楚度", score=91)  # 少一個字
    fixed, notices = repair_sub_scores(bad)
    assert fixed[1].name == "邏輯清晰度"
    assert fixed[1].score == 91
    assert any("歸位" in n for n in notices)


def test_repair_fills_missing_star_part() -> None:
    fixed, notices = repair_star_parts(
        [StarPartDTO(key="S", name="情境", present=True, from_answer="我們辦過活動")]
    )
    assert [p.key for p in fixed] == ["S", "T", "A", "R"]
    assert fixed[3].present is False
    assert notices


def test_repair_clears_collab_for_non_group() -> None:
    report = ReportResponse(
        mode="single",
        collab_dims=[CollabDimDTO(name=COLLAB_DIM_NAMES[0], score=70)],
        resume_grounded=True,
    )
    fixed = repair_report(report)
    assert fixed.collab_dims == []
    assert any("非團體面試" in n for n in fixed.notices)


def test_repair_clears_missing_points_without_resume() -> None:
    """沒有履歷就不可能有漏講點。這是防捏造的最後一道。"""
    report = ReportResponse.model_validate(_load("res_report_single"))
    report.resume_grounded = False
    fixed = repair_report(report)
    assert fixed.missing_points == []
    assert any("尚未建立經歷" in n for n in fixed.notices)


def test_repair_clears_video_dims() -> None:
    report = ReportResponse(mode="single", video_dims=[{"name": "眼神接觸", "score": 70}])
    fixed = repair_report(report)
    assert fixed.video_dims == []


# ---------------------------------------------------------------------------
# 四、verbatim 檢查:starParts.fromAnswer 必須來自逐字稿
# ---------------------------------------------------------------------------


def test_star_verbatim_passes_on_golden() -> None:
    req = _load("req_report_single")
    transcript = "\n".join(t["answer"] for t in req["turns"])
    report = ReportResponse.model_validate(_load("res_report_single"))
    assert verify_star_verbatim(report, transcript) == []


def test_star_verbatim_catches_fabrication() -> None:
    report = ReportResponse(
        mode="single",
        star_parts=[
            StarPartDTO(key="S", name="情境 Situation", present=True, from_answer="我在台積電帶過十人團隊")
        ],
    )
    problems = verify_star_verbatim(report, "我是資管系大三的學生")
    assert len(problems) == 1


# ---------------------------------------------------------------------------
# 五、分數夾範圍:LLM 回超界不得噴 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [(105, 100), (-3, 0), (77, 77)])
def test_score_clamped(raw: int, expected: int) -> None:
    assert SubScoreDTO(name="內容深度", score=raw).score == expected
    dim = FaceDimensionDTO(letter="內", name="內容", score=raw, verdict="x", points=[])
    assert dim.score == expected


# ---------------------------------------------------------------------------
# 六、speaker 必須是開放字串(CareerCategory 事故的預防)
# ---------------------------------------------------------------------------


def test_speaker_accepts_unknown_persona_name() -> None:
    """persona 調校改了顯示名稱時,合約不可以噴錯。"""
    r = TurnResponse(speaker="人資主管", next_question="再說一次你的優勢")
    assert r.speaker == "人資主管"


def test_mode_rejects_unknown_value() -> None:
    """mode 由前端路由決定,值域封閉,這裡要嚴格。"""
    with pytest.raises(ValidationError):
        ReportResponse.model_validate({"mode": "video"})
    with pytest.raises(ValidationError):
        StartInterviewResponse.model_validate(
            {"sessionId": "x", "mode": "quick", "openingQuestion": "x"}
        )


# ---------------------------------------------------------------------------
# 七、Protocol 相容性:Fake 必須符合 Protocol,真實作換上來才不會爆
# ---------------------------------------------------------------------------


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeTranscriptAnalyzer(), TranscriptAnalyzer)
    assert isinstance(FakeGapComputer(), GapComputer)
    assert isinstance(FakeCollabObserver(), CollabObserver)


def test_gap_computer_returns_empty_without_resume() -> None:
    """履歷為空是正常情況,必須回空清單而不是拋例外。"""
    assert FakeGapComputer().compute([], JDInput(description="任何 JD"), "任何逐字稿") == []


def test_collab_observer_returns_four_in_order() -> None:
    rows = FakeCollabObserver().observe(
        [
            Utterance("user", "我覺得先做市場調查", 0, 3000),
            Utterance("AI-邏輯", "母數是多少", 3000, 5000),
            Utterance("user", "同意前面那位", 5500, 8000),
        ]
    )
    assert [r.name for r in rows] == list(COLLAB_DIM_NAMES)


def test_collab_observer_never_assigns_level() -> None:
    """A 只交可觀察值,等第由 B 的 LLM 對照 BARS 指派。

    level 是 None 不是 0——0 分與未評分在報告上是兩件完全不同的事。
    """
    rows = FakeCollabObserver().observe([Utterance("user", "x")])
    assert all(r.level is None for r in rows)


def test_collab_signals_degrade_without_speaker_info() -> None:
    """輸入只有使用者發言時,兩個維度算不出來,要空 signals 加 evidence 說明。

    前端 groupSays 尚未改成記錄所有發言者之前,這是預期狀態。
    """
    rows = {r.name: r for r in FakeCollabObserver().observe([Utterance("user", "x")])}
    assert rows["傾聽與回應"].signals == {}
    assert "無法比對" in rows["傾聽與回應"].evidence


def test_collab_prohibits_volume_indicators() -> None:
    """發言次數不可當等第依據——那是在自動化 babble 偏誤。"""
    from app.contracts.interview_protocols import COLLAB_PROHIBITED_INDICATORS

    for kw in ("發言次數", "babble", "打斷次數"):
        assert kw in COLLAB_PROHIBITED_INDICATORS


def test_group_report_without_rubric_shows_nothing() -> None:
    """rubric 未到位時不補 0 分。補四個 0 會讓使用者以為協作全部拿 0。"""
    r = repair_report(ReportResponse(mode="group", resume_grounded=True))
    assert r.collab_dims == []
    assert any("協作評分尚未啟用" in n for n in r.notices)


def test_gap_candidate_carries_evidence() -> None:
    """沒有 evidence,寫 why 時就只能靠 LLM 自由發揮,會捏造經歷。"""
    for c in FakeGapComputer().compute([{"id": "exp_001"}], JDInput(), "逐字稿"):
        assert c.resume_evidence
        assert c.jd_evidence
        assert c.resume_experience_ids


# ---------------------------------------------------------------------------
# 九、D1 異議定案後新增的檢查
# ---------------------------------------------------------------------------


def test_jd_input_keeps_required_skills_order() -> None:
    """104 的 requiredSkills 排前面的較關鍵,A 的位置衰減依賴這個順序。"""
    skills = ["SQL", "Python", "報表自動化"]
    assert JDInput(required_skills=skills).required_skills == skills


def test_jd_input_records_source() -> None:
    """校準分佈(catalog)與正式分佈(extracted)要能分開統計。"""
    assert JDInput().source == "extracted"
    assert JDInput(source="catalog").source == "catalog"


def test_mentioned_skills_returns_ids_not_display() -> None:
    """join 鍵必須是帶命名空間的 skill_id,不是顯示字串。"""
    for sid in FakeTranscriptAnalyzer().mentioned_skills("我用 SQL 跟 Python"):
        assert sid.startswith(("sk:", "skm:")), f"{sid} 不是 skill_id"


def test_gap_candidate_skill_id_joins_with_mentioned() -> None:
    """兩邊的鍵必須同一個命名空間,否則差集永遠算不對。"""
    for c in FakeGapComputer().compute([{"id": "e1"}], JDInput(), "x"):
        assert c.skill_id.startswith(("sk:", "skm:"))
        assert c.display


def test_gap_candidate_kind_is_hard_or_soft() -> None:
    """硬技能會被指名,軟技能只會被展演,兩者的漏講可信度不同。"""
    kinds = {c.kind for c in FakeGapComputer().compute([{"id": "e1"}], JDInput(), "x")}
    assert kinds <= {"hard", "soft"}
    assert "soft" in kinds, "Fake 應涵蓋軟技能,否則 B 測不到過濾路徑"


def test_text_stats_declares_segmentation() -> None:
    """估算值要標示出來,否則 prompt 會把它當精確測量講。"""
    a = FakeTranscriptAnalyzer()
    assert a.text_stats("我是資管系。過去做行銷。").segmentation == "punctuation"
    assert a.text_stats("我是資管系 過去做行銷").segmentation in {
        "discourse_marker",
        "unavailable",
    }


def test_utterance_carries_speaker_and_order() -> None:
    """協作四項有三項需要發言者身分與時序。"""
    u = Utterance("AI-邏輯", "母數是多少", 3000, 5000)
    assert u.speaker_id == "AI-邏輯"
    assert u.end_ms > u.start_ms


def test_group_says_carries_all_speakers() -> None:
    """群面請求必須包含 AI 同儕的發言,否則傾聽與回應算不出來。"""
    req = ReportRequest.model_validate(_load("req_report_group"))
    speakers = {u.speaker for u in req.group_says}
    assert "user" in speakers
    assert len(speakers) > 1, "只有使用者的發言,前一位發言者的內容不在場"


def test_utterance_dto_defaults_are_safe() -> None:
    """沒有計時資料時給 0,陣列順序仍可用。"""
    u = UtteranceDTO(text="x")
    assert u.speaker == "user"
    assert u.start_ms == 0


def test_why_rule_forbids_resume_narration() -> None:
    """why 是 JD 側論證,不可敘述使用者在哪一段經歷做了什麼。"""
    assert "jd_evidence" in WHY_MUST_BE_JD_SIDE
    assert "禁止" in WHY_MUST_BE_JD_SIDE


# ---------------------------------------------------------------------------
# 八、已問領域的往返:禁則要可執行,不能只寫在 prompt 裡
# ---------------------------------------------------------------------------


def test_turn_request_carries_asked_topics() -> None:
    """「追問不得重複已問領域」沒有清單就無法執行。"""
    req = TurnRequest.model_validate(_load("req_turn"))
    assert req.asked_topics == ["自我介紹"]


def test_turn_response_labels_topic() -> None:
    """後端標記領域,前端累積後回傳,前端不必自己判斷題目屬於哪一類。"""
    for name in ("res_turn_single", "res_turn_panel", "res_turn_group"):
        assert TurnResponse.model_validate(_load(name)).topic


def test_start_response_seeds_first_topic() -> None:
    """開場題也要標領域,否則第一輪的 asked_topics 會是空的。"""
    for name in ("res_start_single", "res_start_panel", "res_start_group"):
        assert StartInterviewResponse.model_validate(_load(name)).opening_topic


def test_asked_topics_defaults_to_empty() -> None:
    """欄位可省略。session 落地後由後端自行查詢,前端不必再帶。"""
    assert TurnRequest(answer="x").asked_topics == []


def test_text_stats_declares_filler_reliability() -> None:
    """引擎若做 disfluency removal,filler_count 反映的是後處理不是使用者。

    不標示的話,流暢度會錨在一個恆為 0 的數字上,每個人拿到一樣的分數,
    而且因為有硬數字撐著而看起來很客觀。
    """
    a = FakeTranscriptAnalyzer()
    assert a.text_stats("嗯 那個 我是資管系").filler_reliability == "measured"
    assert a.text_stats("我是資管系的學生").filler_reliability == "unknown"


# ---------------------------------------------------------------------------
# 十、輸入方式:語音與打字的測量有效性不同
# ---------------------------------------------------------------------------


def test_turn_request_declares_input_mode() -> None:
    """後端只收到文字,分不出是講的還是打的,必須由前端宣告。"""
    assert TurnRequest(answer="x").input_mode == "unknown"
    assert TurnRequest(answer="x", input_mode="voice").input_mode == "voice"


def test_turn_dto_carries_input_mode() -> None:
    """報告端要逐輪知道,因為一場面試可以混用兩種輸入。"""
    assert TurnDTO(question="q", answer="a", input_mode="typed").input_mode == "typed"


def test_utterance_carries_input_mode() -> None:
    """群面同時有語音與文字輸入,搶話偵測需要打字停頓。"""
    assert UtteranceDTO(text="x", input_mode="typed").input_mode == "typed"


def test_input_mode_defaults_to_unknown_not_voice() -> None:
    """預設不可以是 voice。前端沒宣告時當作打字保守處理,
    寧可少講一個維度,也不要用量錯東西的分數指導使用者。"""
    assert TurnDTO().input_mode == "unknown"
    assert UtteranceDTO().input_mode == "unknown"


def test_input_mode_rule_covers_the_four_affected_measures() -> None:
    from app.schemas.interview import INPUT_MODE_RULE

    for measure in ("filler_count", "segmentation", "表達流暢度", "prosody"):
        assert measure in INPUT_MODE_RULE


# ---------------------------------------------------------------------------
# 十一、W1 交接後的合約增補
# ---------------------------------------------------------------------------


def test_mentioned_skills_accepts_candidate_set() -> None:
    """W2 履歷條件式模糊匹配需要把搜尋範圍限縮到履歷技能集。"""
    a = FakeTranscriptAnalyzer()
    assert a.mentioned_skills("我用 SQL 跟 Python") == {"sk:sql", "sk:python"}
    assert a.mentioned_skills("我用 SQL 跟 Python", {"sk:sql"}) == {"sk:sql"}


def test_mentioned_skills_candidates_defaults_to_none() -> None:
    """W1 行為不變,向後相容。"""
    assert FakeTranscriptAnalyzer().mentioned_skills("我用 Excel") == {"sk:excel"}


def test_turn_dto_carries_answer_segments() -> None:
    """段界是實測的語流邊界,靠 \\n 慣例會靜默遺失。"""
    t = TurnDTO(question="q", answer="甲\n乙", answer_segments=["甲", "乙"])
    assert t.answer_segments == ["甲", "乙"]
    assert TurnDTO(question="q", answer="甲").answer_segments == []


def test_demonstration_threshold_separates_behaviour_from_claim() -> None:
    """展演門檻兩邊共用,定義不一致的話評測結果無法解釋。"""
    from app.contracts.interview_protocols import DEMONSTRATION_THRESHOLD

    for kw in ("具體行為", "自我宣稱", "只能推論"):
        assert kw in DEMONSTRATION_THRESHOLD
