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
    CollabScorer,
    FakeCollabScorer,
    FakeGapComputer,
    FakeTranscriptAnalyzer,
    GapComputer,
    TranscriptAnalyzer,
)
from app.schemas.interview import (
    COLLAB_DIM_NAMES,
    FACE_DIMENSIONS,
    STAR_PARTS,
    SUB_SCORE_NAMES,
    CollabDimDTO,
    FaceDimensionDTO,
    ReportResponse,
    StartInterviewResponse,
    StarPartDTO,
    SubScoreDTO,
    TurnRequest,
    TurnResponse,
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
    assert isinstance(FakeCollabScorer(), CollabScorer)


def test_gap_computer_returns_empty_without_resume() -> None:
    """履歷為空是正常情況,必須回空清單而不是拋例外。"""
    assert FakeGapComputer().compute([], "任何 JD", "任何逐字稿") == []


def test_collab_scorer_returns_four_in_order() -> None:
    rows = FakeCollabScorer().score(["我覺得先做市場調查", "同意前面那位"])
    assert [r.name for r in rows] == list(COLLAB_DIM_NAMES)


def test_gap_candidate_carries_evidence() -> None:
    """沒有 evidence,寫 why 時就只能靠 LLM 自由發揮,會捏造經歷。"""
    for c in FakeGapComputer().compute([{"id": "exp_001"}], "JD", "逐字稿"):
        assert c.resume_evidence
        assert c.jd_evidence
        assert c.resume_experience_id


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
