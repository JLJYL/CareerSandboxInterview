"""端點層測試。不打真 LLM、不載入 embedding。

【測什麼】
    三個端點的形狀與狀態碼
    lifespan 有沒有把共用元件放進去
    降級路徑:LLM 失敗時回 200 加 notices,不是 500

【為什麼不是 500】
面試進行中噴 500 等於整場中斷。pipeline 全部設計成降級,
這一層要驗證那個設計真的傳到 HTTP 層。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.contracts.interview_protocols import FakeCollabObserver, GapCandidate, TextStats
from app.schemas.interview import SUB_SCORE_NAMES


class StubAnalyzer:
    def text_stats(self, transcript: str) -> TextStats:
        return TextStats(
            char_count=len(transcript), filler_count=12,
            filler_detail={"那個": 5, "就是": 7}, quantifier_count=3,
            filler_reliability="measured", segmentation="stt_segment",
            avg_sentence_len=78.0,
        )

    def mentioned_skills(self, transcript, candidates=None):
        return set()


class StubGap:
    def compute(self, resume, jd, transcript):
        if not resume:
            return []
        return [GapCandidate(
            skill_id="sk:sql", display="SQL(sk:sql)", kind="hard",
            resume_experience_ids=["e1"], resume_evidence="SQL",
            jd_evidence="需具備 SQL 撰寫能力", weight=0.9,
        )]


OPENING = {"openingQuestion": "先講一個你用資料做判斷的經驗。",
           "openingTopic": "資料分析", "fallbackProbes": ["依據是什麼?"]}
TURN = {"speaker": "", "nextQuestion": "那個數字的母數是多少?",
        "reaction": "嗯。", "isFollowUp": True, "topic": "資料分析",
        "shouldAdvance": False}
GOOD_WHY = "這份職缺把資料查詢列為日常工作,沒有這項能力就無法承擔核心職責。"


def fake_llm(system: str, user: str) -> str:
    if "開場問題" in system:
        return json.dumps(OPENING, ensure_ascii=False)
    if "接話" in system:
        return json.dumps(TURN, ensure_ascii=False)
    if "搶節奏" in system:
        return json.dumps(["我先講。"], ensure_ascii=False)
    if "抽出這份工作實際需要的能力" in system:
        return json.dumps(["SQL", "報表自動化"], ensure_ascii=False)
    if "三個面向" in system:
        return json.dumps([
            {"letter": "內", "verdict": "缺數字", "points": ["加一個量化成果"]},
            {"letter": "構", "verdict": "邏輯清楚", "points": ["補結果段"]},
            {"letter": "達", "verdict": "用詞直接", "points": ["少用填充詞"]},
        ], ensure_ascii=False)
    if "細分分數" in system:
        names = [n for n in SUB_SCORE_NAMES if n not in ("表達流暢度", "內容深度")]
        return json.dumps([{"name": n, "score": v} for n, v in zip(names, [82, 68, 64, 80])],
                          ensure_ascii=False)
    if "逐題" in system:
        return json.dumps([{"id": 1, "comment": "有交代背景", "better": "把結論提前講"}],
                          ensure_ascii=False)
    if "STAR" in system:
        return json.dumps([
            {"key": "S", "present": True, "fromAnswer": "我用 SQL 重寫查詢", "hint": ""},
            {"key": "T", "present": False, "fromAnswer": "", "hint": "缺任務"},
            {"key": "A", "present": False, "fromAnswer": "", "hint": "缺行動"},
            {"key": "R", "present": False, "fromAnswer": "", "hint": "缺結果"},
        ], ensure_ascii=False)
    if "漏講加分點" in system:
        return json.dumps([{"id": 1, "why": GOOD_WHY}], ensure_ascii=False)
    return "{}"


def boom(system: str, user: str) -> str:
    raise RuntimeError("timeout")


@pytest.fixture
def client(monkeypatch):
    """繞過 lifespan 的真實載入。TranscriptAnalyzer 會載 2.3GB,測試不能碰。"""
    from app.api import deps

    monkeypatch.setattr(deps, "build_components",
                        lambda: (StubAnalyzer(), StubGap(), FakeCollabObserver()))
    for name in ("get_llm", "get_llm_live", "get_llm_verbatim"):
        monkeypatch.setattr(deps, name, lambda: fake_llm)
    import app.api.routes as routes
    for name in ("get_llm", "get_llm_live", "get_llm_verbatim"):
        monkeypatch.setattr(routes, name, lambda: fake_llm)

    from app.main import app
    with TestClient(app) as c:
        yield c


CTX = {"customRole": "資料分析實習生", "customJd": "需具備 SQL 撰寫能力。", "type": "行為"}


# ---------------------------------------------------------------------------


def test_health(client) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_start_returns_session_and_question(client) -> None:
    r = client.post("/interviews", json={"mode": "single", "context": CTX})
    assert r.status_code == 200
    d = r.json()
    assert d["sessionId"]
    assert d["openingQuestion"]
    assert d["openingTopic"]


def test_start_group_gets_interrupt_lines(client) -> None:
    d = client.post("/interviews", json={"mode": "group", "context": CTX}).json()
    assert d["interruptLines"]
    assert d["interruptCap"] == len(d["interruptLines"])
    assert {p["displayName"] for p in d["personas"]} >= {"主考官", "AI-邏輯"}


def test_start_single_has_no_interrupts(client) -> None:
    d = client.post("/interviews", json={"mode": "panel", "context": CTX}).json()
    assert d["interruptLines"] == []


def test_turn_returns_next_question(client) -> None:
    r = client.post("/interviews/itv_x/turns", json={
        "answer": "我用 SQL 重寫查詢", "followUpIdx": 0, "question": "請自我介紹",
    })
    assert r.status_code == 200
    assert r.json()["nextQuestion"]
    assert r.json()["topic"], "topic 不可為空,前端要拿它累積"


def test_turn_infers_mode_from_spoken_by(client) -> None:
    """TurnRequest 沒有 mode 欄位,從 spokenBy 反推。這是暫時做法。"""
    d = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": ["技術主管"],
    }).json()
    assert d["speaker"], "panel 模式必須回說話者"


def test_report_returns_all_blocks(client) -> None:
    r = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": CTX,
        "turns": [{"question": "請自我介紹", "answer": "我用 SQL 重寫查詢", "inputMode": "voice"}],
        "experiences": [{"id": "e1", "title": "實習", "tags": ["SQL"]}],
    })
    assert r.status_code == 200
    d = r.json()
    assert [s["name"] for s in d["subScores"]] == list(SUB_SCORE_NAMES)
    assert [p["key"] for p in d["starParts"]] == ["S", "T", "A", "R"]
    assert len(d["faceDimensions"]) == 3
    assert d["questionFeedbacks"]
    assert d["improvements"]
    assert d["resumeGrounded"] is True


def test_report_without_resume_has_no_missing_points(client) -> None:
    """沒有履歷就不可能有漏講點。這是防捏造的最後一道。"""
    d = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": CTX,
        "turns": [{"question": "q", "answer": "我用 SQL 重寫查詢"}],
        "experiences": [],
    }).json()
    assert d["missingPoints"] == []
    assert d["resumeGrounded"] is False
    assert any("尚未建立經歷" in n for n in d["notices"])


def test_report_with_resume_has_missing_points(client) -> None:
    d = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": CTX,
        "turns": [{"question": "q", "answer": "我做過一些報表"}],
        "experiences": [{"id": "e1", "title": "實習", "tags": ["SQL"]}],
    }).json()
    assert d["missingPoints"]
    assert d["missingPoints"][0]["why"]


def test_turn_infers_group_mode_from_context(client) -> None:
    """群面的名單依 groupInterviewers 與 groupSize 而變,
    3 位主管的配置用預設參數會對不上。"""
    d = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": ["用人主管"],
        "context": {**CTX, "groupInterviewers": 3, "groupSize": 5},
    }).json()
    assert d["speaker"], "群面必須回說話者"


def test_turn_passes_context_through(client) -> None:
    """設定要逐輪生效,否則第二輪之後全部退回預設值。"""
    d = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0,
        "context": {**CTX, "difficulty": "困難"},
    })
    assert d.status_code == 200


def test_group_report_has_no_collab_without_rubric(client) -> None:
    d = client.post("/interviews/itv_x/report", json={
        "mode": "group", "context": CTX,
        "turns": [{"question": "q", "answer": "我覺得先做市場調查"}],
        "groupSays": [
            {"speaker": "user", "content": "我覺得先做市場調查", "isUser": True},
            {"speaker": "AI-邏輯", "content": "母數是多少", "isUser": False},
        ],
    }).json()
    assert d["collabDims"] == []
    # rubric 未到位時要說明原因,不能只是靜靜地不顯示
    assert any("rubric" in n or "協作評分尚未啟用" in n for n in d["notices"])
    assert any("協作訊號已抽取" in n for n in d["notices"]), \
        "訊號有抽,只是還不能指派等第——這兩件事要分得出來"


def test_report_rejects_empty_turns(client) -> None:
    r = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": CTX, "turns": [],
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 降級:面試進行中噴 500 等於整場中斷
# ---------------------------------------------------------------------------


def test_start_degrades_when_llm_fails(client, monkeypatch) -> None:
    import app.api.routes as routes
    monkeypatch.setattr(routes, "get_llm_live", lambda: boom)
    monkeypatch.setattr(routes, "get_llm", lambda: boom)
    r = client.post("/interviews", json={"mode": "single", "context": CTX})
    assert r.status_code == 200
    assert r.json()["openingQuestion"], "要有預設題,不能空手回去"


def test_turn_degrades_when_llm_fails(client, monkeypatch) -> None:
    import app.api.routes as routes
    monkeypatch.setattr(routes, "get_llm_live", lambda: boom)
    r = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "fallback": ["備援題"],
    })
    assert r.status_code == 200
    assert r.json()["nextQuestion"] == "備援題"


def test_report_degrades_when_llm_fails(client, monkeypatch) -> None:
    import app.api.routes as routes
    monkeypatch.setattr(routes, "get_llm", lambda: boom)
    monkeypatch.setattr(routes, "get_llm_verbatim", lambda: boom)
    r = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": CTX,
        "turns": [{"question": "q", "answer": "我用 SQL 重寫查詢"}],
    })
    assert r.status_code == 200
    d = r.json()
    assert [s["name"] for s in d["subScores"]] == list(SUB_SCORE_NAMES), "區塊全失敗仍要有形狀"
    assert d["notices"]


def test_mode_field_wins_over_inference(client) -> None:
    """前端帶了 mode 就用它,不要反推。

    反推有兩個失準點:第一輪 spokenBy 是空的、一對一不回 speaker。
    """
    d = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "mode": "panel", "spokenBy": [],
    }).json()
    assert d["speaker"], "帶了 panel 就該回說話者,即使 spokenBy 是空的"


def test_first_turn_of_group_needs_mode(client) -> None:
    """第一輪 spokenBy 是空的,沒有 mode 就會被當成 single。"""
    without = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": [],
    }).json()
    assert without["speaker"] == "", "沒帶 mode 時退回 single"

    with_mode = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "mode": "group",
        "context": {**CTX, "groupInterviewers": 3, "groupSize": 5},
    }).json()
    assert with_mode["speaker"], "帶了 group 第一輪就正確"


def test_mode_falls_back_to_inference_when_absent(client) -> None:
    """前端還沒改之前,反推要繼續可用。"""
    d = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": ["技術主管"],
    }).json()
    assert d["speaker"]


def test_mode_rejects_unknown_value(client) -> None:
    """mode 由前端路由決定,值域封閉,這裡要嚴格。"""
    r = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "mode": "video",
    })
    assert r.status_code == 422
