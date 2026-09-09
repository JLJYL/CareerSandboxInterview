"""端到端:開場 → 多輪 → 報告,用**前端目前真的送得出的欄位**。

【為什麼不用合約的完整形狀】
合約的欄位比前端現在能給的多。照理想形狀測會測到一個不存在的系統,
而 demo 當天跑的是這裡測的這一種。

前端目前送不出的(2026-09-02 的 main):
    turns[].segmentStartsMs / endedBy  —— InterviewSession.Turn 只有 question/answer
    context.groupSize / groupRole      —— InterviewConfig 沒有這兩欄
    mode                               —— format 是畫面內的區域變數,沒進 config

送得出的:
    groupSays[]  —— GroupUtterance(speaker, content, isUser, segments, segmentStartsMs)
                    注意 speaker 是「你」不是「user」

【這一支要驗什麼】
    三個端點串起來不會壞
    前端不送的欄位走預設值,不是 422
    後端自己累積的狀態(askedTopics、spokenBy)前後對得上
    群面的 speaker 是「你」時切片不會切錯
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.contracts.interview_protocols import FakeCollabObserver, GapCandidate, TextStats
from app.schemas.interview import COLLAB_DIM_NAMES, SUB_SCORE_NAMES


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


GOOD_WHY = "這份職缺把資料查詢列為日常工作,沒有這項能力就無法承擔核心職責。"


def fake_llm(system: str, user: str) -> str:
    if "決定這一輪由誰開口" in system:
        return "技術主管"
    if "開場問題" in system:
        return json.dumps({
            "openingQuestion": "先講一個你用資料做判斷的經驗。",
            "openingTopic": "資料分析",
            "fallbackProbes": ["依據是什麼?", "如果重來會怎麼改?"],
        }, ensure_ascii=False)
    if "接話" in system:
        return json.dumps({
            "speaker": "技術主管", "nextQuestion": "那個數字的母數是多少?",
            "reaction": "嗯。", "isFollowUp": True, "topic": "資料分析",
        }, ensure_ascii=False)
    if "搶節奏" in system:
        return json.dumps(["我先講。", "我插一句。"], ensure_ascii=False)
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
        return json.dumps([
            {"id": i, "comment": "有交代背景", "better": "把結論提前講"} for i in range(1, 6)
        ], ensure_ascii=False)
    if "STAR" in system:
        return json.dumps([
            {"key": "S", "present": True, "fromAnswer": "我用 SQL 重寫查詢", "hint": ""},
            {"key": "T", "present": False, "fromAnswer": "", "hint": "缺任務"},
            {"key": "A", "present": False, "fromAnswer": "", "hint": "缺行動"},
            {"key": "R", "present": False, "fromAnswer": "", "hint": "缺結果"},
        ], ensure_ascii=False)
    if "漏講加分點" in system:
        return json.dumps([{"id": 1, "why": GOOD_WHY}], ensure_ascii=False)
    if "這個維度評分" in system:
        return json.dumps({"level": 3, "evidence": "我覺得先做客群分析",
                           "hint": "開場就先表態,搶到定錨位置"}, ensure_ascii=False)
    return "{}"


@pytest.fixture
def client(monkeypatch):
    from app.api import deps

    monkeypatch.setattr(deps, "build_components",
                        lambda: (StubAnalyzer(), StubGap(), FakeCollabObserver()))
    import app.api.routes as routes
    for name in ("get_llm", "get_llm_live", "get_llm_verbatim"):
        monkeypatch.setattr(routes, name, lambda: fake_llm)

    from app.main import app
    with TestClient(app) as c:
        yield c


# 前端真的送得出的 context:InterviewConfig 的欄位,一個不多一個不少
FRONTEND_CTX = {
    "round": "初試",
    "language": "中文",
    "type": "行為",
    "difficulty": "中等",
    "groupInterviewers": 1,
    "customRole": "資料分析實習生",
    "customCompany": "某電商",
    "customSeniority": "新鮮人",
    "customIndustry": "電子商務",
    "customJd": "需具備 SQL 撰寫能力,能建立自動化報表流程。",
}

ANSWERS = [
    "我之前在系學會做過行銷 然後也有去電商公司實習",
    "實習的時候我用 SQL 重寫了週報的查詢 把產出時間從四小時縮短到一小時",
    "那時候行銷跟工程對需求的優先順序有分歧 我有去協調兩邊的排程",
]


def run_session(client, mode_hint: str | None = None) -> dict:
    """跑完一場:開場 → 三輪 → 報告。回傳報告。

    模擬前端的狀態累積:askedTopics 收 topic、spokenBy 收 speaker。
    這正是要驗的——那兩個目前由前端負責累積,累積邏輯錯了後端不會發現。
    """
    start = client.post("/interviews", json={
        "mode": mode_hint or "single", "context": FRONTEND_CTX,
    })
    assert start.status_code == 200, start.text
    s = start.json()

    asked = [s["openingTopic"]] if s["openingTopic"] else []
    spoken = [s["openingSpeaker"]] if s["openingSpeaker"] else []
    question = s["openingQuestion"]
    turns = []

    for i, ans in enumerate(ANSWERS):
        body = {
            "answer": ans,
            "followUpIdx": i,
            "question": question,
            "askedTopics": asked,
            "spokenBy": spoken,
            "fallback": s["fallbackProbes"],
            "inputMode": "voice",
            "context": FRONTEND_CTX,
        }
        r = client.post(f"/interviews/{s['sessionId']}/turns", json=body)
        assert r.status_code == 200, r.text
        t = r.json()
        turns.append({"question": question, "answer": ans, "inputMode": "voice"})
        question = t["nextQuestion"]
        if t["topic"]:
            asked.append(t["topic"])
        if t["speaker"]:
            spoken.append(t["speaker"])

    rep = client.post(f"/interviews/{s['sessionId']}/report", json={
        "mode": mode_hint or "single",
        "context": FRONTEND_CTX,
        "turns": turns,
        "experiences": [{"id": "e1", "title": "電商實習", "tags": ["SQL"]}],
    })
    assert rep.status_code == 200, rep.text
    return rep.json()


# ---------------------------------------------------------------------------


def test_single_session_end_to_end(client) -> None:
    """一對一:三個端點串起來不會壞,報告六個區塊都有。"""
    d = run_session(client)
    assert [s["name"] for s in d["subScores"]] == list(SUB_SCORE_NAMES)
    assert [p["key"] for p in d["starParts"]] == ["S", "T", "A", "R"]
    assert len(d["faceDimensions"]) == 3
    assert len(d["questionFeedbacks"]) == len(ANSWERS)
    assert d["improvements"]
    assert d["missingPoints"], "有履歷就該有漏講點"


def test_missing_optional_fields_do_not_break(client) -> None:
    """前端送不出 segmentStartsMs / endedBy / groupSize / mode,
    那些走預設值不是 422。"""
    for body in (
        {"answer": "x", "followUpIdx": 0},
        {"answer": "x", "followUpIdx": 0, "context": FRONTEND_CTX},
        {"answer": "x", "followUpIdx": 0, "question": "q", "askedTopics": []},
    ):
        r = client.post("/interviews/itv_x/turns", json=body)
        assert r.status_code == 200, f"{body} → {r.status_code} {r.text}"


def test_report_without_segment_data(client) -> None:
    """Turn 沒有 segments 與 endedBy,報告照樣要出得來。"""
    d = client.post("/interviews/itv_x/report", json={
        "mode": "single", "context": FRONTEND_CTX,
        "turns": [{"question": "q", "answer": "我用 SQL 重寫查詢", "inputMode": "voice"}],
        "experiences": [],
    })
    assert d.status_code == 200
    j = d.json()
    assert j["subScores"]
    assert not any("被語音辨識切斷" in n for n in j["notices"]), \
        "沒有 endedBy 時不該宣稱有輪次被切斷"


def test_group_session_with_real_speaker_label(client) -> None:
    """群面:前端送的 speaker 是「你」不是「user」。

    早期版本的切片寫 speaker_id == "user",照這個 payload 會全部切錯,
    而且不報錯——切片是空的,LLM 回 level 0,看起來像正常降級。
    """
    d = client.post("/interviews/itv_x/report", json={
        "mode": "group", "context": FRONTEND_CTX,
        "turns": [{"question": "q", "answer": "我覺得先做客群分析", "inputMode": "voice"}],
        "groupSays": [
            {"speaker": "主考官", "content": "題目是會員制度", "isUser": False},
            {"speaker": "你", "content": "我覺得先做客群分析", "isUser": True,
             "segments": ["我覺得先做客群分析"], "segmentStartsMs": [1200]},
            {"speaker": "AI-邏輯", "content": "母數是多少", "isUser": False},
            {"speaker": "你", "content": "同意前面那位 可以先做小規模驗證", "isUser": True},
        ],
    })
    assert d.status_code == 200
    j = d.json()
    assert j["collabDims"], "切片正確的話四個維度該評得出來"
    assert {c["name"] for c in j["collabDims"]} <= set(COLLAB_DIM_NAMES)


def test_asked_topics_accumulate_across_turns(client) -> None:
    """askedTopics 由前端累積,累積錯了後端不會發現。

    這裡驗每一輪都回得出 topic——空的那一筆會讓序列斷掉,
    而斷掉的地方就是之後重複提問的破口。
    """
    s = client.post("/interviews", json={"mode": "single", "context": FRONTEND_CTX}).json()
    asked = [s["openingTopic"]]
    question = s["openingQuestion"]
    for i, ans in enumerate(ANSWERS):
        t = client.post(f"/interviews/{s['sessionId']}/turns", json={
            "answer": ans, "followUpIdx": i, "question": question,
            "askedTopics": asked, "context": FRONTEND_CTX,
        }).json()
        assert t["topic"], f"第 {i + 1} 輪沒有 topic,前端會累積到一個空字串"
        asked.append(t["topic"])
        question = t["nextQuestion"]
    assert len(asked) == len(ANSWERS) + 1


def test_panel_first_turn_without_mode(client) -> None:
    """前端目前送不出 mode,而第一輪 spokenBy 是空的。

    已知後果:panel 的第一輪會被當成 single,不回說話者。
    這是交接清單上的項目,不是 bug——但要有測試釘住這個行為,
    前端補上 mode 之後這個測試會紅燈,那時就知道該改了。
    """
    t = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": [], "context": FRONTEND_CTX,
    }).json()
    assert t["speaker"] == "", "沒有 mode 又沒有 spokenBy 時退回 single"

    t2 = client.post("/interviews/itv_x/turns", json={
        "answer": "x", "followUpIdx": 0, "spokenBy": ["HR 主管"], "context": FRONTEND_CTX,
    }).json()
    assert t2["speaker"], "第二輪起 spokenBy 有東西,反推得出 panel"


# ---------------------------------------------------------------------------
# gap 要看使用者在討論裡講的全部內容
# ---------------------------------------------------------------------------


class SpyGap:
    """記下 compute() 實際收到的逐字稿。

    gap 的正確性沒辦法用回傳值驗——它吃錯文字時照樣回得出候選,
    只是候選變多(提及集變小)。要驗的是它看到了什麼。
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def compute(self, resume, jd, transcript):
        self.seen.append(transcript)
        return []


def test_gap_sees_group_discussion_not_just_turns(client, monkeypatch) -> None:
    """群面時使用者的發言在 groupSays 裡,turns 可能只有主問題的簡短回答。

    早期版本 gap 只拿 turns.answer,群面時使用者在討論裡講的東西
    完全不在裡面。實測:提及集從 3 個技能變成空。

    gap = 履歷 ∩ JD − 提及。提及漏掉 → gap 變大 →
    系統對使用者說「你漏講了 SQL」而他明明講了。
    那是假指控,不是少給建議,而且靜默——報告照樣產出、測試照樣全綠。
    """
    from app.api import deps

    spy = SpyGap()
    monkeypatch.setattr(
        deps, "build_components",
        lambda: (StubAnalyzer(), spy, FakeCollabObserver()),
    )
    import app.api.routes as routes
    for name in ("get_llm", "get_llm_live", "get_llm_verbatim"):
        monkeypatch.setattr(routes, name, lambda: fake_llm)

    from app.main import app

    with TestClient(app) as c:
        c.post("/interviews/itv_x/report", json={
            "mode": "group", "context": FRONTEND_CTX,
            "turns": [{"question": "請自我介紹", "answer": "我是資管系的"}],
            "groupSays": [
                {"speaker": "主考官", "content": "題目是會員制度", "isUser": False},
                {"speaker": "你", "content": "我用 SQL 重寫過週報查詢",
                 "isUser": True, "segments": ["我用 SQL 重寫過週報查詢"]},
                {"speaker": "AI-邏輯", "content": "母數是多少", "isUser": False},
                {"speaker": "你", "content": "也用 Excel 做過報表", "isUser": True},
            ],
            "experiences": [{"id": "e1", "title": "實習", "tags": ["SQL"]}],
        })

    assert spy.seen, "gap 應該被呼叫"
    seen = spy.seen[0]
    assert "SQL" in seen, "使用者在討論裡講的 SQL 必須進 gap 的輸入"
    assert "Excel" in seen
    assert "母數是多少" not in seen, "AI 同儕的發言不該算成使用者講過"


def test_gap_uses_turns_in_single_mode(client, monkeypatch) -> None:
    """一對一的使用者發言本來就全在 turns.answer,只有群面中彈。"""
    from app.api import deps

    spy = SpyGap()
    monkeypatch.setattr(
        deps, "build_components",
        lambda: (StubAnalyzer(), spy, FakeCollabObserver()),
    )
    import app.api.routes as routes
    for name in ("get_llm", "get_llm_live", "get_llm_verbatim"):
        monkeypatch.setattr(routes, name, lambda: fake_llm)

    from app.main import app

    with TestClient(app) as c:
        c.post("/interviews/itv_x/report", json={
            "mode": "single", "context": FRONTEND_CTX,
            "turns": [{"question": "q", "answer": "我用 SQL 重寫過週報查詢"}],
            "experiences": [{"id": "e1", "title": "實習", "tags": ["SQL"]}],
        })

    assert "SQL" in spy.seen[0]
