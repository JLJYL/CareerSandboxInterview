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
    # fake_llm 不認得協作 prompt,四個維度都會回空 dict → 全部未評分
    assert d["collabDims"] == []
    assert any("協作" in n for n in d["notices"]), "未評分要說明原因,不能靜靜地不顯示"


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


# ---------------------------------------------------------------------------
# 協作評分:逐維度分開評
# ---------------------------------------------------------------------------


def test_collab_scored_per_dimension(client, monkeypatch) -> None:
    """一次評四維的維度間相關 r̄ ≈ .86–.92,分開評掉到 .26–.35。

    這裡驗四次獨立呼叫,每次只帶一個維度。
    """
    seen = []

    def spy(system: str, user: str) -> str:
        if "這個維度評分" in system:
            seen.append(system)
            return json.dumps({"level": 3, "evidence": "我覺得先做客群分析",
                               "hint": "開場就先表態,搶到定錨位置"}, ensure_ascii=False)
        return fake_llm(system, user)

    import app.api.routes as routes
    monkeypatch.setattr(routes, "get_llm", lambda: spy)

    d = client.post("/interviews/itv_x/report", json={
        "mode": "group", "context": CTX,
        "turns": [{"question": "q", "answer": "我覺得先做市場調查"}],
        "groupSays": [
            {"speaker": "AI-邏輯", "content": "母數是多少", "isUser": False},
            {"speaker": "user", "content": "我覺得先做客群分析", "isUser": True},
        ],
    }).json()

    assert len(seen) == 4, "四個維度要分開呼叫,不是一次評完"
    names = [n for n in ("參與主動性", "傾聽與回應", "論點建構", "協作姿態")
             if any(n in s for s in seen)]
    assert len(names) == 4, "每次呼叫只帶一個維度"
    assert d["collabDims"], "評出來的維度要進報告"


def test_collab_evidence_must_be_verbatim() -> None:
    """協作是零量化錨點的評分,沒有原文檢查就無法驗證等第不是編的。

    跟 starParts.fromAnswer 同一條規則、同一種檢查。
    """
    from app.pipeline.collab_score import verify_evidence
    from app.schemas.interview import CollabDimDTO

    transcript = "我覺得先做客群分析\n那樣不行啦 還是要先調查"
    real = CollabDimDTO(name="論點建構", score=76, evidence="我覺得先做客群分析")
    fake = CollabDimDTO(name="參與主動性", score=88, evidence="我在台積電帶過十人團隊")

    assert verify_evidence([real], transcript) == []
    assert len(verify_evidence([fake], transcript)) == 1
    assert verify_evidence([CollabDimDTO(name="x", score=76)], transcript) == [], \
        "找不到對應句子時留空是對的,不該被當成違規"


def test_collab_stance_bars_marks_negation() -> None:
    """等第 2 與 3 的界線是有沒有否定的語言,不是後面有沒有接理由。

    實測:「那樣不行啦 還是要先調查」被判成 3——
    那句話是等第 2 的教科書範例。
    """
    from app.prompts.collab_rubric import BARS

    stance = BARS["協作姿態"]
    assert "那樣不行" in stance, "要給出否定詞的具體標記"
    assert "即使後面接了理由" in stance
    assert "以否定詞開頭 → 2" in stance, "要給一個可以照著做的判斷步驟"


def test_evidence_survives_slice_markers() -> None:
    """切片會加標記,但 verify_evidence 比對的是純發言內容。

    prompt 已經寫明「只引「」裡面的內容」,而且切片用引號劃了邊界——
    但那是判斷型規則,會有殘留率。

    殘留的後果特別糟:連標記一起抄的引用**跟真的編造分不出來**,
    兩者都會被標成「依據不可信」。剝掉標記之後,被擋下來的就只剩真的編造。
    """
    from app.pipeline.collab_score import verify_evidence
    from app.schemas.interview import CollabDimDTO

    transcript = "題目是會員制度\n我覺得先做客群分析\n母數是多少"

    def check(ev: str) -> bool:
        return verify_evidence([CollabDimDTO(name="x", score=76, evidence=ev)], transcript) == []

    # 四種切片格式都要通過
    assert check("我覺得先做客群分析")
    assert check("[接續他人之後]「我覺得先做客群分析」")
    assert check("你：「我覺得先做客群分析」")
    assert check("他人（主考官）：「題目是會員制度」")
    # 真的編造仍然要被擋
    assert not check("我在台積電帶過十人團隊")
    assert not check("我覺得先做客群分析然後再看資料"), "半真半編也要擋"


def test_real_collab_observer_matches_protocol() -> None:
    """成員 A 的真實作要符合 Protocol,而且切片格式要跟 verify_evidence 對得上。"""
    try:
        from app.pipeline.collab import CollabObserver
    except ImportError:
        import pytest

        pytest.skip("成員 A 的 collab.py 尚未進 repo")

    from app.contracts.interview_protocols import CollabObserver as Proto
    from app.contracts.interview_protocols import Utterance
    from app.pipeline.collab_score import strip_markers

    o = CollabObserver()
    assert isinstance(o, Proto)

    us = [
        Utterance("主考官", "題目是會員制度"),
        Utterance("你", "我覺得先做客群分析", is_user=True),
        Utterance("AI-邏輯", "母數是多少"),
    ]
    src = "".join(u.text for u in us)
    for sl in o.observe(us):
        for ex in sl.excerpts:
            # 每一行剝掉標記之後都該落在逐字稿裡
            for line in ex.split("\n"):
                bare = strip_markers(line).replace(" ", "")
                if bare:
                    assert bare in src, f"{sl.name} 的片段剝不出原文:{line}"


def test_interviewer_derived_from_persona_registry() -> None:
    """is_interviewer 由後端查 personas_for() 推導,不需要前端送。

    那份註冊表就是定義顯示名稱的地方——persona 改名會同時改到兩邊,
    不會出現名稱變了而判斷邏輯沒變的情況。
    """
    from app.api.routes import _interviewer_names
    from app.schemas.interview import InterviewContext

    solo = _interviewer_names("group", InterviewContext(group_interviewers=1, group_size=5))
    assert solo == {"主考官"}

    panel = _interviewer_names("group", InterviewContext(group_interviewers=3, group_size=5))
    assert panel == {"用人主管", "技術主管", "HR 主管"}
    assert not any(n.startswith("AI-") for n in panel), "AI 應徵者是同儕不是面試官"


# ---------------------------------------------------------------------------
# 版控防護:這支檔被 .gitignore 誤擋過三次
# ---------------------------------------------------------------------------


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[1]


def test_app_main_is_tracked_by_git() -> None:
    """app/main.py 必須在版控裡。

    .gitignore 有一段(27–54 行)是 CareerSandboxModule 的路徑清單,
    某次 merge 帶過來的。它擋過 app/api/,也擋過 app/main.py。

    這種失敗特別難查:git add -A 不報錯、git status 不顯示、
    commit 訊息寫了但檔案沒進去,而症狀是「lifespan 沒有跑到」——
    錯誤訊息指向 deps.py,不會有人聯想到是一支沒進版控的檔案。
    """
    import subprocess

    r = subprocess.run(
        ["git", "ls-files", "app/main.py"],
        capture_output=True, text=True, cwd=str(_repo_root()),
    )
    if r.returncode != 0:
        import pytest

        pytest.skip("不在 git 工作目錄裡")
    assert r.stdout.strip() == "app/main.py", (
        "app/main.py 沒有進版控。檢查 .gitignore 有沒有擋到它:\n"
        "    git check-ignore -v app/main.py"
    )


def test_lifespan_does_not_bind_deps_at_import() -> None:
    """lifespan 要呼叫 deps.build_components(),不能在頂端 import。

    頂端匯入會在匯入當下綁死原始函式,測試的 monkeypatch 換不到,
    而且模組只匯入一次,第一個測試的綁定會留到後面全部——
    那會讓後面的測試真的去載 bge-m3。
    """
    import pathlib

    src = (_repo_root() / "app" / "main.py").read_text(encoding="utf-8")

    # 只看實際的 import 陳述,不看註解——檔案裡有一段寫明「不要這樣寫」的反例,
    # 整份比對會抓到那一行。
    imports = [
        ln.strip()
        for ln in src.splitlines()
        if ln.startswith(("import ", "from ")) and "build_components" in ln
    ]
    assert imports == [], f"lifespan 依賴不可在頂端匯入:{imports}"
    assert "deps.build_components()" in src


def test_stance_weights_peer_conflict_higher() -> None:
    """跟面試官的分歧情境不對等:平行競爭 vs 權力落差。

    協作姿態測的是團隊協作,而團隊是同儕之間的事。
    成員 A 保留面試官發言並標明身分是對的——排掉會漏掉觀察機會——
    但 BARS 要說明怎麼權衡,否則 LLM 會把兩者當成等價。
    """
    from app.prompts.collab_rubric import BARS

    stance = BARS["協作姿態"]
    assert "同儕之間的分歧  → 主要依據" in stance
    assert "情境不對等" in stance
    assert "同儕協作的部分這次觀察不到" in stance, "只有跟面試官的分歧時要說明"


def test_fold_fixes_verbatim_false_positives() -> None:
    """Whisper 的繁簡輸出不一致,逐字稿是簡體時 LLM 會把引用轉成繁體。

    直接比對會把**正確的引用判成編造**——
    starParts 降成未命中、協作維度標成依據不可信。
    """
    from app.pipeline.collab_score import verify_evidence
    from app.schemas.interview import CollabDimDTO, ReportResponse, StarPartDTO
    from app.schemas.interview_repair import verify_star_verbatim

    simp = "我觉得先做客群分析 样本低于三十就不采用"
    trad = "我覺得先做客群分析"

    assert verify_evidence([CollabDimDTO(name="x", score=76, evidence=trad)], simp) == []
    r = ReportResponse(
        mode="single",
        star_parts=[StarPartDTO(key="S", name="情境", present=True, from_answer=trad)],
    )
    assert verify_star_verbatim(r, simp) == []
    # 真的編造仍然要擋
    assert verify_evidence(
        [CollabDimDTO(name="x", score=76, evidence="我在台積電帶過十人團隊")], simp
    )


def test_fold_does_not_touch_ambiguous_chars() -> None:
    """一對多的字不收——收了會把「頭髮」折成「頭發」,製造新的比對失敗。"""
    from app.pipeline.text_fold import fold_variants

    for c in "發髮乾幹餘余":
        assert fold_variants(c) == c
