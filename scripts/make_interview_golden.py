"""產生面試合約的 golden JSON。

用途:
  1. 給怡君看的實際回應長相,比讀 schema 快
  2. 合約測試的固定輸入,任何一次改動只要動到形狀就會紅燈
  3. 前端在後端還沒好之前可以直接拿去當假資料

執行:
  python scripts/make_interview_golden.py
產出:
  fixtures/interview/golden/*.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.schemas.interview import (  # noqa: E402
    COLLAB_DIM_NAMES,
    ExperienceDTO,
    FaceDimensionDTO,
    InterviewContext,
    MissingPointDTO,
    Persona,
    ProsodyItem,
    QuestionFeedbackDTO,
    ReportRequest,
    ReportResponse,
    StartInterviewRequest,
    StartInterviewResponse,
    StarPartDTO,
    SubScoreDTO,
    CollabDimDTO,
    TurnDTO,
    TurnRequest,
    TurnResponse,
    SUB_SCORE_NAMES,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "interview" / "golden"


def _dump(name: str, model) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(by_alias=True, exclude_none=False)
    (OUT / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  {name}.json")


# --- 共用素材 -------------------------------------------------------------

CONTEXT = InterviewContext(
    round="初試",
    language="中文",
    type="行為",
    difficulty="中等",
    group_interviewers=1,
    custom_role="資料分析實習生",
    custom_company="某電商公司",
    custom_seniority="新鮮人",
    custom_industry="電子商務",
    custom_jd="協助建立銷售報表與資料清理,需具備 SQL 撰寫能力,能建立自動化報表流程,需與產品、工程團隊密切協作。",
)

EXPERIENCES = [
    ExperienceDTO(
        id="exp_001",
        title="電商公司 資料分析實習",
        category="實習",
        period="2025/07 - 2025/09",
        role="資料分析實習生",
        action="用 SQL 重寫週報查詢,並排程自動產出",
        result="週報產出時間由 4 小時縮短為 1 小時",
        learning="學到先確認需求再動手寫查詢",
        description="協助行銷部門建立銷售追蹤報表",
        tags=["SQL", "報表自動化", "跨部門溝通"],
    ),
    ExperienceDTO(
        id="exp_002",
        title="系學會 行銷組",
        category="社團",
        period="2024/09 - 2025/06",
        role="行銷組長",
        action="經營系學會 IG,規劃聯名活動",
        result="追蹤數由 0 成長至 1200",
        learning="內容排程比單篇品質更重要",
        description="負責系上活動的對外宣傳",
        tags=["社群經營", "活動企劃"],
    ),
]

TURNS = [
    TurnDTO(
        question="請先簡單自我介紹",
        answer="我是資管系大三的學生 過去在系學會做行銷 後來去電商公司實習做資料分析",
    ),
    TurnDTO(
        question="講一個你解決問題的經驗",
        answer="我們辦過一場聯名活動 前期要決定推廣規模 我沒先測試就直接全推",
    ),
]

TRANSCRIPT_TEXT = "\n".join(t.answer for t in TURNS)


# --- 端點一:開場 ---------------------------------------------------------


def build_start() -> None:
    _dump(
        "req_start_single",
        StartInterviewRequest(mode="single", context=CONTEXT, experiences=EXPERIENCES),
    )
    _dump(
        "res_start_single",
        StartInterviewResponse(
            session_id="itv_2026081201HZ",
            mode="single",
            opening_question="先請你用一分鐘自我介紹,重點放在跟資料分析相關的經驗。",
            opening_speaker="",
            opening_topic="自我介紹",
            personas=[
                Persona(
                    id="interviewer_main",
                    display_name="面試官",
                    role="interviewer",
                    blurb="電商公司資料團隊,問題偏行為面。",
                )
            ],
            fallback_probes=[
                "這個判斷你是看哪些數據下的?",
                "如果時間只剩一半,你會先砍掉哪一段?",
                "這件事你學到什麼,下次會怎麼做?",
            ],
        ),
    )

    _dump(
        "res_start_panel",
        StartInterviewResponse(
            session_id="itv_2026081202KQ",
            mode="panel",
            opening_question="請先自我介紹,然後說明你為什麼想投這個職位。",
            opening_speaker="HR 主管",
            opening_topic="自我介紹",
            personas=[
                Persona(id="hr", display_name="HR 主管", role="interviewer", blurb="看人格特質與團隊適配。"),
                Persona(id="tech", display_name="技術主管", role="interviewer", blurb="追問工具與方法細節。"),
                Persona(id="hiring", display_name="用人主管", role="interviewer", blurb="追問成果與取捨判斷。"),
            ],
            fallback_probes=[
                "可以再具體一點嗎?那個判斷你是看哪些數據下的?",
                "團隊合作裡你遇過最大的衝突是什麼?",
                "如果這個專案時程被砍一半,你會怎麼重排優先順序?",
            ],
        ),
    )

    _dump(
        "res_start_group",
        StartInterviewResponse(
            session_id="itv_2026081203MW",
            mode="group",
            opening_question="今天的討論題目是:公司要推新的會員制度,你們認為第一步該做什麼?",
            opening_speaker="主考官",
            opening_topic="商業判斷",
            personas=[
                Persona(id="moderator", display_name="主考官", role="moderator", blurb="主持討論,不評論。"),
                Persona(id="peer_logic", display_name="AI-邏輯", role="peer", blurb="質疑數據與母數。"),
                Persona(id="peer_assertive", display_name="AI-強勢", role="peer", blurb="搶快,主張先做再修。"),
                Persona(id="peer_friendly", display_name="AI-親切", role="peer", blurb="補位,傾向找共識。"),
            ],
            interrupt_lines=[
                "我先講一下我的想法,我覺得應該先鎖定高價值客群。",
                "打斷一下,我們時間不多,我直接給結論。",
                "我補一句,這件事的順序我覺得反了。",
            ],
            interrupt_cap=3,
            fallback_probes=[
                "如果資源只夠做一件事,你會先砍掉哪個?",
                "你剛剛提到的假設,有什麼資料可以支撐?",
            ],
        ),
    )


# --- 端點二:每輪 ---------------------------------------------------------


def build_turns() -> None:
    _dump(
        "req_turn",
        TurnRequest(
            answer="我在實習的時候用 SQL 做了一些報表 幫行銷部門看銷售狀況",
            follow_up_idx=0,
            question="請先簡單自我介紹",
            fallback=["這個判斷你是看哪些數據下的?"],
            asked_topics=["自我介紹"],
        ),
    )
    _dump(
        "res_turn_single",
        TurnResponse(
            speaker="",
            next_question="那份報表本來多久產一次?你改完之後差多少?",
            reaction="嗯。",
            is_follow_up=True,
            should_advance=False,
            topic="自我介紹",
        ),
    )
    _dump(
        "res_turn_panel",
        TurnResponse(
            speaker="技術主管",
            next_question="工具是手段。講一次你用數據推翻原本決定的經驗。",
            reaction="(技術主管點了點頭)",
            is_follow_up=True,
            should_advance=False,
            topic="資料分析",
        ),
    )
    _dump(
        "res_turn_group",
        TurnResponse(
            speaker="AI-邏輯",
            next_question="等等,這個數字的母數是多少?沒有對照組我不敢下結論。",
            reaction="",
            is_follow_up=False,
            should_advance=True,
            topic="數據驗證",
        ),
    )


# --- 端點三:報告 ---------------------------------------------------------


def _face_dims() -> list[FaceDimensionDTO]:
    return [
        FaceDimensionDTO(
            letter="內",
            name="內容",
            score=71,
            verdict="有講到重點,但缺乏具體數字與亮點。",
            points=[
                "自我介紹加 1-2 個量化成就",
                "回答時多舉一個具體例子,少用空泛形容詞",
            ],
        ),
        FaceDimensionDTO(
            letter="構",
            name="結構",
            score=82,
            verdict="邏輯清楚,但講失敗經歷時 STAR 的 Result 段常缺。",
            points=["講經歷一律用 STAR", "結尾補一句我從中學到什麼"],
        ),
        FaceDimensionDTO(
            letter="達",
            name="表達",
            score=70,
            verdict="用詞直接,但口語填充詞偏多。",
            points=["重點句講完停半秒", "減少嗯與那個的使用"],
            # 本期只有填充詞可算,其餘四項等前端補時間資料
            prosody=[ProsodyItem(label="填充詞", value="「嗯 / 那個」3 次")],
        ),
    ]


def _sub_scores() -> list[SubScoreDTO]:
    values = [78, 82, 71, 68, 64, 80]
    return [SubScoreDTO(name=n, score=v) for n, v in zip(SUB_SCORE_NAMES, values)]


def _star_parts() -> list[StarPartDTO]:
    return [
        StarPartDTO(key="S", name="情境 Situation", present=True, from_answer="我們辦過一場聯名活動", hint="有交代背景"),
        StarPartDTO(key="T", name="任務 Task", present=True, from_answer="前期要決定推廣規模", hint="任務算清楚"),
        StarPartDTO(key="A", name="行動 Action", present=True, from_answer="我沒先測試就直接全推", hint="行動講了,但偏簡略"),
        StarPartDTO(key="R", name="結果 Result", present=False, from_answer="", hint="缺這段:結果數字 + 你學到什麼"),
    ]


def build_report() -> None:
    _dump(
        "req_report_single",
        ReportRequest(
            mode="single",
            context=CONTEXT,
            experiences=EXPERIENCES,
            turns=TURNS,
        ),
    )

    _dump(
        "res_report_single",
        ReportResponse(
            mode="single",
            face_dimensions=_face_dims(),
            sub_scores=_sub_scores(),
            question_feedbacks=[
                QuestionFeedbackDTO(
                    question=TURNS[0].question,
                    answer=TURNS[0].answer,
                    comment="有交代背景,但兩段經歷都停在做過什麼,沒有帶出成果。",
                    better="我是資管系大三,系學會行銷把 IG 從 0 經營到 1200 追蹤,實習時用 SQL 把週報產出從 4 小時縮到 1 小時。",
                ),
                QuestionFeedbackDTO(
                    question=TURNS[1].question,
                    answer=TURNS[1].answer,
                    comment="情境與行動清楚,但沒有講結果,也沒有講學到什麼。",
                    better="補上一句:那次全推之後成效不如預期,後來我改成先小規模測試再放大。",
                ),
            ],
            star_parts=_star_parts(),
            improvements=[
                "自我介紹加 1-2 個量化成就",
                "講經歷一律補上 Result 段",
                "減少嗯與那個的使用",
            ],
            missing_points=[
                MissingPointDTO(
                    point="用 SQL 把週報產出時間從 4 小時縮短為 1 小時",
                    why="這份 JD 明列需具備 SQL 撰寫能力與自動化報表流程,這是最直接的對應證據,整場沒有提到。",
                ),
                MissingPointDTO(
                    point="協調行銷與工程兩邊的需求排程",
                    why="JD 要求與產品、工程團隊密切協作,這段可以證明你有跨部門經驗。",
                ),
            ],
            resume_grounded=True,
        ),
    )

    _dump(
        "res_report_group",
        ReportResponse(
            mode="group",
            face_dimensions=_face_dims(),
            sub_scores=_sub_scores(),
            question_feedbacks=[],
            star_parts=_star_parts(),
            collab_dims=[
                CollabDimDTO(name=COLLAB_DIM_NAMES[0], score=74, hint="有主動發言。可在開頭就先表態,搶到定錨位置。"),
                CollabDimDTO(name=COLLAB_DIM_NAMES[1], score=68, hint="較常各說各話。試著明確回應前一位。"),
                CollabDimDTO(name=COLLAB_DIM_NAMES[2], score=71, hint="論點清楚但偏短,可把推理講完整。"),
                CollabDimDTO(name=COLLAB_DIM_NAMES[3], score=70, hint="姿態不錯,但可適時收斂分歧、推進共識。"),
            ],
            improvements=["開場就先表態", "明確回應前一位發言者"],
            missing_points=[],
            resume_grounded=True,
        ),
    )

    # 降級案例:使用者還沒建立任何經歷
    _dump(
        "res_report_no_resume",
        ReportResponse(
            mode="single",
            face_dimensions=_face_dims(),
            sub_scores=_sub_scores(),
            question_feedbacks=[
                QuestionFeedbackDTO(
                    question=TURNS[0].question,
                    answer=TURNS[0].answer,
                    comment="兩段經歷都停在做過什麼,沒有帶出成果。",
                    # 無履歷時只能重組措辭,不得補任何數字
                    better="把結論放前面:我是資管系大三,主要專長是資料分析,接著再展開兩段經歷。",
                )
            ],
            star_parts=_star_parts(),
            improvements=["建立經歷資料後可取得漏講點分析"],
            missing_points=[],
            resume_grounded=False,
            notices=["尚未建立經歷,漏講點分析與成果補強暫不可用"],
        ),
    )


if __name__ == "__main__":
    print("產生 golden JSON:")
    build_start()
    build_turns()
    build_report()
    print(f"完成,輸出至 {OUT}")
