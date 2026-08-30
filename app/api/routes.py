"""面試模組的三個 HTTP 端點。

    POST /interviews                  開場:主問題、備援追問、搶話台詞(群面)
    POST /interviews/{id}/turns       每輪:接話
    POST /interviews/{id}/report      賽後:A4 六件 + A5 漏講點

【五個接點怎麼對應到三個端點】
    A1 一對一   turns(mode=single)
    A2 panel    turns(mode=panel)
    A3 群面     turns(mode=group)+ 開場帶搶話台詞
    A4 報告     report
    A5 漏講點   report

A1/A2/A3 輸入輸出形狀相同,只有 persona 與派發不同,所以共用一個端點。
A4/A5 吃的輸入完全相同(逐字稿 + JD + 履歷),拆兩個端點等於同一份資料送兩趟。

【這個服務沒有資料庫】
履歷由前端在 request body 帶入,session 狀態歸資料庫組的
Interview_Sessions / Interview_Turns。所以 session_id 目前只是識別字串,
後端不查也不存——路徑參數收下來原樣回傳,等資料庫組接上再改。

【錯誤處理的原則】
面試進行中噴 500 等於整場中斷。所有 pipeline 都設計成降級不拋例外,
這一層只處理輸入驗證失敗(那由 Pydantic 擋)與真正無法預期的例外。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path

from app.api.deps import (
    get_collab,
    get_gap,
    get_llm,
    get_llm_live,
    get_llm_verbatim,
    get_analyzer,
)
from app.contracts.interview_protocols import JDInput, Utterance
from app.pipeline.interview_live import next_turn, start_interview
from app.pipeline.interview_report import generate_report
from app.pipeline.jd_extract import extract_jd
from app.pipeline.missing_points import generate_missing_points
from app.schemas.interview import (
    CollabDimDTO,
    ReportRequest,
    ReportResponse,
    StartInterviewRequest,
    StartInterviewResponse,
    TurnRequest,
    TurnResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/interviews", tags=["interview"])

SESSION_ID = Path(..., min_length=1, max_length=64, description="開場時回傳的 sessionId")


def _resolve_jd(custom_jd: str, llm) -> tuple[JDInput | None, list[str]]:
    """把使用者手貼的散文 JD 抽成結構化需求。

    【為什麼要這一步】
    面試的 JD 是使用者在設定頁自己打的自由文字,而且是選填欄位。
    成員 A 實測:1,010 筆同時具備 requiredSkills 與 description 的 JD 中,
    5,988 條 requiredSkills 只有 5.5% 字面出現在散文裡。
    直接把散文交給下游,等於要它從一份幾乎不提需求的文章裡反推需求。

    JD 為空時回 None——不要硬跑抽取器,那只會產生幻覺。
    抽取失敗也回 None,下游會退回只用職稱與產業出題。
    """
    if not custom_jd.strip():
        return None, []
    try:
        result = extract_jd(custom_jd, llm)
    except Exception as exc:  # noqa: BLE001
        log.warning("JD 抽取失敗: %s", exc)
        return None, [f"JD 抽取失敗({type(exc).__name__}),改用原文出題"]
    return result.jd, list(result.notices)


@router.post("", response_model=StartInterviewResponse, summary="開場")
async def create_interview(req: StartInterviewRequest) -> StartInterviewResponse:
    jd, notices = _resolve_jd(req.context.custom_jd, get_llm())
    resp = await start_interview(
        mode=req.mode, context=req.context, llm=get_llm_live(), jd=jd
    )
    resp.notices = notices + list(resp.notices)
    return resp


@router.post("/{session_id}/turns", response_model=TurnResponse, summary="每輪接話")
async def submit_turn(req: TurnRequest, session_id: str = SESSION_ID) -> TurnResponse:
    """一輪接話。

    session_id 目前不使用——狀態全部由前端在 request 裡帶來
    (question、askedTopics、spokenBy、followUpIdx、context)。
    資料庫組的 Interview_Turns 接上之後,這些欄位可以改成後端自己查,
    合約不用改,前端也不用改。
    """
    return await next_turn(
        mode=_mode_from_request(req),
        answer=req.answer,
        follow_up_idx=req.follow_up_idx,
        asked_questions=req.asked_topics,
        spoken_by=req.spoken_by,
        question=req.question,
        prev_topic=req.asked_topics[-1] if req.asked_topics else "",
        ended_by=req.ended_by,
        fallback=req.fallback,
        llm=get_llm_live(),
        context=req.context,
    )


def _mode_from_request(req: TurnRequest) -> str:
    """決定這一輪的模式。

    前端帶了 mode 就用它。沒帶時從 spokenBy 反推——那是過渡做法,
    有兩個已知的失準點:

        第一輪 spokenBy 還是空的,一律當成 single
        一對一不回 speaker,所以要靠開場的 openingSpeaker 被放進 spokenBy
        才分得出 panel 與 group

    群面的名單依 context 的 groupInterviewers 與 groupSize 而變,
    所以反推時要帶那兩個參數,否則 3 位主管的配置會對不上。
    """
    if req.mode:
        return req.mode

    from app.prompts.interview_personas import speaker_names

    seen = set(req.spoken_by)
    if not seen:
        return "single"
    gi, gs = req.context.group_interviewers, req.context.group_size
    if seen & set(speaker_names("group", gi, gs)):
        return "group"
    if seen & set(speaker_names("panel")):
        return "panel"
    return "single"


@router.post("/{session_id}/report", response_model=ReportResponse, summary="賽後報告")
async def create_report(req: ReportRequest, session_id: str = SESSION_ID) -> ReportResponse:
    """A4 六件 + A5 漏講點。

    兩者合併成一個端點的理由:吃的輸入完全相同,拆開等於同一份資料送兩趟、
    LLM 跑兩趟。前端拿一份 response 餵兩個 provider。
    """
    if not req.turns:
        raise HTTPException(status_code=422, detail="turns 不可為空,沒有逐字稿就沒有報告")

    transcript = "\n".join(t.answer for t in req.turns)
    input_mode = _dominant_input_mode(req)
    stats = get_analyzer().text_stats(transcript)

    report = await generate_report(
        mode=req.mode,
        turns=req.turns,
        stats=stats,
        input_mode=input_mode,
        resume_text=_resume_text(req),
        resume_grounded=bool(req.experiences),
        llm=get_llm(),
        llm_verbatim=get_llm_verbatim(),
    )

    # A5:履歷為空時 GapComputer 回空清單,generate_missing_points 也不會呼叫 LLM。
    # 那是正常情況——使用者還沒建經歷,不是錯誤。
    jd, jd_notices = _resolve_jd(req.context.custom_jd, get_llm())
    if jd is None:
        jd = JDInput(description=req.context.custom_jd, source="extracted")
    candidates = get_gap().compute(
        [e.model_dump() for e in req.experiences], jd, transcript
    )
    points, mp_notices = generate_missing_points(
        candidates, transcript, req.context.custom_jd, get_llm()
    )
    report.missing_points = points
    report.notices = list(report.notices) + jd_notices + mp_notices

    # 協作訊號:僅群面。目前只抽可觀察值,等第等後端組的 BARS rubric。
    # level 全是 None 時 repair_report 會清空 collab_dims 並記入 notices,
    # 前端不顯示協作區塊——那比補四個 0 分好,0 分會讓使用者以為自己拿了零分。
    if req.mode == "group" and req.group_says:
        try:
            signals = get_collab().observe(
                [
                    Utterance(
                        speaker_id=u.speaker,
                        text=u.content,
                        start_ms=u.start_ms or (u.segment_starts_ms[0] if u.segment_starts_ms else 0),
                        end_ms=u.end_ms,
                    )
                    for u in req.group_says
                ]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("協作訊號抽取失敗: %s", exc)
            report.notices.append(f"協作訊號抽取失敗({type(exc).__name__}),本次不顯示協作區塊")
        else:
            scored = [s for s in signals if s.level is not None]
            if scored:
                report.collab_dims = [
                    CollabDimDTO(name=s.name, score=int(s.level), hint=s.evidence)
                    for s in scored
                ]
            else:
                report.notices.append(
                    f"協作訊號已抽取({len(signals)} 項),但 BARS rubric 尚未到位,無法指派等第"
                )

    # 這一行必須在最後:repair_report 會在 resume_grounded=False 時清空
    # missing_points,而上面才剛把它填進去。
    from app.schemas.interview_repair import repair_report

    return repair_report(report)


def _dominant_input_mode(req: ReportRequest) -> str:
    """整場的輸入方式。混用時取多數,平手或全部未知時回 unknown。

    保守方向:未知比照打字處理——不確定時不要宣稱評估了口語表現。
    打字的答案沒有填充詞、標點乾淨,會拿到高流暢度,
    那在面試模擬器裡是在獎勵錯誤的行為。
    """
    modes = [t.input_mode for t in req.turns if t.input_mode]
    if not modes:
        return "unknown"
    voice = modes.count("voice")
    return "voice" if voice * 2 > len(modes) else "unknown"


def _resume_text(req: ReportRequest) -> str:
    lines = []
    for e in req.experiences:
        bits = [e.title, e.role, e.action, e.result]
        tags = "、".join(e.tags)
        lines.append("・" + " / ".join(b for b in bits if b) + (f"(標籤:{tags})" if tags else ""))
    return "\n".join(lines)
