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

from collections.abc import Sequence

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
from app.pipeline.collab_score import score_collab
from app.pipeline.missing_points import generate_missing_points
from app.schemas.interview import (
    ReportRequest,
    ReportResponse,
    StartInterviewRequest,
    StartInterviewResponse,
    TurnDTO,
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


def _interviewer_names(mode: str, ctx) -> set[str]:
    """該場的面試官(主持人)顯示名稱。

    【為什麼可以用名稱查】
    合約說「speaker 是顯示名稱,不可用來判斷身分」——那是指不要在各處
    硬寫 "user"、"主考官" 這類字串。這裡查的是 personas_for(),
    也就是**定義那些名稱的同一份註冊表**:persona 改名會同時改到兩邊,
    不會出現名稱變了而判斷邏輯沒變的情況。

    查不到的名稱視為同儕。那是保守方向:誤判成同儕只是回到加這一欄
    之前的行為,誤判成面試官會讓真正的同儕發言從配對裡消失。
    """
    from app.prompts.interview_personas import personas_for

    return {
        p.display_name
        for p in personas_for(mode, ctx.group_interviewers, ctx.group_size)
        if p.display_name and p.role != "peer"
    }


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

    # 【三份文字,用途不同】
    #
    # transcript     給 LLM 讀。純內容,不含段界換行——段界對 LLM 沒有意義,
    #                而且會讓它以為那是句子邊界。
    # user_text      給 TextStats 與 GapComputer。**使用者講的全部內容**,
    #                段界用換行表示。
    #
    # 群面時使用者的發言在 groupSays 裡,turns 可能只有主問題的簡短回答。
    # 這件事對 TextStats 與 gap 同樣成立,兩邊都要用 groupSays。
    #
    # 【為什麼 gap 也用帶換行的版本】
    # 實測換行不影響 mentioned_skills:正常分段的偵測結果與同段完全一樣。
    # 詞真的被切斷時抓不到(SQ\nL → 空),但那是 Whisper 分段切在詞中間造成的,
    # 換成空白接一樣會斷。所以兩邊共用同一份,不需要兩個版本。
    transcript = "\n".join(t.answer for t in req.turns)

    user_text = build_stats_text(req.turns)
    if req.mode == "group" and req.group_says:
        group_text = build_group_text(req.group_says)
        if group_text.strip():
            user_text = group_text

    input_mode = _dominant_input_mode(req)
    stats = get_analyzer().text_stats(user_text)

    report = await generate_report(
        mode=req.mode,
        turns=req.turns,
        stats=stats,
        input_mode=input_mode,
        resume_text=_resume_text(req),
        resume_grounded=bool(req.experiences),
        llm=get_llm(),
        llm_verbatim=get_llm_verbatim(),
        engine=req.context.transcription_engine,
    )

    # A5:履歷為空時 GapComputer 回空清單,generate_missing_points 也不會呼叫 LLM。
    # 那是正常情況——使用者還沒建經歷,不是錯誤。
    jd, jd_notices = _resolve_jd(req.context.custom_jd, get_llm())
    if jd is None:
        jd = JDInput(description=req.context.custom_jd, source="extracted")
    # gap = 履歷 ∩ JD − 提及。提及漏掉會讓 gap 變大,
    # 系統對使用者說「你漏講了 SQL」而他在討論裡明明講了——
    # 那是假指控,不是少給建議,而且靜默:報告照樣產出、測試照樣全綠。
    #
    # 早期版本用 transcript(只含 turns.answer),群面時使用者在討論裡
    # 講的東西完全不在裡面。實測:提及集從 3 個技能變成空。
    candidates = get_gap().compute(
        [e.model_dump() for e in req.experiences], jd, user_text
    )
    points, mp_notices = generate_missing_points(
        candidates, user_text, req.context.custom_jd, get_llm()
    )
    report.missing_points = points
    report.notices = list(report.notices) + jd_notices + mp_notices

    # 協作維度:僅群面。A 切片,B 的 LLM 逐維度對照 BARS 評。
    #
    # 四次獨立呼叫不是浪費。rubric 第八節第 2 點實測:一次讀完整段吐四個分數,
    # 維度間相關 r̄ ≈ .86–.92——那個相關高到四個分數等於同一個分數。
    # 分開評掉到 .26–.35,接近人類評審的 .34。
    if req.mode == "group" and req.group_says:
        try:
            hosts = _interviewer_names(req.mode, req.context)
            slices = get_collab().observe(
                [
                    Utterance(
                        speaker_id=u.speaker,
                        text=u.content,
                        is_user=u.is_user,
                        is_interviewer=not u.is_user and u.speaker in hosts,
                        start_ms=u.start_ms or (u.segment_starts_ms[0] if u.segment_starts_ms else 0),
                        end_ms=u.end_ms,
                    )
                    for u in req.group_says
                ]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("協作切片失敗: %s", exc)
            report.notices.append(
                f"協作切片失敗({type(exc).__name__}),本次不顯示協作區塊"
            )
        else:
            dims, collab_notices = await score_collab(
                slices, get_llm(), "\n".join(u.content for u in req.group_says)
            )
            report.collab_dims = dims
            report.notices += collab_notices

    # 這一行必須在最後:repair_report 會在 resume_grounded=False 時清空
    # missing_points,而上面才剛把它填進去。
    from app.schemas.interview_repair import repair_report

    return repair_report(report)


def build_stats_text(turns: Sequence[TurnDTO]) -> str:
    """組給 TranscriptAnalyzer 的文字。段界用換行表示。

    【為什麼不能只用 answer】
    analyzer 的 _segment() 看的是換行:有換行就是 stt_segment(實測的段界),
    沒有就退回 discourse_marker(以語氣詞估算)。

    "\n".join(t.answer ...) 只在**題與題之間**加換行,段界完全沒進去,
    所以 segmentation 一直是估算值。實測差別:
        只用 answer     discourse_marker  平均段長 34.5
        改讀 segments   stt_segment       平均段長 13.8

    差的不只是數字。估算值不能講成量出來的,
    prompt 裡的「平均每段 N 字」對 discourse_marker 是不成立的宣稱。

    【退回那條要留著】
    answer_segments 為空是常態,不是錯誤:
        前端還沒接完之前一直是空的
        之後也可能遇到真的沒有分段資料的來源(貼上的文字、匯入的逐字稿)
    退回之後 segmentation 是 discourse_marker,那是誠實的降級。
    """
    parts: list[str] = []
    for t in turns:
        segs = [x for x in (t.answer_segments or []) if x.strip()]
        parts.append("\n".join(segs) if segs else t.answer)
    return "\n".join(parts)


def build_group_text(says: Sequence) -> str:
    """群面的逐字稿。同樣優先用 segments。

    只取使用者的發言——TextStats 評的是使用者,
    把 AI 同儕的話算進填充詞率與段長會讓數字失去意義。
    """
    parts: list[str] = []
    for u in says:
        if not u.is_user:
            continue
        segs = [x for x in (u.segments or []) if x.strip()]
        parts.append("\n".join(segs) if segs else u.content)
    return "\n".join(parts)


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
