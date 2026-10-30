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
from collections import OrderedDict

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
from app.pipeline.interview_live import next_turn, resume_block, start_interview
from app.pipeline.interview_report import generate_report
from app.pipeline.interviewer_spec import derive_specs
from app.pipeline.turn_decision import decide_turn
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

SESSION_ID = Path(
    ..., min_length=1, max_length=64, description="開場時回傳的 sessionId"
)

_JD_CACHE: "OrderedDict[str, tuple[JDInput | None, list[str]]]" = OrderedDict()
_JD_CACHE_MAX = 64
"""JD 抽取的行程內快取,鍵是使用者貼的原文。

【為什麼需要】
每輪接話要職缺需求才算得出「這個缺口值不值得追」。前端有帶 jdSkills 的時候
不用抽;沒帶的時候只能從 `context.custom_jd` 現抽,而那是**每一輪一次呼叫**。
同一場面試五輪就是五次,而五次的答案一模一樣——抽取用的是 temp 0 的模型。

只存**抽到東西**的結果。`extract_jd` 把呼叫例外吃在裡面、回一個空的 JDInput,
所以「失敗」跟「這份 JD 真的沒有可抽的需求」長得一樣——照單全收的話
一次逾時會變成那份 JD 的永久答案。空結果重算不花錢:內容過短那條路
在打 API 之前就短路了。上限 64 筆:這是行程內的東西,不設限只會長大。
"""


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
    text = custom_jd.strip()
    if not text:
        return None, []
    hit = _JD_CACHE.get(text)
    if hit is not None:
        return hit
    try:
        result = extract_jd(custom_jd, llm)
    except Exception as exc:  # noqa: BLE001
        log.warning("JD 抽取失敗: %s", exc)
        # 失敗不進快取。存起來的話一次逾時會變成這份 JD 的永久答案,
        # 而那不像故障——使用者重開一場還是沒有職缺需求。
        return None, [f"JD 抽取失敗({type(exc).__name__}),改用原文出題"]
    out = (result.jd, list(result.notices))
    # ★ 只快取抽到東西的。`extract_jd` 自己吃掉呼叫例外,回一個空的 JDInput
    #   加一條 notice——所以上面那個 except 根本不會觸發,失敗是以「成功但空的」
    #   形式回來的。照單全收的話一次逾時會變成那份 JD 的永久答案。
    #   空結果重算的成本是零:內容過短那條路在打 API 之前就短路了。
    #   跟 `AdjacentCache.put` 拒收降級結果是同一條規則。
    if out[0] is not None and out[0].required_skills:
        _JD_CACHE[text] = out
        while len(_JD_CACHE) > _JD_CACHE_MAX:
            _JD_CACHE.pop(next(iter(_JD_CACHE)))
    return out


def clear_jd_cache() -> None:
    """清掉 JD 抽取快取。測試用。

    不清的話同一份 JD 文字在不同測試之間共用結果,
    而「抽取失敗會怎樣」那種測試會因為快取是熱的而測不到。
    """
    _JD_CACHE.clear()


@router.post("", response_model=StartInterviewResponse, summary="開場")
async def create_interview(req: StartInterviewRequest) -> StartInterviewResponse:
    jd, notices = _resolve_jd(req.context.custom_jd, get_llm())
    resp = await start_interview(
        mode=req.mode,
        context=req.context,
        llm=get_llm_live(),
        # 競爭者的相鄰技能挑選走這一支。那是分類不是生成,temp 0.4 會讓
        # 同一個職缺每次挑出不同的技能池,而那個變化會疊在 seed 上面。
        llm_exact=get_llm(),
        jd=jd,
        # 欄位一直都在,只是開場那支從來沒讀過它——三處使用全在報告那條路徑。
        # 使用者在應徵頁選了哪一版履歷,開場題就該跟著那一版走。
        experiences=req.experiences,
    )
    resp.notices = notices + list(resp.notices)
    resp.jd_skills = list(jd.required_skills) if jd else []
    return resp


@router.post("/{session_id}/turns", response_model=TurnResponse, summary="每輪接話")
async def submit_turn(req: TurnRequest, session_id: str = SESSION_ID) -> TurnResponse:
    """一輪接話。

    session_id 目前不使用——狀態全部由前端在 request 裡帶來
    (question、askedTopics、spokenBy、followUpIdx、context)。
    資料庫組的 Interview_Turns 接上之後,這些欄位可以改成後端自己查,
    合約不用改,前端也不用改。
    """
    mode = _mode_from_request(req)
    decision = await _decide(req, mode)
    resp = await next_turn(
        mode=mode,
        decision=decision,
        answer=req.answer,
        follow_up_idx=req.follow_up_idx,
        # 問句原文只給查重,領域標籤只給 prompt。兩者混在一起會讓查重
        # 誤判——見 `next_turn` 的 asked_topics 那段說明。
        asked_questions=[q for q in req.asked_questions if str(q or '').strip()],
        asked_topics=[t for t in req.asked_topics if str(t or '').strip()],
        spoken_by=req.spoken_by,
        question=req.question,
        prev_topic=req.asked_topics[-1] if req.asked_topics else "",
        ended_by=req.ended_by,
        fallback=req.fallback,
        llm=get_llm_live(),
        context=req.context,
        # 職缺那一段跟履歷那一段的材料。原本每輪都沒帶,所以
        # context_block 會寫「使用者沒有提供 JD」而 _turn_jd 早就抽出來了。
        jd=_turn_jd(req),
        experiences=req.experiences,
        group_says=req.group_says,
    )
    # 追問深度的狀態由前端保管,後端只算下一步該是多少——跟 askedTopics
    # 與 followUpIdx 同一個模式。前端不解讀這兩個值,原樣送回就好。
    resp.probe_gap, resp.probe_depth = _next_probe(req, decision)

    # 前端沒帶問句原文的時候要講出來。
    #
    # 【這條接線錯過一次,而且錯了兩種方式】
    # 原本是 `asked_questions=req.asked_topics`——領域標籤不是問句原文,
    # 而 `compose_probe_rules` 的註解早就寫明傳的該是原文。兩個欄位都是
    # list[str],所以接錯沒有任何東西擋得住。
    #
    # 修第一次的時候以為降級是「查重形同關閉」,所以留了退回標籤那條路。
    # 那個判斷也是錯的:標籤被問句整個包住時 `_overlapping` 的比例衝到 1.0,
    # 是**誤判**不是失效,模型的新主問題會被換成通用的備援題。
    # 所以現在標籤只進 prompt,不進查重,降級才真的是關閉。
    if req.asked_topics and not any(str(q or "").strip() for q in req.asked_questions):
        resp.notices.append(
            "本輪:前端未帶 askedQuestions,防重複改用領域標籤比對,"
            "重複的主問題擋不下來"
        )
    return resp


async def _decide(req: TurnRequest, mode: str):
    """這一輪要做什麼。決定在這一層算,不在 next_turn 裡發起。

    【決定怎麼來的】
    問模型這段話裡有什麼(三次並行),再用 `AcceptRule.accepts()` 算要不要追問。
    一輪三次呼叫,2d 實測中位數 1.54s、p90 2.04s(concurrency=6,含排隊)。

    舊路是把放行條件寫進 prompt 問模型挑動作。1d 量到三種難度的追問率都是
    0.500,2d 量到換成程式算之後是 0.143/0.429/0.857、0/630 次降級,
    所以舊路在 2d 之後刪掉了。

    【為什麼 jd 是 None】
    結構化 JD 目前只在開場抽一次(`_resolve_jd`),每輪的請求只帶散文
    `context.customJd`。每輪重抽一次要多一次呼叫,而 JD 整場不會變。

    所以這一版的決定只依 STAR 與放行條件判斷,不看職缺需求。
    facts 路徑本來就不把職缺需求帶進判讀——判讀問的是「這段話裡有什麼」,
    跟職缺要什麼無關,所以 `has_jd()` 為 False 在這條路上不會讓模型腦補。

    【職缺需求從哪裡來】
    兩條路,見 `_turn_jd`:前端帶 `jdSkills`(開場回傳的或型錄的結構化欄位),
    或退回用 `context.custom_jd` 現抽並走行程內快取。
    兩條都沒有才是 None,那時候只能看 STAR 四段判缺口。

    【追問深度從哪裡來】
    `probeDepth` 由前端保管、每輪帶回來,跟 askedTopics 與 followUpIdx 同一個模式。
    這支服務無狀態,所以跨輪的計數只能這樣走。不拿 followUpIdx 頂替——
    那是整場第幾題,跟「同一個缺口追過幾次」是不同的東西。
    """
    # 群面不做這個決定。它判的是「這段回答的 STAR 缺什麼、要不要追問」,
    # 那是面試官評估一個人的問法;群面的接話者多半是同場的競爭者。
    # 決定一旦傳下去,生成端會收到「接著他剛剛那段追問」,蓋過競爭者
    # 「先表態再發問」的人格——group-1(10/29)就是這樣變成三位追問者。
    # 群面的接話規則見 GROUP_TURN_RULES。順帶省下每輪讀事實的三到六次呼叫。
    if mode == "group":
        return None
    specs = derive_specs(
        mode,
        jd=_turn_jd(req),
        difficulty=req.context.difficulty,
        group_interviewers=req.context.group_interviewers,
        group_size=req.context.group_size,
    )
    if not specs:
        return None
    return await decide_turn(
        spec=specs[0],
        answer=req.answer,
        llm=get_llm_live(),
        question=req.question,
        depth=req.probe_depth,
        ended_by=req.ended_by,
        transcription_engine=req.context.transcription_engine,
    )


def _turn_jd(req: TurnRequest) -> JDInput | None:
    """這一輪的職缺需求。兩條來源,前端帶的優先。

        req.jd_skills          開場回傳的原樣帶回,或前端自己的結構化職缺
        context.custom_jd      沒帶時現抽,走 `_JD_CACHE`,同一份文字只抽一次

    前端帶的優先,因為它可能是型錄的結構化欄位——那比從散文抽的準。
    兩條都沒有就回 None,下游退回只看 STAR 四段判缺口。

    `source` 標成 catalog 只在前端帶的時候:那一批確實不是抽取器產的,
    而合約要求評測報表把兩種來源分開統計。
    """
    skills = [s for s in req.jd_skills if str(s or "").strip()]
    if skills:
        return JDInput(required_skills=skills, source="catalog")
    jd, _notices = _resolve_jd(req.context.custom_jd, get_llm())
    return jd


def _next_probe(req: TurnRequest, decision) -> tuple[str, int]:
    """下一輪要帶回來的 (缺口, 已追次數)。

    【為什麼歸零的條件是這三個】
        換題          這個缺口的追問結束了
        缺口為空      被切斷那條路刻意不帶 gap,而「被打斷不是他的問題」,
                      不該算進上限——`forced_decision` 把它排在深度檢查前面
                      就是這個意思,這裡要跟它一致
        換了缺口      上限是「同一個缺口」的上限,換了就重新算

    計數的定義:送進去的 depth 是「這個缺口已經追過幾次」,
    所以 max_depth=2 會讓同一個缺口被追兩次,第三輪強制換題。
    """
    if decision is None or decision.action == "NEXT" or not decision.gap:
        return "", 0
    if decision.gap == req.probe_gap:
        return decision.gap, req.probe_depth + 1
    return decision.gap, 1


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
async def create_report(
    req: ReportRequest, session_id: str = SESSION_ID
) -> ReportResponse:
    """A4 六件 + A5 漏講點。

    兩者合併成一個端點的理由:吃的輸入完全相同,拆開等於同一份資料送兩趟、
    LLM 跑兩趟。前端拿一份 response 餵兩個 provider。
    """
    if not req.turns:
        raise HTTPException(
            status_code=422, detail="turns 不可為空,沒有逐字稿就沒有報告"
        )

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
    #
    # 職缺需求先用前端帶來的 jdSkills,跟面試那幾輪同一份;沒帶才從散文抽。
    # 見 `ReportRequest.jd_skills`。
    skills = [s for s in req.jd_skills if str(s or "").strip()]
    if skills:
        jd = JDInput(required_skills=skills, description=req.context.custom_jd, source="catalog")
        jd_notices: list[str] = []
    else:
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
    # 履歷側只吃 tags(gap.py:履歷側寧缺勿濫)。有經歷但一筆 tags 都沒有的時候
    # 漏講點必然是空的,而那跟「比過了、沒有漏講」在畫面上一模一樣。
    if req.experiences and not any(
        str(t or "").strip() for e in req.experiences for t in e.tags
    ):
        mp_notices = list(mp_notices) + [
            "漏講點:履歷的經歷都沒有標籤(tags),比對不出來"
        ]
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
                        start_ms=u.start_ms
                        or (u.segment_starts_ms[0] if u.segment_starts_ms else 0),
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
    """報告那條路的履歷文字。格式與開場共用同一支,見 `resume_block`。

    原本這裡自己攤一份,而開場那邊根本沒用到履歷。開場接上之後
    兩邊各留一份的話格式會分岔,而分岔的方向通常是「其中一邊忘了
    帶某個欄位」,然後沒有人發現。
    """
    return resume_block(req.experiences)
