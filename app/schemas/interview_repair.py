"""固定值域的機械修復層。

【為什麼需要這一層】
subScores 六個名稱、starParts 四個鍵、collabDims 四個名稱、faceDimensions
三個 letter 都是固定值域,前端 UI 照名稱排版,少一項就破版。

但這些值有 LLM 產出成分。把它們寫成 Literal 會在 LLM 回「邏輯清楚度」
(少一個「晰」)時直接噴 500。CareerCategory 事故就是這樣來的。

正確做法是:型別放開,產出後過一道機械修復——缺的補上、多的丟掉、
順序歸位、名稱對不上的用位置對齊。修復發生時記錄到 notices,
讓問題看得見,但不讓使用者看到 500。

【呼叫時機】
在 pipeline 組完 ReportResponse 之後、回傳之前呼叫 repair_report()。
"""

from __future__ import annotations

from app.schemas.interview import (
    COLLAB_DIM_NAMES,
    FACE_DIMENSIONS,
    STAR_PARTS,
    SUB_SCORE_NAMES,
    CollabDimDTO,
    FaceDimensionDTO,
    ReportResponse,
    StarPartDTO,
    SubScoreDTO,
)


def repair_sub_scores(items: list[SubScoreDTO]) -> tuple[list[SubScoreDTO], list[str]]:
    """六項固定名稱,缺補 0、多丟棄、順序歸位。

    對不上名稱時用位置對齊:LLM 常寫成「邏輯清楚度」這種近義變體,
    位置通常是對的,直接改名比丟掉整項好。
    """
    notices: list[str] = []
    by_name = {i.name: i for i in items}
    out: list[SubScoreDTO] = []

    for idx, name in enumerate(SUB_SCORE_NAMES):
        if name in by_name:
            out.append(SubScoreDTO(name=name, score=by_name[name].score))
        elif idx < len(items):
            # 位置對齊救援
            out.append(SubScoreDTO(name=name, score=items[idx].score))
            notices.append(f"subScores 第 {idx + 1} 項名稱不符,已依位置歸位為「{name}」")
        else:
            out.append(SubScoreDTO(name=name, score=0))
            notices.append(f"subScores 缺少「{name}」,已補 0")

    extra = len(items) - len(SUB_SCORE_NAMES)
    if extra > 0:
        notices.append(f"subScores 多出 {extra} 項,已丟棄")
    return out, notices


def repair_star_parts(items: list[StarPartDTO]) -> tuple[list[StarPartDTO], list[str]]:
    """S/T/A/R 四段,缺的補成 present=False。"""
    notices: list[str] = []
    by_key = {i.key.strip().upper(): i for i in items}
    out: list[StarPartDTO] = []

    for key, name in STAR_PARTS:
        src = by_key.get(key)
        if src is None:
            out.append(
                StarPartDTO(key=key, name=name, present=False, from_answer="", hint="這段沒有講到")
            )
            notices.append(f"starParts 缺少「{key}」,已補成未命中")
            continue
        out.append(
            StarPartDTO(
                key=key,
                name=name,  # 名稱一律以常數為準,不採用 LLM 版本
                present=src.present,
                from_answer=src.from_answer if src.present else "",
                hint=src.hint,
            )
        )
    return out, notices


def repair_collab_dims(items: list[CollabDimDTO]) -> tuple[list[CollabDimDTO], list[str]]:
    """協作四項,缺補 0、順序歸位。"""
    notices: list[str] = []
    by_name = {i.name: i for i in items}
    out: list[CollabDimDTO] = []

    for idx, name in enumerate(COLLAB_DIM_NAMES):
        if name in by_name:
            src = by_name[name]
            out.append(CollabDimDTO(name=name, score=src.score, hint=src.hint))
        elif idx < len(items):
            out.append(CollabDimDTO(name=name, score=items[idx].score, hint=items[idx].hint))
            notices.append(f"collabDims 第 {idx + 1} 項名稱不符,已依位置歸位為「{name}」")
        else:
            out.append(CollabDimDTO(name=name, score=0, hint=""))
            notices.append(f"collabDims 缺少「{name}」,已補 0")
    return out, notices


def repair_face_dimensions(
    items: list[FaceDimensionDTO],
) -> tuple[list[FaceDimensionDTO], list[str]]:
    """三個面向,letter 與 name 一律以常數為準,順序歸位。"""
    notices: list[str] = []
    by_letter = {i.letter: i for i in items}
    out: list[FaceDimensionDTO] = []

    for idx, (letter, name) in enumerate(FACE_DIMENSIONS):
        src = by_letter.get(letter)
        if src is None and idx < len(items):
            src = items[idx]
            notices.append(f"faceDimensions 第 {idx + 1} 項標記不符,已依位置歸位為「{letter}」")
        if src is None:
            out.append(
                FaceDimensionDTO(letter=letter, name=name, score=0, verdict="", points=[])
            )
            notices.append(f"faceDimensions 缺少「{name}」,已補 0")
            continue
        out.append(
            FaceDimensionDTO(
                letter=letter,
                name=name,
                score=src.score,
                verdict=src.verdict,
                points=src.points,
                # prosody 只允許出現在「表達」面向
                prosody=src.prosody if letter == "達" else None,
            )
        )
    return out, notices


def repair_report(report: ReportResponse) -> ReportResponse:
    """組完報告後的最後一道。所有修復動作都會寫進 notices。"""
    notices = list(report.notices)

    report.face_dimensions, n = repair_face_dimensions(report.face_dimensions)
    notices += n
    report.sub_scores, n = repair_sub_scores(report.sub_scores)
    notices += n
    report.star_parts, n = repair_star_parts(report.star_parts)
    notices += n

    if report.mode == "group":
        report.collab_dims, n = repair_collab_dims(report.collab_dims)
        notices += n
    elif report.collab_dims:
        report.collab_dims = []
        notices.append("非團體面試,已清空 collabDims")

    # 本期不做影像,恆空
    if report.video_dims:
        report.video_dims = []
        notices.append("本期未啟用影像面試,已清空 videoDims")

    # 沒有履歷就不可能有漏講點,也不可以讓 better 補事實
    if not report.resume_grounded:
        if report.missing_points:
            report.missing_points = []
            notices.append("無履歷資料,已清空 missingPoints")
        notices.append("尚未建立經歷,漏講點分析與成果補強暫不可用")

    report.notices = notices
    return report


def verify_star_verbatim(report: ReportResponse, turns_text: str) -> list[str]:
    """機械檢查:from_answer 必須是逐字稿的子字串。

    回傳違規訊息清單,空清單代表通過。這一項不做自動修復——
    引用不存在的原文是嚴重錯誤,應該讓測試紅燈而不是默默吞掉。
    """
    problems: list[str] = []
    normalized = turns_text.replace(" ", "").replace("\n", "")
    for part in report.star_parts:
        if not part.present or not part.from_answer:
            continue
        needle = part.from_answer.strip("「」\"' ").replace(" ", "")
        if needle and needle not in normalized:
            problems.append(f"starParts[{part.key}].fromAnswer 不存在於逐字稿:{needle[:40]}")
    return problems
