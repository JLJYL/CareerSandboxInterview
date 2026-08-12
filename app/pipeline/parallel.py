"""LLM 呼叫並行化工具。

【為什麼需要】
A4 報告有五個彼此獨立的 LLM 呼叫。序列執行的總時間是每一個的加總
(GPT-4o 約 3–8 秒,五個就是 15–40 秒);並行執行的總時間等於最慢的那一個。

原理:LLM 呼叫的時間幾乎全花在等待網路回應,那段時間本機 CPU 閒置。
asyncio 讓程式在等待期間先把其他請求也發出去。這只對等待型工作有效,
本地計算不會因此變快。

【與參考來源的差異】
手法借自 manthan89-py/AI-Interview-System 的 analyze_candidate.py,
但硬化了三處,原因見 docs/NOTES_reference_repo.md:

  1. 逐工作計時,而非整包計時。原版把 timeout 包在 gather 外層,
     一個慢就取消全部;這裡每個工作各自計時,慢的單獨逾時。
  2. return_exceptions=True。原版一個失敗會讓整包 gather 失敗,
     其餘成功的結果全部丟掉。
  3. 號誌限制同時數。interview_eval.py 要跑 10 份測試集 × 5 個區塊,
     不限制會撞上 OpenAI 的每分鐘上限。

【重要】
成本不變。並行不減少呼叫次數,省的只有等待時間。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_TIMEOUT = 45.0
"""單一工作的逾時秒數。設在 45 是因為 gpt-4o 產出 STAR 拆解
(需要跨行 verbatim 引用)實測偏慢,留餘裕。"""

DEFAULT_CONCURRENCY = 5
"""同時進行的呼叫數上限。一份報告五個區塊剛好不受限制;
批次評測時才會真正生效。"""


class BlockFailure:
    """某個區塊失敗的標記。

    不拋例外的理由:報告少一塊仍然可用,整份 500 就完全不可用。
    上層拿到這個物件時應該把 reason 寫進 ReportResponse.notices,
    再交給 repair_report() 把該區塊補成空值。
    """

    __slots__ = ("name", "reason")

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def __repr__(self) -> str:
        return f"BlockFailure({self.name!r}, {self.reason!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockFailure)
            and self.name == other.name
            and self.reason == other.reason
        )


async def _guarded(
    name: str,
    coro: Awaitable[T],
    timeout: float,
    sem: asyncio.Semaphore,
) -> tuple[str, T | BlockFailure]:
    """單一工作的保護層:號誌 + 逾時 + 例外攔截。

    wait_for 放在號誌內側,計時從實際開始執行才起算。
    放外側的話,排隊等號誌的時間會被算進逾時,批次執行時後面的工作
    會全部誤判為逾時。
    """
    async with sem:
        try:
            return name, await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return name, BlockFailure(name, f"逾時({timeout:.0f} 秒)")
        except Exception as exc:  # noqa: BLE001 — 這裡就是要攔全部
            return name, BlockFailure(name, f"{type(exc).__name__}: {exc}")


async def gather_blocks(
    blocks: dict[str, Awaitable[Any]],
    timeout: float = DEFAULT_TIMEOUT,
    max_concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[dict[str, Any], list[str]]:
    """並行執行多個具名工作,回傳 (成功結果, notices)。

    失敗的工作不會出現在結果字典裡,而是在 notices 留下一行說明。
    呼叫端用 dict.get(name, 預設值) 取值即可,不需要逐一 try。

    用法:

        results, notices = await gather_blocks({
            "face_dimensions": gen_face_dimensions(turns, stats),
            "sub_scores": gen_sub_scores(turns, stats),
            "question_feedbacks": gen_question_feedbacks(turns, resume, jd),
            "star_parts": gen_star_parts(turns),
            "missing_points": gen_missing_why(gaps, jd),
        })

        report = ReportResponse(
            mode=mode,
            face_dimensions=results.get("face_dimensions", []),
            ...
            notices=notices,
        )
    """
    if not blocks:
        return {}, []

    sem = asyncio.Semaphore(max_concurrency)
    pairs = await asyncio.gather(
        *(_guarded(name, coro, timeout, sem) for name, coro in blocks.items())
    )

    results: dict[str, Any] = {}
    notices: list[str] = []
    for name, value in pairs:
        if isinstance(value, BlockFailure):
            notices.append(f"{value.name} 產生失敗,已略過({value.reason})")
        else:
            results[name] = value
    return results, notices
