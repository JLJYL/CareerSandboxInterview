"""協作維度評分:切片 → 四次獨立 LLM 呼叫 → CollabDimDTO。

【流程】
    A 的 CollabObserver  →  list[CollabSlice]   四份切片,沒有數字
            ↓
    四次獨立呼叫,並行。每次只看一份切片、一份 BARS
            ↓
    等第 1–4 → 0–100,level 0 代表未評分不進報告

【為什麼是四次而不是一次】
rubric 第八節第 2 點的實測:一次讀完整段吐四個分數,
維度間相關 r̄ ≈ .86–.92——那個相關高到四個分數等於同一個分數。
分開評掉到 .26–.35,接近人類評審的 .34。

四次呼叫的成本換四個真的不一樣的分數。合併省下的三次呼叫
換來的是一份看起來有四個維度、實際上只有一個維度的報告。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from app.contracts.interview_protocols import CollabSlice
from app.pipeline.jd_normalize import normalize_locale
from app.pipeline.text_fold import contains
from app.pipeline.parallel import gather_blocks
from app.prompts.collab_rubric import (
    LEVEL_TO_SCORE,
    MAX_LEVEL,
    MIN_LEVEL,
    compose_collab_prompt,
)
from app.schemas.interview import CollabDimDTO

LLMCall = Callable[[str, str], str]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

MAX_EXCERPTS = 40
"""單一維度最多送幾則片段。

長討論會讓切片很大,而 BARS 的判斷只需要看得出模式,不需要看完全部。
超過就取前 N 則並在 hint 註明——截斷要看得見,不要靜默丟掉。
"""


def _parse(raw: str) -> dict:
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i : j + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def _score_one(llm: LLMCall, sl: CollabSlice) -> dict:
    import asyncio

    has = bool(sl.excerpts)
    excerpts = sl.excerpts[:MAX_EXCERPTS]
    body = ["【切片】"] + [f"{i}. {x}" for i, x in enumerate(excerpts, 1)]
    if sl.note:
        body += ["", f"【切片說明】{sl.note}"]
    if len(sl.excerpts) > MAX_EXCERPTS:
        body += ["", f"(共 {len(sl.excerpts)} 則,只顯示前 {MAX_EXCERPTS} 則)"]

    raw = await asyncio.to_thread(
        llm, compose_collab_prompt(sl.name, has_excerpts=has), "\n".join(body)
    )
    return _parse(raw)


def _to_dim(name: str, data: dict) -> tuple[CollabDimDTO | None, str | None]:
    """把單一維度的裁決轉成 DTO。回傳 (DTO, 問題說明)。

    level 0 或缺漏時回 (None, 說明)——那代表未評分,不是評了 0 分。
    補一個 0 分進報告會讓使用者以為自己這一項拿了零,
    而實際上是這場討論沒有可以評的內容。
    """
    try:
        level = int(data.get("level", 0))
    except (TypeError, ValueError):
        return None, f"{name}:模型回的 level 不是數字,本次不評"
    if level == 0:
        return None, f"{name}:{data.get('hint') or '這次沒有可觀察的內容'}"
    if not MIN_LEVEL <= level <= MAX_LEVEL:
        return None, f"{name}:模型回的 level {level} 超出 1–4,本次不評"

    return (
        CollabDimDTO(
            name=name,
            score=LEVEL_TO_SCORE[level],
            hint=normalize_locale(str(data.get("hint", "")).strip()),
            # evidence 不過 normalize_locale——那是原文引用,改了就不是原文
            evidence=str(data.get("evidence", "") or "").strip(),
        ),
        None,
    )


_MARKER = re.compile(r"^\s*(?:\[[^\]]*\]|[^:：]{1,12}[:：])\s*")
"""切片加在發言前面的標記。

    [接續他人之後]「我覺得先做客群分析」
    你:「我覺得先做客群分析」
    他人(主考官):「題目是會員制度」

【為什麼要剝掉再比對】
切片會加標記,但 routes.py 組給 verify_evidence 的逐字稿是純發言內容。
prompt 已經寫明「只引「」裡面的內容」,而且切片用引號劃了邊界——
但那是判斷型規則,會有殘留率。

殘留的後果特別糟:連標記一起抄的引用**跟真的編造分不出來**,
兩者都會被標成「依據不可信」。剝掉標記之後,
被擋下來的就只剩真的編造。
"""


def strip_markers(evidence: str) -> str:
    """剝掉切片標記與外層引號,取出真正被引用的話。"""
    t = evidence.strip()
    t = _MARKER.sub("", t)
    return t.strip("「」\"' 　")


def verify_evidence(dims: Sequence[CollabDimDTO], transcript: str) -> list[str]:
    """evidence 必須是逐字稿的子字串。回傳違規說明,空清單代表通過。

    跟 starParts.fromAnswer 同一條規則、同一種檢查。
    協作是零量化錨點的評分,沒有這道檢查就無法驗證等第不是編的。

    比對前先剝掉切片標記——見 _MARKER。

    不做自動修復,也不丟掉該維度:
    evidence 不會顯示給使用者(前端的 CollabDim 沒有這一欄),
    而且「引用錯」不必然代表「等第錯」——模型可能判斷正確但引述時記錯句子。
    為了一個不可靠的推論丟掉整個維度,代價比留著加註記大。
    """
    out: list[str] = []
    for d in dims:
        if not d.evidence:
            continue
        needle = strip_markers(d.evidence)
        # 用 contains 而不是 in:Whisper 的繁簡輸出不一致,
        # 逐字稿是簡體而 prompt 是繁體時,LLM 會把引用順手轉成繁體。
        # 直接比對會把正確的引用判成編造。
        if needle and not contains(transcript, needle):
            out.append(f"{d.name} 的引用不存在於逐字稿:{needle[:30]}")
    return out


async def score_collab(
    slices: Sequence[CollabSlice], llm: LLMCall, transcript: str = ""
) -> tuple[list[CollabDimDTO], list[str]]:
    """回傳 (協作維度, notices)。

    任一維度失敗不影響其他——gather_blocks 的 return_exceptions。
    全部失敗時回空清單,repair_report 會清掉協作區塊並記入 notices。
    """
    if not slices:
        return [], []

    results, notices = await gather_blocks(
        {sl.name: _score_one(llm, sl) for sl in slices}
    )

    dims: list[CollabDimDTO] = []
    for sl in slices:
        data = results.get(sl.name)
        if data is None:
            continue
        dim, problem = _to_dim(sl.name, data)
        if dim:
            dims.append(dim)
        elif problem:
            notices.append(f"協作:{problem}")

    if transcript:
        for problem in verify_evidence(dims, transcript):
            notices.append(f"協作:{problem},該維度的依據不可信")

    if dims and len(dims) < len(slices):
        notices.append(
            f"協作:{len(slices) - len(dims)} 個維度這次沒有可觀察的內容,未評分"
        )
    return dims, notices
