"""JD 需求抽取:散文 → JDInput.required_skills。

【在整條鏈的位置】

    使用者手貼的散文 JD(InterviewConfig.customJd,選填)
          ↓  這支
    JDInput(required_skills=[...], source="extracted")
          ↓  成員 A 的 GapComputer
    漏講候選

型錄職缺走另一條路:直接取 requiredSkills,source="catalog"。
兩條路交給 A 的形狀相同,校準才轉移得過去。

【LLM 用注入的】
llm 參數是一個 (system, user) -> str 的可呼叫物件。
測試傳假的,正式環境傳真的。這支本身不 import 任何 LLM SDK,
所以 conftest 不用 mock 任何東西,測試也不會意外打到 API。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.contracts.interview_protocols import JDInput
from app.pipeline.jd_normalize import normalize_skills
from app.prompts.jd_extract import MAX_SKILLS, compose_jd_prompt

LLMCall = Callable[[str, str], str]
"""(system_prompt, user_prompt) -> 模型回的原始文字。"""


@dataclass(frozen=True)
class ExtractResult:
    """抽取結果。notices 記錄所有機械修復動作,問題看得見但不噴例外。"""

    jd: JDInput
    notices: list[str] = field(default_factory=list)
    raw: str = ""


# ---------------------------------------------------------------------------
# 解析與修復
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_skills(raw: str) -> tuple[list[str], list[str]]:
    """把模型回的文字解析成技能清單。

    不拋例外。解析失敗回空清單加 notices——一份抽不出東西的 JD 是
    正常情況(散文太空泛),而空清單下游能優雅降級,例外不行。
    """
    notices: list[str] = []
    text = _FENCE.sub("", raw or "").strip()
    if not text:
        return [], ["模型回了空字串"]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _ARRAY.search(text)
        if not m:
            return [], ["模型輸出不是 JSON 陣列,已略過"]
        notices.append("模型輸出夾雜其他文字,已擷取其中的陣列")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [], ["模型輸出的陣列無法解析,已略過"]

    if not isinstance(data, list):
        return [], [f"模型輸出的頂層是 {type(data).__name__} 不是陣列,已略過"]

    return data, notices


def clean_skills(items: list, existing_notices: list[str]) -> tuple[list[str], list[str]]:
    """清洗:去空白、丟非字串、去重、截斷。**順序一律保留。**

    順序即重要性順序,成員 A 的 JD_POSITION_DECAY 依賴它,任何排序動作
    都會破壞計分。所以這裡只做刪除,不做重排。
    """
    notices = list(existing_notices)
    out: list[str] = []
    seen: set[str] = set()
    dropped_type = 0

    for item in items:
        if not isinstance(item, str):
            dropped_type += 1
            continue
        s = item.strip().strip("、,,。.;;")
        if not s:
            continue
        key = s.casefold().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    # 機械正規化:用語與專有名詞後綴。prompt 修過兩次都有殘留,
    # 這類確定性字串問題交給程式,見 app/pipeline/jd_normalize.py。
    out, norm_notices = normalize_skills(out)
    notices += norm_notices

    if dropped_type:
        notices.append(f"模型輸出含 {dropped_type} 個非字串項目,已丟棄")
    if len(items) - dropped_type > len(out):
        notices.append(f"去重後由 {len(items) - dropped_type} 項縮為 {len(out)} 項")
    if len(out) > MAX_SKILLS:
        notices.append(f"抽出 {len(out)} 項超過上限 {MAX_SKILLS},已截斷後段")
        out = out[:MAX_SKILLS]
    return out, notices


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

MIN_DESCRIPTION_CHARS = 20
"""短於此長度視為沒有有效 JD,直接回空,不浪費一次呼叫。

customJd 是選填欄位,使用者跳過或只打幾個字是常態。"""


def extract_jd(
    description: str,
    llm: LLMCall,
    *,
    title: str = "",
    company: str = "",
) -> ExtractResult:
    """散文 JD → JDInput。

    永遠回得出 JDInput,不拋例外。抽不到東西時 required_skills 為空清單,
    下游(GapComputer)收到空清單會回空候選,端點再降級成「未提供 JD」。
    """
    desc = (description or "").strip()

    if len(desc) < MIN_DESCRIPTION_CHARS:
        return ExtractResult(
            jd=JDInput(required_skills=[], description=desc, source="extracted"),
            notices=["JD 內容過短或未提供,略過抽取"],
        )

    system = compose_jd_prompt(title=title, company=company)
    try:
        raw = llm(system, desc)
    except Exception as exc:  # noqa: BLE001 — 抽取失敗不可讓整份報告掛掉
        return ExtractResult(
            jd=JDInput(required_skills=[], description=desc, source="extracted"),
            notices=[f"JD 抽取呼叫失敗,已略過({type(exc).__name__}: {exc})"],
        )

    items, notices = parse_skills(raw)
    skills, notices = clean_skills(items, notices)

    if not skills:
        notices.append("這份 JD 抽不出明確的能力需求")

    return ExtractResult(
        jd=JDInput(required_skills=skills, description=desc, source="extracted"),
        notices=notices,
        raw=raw,
    )


def from_catalog(required_skills: list[str], description: str = "") -> JDInput:
    """型錄職缺:直接用結構化欄位,不呼叫 LLM。

    順序照原樣保留——104 的 requiredSkills 排前面的通常較關鍵。
    """
    return JDInput(
        required_skills=list(required_skills),
        description=description or "",
        source="catalog",
    )
