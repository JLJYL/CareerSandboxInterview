"""漏講點(A5)管線:GapCandidate → MissingPointDTO。

【流程】
    GapComputer(A,確定性)  → list[GapCandidate]
            ↓
    這裡:一次 LLM 呼叫,同時做否決與 why 生成
            ↓
    機械檢查(否決 null、長度、禁詞)
            ↓
    取前 MAX_POINTS 條 → list[MissingPointDTO]

【為什麼否決與 why 合併成一次呼叫】
兩者吃的輸入完全相同(候選 + 逐字稿 + JD 證據),拆兩次等於同一份資料
送兩趟。而且否決的判斷本身就需要想過 why 寫不寫得出來——
「寫不出一個誠實的理由」正是否決條件之一。

【降級】
LLM 失敗時不拋例外,回空清單並記入 notices。
一份少了漏講點區塊的報告仍然可用,500 則完全不可用。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from app.contracts.interview_protocols import GapCandidate
from app.pipeline.jd_normalize import normalize_locale
from app.prompts.missing_points import MAX_POINTS, compose_why_prompt
from app.schemas.interview import MissingPointDTO

LLMCall = Callable[[str, str], str]
"""(system, user) -> str。注入式,所以這支不 import 任何 SDK。"""


# ---------------------------------------------------------------------------
# 輸出解析
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_verdicts(raw: str, expected: int) -> tuple[dict[int, dict], list[str]]:
    """把模型輸出解析成 {候選編號: 裁決}。

    **依 id 對齊,不依位置。** 早期版本靠陣列位置,模型多輸出一個元素就整批
    位移一格,保留下來的理由被安到別的技能上——實測十格裡有四格出現這個現象。

    沒出現在輸出裡的候選視為否決。解析失敗回空 dict,等於全部否決。
    """
    notices: list[str] = []
    text = (raw or "").strip()

    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("["):
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j > i:
            text = text[i : j + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}, ["漏講點:模型輸出不是合法 JSON,本次不顯示漏講點"]

    if not isinstance(data, list):
        return {}, ["漏講點:模型輸出不是陣列,本次不顯示漏講點"]

    out: dict[int, dict] = {}
    unknown = 0
    missing_id = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        try:
            idx = int(raw_id)
        except (TypeError, ValueError):
            missing_id += 1
            continue
        if not 1 <= idx <= expected:
            unknown += 1
            continue
        out[idx] = {"why": str(item.get("why", "") or "").strip()}

    if missing_id:
        notices.append(f"漏講點:{missing_id} 筆裁決沒有 id,已丟棄")
    if unknown:
        notices.append(f"漏講點:{unknown} 筆裁決的 id 不在候選範圍,已丟棄")
    absent = expected - len(out)
    if absent > 0:
        notices.append(f"漏講點:{absent} 條候選模型沒有回覆理由")
    return out, notices


# ---------------------------------------------------------------------------
# 機械檢查
# ---------------------------------------------------------------------------

_RESUME_NARRATION = re.compile(
    r"(你|您|候選人|應徵者|求職者|他|她)?(在|於)?[^,,。.;;]{0,16}"
    r"(經歷|經驗|專案|專題|實習|社團|打工|工作|履歷)(中|裡|時|上|裡面)?"
    r"[^,,。.;;]{0,10}(展現|展示|做過|有過|提到|具備|累積|培養|表現)"
)
"""偵測 why 敘述了使用者做過什麼。

WHY_MUST_BE_JD_SIDE 寫在 prompt 裡,但那是判斷型規則,會有殘留率。
實測第一版漏掉「候選人在專題中展現了…」——主詞是「候選人」不是「你」、
名詞是「專題」不是「專案」。所以這一版把主詞與名詞都放寬,
並改成偵測「主體 + 場域 + 動作」的組合形狀。

命中就丟掉那一條,不改寫——改寫等於我們自己編一個理由,
而問題本來就是理由編得太好。
"""

_ADVICE = re.compile(r"(你應該|建議你|你可以多|記得要|下次要|不妨)")
"""why 回答的是「為什麼重要」,不是「你該怎麼做」。
建議句屬於 improvements 欄位,不是這裡。"""

# 非台灣用語不丟棄,機械替換。
#
# 早期版本是丟棄,理由寫「替換會讓句子讀起來不順」——那個理由是錯的。
# 用詞正確與否跟這條漏講點對不對完全無關,丟掉等於為了措辭殺掉一個可能正確的點。
# 而 normalize_locale 做的就是同一種替換,jd_normalize 那邊已經證明可行。
#
# 實測:第一版丟掉的三條(Angular、Git、HTML/CSS)剛好都是假指控,
# 看起來像幫上忙,但那是運氣不是設計。

MIN_WHY_CHARS = 15
MAX_WHY_CHARS = 120


def check_why(why: str) -> str | None:
    """回傳問題描述,通過則回 None。"""
    if not why:
        return "why 為空"
    if len(why) < MIN_WHY_CHARS:
        return f"why 只有 {len(why)} 字,過短"
    if len(why) > MAX_WHY_CHARS:
        return f"why 有 {len(why)} 字,超過上限"
    if _RESUME_NARRATION.search(why):
        return "why 敘述了使用者的經歷,違反 WHY_MUST_BE_JD_SIDE"
    if _ADVICE.search(why):
        return "why 寫成了建議句"
    return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_user_message(
    candidates: Sequence[GapCandidate], transcript: str, jd_description: str
) -> str:
    lines = ["【本場逐字稿】", transcript or "(無)", "", "【JD 原文】", jd_description or "(無)", "", "【候選清單】"]
    for i, c in enumerate(candidates, 1):
        lines += [
            f"{i}. {c.display}",
            f"   類型: {c.kind}",
            f"   JD 提到這項的原文: {c.jd_evidence or '(無)'}",
        ]
    return "\n".join(lines)


def generate_missing_points(
    candidates: Sequence[GapCandidate],
    transcript: str,
    jd_description: str,
    llm: LLMCall,
    *,
    max_points: int = MAX_POINTS,
) -> tuple[list[MissingPointDTO], list[str]]:
    """回傳 (漏講點, notices)。

    候選為空時直接回空,不呼叫 LLM——沒有履歷或沒有交集是正常情況。

    【這一層不做過濾】
    早期版本讓 LLM 同時判斷保留與否。實測兩輪都是虧的:

        不經 LLM   17 條候選,7 命中   precision 0.412   recall 7/8
        經過 LLM   10 條保留,4 命中   precision 0.400   recall 4/8

    否決掉的裡面有 3 條是真漏講,留下的全是假指控,而且否決集中在候選清單
    後段——那是注意力衰減,不是判斷。所以這一層只負責寫 why。

    過濾交給更可靠的機制:成員 A 的語意比對與履歷條件式模糊匹配(W2)。

    仍然會丟掉的只有兩種,都是安全性而非正確性的理由:
        why 敘述了使用者的經歷 —— 會產生看起來很具體但可能全錯的歸因
        why 為空或長度異常     —— 沒有理由的漏講點不能顯示
    """
    notices: list[str] = []
    if not candidates:
        return [], notices

    # 材料不足的候選在程式端濾掉,不問模型。
    # jd_evidence 是 why 唯一允許引用的素材,沒有它就寫不出有依據的理由。
    # 這一步放在程式端的理由:「有沒有材料」是確定性事實,
    # 交給模型判斷會讓「材料不足」與「模型自己想砍」混在一起,查不出是哪一種。
    usable = [c for c in candidates if c.jd_evidence]
    skipped = len(candidates) - len(usable)
    if skipped:
        notices.append(f"漏講點:{skipped} 條候選沒有 JD 依據,無法寫理由")
    if not usable:
        return [], notices

    try:
        raw = llm(compose_why_prompt(), build_user_message(usable, transcript, jd_description))
    except Exception as exc:  # noqa: BLE001
        return [], [f"漏講點:生成失敗({type(exc).__name__}),本次不顯示漏講點"]

    verdicts, parse_notices = parse_verdicts(raw, len(usable))
    notices += parse_notices

    out: list[MissingPointDTO] = []
    no_reply = 0
    rejected = 0
    localized = 0
    for i, cand in enumerate(usable, 1):
        v = verdicts.get(i)
        if v is None or not v["why"]:
            no_reply += 1
            continue
        why = normalize_locale(v["why"])
        if why != v["why"]:
            localized += 1
        problem = check_why(why)
        if problem:
            rejected += 1
            notices.append(f"漏講點「{cand.display}」已丟棄:{problem}")
            continue
        out.append(MissingPointDTO(point=cand.display, why=why))

    if no_reply:
        # 模型漏回。實測集中在清單後段,是注意力衰減。
        # 記成明確的模型失誤,不要跟「材料不足」混為一談。
        notices.append(f"漏講點:{no_reply} 條候選模型漏寫理由,無法顯示")
    if localized:
        notices.append(f"漏講點:{localized} 條的用詞已機械正規化為台灣用語")
    if len(out) > max_points:
        notices.append(f"漏講點:保留 {len(out)} 條,依重要性取前 {max_points} 條")
        out = out[:max_points]

    return out, notices
