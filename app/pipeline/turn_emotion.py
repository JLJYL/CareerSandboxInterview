"""判讀逐題情緒,並把不該標註的濾掉。

【這一塊只產出要標註的那幾題】
`平穩` 不標註,所以回傳的是稀疏清單:沒有被標的題不出現。

這樣設計不只是省渲染。1d 實測過一個現象:模型列幾項取決於 prompt 寫了幾條,
不取決於內容。要求它「每一題都給一個狀態」然後全部顯示,等於邀請它
每一題都找一個不是平穩的標籤。濾在程式這一端,模型就沒有這個誘因。

【程式先判,判不出來才問模型】
`明確說不知道` 有現成的字面集合(`interview_live._ADMITS_UNKNOWN`),
那一項不問模型——「能用程式算的不要問 LLM」。命中就是卡住,而且 `source`
記成 code,之後驗證時分得開是誰判的。

【引文沒落地就退回平穩】
它說他焦慮可以是編的,但它抄的那句話在不在那一題的回答裡,程式查得出來。
查不到就不標註。這是把一個主觀判斷的誤報成本壓下來的唯一手段——
而誤報的代價在這一塊特別高:把人標成焦慮是對他的判斷,
不是「少講一個缺結果」那種可以一笑置之的錯。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from app.pipeline.interview_live import LLMCall, _ADMITS_UNKNOWN, parse_obj
from app.pipeline.star_read import grounded
from app.prompts.turn_emotion import (
    EMOTION_LABELS,
    PLAIN,
    compose_emotion_prompt,
    emotion_user_message,
)

EMOTION_SOURCES: tuple[str, ...] = ("code", "llm")
CODE, LLM = EMOTION_SOURCES
"""`TurnEmotion.source` 的值域。

    code  程式用字面判的(明確說不知道)
    llm   模型判的,而且引文落地了

降級沒有自己的值:判讀失敗就是不標註,而不標註本來就不會出現在清單裡。
沒有「我們想標但失敗了」這個狀態要表達。
"""

STUCK = "卡住"
"""`_ADMITS_UNKNOWN` 命中時的標籤。單獨拉出來是因為程式會直接用它,
拼錯不會壞在 import 時而是壞在線上——檔尾有檢查它在值域裡。"""

NO_QUOTE_BACK = "過度自責"
"""這一類查證要引文,但報告裡不顯示引文。見 `TurnEmotion.shown_quote`。"""


@dataclass(frozen=True)
class EmotionTurn:
    """判讀一題需要的東西。

    【為什麼不是 (題目, 回答) 的 tuple】
    多了停頓那一行之後就是三個位置,而三個同型別的位置排錯不會壞在 import,
    會壞在線上而且看不出來。宣告成結構之後排錯是 TypeError。

    停頓那一行**不能併進 answer**:`grounded()` 拿 answer 查引文,
    併進去之後模型可以引那一行當證據,而那一行是我們寫的不是他講的。
    """

    question: str
    answer: str
    pause_note: str = ""
    """`speech_timing.describe_pauses()` 的結果。空字串表示沒有時間資料。

    空字串有兩種來源(打字、前端還沒送 segment_ends_ms),兩者對判讀的
    影響一樣:「長停頓」這個訊號不可用。所以不分開表達。
    """


@dataclass(frozen=True)
class TurnEmotion:
    """某一題的情緒標註。只有要標註的題才會有。"""

    turn_index: int
    """第幾題,從 1 開始。對齊逐題回顧的題號。"""

    label: str
    """EMOTION_LABELS 其中一個,而且不會是 PLAIN——平穩不標註。"""

    quote: str = ""
    """證據原文。查證用,不等於報告裡會顯示的那一句——見 `shown_quote`。

    `source == CODE` 時為空:那一項是字面命中,沒有「它抄了哪一句」可查。
    """

    source: str = LLM

    @property
    def shown_quote(self) -> str:
        """報告裡該顯示的引文。`過度自責` 回空字串。

        查證與呈現是兩件事。那一類一定要有引文才採用(不然無法確認模型
        不是在編),但把「我真的很爛」引回去給使用者看不是回饋是傷害。
        `EMOTION_WRITEUP` 那一欄寫的是「只描述行為不評價」,這裡是它的執行點。
        """
        return "" if self.label == NO_QUOTE_BACK else self.quote


def batch_input_mode(modes: Sequence[str]) -> str:
    """一批回答共用的輸入模式。不一致就回 "unknown"。

    一次呼叫問所有題,而 prompt 會按模式告訴模型哪些訊號可用
    (打字沒有停頓資料、語音的碎句可能是引擎造成的)。
    混用的時候兩個分支都不對,所以退回 unknown——那一支明寫
    「斷句與停頓都不可靠,只從內容判斷」。

    這不是理論上的情況:同一場裡語音辨識失敗而改打字是會發生的,
    而那一題的碎句程度跟其他題不可比。拿整場的模式去套,
    等於用「這場大多是語音」去解釋一題打字的回答。

    `generate_report` 另外有一個整場的 `input_mode`,那是給流暢度公式用的,
    跟這裡問的不是同一件事,所以不共用。
    """
    seen = {m for m in modes if m}
    return seen.pop() if len(seen) == 1 else "unknown"


def admits_unknown(answer: str) -> bool:
    """他有沒有明確說答不出來。與 `forced_decision` 共用同一組字面。

    共用是刻意的:決策那邊靠它決定不要追問,報告這邊靠它標卡住,
    兩邊說的是同一件事。各自列一份就會出現「決策覺得他答不出來、
    報告覺得他還好」這種自相矛盾的輸出。
    """
    return any(k in (answer or "") for k in _ADMITS_UNKNOWN)


def parse_emotions(raw: str, turns: Sequence[EmotionTurn]) -> dict[int, TurnEmotion]:
    """解析模型的分派結果,濾掉平穩與引文沒落地的。

    丟掉的四種情況,都是靜默丟:
        認不得的標籤      封閉集合外的值不要猜
        超出範圍的題號    對不到題的標註沒有地方放
        平穩              不標註
        引文沒落地        宣稱無法查證就不採用

    靜默是對的:這一塊不標註是正常結果,不是失敗。把「沒標到」當成錯誤
    回報會讓報告出現一堆雜訊通知,而那些通知對使用者沒有意義。
    """
    data = parse_obj(raw) or {}
    out: dict[int, TurnEmotion] = {}
    for key, cell in data.items():
        if not isinstance(cell, dict):
            continue
        try:
            idx = int(str(key).strip())
        except (TypeError, ValueError):
            continue
        if not 1 <= idx <= len(turns):
            continue
        label = str(cell.get("label", "")).strip()
        if label not in EMOTION_LABELS or label == PLAIN:
            continue
        quote = str(cell.get("quote", "") or "").strip()
        # 查那一題的 answer,不查整場——整場逐字稿裡找得到不算證據。
        if not grounded(quote, turns[idx - 1].answer):
            continue
        out[idx] = TurnEmotion(idx, label, quote, LLM)
    return out


async def read_emotions(
    *,
    turns: Sequence[EmotionTurn],
    llm: LLMCall,
    input_mode: str = "unknown",
) -> list[TurnEmotion]:
    """逐題情緒。回傳要標註的那幾題,照題號排序。

    一次呼叫問所有題:事實判讀那一輪實測拆開問與一次問六題差 0.00,
    而這裡拆開還會失去「某幾題比其他題碎」這個比較基準——
    那正是 `SIGNAL_BY_MODE` 要模型看的東西。

    呼叫失敗就只剩程式判出來的那些。不標註是正常結果,所以不降級、不通知。
    """
    if not turns:
        return []

    by_code = {
        i: TurnEmotion(i, STUCK, "", CODE)
        for i, t in enumerate(turns, 1)
        if admits_unknown(t.answer)
    }

    try:
        raw = await asyncio.to_thread(
            llm,
            compose_emotion_prompt(input_mode),
            emotion_user_message([(t.question, t.answer, t.pause_note) for t in turns]),
        )
    except Exception:  # noqa: BLE001
        return [by_code[i] for i in sorted(by_code)]

    # 程式判出來的優先:字面命中是確定的事,模型的判讀是推論。
    merged = {**parse_emotions(raw, turns), **by_code}
    return [merged[i] for i in sorted(merged)]


_undeclared = {STUCK, PLAIN, NO_QUOTE_BACK} - set(EMOTION_LABELS)
if _undeclared:
    raise RuntimeError(
        f"這些標籤不在 EMOTION_LABELS 裡:{sorted(_undeclared)}。"
        "程式直接用的標籤拼錯的話,前端會收到一個它查不到的值。"
    )
