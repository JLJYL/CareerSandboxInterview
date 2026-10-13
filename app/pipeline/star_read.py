"""判讀一段回答裡有哪些事實,供決定用。

【這一步為什麼存在】
1d 實測:三種難度的追問率都是 0.500,而「有沒有可查證的數字」一個 bit
就能預測 180/180 次決定。放行條件寫進 prompt 兩次都沒生效,所以判斷
要搬回程式——`AcceptRule` 的五個欄位本來就是可判定的。

搬回程式之後缺的只有一件事:誰供應那些事實。這個檔供應。

    這個檔     這段話裡有什麼            問模型,但只問它讀得準的
    accepts()  這些事實夠不夠放行        程式算,一次 100%

【呼叫數】
一輪三次:一次句子分派(S/T/A/R)加兩次單題(量化、取捨)。三次並行。

量化與取捨不走分派,因為那兩項不是 STAR 段落而是結果的屬性,
分派表達不出來;而且那兩項在單題問法下一直是 1.00,沒有換的理由。

延遲是已知代價,還沒量。先把判斷做對,再決定要不要省呼叫。

【失敗一定要看得見】
這是 1d 學到最貴的一課。`_REFUSAL_MARKS` 那次降級了 19/180 次決定,
而降級後的決定長得跟「模型判斷講完了」完全一樣,只看追問率看不出來,
多燒了一輪才查到。

所以這裡每一項事實都記它是怎麼來的(`read_by`),呼叫失敗記成 `failed`
而不是悄悄當成「沒有」——「四段都沒讀到」與「讀出來四段都沒有」在
`accepts()` 眼裡是同一件事,但前者該退回問模型,後者該直接追問。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import NamedTuple

from app.pipeline.interview_live import LLMCall, parse_obj
from app.pipeline.interviewer_spec import STAR_PART_KEYS, AcceptRule
from app.pipeline.transcript import segment_text
from app.prompts.star_read import (
    ASSIGN_SYSTEM,
    FACTS,
    STAR_LABELS,
    assign_user_message,
    compose_fact_prompt,
    fact_user_message,
)

STAR_READ_SOURCES: tuple[str, ...] = ("assign", "question", "failed")
ASSIGN, QUESTION, FAILED = STAR_READ_SOURCES
"""`AnswerFacts.read_by` 的值域。

宣告出來跟 `DECISION_SOURCES` 同一個理由:報表自己列一份的話,
這裡多一個值而它沒跟著加,那一整類會被漏掉而且報表照樣印得出來。
"""

ASSIGNABLE_SEGMENTATION: tuple[str, ...] = ("punctuation",)
"""只有真句末標點切出來的邊界才拿去做句子分派。

【為什麼不收 stt_segment 與 discourse_marker】
分派的前提是「一句只能掛一個標籤」,那個互斥性是它能治 T 的原因。
前提是切出來的真的是一句話——實測的 1.00 是在句末標點切出來的句子上量的。

語氣詞估出來的邊界會把一句話切成兩半(「後來每週快很多」/「所以就不用對帳」),
半句話掛哪個標籤沒有正確答案,而模型照樣會掛一個。那不是分派,是亂數。

`stt_segment`(STT 自動送出點)是最有機會補上來的一個,但沒量過,
所以現在不收。要收就先量——這個常數就是那次量測要改的地方。

【代價,寫清楚】
裝置端引擎沒有標點,所以 device 的逐字稿一律走單題問法,而 T 在那條路徑上
是 0.29。後果是困難難度會把該追問的放行(A7 那一格)。
這件事由 `read_by["T"] == "question"` 標出來,不是靜悄悄發生的。
"""

ASSIGN_MIN_SENTENCES = 1
"""至少要有幾句才做分派。

【為什麼是 1 不是 2】
一開始寫 2,理由是「一句話只掛一個標籤,其他三段必然判成沒有」。
那個理由是對的,但方向判錯了:

    只有一句而走分派    最多讀出一段 → 少報 → 傾向追問
    只有一句而走單題    T 一律答「有」 → 多報 → 傾向放行

只有一句的回答本來就該被追問(A1 在三種難度都是追問),少報剛好得到
正確的結果,多報會把該追的放掉。而且實測的 1.00 裡 A1 就是一句話、
走的就是分派——把門檻設成 2 等於上線的設定跟量過的設定不一樣,
然後拿量到的數字去宣稱它。

留著這個常數是為了擋 0 句(回答只有一個句號的時候 segmentation 仍是
punctuation 但切不出句子),順便把上面這段推導寫在它旁邊。
"""

MIN_ANSWER_CHARS = 2
"""短於這個長度不呼叫模型。與 `turn_decision.MIN_ANSWER_CHARS` 同值,
但那邊是「要不要追問」,這邊是「要不要花三次呼叫」,兩個判斷各自獨立。"""

MIN_QUOTE_CHARS = 4
"""短於這個長度的引文不算證據。「報表」在任何一筆回答裡都找得到。"""

_PUNCT = "，,。.、;；:：「」『』()（）?？!！…—　 \n\t"
_STRIP = str.maketrans("", "", _PUNCT)


class FactRead(NamedTuple):
    """一項事實的判讀結果。"""

    present: bool
    quote: str
    grounded: bool


def _norm(text: str) -> str:
    return (text or "").translate(_STRIP)


def grounded(quote: str, answer: str) -> bool:
    """引文是不是真的在答案裡。標點與空白正規化後做子字串比對。

    這是整個判讀裡唯一不靠模型的判斷:它說「有交代」可以是編的,
    但它抄的那句話在不在原文裡,程式查得出來。

    「能用程式算的不要問 LLM」在這裡的形式就是不問它有沒有說謊,直接查。
    """
    q = _norm(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _norm(answer)


def read_one(raw: str, answer: str) -> FactRead:
    """解析單題輸出。解析不出來當成「沒有」。

    `present` 說有但引文沒落地 → 判讀改成「沒有」。
    那是抄原文這條規則的執行點:宣稱無法查證就不採用。

    【為什麼是 `if not data` 不是 `if data is None`】
    `interview_live.parse_obj` 解析失敗回空 dict,不回 None。寫成 `is None`
    的話失敗會掉到下面,`get("present")` 取不到值變成「模型說沒有」,
    而且 `grounded` 會是 True——一個沒讀到的結果長得跟一個乾淨的否認一樣。

    `tools/star_read_eval.py` 裡那份 `parse_obj` 回的是 None,兩邊不同。
    實測那幾輪格式遵從率 100%,沒有任何一列走到這條路,所以數字沒受影響——
    但這正是「同一件事有兩份實作」會怎麼咬人的樣本。工具已改成共用這一份。
    """
    data = parse_obj(raw)
    if not data:
        return FactRead(False, "", False)
    quote = str(data.get("quote", "") or "")
    if not bool(data.get("present")):
        return FactRead(False, quote, True)
    ok = grounded(quote, answer)
    return FactRead(ok, quote, ok)


def parse_assign(raw: str, sentences: list[str]) -> dict[str, FactRead]:
    """從分派結果推出 S/T/A/R 在不在。

    引文由程式供給(就是被分派的那一句),所以 `grounded` 必然為真——
    那一欄在分派模式下不是量測,是為了讓報表的欄位對齊。

    漏掉的句子、認不得的標籤、超出範圍的編號一律丟掉:分派是封閉集合,
    修不回來的值不要猜。全部丟掉的結果是四段都判成沒有,而 `read_by`
    仍然記 assign——那是「分派說沒有」,跟「分派失敗」不一樣。
    """
    data = parse_obj(raw) or {}
    chosen: dict[int, str] = {}
    for key, label in data.items():
        if not isinstance(label, str) or label.strip() not in STAR_LABELS:
            continue
        try:
            idx = int(str(key).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(sentences):
            chosen[idx] = label.strip()

    out: dict[str, FactRead] = {}
    for part in STAR_PART_KEYS:
        hit = [sentences[i - 1] for i, lab in sorted(chosen.items()) if lab == part]
        out[part] = FactRead(bool(hit), hit[0] if hit else "", True)
    return out


_ABSENT = FactRead(False, "", False)


@dataclass(frozen=True)
class AnswerFacts:
    """一段回答裡有什麼。只記事實,不記夠不夠。

    【為什麼存 reads 而不是存六個 bool】
    `tools/star_read_eval.py` 要算引文落地率,那需要每一項的引文與落地與否。
    只存結論的話工具就得自己再組一次呼叫才拿得到那些欄位——
    於是工具量的是它自己那份組法,不是上線跑的這一份。

    1d 之後唯一確定有效的做法是量測與線上共用同一條路徑,所以判讀的
    原始結果留在這裡,結論用 property 導出去。
    """

    reads: dict[str, FactRead] = field(default_factory=dict)
    """每一項的判讀結果。鍵是 FACTS,缺的鍵表示沒讀到。"""

    read_by: dict[str, str] = field(default_factory=dict)
    """每一項是怎麼來的:assign / question / failed。

    空的 dict 表示沒有讀(回答太短,沒有呼叫模型)。
    """

    segmentation: str = "unavailable"
    """斷句的來源,直接取 `segment_text()` 的第二個回傳值。

    留著是因為「為什麼這一輪沒走分派」的答案在這裡,而那決定了 T 可不可信。
    """

    def read(self, fact: str) -> FactRead:
        return self.reads.get(fact, _ABSENT)

    @property
    def star_parts(self) -> tuple[str, ...]:
        """交代到的 STAR 段落,照 STAR_PART_KEYS 的順序。"""
        return tuple(p for p in STAR_PART_KEYS if self.read(p).present)

    @property
    def quantified(self) -> bool:
        return self.read("quantified").present

    @property
    def tradeoff(self) -> bool:
        return self.read("tradeoff").present

    @property
    def quotes(self) -> dict[str, str]:
        """每一項的證據原文。報告要引用、人工標注要查,都靠這個。"""
        return {f: r.quote for f, r in self.reads.items() if r.quote}

    @property
    def failed(self) -> tuple[str, ...]:
        """哪幾項的呼叫沒回來。"""
        return tuple(f for f in FACTS if self.read_by.get(f) == FAILED)

    @property
    def complete(self) -> bool:
        """六項都讀到了。沒讀到的時候不該讓程式直接下判斷。"""
        return all(self.read_by.get(f) in (ASSIGN, QUESTION) for f in FACTS)

    @property
    def task_reliable(self) -> bool:
        """任務這一項可不可信——只有走分派才可信。

        單獨拉出來是因為只有 T 有這個問題:其他五項在兩種路徑下都是 1.00。
        """
        return self.read_by.get("T") == ASSIGN

    def accepted_by(self, rule: AcceptRule) -> bool:
        """這些事實夠不夠放行。判斷在 `AcceptRule`,這裡只是把欄位遞過去。"""
        return rule.accepts(
            star_parts=self.star_parts,
            quantified=self.quantified,
            tradeoff=self.tradeoff,
        )

    def gap_for(self, rule: AcceptRule) -> str:
        """最該追的那一個缺口,放行時是空字串。

        這一欄以前是問模型的。1d 實測模型指錯:A3 講了結果只是沒數字,
        20/20 次都寫「缺結果」,下一句就會去問他已經講過的東西。
        算出來的版本在同一筆上是「結果沒量化」,而且不可能錯——
        哪一條條件沒過是確定的事。
        """
        return rule.first_gap(
            star_parts=self.star_parts,
            quantified=self.quantified,
            tradeoff=self.tradeoff,
        )

    def as_label(self) -> dict:
        """擺成 `fixtures/golden/decision/answers_v1.json` 的 label 形狀。

        標準答案與判讀結果共用一個形狀,`accepts()` 才能同一支函式吃兩邊——
        量測跟線上算的是同一件事。
        """
        return {
            "star_parts": list(self.star_parts),
            "quantified": self.quantified,
            "tradeoff": self.tradeoff,
        }

    @classmethod
    def from_label(cls, label: dict, *, read_by: str = QUESTION) -> AnswerFacts:
        """`as_label()` 的反向。給測試與量測工具拿標註當判讀結果用。

        引文留空:標註沒有引文,而捏一個假的會讓落地率變成假的。
        """
        present = set(label.get("star_parts") or ())
        flags = {
            **{p: p in present for p in STAR_PART_KEYS},
            "quantified": bool(label.get("quantified")),
            "tradeoff": bool(label.get("tradeoff")),
        }
        return cls(
            reads={f: FactRead(flags[f], "", True) for f in FACTS},
            read_by={f: read_by for f in FACTS},
            segmentation="punctuation",
        )


async def read_facts(*, answer: str, llm: LLMCall) -> AnswerFacts:
    """讀出這段回答裡的六項事實。永遠回一個結果,不拋例外。

    走哪條路徑由斷句決定:切得出真句子就用分派,否則退回單題。
    兩條路徑都會把來源記在 `read_by` 上,因為兩者的 T 一個是 1.00
    一個是 0.29,而結果的資料形狀一模一樣。
    """
    text = (answer or "").strip()
    if len(text) < MIN_ANSWER_CHARS:
        # 沒有東西可讀。不呼叫模型,也不假裝讀過——read_by 空著,complete 為 False。
        return AnswerFacts()

    sentences, segmentation = segment_text(text, level="sentence")
    use_assign = (
        segmentation in ASSIGNABLE_SEGMENTATION and len(sentences) >= ASSIGN_MIN_SENTENCES
    )
    asked = ("quantified", "tradeoff") if use_assign else FACTS

    async def one(fact: str) -> tuple[str, FactRead | None]:
        try:
            raw = await asyncio.to_thread(
                llm, compose_fact_prompt(fact), fact_user_message(text)
            )
        except Exception:  # noqa: BLE001
            return fact, None
        return fact, read_one(raw, text)

    async def assignment() -> dict[str, FactRead] | None:
        try:
            raw = await asyncio.to_thread(
                llm, ASSIGN_SYSTEM, assign_user_message(sentences)
            )
        except Exception:  # noqa: BLE001
            return None
        return parse_assign(raw, sentences)

    jobs: list = [one(f) for f in asked]
    if use_assign:
        jobs.append(assignment())
    done = await asyncio.gather(*jobs)

    # 比的是 None 不是真假值:FactRead(False, "", False) 是一個三元組,
    # 它永遠是真值。寫成 `if got` 現在剛好對,但那是靠 tuple 的真假規則,
    # 而這條分支決定的是「失敗有沒有被標出來」,不該靠這種東西。
    reads: dict[str, FactRead] = {}
    by: dict[str, str] = {}
    for fact, got in done[: len(asked)]:
        reads[fact] = _ABSENT if got is None else got
        by[fact] = FAILED if got is None else QUESTION
    if use_assign:
        parts = done[-1]
        for p in STAR_PART_KEYS:
            reads[p] = _ABSENT if parts is None else parts[p]
            by[p] = FAILED if parts is None else ASSIGN

    return AnswerFacts(reads=reads, read_by=by, segmentation=segmentation)
