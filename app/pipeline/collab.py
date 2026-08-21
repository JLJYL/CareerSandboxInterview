"""群面協作可觀察值（成員 A，W2 D1–D3）。

`CollabObserver` 的真實作。**只抽可觀察值，不指派等第**——
`level` 一律 `None`，由 B 的 LLM 對照 BARS 錨點指派。

## 為什麼現在只交兩項

四項裡有兩項的輸入不在場：

    參與主動性   陣列順序就夠            → 現在可做
    論點建構     純文字統計              → 現在可做
    傾聽與回應   要比對前一位發言者說什麼  → 等前端
    協作姿態     要知道別人何時在講        → 等前端

前端目前只把使用者自己的發言寫進 session，AI 同儕的話留在畫面的 messages
裡沒有存。所以後兩項**不管 rubric 怎麼寫都算不出來**——那不是難，
是資訊不在場。它們回空 signals，並在 evidence 註明原因。

★ 不猜。rubric 還沒回來，猜錯的實作比沒有實作糟，因為它看起來像做完了。

## 發言量指標的處理

合約 `COLLAB_PROHIBITED_INDICATORS` 禁用發言次數等「量」的指標，理由是
babble 假說：講最多話的人會被認為是領導者，與內容好壞無關；而我們是
回饋產品，用發言量計分等於在訓練使用者 babble。

合約允許把次數放進 `signals` 當背景資訊。**本實作照做，但把它跟計量訊號
分開命名**（`bg_utterance_count`，`bg_` 前綴）——理由見 `_BG_PREFIX`。
"""

from __future__ import annotations

import re
from typing import Iterable

from app.contracts.interview_protocols import CollabSignal, Utterance
from app.schemas.interview import COLLAB_DIM_NAMES

#: 背景資訊的鍵名前綴。
#:
#: ★ 合約說發言次數「可放進 signals 當背景資訊,但不可以是等第的主要依據」。
#:   問題是:把「共發言 4 次」跟其他訊號平鋪在同一個 dict 裡交給 LLM,
#:   再在 prompt 裡說「不要依賴這個」——那個保護很弱。它是那堆數字裡最
#:   直觀的一個,LLM 會錨定上去,babble 偏誤就從資料欄位漏進來,
#:   而合約前面才剛花一整段說明為什麼不能這樣。
#:
#:   加前綴讓它在**結構上**可辨識,B 的 prompt 可以據此把 bg_ 開頭的鍵
#:   放進「背景」而不是「依據」。這是我單方面加的慣例,B 若要拆成獨立欄位
#:   我跟著改——重點是那道界線要在資料裡看得見,不能只存在於註解。
_BG_PREFIX = "bg_"

#: 因果／推理連接詞。「論點建構」的核心訊號:有沒有把主張跟理由接起來。
#: 只收**顯性**連接詞——「我覺得不錯」有主張沒理由,不該算進來。
CAUSAL_CONNECTORS: tuple[str, ...] = (
    "因為", "所以", "因此", "由於", "導致", "造成", "使得",
    "如果", "假如", "的話", "才能", "為了", "以便",
    "但是", "不過", "然而", "雖然", "儘管", "反而",
    "首先", "再來", "第二", "第三", "最後", "另外", "而且", "此外",
)

#: 議題框架詞。「參與主動性」的替代訊號——委外文件明確要求不靠發言量,
#: 改看「首次發言是否在他人之前提出可討論的框架」。
#: 這些詞出現時，那則發言是在**組織討論**而不只是表達意見。
FRAMING_MARKERS: tuple[str, ...] = (
    "我們可以", "我建議", "先", "第一步", "順序", "分成", "兩個方向",
    "要不要", "不如", "我提議", "定義", "釐清", "確認一下", "回到",
)

_SENT_SPLIT = re.compile(r"[。！？；，、\n\.!\?;]+")


def _count_any(text: str, terms: Iterable[str]) -> int:
    """計算 terms 在 text 出現的總次數（可重複）。"""
    return sum(text.count(t) for t in terms if t)


def _char_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


class CollabObserver:
    """CollabObserver Protocol 的真實作。

    用法：

        observer = CollabObserver()
        signals = observer.observe(session.utterances)   # 四筆，level 皆 None
    """

    def observe(self, utterances: list[Utterance]) -> list[CollabSignal]:
        """回傳四筆，順序對齊 COLLAB_DIM_NAMES，level 一律 None。"""
        utts = list(utterances or [])
        mine = [u for u in utts if u.speaker_id == "user"]
        others = [u for u in utts if u.speaker_id != "user"]

        # ★ 判斷依據是「輸入裡有沒有別人的發言」,不是「有沒有 Utterance」。
        #   前端未改完之前,groupSays 只有使用者自己的話,長度不為零但
        #   others 是空的——那時後兩項算不出來。
        has_speakers = bool(others)

        return [
            self._participation(utts, mine, has_speakers),
            self._responsiveness(has_speakers),
            self._argument(mine),
            self._collaboration(has_speakers),
        ]

    # ---------------------------------------------------------------- 可做的兩項

    def _participation(self, utts: list[Utterance], mine: list[Utterance],
                       has_speakers: bool) -> CollabSignal:
        """參與主動性。**不用發言量當依據**——見 COLLAB_PROHIBITED_INDICATORS。

        改用三個位置／內容訊號：

          first_speak_position  首次發言的相對位置（0=最先，1=最後）
          framing_in_first      首次發言裡有沒有提出可討論的框架
          initiated_after_gap   在無人接續的空檔主動開口的次數

        第一項測「什麼時候開口」而不是「開了幾次口」;
        第二項測「開口時在組織討論還是只在表達意見」。
        兩者都跟講多少話無關。
        """
        signals: dict[str, float] = {}
        parts: list[str] = []

        if not mine:
            return CollabSignal(COLLAB_DIM_NAMES[0], {}, "使用者沒有發言")

        # ★ 分母是「使用者開口前,別人講了幾則」加一,不是總則數。
        #
        #   用總則數會讓發言量從後門混進來:同樣在第 2 則開口,若使用者後面
        #   多講 5 次,位置就從 1.00 變成 0.17,看起來「開口早很多」——
        #   而 babble 假說正是要避免用講多少話來評分。
        #
        #   用「別人給了幾次機會」當分母,測的才是「在別人講了多少之後才開口」,
        #   跟使用者自己講幾次無關。
        first = utts.index(mine[0])
        others_before = sum(1 for u in utts[:first] if u.speaker_id != "user")
        others_total = sum(1 for u in utts if u.speaker_id != "user")
        pos = others_before / max(others_total, 1)
        signals["first_speak_position"] = round(pos, 3)
        signals["others_before_first"] = float(others_before)
        parts.append(f"在他人發言 {others_before}/{others_total} 則之後首次開口"
                     f"（相對位置 {pos:.2f}）")

        framing = _count_any(mine[0].text, FRAMING_MARKERS)
        signals["framing_in_first"] = float(framing)
        parts.append("首次發言" + ("有提出討論框架" if framing else "未提出討論框架"))

        if has_speakers:
            # 「無人回應時主動接續」：前一則不是 user、且再前一則也不是 user，
            # 代表討論停在別人那裡而使用者主動接手。
            gaps = sum(
                1 for i, u in enumerate(utts)
                if u.speaker_id == "user" and i >= 2
                and utts[i - 1].speaker_id != "user"
                and utts[i - 2].speaker_id != "user"
            )
            signals["initiated_after_gap"] = float(gaps)
            parts.append(f"在他人連續發言後主動接續 {gaps} 次")

        # 背景資訊：合約允許帶，但用 bg_ 前綴標記它不參與評分依據
        signals[f"{_BG_PREFIX}utterance_count"] = float(len(mine))
        parts.append(f"（背景：共發言 {len(mine)} 次，依 babble 假說不作為等第依據）")

        return CollabSignal(COLLAB_DIM_NAMES[0], signals, "；".join(parts))

    def _argument(self, mine: list[Utterance]) -> CollabSignal:
        """論點建構。因果連接詞密度 ＋ 發言長度分佈。

        密度用「每百字」而非「每則發言」——後者會讓話多的人分數高，
        那又繞回發言量了。
        """
        if not mine:
            return CollabSignal(COLLAB_DIM_NAMES[2], {}, "使用者沒有發言")

        joined = "".join(u.text for u in mine)
        chars = _char_len(joined)
        connectors = _count_any(joined, CAUSAL_CONNECTORS)
        rate = (connectors / chars * 100) if chars else 0.0

        lengths = [_char_len(u.text) for u in mine]
        avg = sum(lengths) / len(lengths) if lengths else 0.0
        # 變異數大 = 有長有短,通常代表有實質論述也有簡短回應;
        # 全部一樣長多半是罐頭回答。這是描述性的,不預設哪個好。
        var = (sum((x - avg) ** 2 for x in lengths) / len(lengths)) ** 0.5 if lengths else 0.0

        signals = {
            "causal_connector_rate": round(rate, 3),
            "causal_connector_count": float(connectors),
            "avg_utterance_chars": round(avg, 1),
            "utterance_length_sd": round(var, 1),
        }
        return CollabSignal(
            COLLAB_DIM_NAMES[2], signals,
            f"因果連接詞 {connectors} 次、每百字 {rate:.1f}；"
            f"平均發言 {avg:.0f} 字（標準差 {var:.0f}）",
        )

    # ---------------------------------------------------------------- 等前端的兩項

    def _responsiveness(self, has_speakers: bool) -> CollabSignal:
        """傾聽與回應。**要比對前一位發言者說了什麼**。

        ★ 沒有他人發言時回空,不做任何近似。
          可以想到的替代品(例如數「剛剛那位」「我同意」這類詞)測的是
          「有沒有做出回應的姿態」,不是「有沒有接住論點」——
          那會產出一個看起來合理、實際上量錯東西的數字,比沒有數字糟。
        """
        if not has_speakers:
            return CollabSignal(
                COLLAB_DIM_NAMES[1], {},
                "輸入僅含使用者發言，無他人論點可比對——此項不可用",
            )
        return CollabSignal(
            COLLAB_DIM_NAMES[1], {},
            "有他人發言，但指涉偵測待 rubric 定案後實作",
        )

    def _collaboration(self, has_speakers: bool) -> CollabSignal:
        """協作姿態。

        ★ 打斷次數合約明確禁用——群面畫面在 AI 發言時以 isTyping 擋住輸入,
          使用者實際上打斷不了,這個指標結構上恆為 0。
          同意／反對詞比例要知道是在回應誰,一樣要等發言者資料。
        """
        if not has_speakers:
            return CollabSignal(
                COLLAB_DIM_NAMES[3], {},
                "輸入僅含使用者發言，無法判定回應對象——此項不可用"
                "（打斷偵測依合約禁用：介面結構上無法打斷）",
            )
        return CollabSignal(
            COLLAB_DIM_NAMES[3], {},
            "有他人發言，但同意／反對比待 rubric 定案後實作",
        )


class FakeCollabObserver:
    """跟合約的 Fake 同義，放這裡讓 A 側測試不必 import contracts 的 Fake。"""

    def observe(self, utterances: list[Utterance]) -> list[CollabSignal]:
        return [CollabSignal(n, {}, "fake") for n in COLLAB_DIM_NAMES]
