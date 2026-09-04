"""群面協作切片器(成員 A,W3)。

## 為什麼整個重寫

回應 B 在 `feat/b-collab-e2e` 分支對合約的改版:`CollabObserver.observe()`
現在回 `list[CollabSlice]`,不再回 `list[CollabSignal]`。

W2 版本交的四組可觀察值裡,兩組已知失效:

    first_speak_position   rubric 明文列入禁用(首次發言早晚),
                            且與發言量指標同屬「babble 假說」的偏誤來源
    causal_connector_rate  A 自己實測沒有鑑別力:兩段同樣「正常發揮」的
                            錄音,每百字連接詞 0.00 vs 1.50,橫跨整個量程

框架詞(framing_in_first)、發言長度分佈(avg/sd)也一併拿掉——
不是這兩個被證明沒用,是**整個「A 產出數字」的設計被 rubric 推翻**:
剩下能問的問題(「是否推動討論收斂」「是否正確指涉他人」「否定前有沒有
先承接」)本質是語意判斷,確定性規則做不到,硬擠數字出來就是在編。

所以這版不產出任何 `signals`。A 的角色改成**切片**:依照
`app/prompts/collab_rubric.py` 裡四個維度各自的判準,把逐字稿切成
B 的 LLM 需要看的那一份。切片本身是機械操作、可測試、跟語意判斷無關;
語意判斷(這一段算幾級)全部留給 B 那邊的四次獨立 LLM 呼叫。

## 切片規則(對照 collab_rubric.py 的 BARS 逐條核對)

    參與主動性  使用者每一則發言,標記「接續他人之後」或「自己起頭」。
                標記只是脈絡,不是評分依據——rubric 原文如此。
    傾聽與回應  使用者發言 + 往前找到的最近一則他人發言,成對出現。
                沒有這種配對時回空——資訊不在場,不是難。
    論點建構    使用者每一則發言,單則獨立。這是唯一不需要他人發言
                在場就能切的維度,rubric 明訂「只看一段發言內部」。
    協作姿態    整段討論,不切。立場衝突可能出現在任何地方,
                切成段落反而會漏掉,交給 LLM 自己找。

## 關於 is_user,不要用 speaker_id 判斷

前端實際送的 `speaker_id` 是「你」,不是 `"user"`。W2 版本寫
`speaker_id == "user"` 曾經讓四個維度全部切錯,而且不會報錯——
切片是空的,LLM 收到空切片回 `level 0`,報告顯示「這次沒有可觀察的
內容」,看起來像正常降級。這版全部改用合約新加的 `Utterance.is_user`。

## 「接續 / 起頭」判斷粒度:跟 B 核對過的決策

B 在合約檔裡示範的 Fake,「參與主動性」的接續/起頭標記是**整場只判一次**
(看整段逐字稿裡有沒有出現過他人,不分則)。Fake 只是為了讓管線跑得起來,
不要求語意正確。這版採**逐則、且用浮動的「最近一次他人」**判斷,理由:

    rubric 原文「每則發言標了它是接續還是起頭」,字面是逐則,排除整場一判。

    剩下要選的是「起頭」的定義:嚴格看陣列位置緊鄰的前一則,還是往前找
    最近一次真正的話題轉換點。選浮動版——使用者把一次回應拆成連續兩則,
    中間沒有人插話,那兩則在功能上仍是同一次「接續」,不該因為斷句方式
    被標成「自己起頭」。這跟「傾聽與回應」要不要配對用的是同一個判斷,
    兩個維度現在共用同一次掃描(`_with_nearest_other`),邏輯只寫一次。

這不在合約凍結範圍內(合約只定義資料形狀,沒規定切法的演算法),
但這個標記會直接進 B 的 prompt,已跟 B 核對過這個定義符合他寫 prompt
時的預期。

## 引號邊界:evidence 逐字檢查的前提

B 的 `verify_evidence` 要求 LLM 回的 `evidence` 是逐字稿裡真的出現過的
子字串,用來防止亂編引用。這代表**我在切片裡加的標籤(接續/起頭、
講者名稱)絕對不能跟真正的發言文字黏在一起沒有邊界**——LLM 照 prompt
指示「引用逐字稿裡的話」,很可能連標籤一起抄進去,那樣的話跟原始逐字稿
(`routes.py` 組的是純發言內容,不含這些標籤)對不起來,會被誤判成編造。

所以四個維度裡,只要標籤跟文字同一則出現,文字一律用「」包起來——
讓 LLM 有清楚的邊界知道「這裡面才是真正的話」。這是實際串起
`CollabObserver.observe()` → `collab_score.score_collab()` 走一次
才發現的,單獨測切片格式測不出來,因為問題出在下游怎麼消費這個格式。
"""

from __future__ import annotations

from app.contracts.interview_protocols import CollabSlice, Utterance
from app.schemas.interview import COLLAB_DIM_NAMES


def _clean(text: str) -> str:
    return (text or "").strip()


def _with_nearest_other(utts: list[Utterance]) -> list[Utterance | None]:
    """對每一則發言，標出「往前看最近一次別人講的話」是哪一則。

    同一份掃描結果同時餵給「參與主動性」（有沒有 = 接續 / 起頭）與
    「傾聽與回應」（要跟誰配對）——兩者問的其實是同一個底層問題，
    只是一個只要「有沒有」，一個還要「內容是什麼」。

    「最近一次」用浮動視窗，不是嚴格陣列相鄰：使用者連續講兩則，
    兩則都對應同一則他人發言，直到下一位他人出現才更新。
    這則本身若是他人發言，對應值填 None（呼叫端只會用到 is_user 那些列）。

    回傳跟 utts 等長的清單。
    """
    result: list[Utterance | None] = []
    pending_other: Utterance | None = None
    for u in utts:
        if u.is_user:
            result.append(pending_other)
        else:
            pending_other = u
            result.append(None)
    return result


class CollabObserver:
    """CollabObserver Protocol 的真實作(W3,切片版)。

    用法:

        observer = CollabObserver()
        slices = observer.observe(session.utterances)   # 四份,依 COLLAB_DIM_NAMES 順序
    """

    def observe(self, utterances: list[Utterance]) -> list[CollabSlice]:
        """回傳四份切片,順序對齊 COLLAB_DIM_NAMES。"""
        utts = list(utterances or [])

        return [
            self._participation(utts),
            self._responsiveness(utts),
            self._argument(utts),
            self._collaboration(utts),
        ]

    # ---------------------------------------------------------------- 參與主動性

    def _participation(self, utts: list[Utterance]) -> CollabSlice:
        """使用者每一則發言,標記接續他人之後,還是自己起頭。

        用浮動的「最近一次他人」判斷(`_with_nearest_other`),不是嚴格
        陣列相鄰:使用者連續講兩則,只要中間沒有人插話,兩則都算接續
        同一次他人發言。「起頭」代表往前完全沒有他人發言過(整場第一次
        開口,或這之前只有使用者自己在講)。標記不預設哪個好——rubric
        原文:「接續也可以是主動推進」。
        """
        nearest = _with_nearest_other(utts)
        excerpts: list[str] = []
        for u, prior_other in zip(utts, nearest):
            if not u.is_user:
                continue
            text = _clean(u.text)
            if not text:
                continue
            tag = "[接續他人之後]" if prior_other is not None else "[自己起頭]"
            excerpts.append(f"{tag}「{text}」")

        if not excerpts:
            return CollabSlice(COLLAB_DIM_NAMES[0], [], "使用者沒有發言")
        return CollabSlice(
            COLLAB_DIM_NAMES[0],
            excerpts,
            "每則發言已標記是接續他人還是自己起頭；"
            "標記只是脈絡，不是評分依據，不得以此判斷主動或被動。",
        )

    # ---------------------------------------------------------------- 傾聽與回應

    def _responsiveness(self, utts: list[Utterance]) -> CollabSlice:
        """使用者發言 + 往前找到的最近一則他人發言，成對出現。

        跟「參與主動性」共用同一次掃描（`_with_nearest_other`）：使用者
        若把回應拆成連續兩則，兩則仍配對同一位他人；直到下一則他人發言
        出現才更新配對對象。

        整段都沒有這種組合時回空——沒有前一位發言者的內容可比對，
        不是難，是資訊不在場，不做任何近似。
        """
        nearest = _with_nearest_other(utts)
        pairs: list[str] = []
        for u, prior_other in zip(utts, nearest):
            if not u.is_user or prior_other is None:
                continue
            text = _clean(u.text)
            other_text = _clean(prior_other.text)
            if not text or not other_text:
                continue
            pairs.append(
                f"他人（{prior_other.speaker_id}）：「{other_text}」\n你：「{text}」"
            )

        if not pairs:
            return CollabSlice(
                COLLAB_DIM_NAMES[1],
                [],
                "沒有「他人發言後使用者接著回應」的組合，此維度這次無法評分",
            )
        return CollabSlice(
            COLLAB_DIM_NAMES[1],
            pairs,
            "每組是一則他人發言與使用者其後的回應；只看這一對，"
            "不含更早的對話內容。",
        )

    # ---------------------------------------------------------------- 論點建構

    def _argument(self, utts: list[Utterance]) -> CollabSlice:
        """使用者每一則發言，單則獨立——只看發言內部的主張與理由。

        唯一不需要他人發言在場就能切的維度：純粹是使用者自己怎麼
        組織論點，跟群面裡有沒有其他人講話無關。
        """
        excerpts = [_clean(u.text) for u in utts if u.is_user and _clean(u.text)]
        if not excerpts:
            return CollabSlice(COLLAB_DIM_NAMES[2], [], "使用者沒有發言")
        return CollabSlice(
            COLLAB_DIM_NAMES[2],
            excerpts,
            "每則獨立評分；不因單一則特別好或特別差就拉高或拉低整體判斷，"
            "看的是這個人平常怎麼組織論點。",
        )

    # ---------------------------------------------------------------- 協作姿態

    def _collaboration(self, utts: list[Utterance]) -> CollabSlice:
        """整段討論，不切——立場衝突可能出現在任何地方，切片會漏掉。

        找不找得到分歧、分歧發生在哪一句，是語意判斷，交給 B 的 LLM。
        沒有他人發言時一樣把整段（僅使用者發言）給出，讓 LLM 自己判斷
        「這場討論沒有出現立場分歧」並回 level 0，而不是由 A side 先猜。
        """
        cleaned = [(u, _clean(u.text)) for u in utts]
        excerpts = [
            f"{'你' if u.is_user else u.speaker_id}：「{text}」"
            for u, text in cleaned
            if text
        ]

        if not excerpts:
            return CollabSlice(COLLAB_DIM_NAMES[3], [], "沒有任何發言")

        has_others = any(not u.is_user for u, text in cleaned if text)
        note = (
            "整段討論依原順序給出；找出分歧發生的段落、判斷是先承接還是"
            "直接否定，是語意判斷，交給評分者自行辨識。"
            if has_others
            else "輸入僅含使用者發言，沒有他人可能造成立場分歧，"
            "此維度預期無法觀察。"
        )
        return CollabSlice(COLLAB_DIM_NAMES[3], excerpts, note)


class FakeCollabObserver:
    """跟合約的 Fake 同義，放這裡讓 A 側測試不必 import contracts 的 Fake。"""

    def observe(self, utterances: list[Utterance]) -> list[CollabSlice]:
        return [CollabSlice(n, [], "fake") for n in COLLAB_DIM_NAMES]
