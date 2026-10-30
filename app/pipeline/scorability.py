"""可評性閘門。在算分之前先問:這一輪有沒有東西可以評?

【要解決什麼】
實測(`score_fluency` / `score_depth` 直接呼叫):

    「不知道。」            4 字   表達流暢度 90   內容深度 55
    「喔好啊這樣啊對啊。」   14 字   表達流暢度 90   內容深度 55
    正常回答(填充詞 3.0/百字)      表達流暢度 82
    刻意講亂(填充詞 9.8/百字)      表達流暢度 60

亂答比認真回答高 8 分。原因不是公式寫錯:兩個維度量的都是密度與處數,
而短又乾淨的回答填充詞密度是 0,自然落在最高的 band。公式對填充詞率是
單調的,但填充詞率對品質不是。

注意失誤出在**短而乾淨**那一端,不是「講得亂」那一端——刻意講亂拿 60,
那是對的。所以閘門要看的是長度與內容,不是填充詞率。

【為什麼不是把分數調低就好】
低分的語意是「評過了,表現不好」。沒有回答的那一輪沒有被評過,
輸出一個低分是在宣稱一件沒有發生的事。這跟協作四維的既有做法一致:
`collab_score._to_dim` 遇到 level 0 回 None,註解寫著「那代表未評分,
不是評了 0 分。補一個 0 分進報告會讓使用者以為自己這一項拿了零」。

六項分數一直沒有這一層,所以同一份報告裡協作那半邊會說「這次觀察不到」,
六項那半邊會補一個數字。

【對齊既有實務】
e-rater 的 advisory flags 走同一條路:Off topic / Too short /
Insufficient development 在 prescreening 階段標記,部分回答直接
bypass 自動評分系統。見 Zhang et al. (2016),ETS RR,ERIC EJ1124780。

ETS 的 constructed-response 最佳實務也把「確認演算法不會給無意義的回答
高分」列為自動評分的效度證據之一(McCaffrey et al., 2022, §1.2.3)。
那一句就是這個模組存在的理由。

【「不知道」為什麼算可評,不算不可評】
他有回應。「我沒做過這件事」本身就是關於那項能力的觀察證據,
給最低錨點是誠實的評分,不是懲罰。不可評要留給**看不出他想講什麼**的
情況——空白、太短、離題。這是 B 與 A、C 的分界,也是 e-rater 把
`Insufficient development` 和 `Off topic` 分成兩個 flag 的同一個理由。

反過來說也成立:全場答「不知道」如果輸出一份六項全空的報告,
使用者什麼都得不到。最低錨點加上說明才有資訊。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# 與 interview_live、turn_decision、turn_emotion 共用同一組字面,改要一起改。
from app.pipeline.interview_live import _ADMITS_UNKNOWN
from app.prompts.collab_rubric import LEVEL_TO_SCORE

SCORABLE = "SCORABLE"
"""有內容可以評,照常走評分。"""

ADMITS_UNKNOWN = "ADMITS_UNKNOWN"
"""明確表示答不出來。**可評**,給最低錨點。"""

TOO_SHORT = "TOO_SHORT"
"""短到看不出他想講什麼。不可評。"""

LABELS: tuple[str, ...] = (SCORABLE, ADMITS_UNKNOWN, TOO_SHORT)
"""目前真的會出現的標籤。

【為什麼沒有 OFF_TOPIC 與 NONSENSE】
兩個都判不出來:

  OFF_TOPIC   需要把回答跟題目做語意比對。現有的 `is_near_duplicate`
              是純字面重疊,對這件事沒有用——答非所問的回答跟題目
              字面重疊可能很高(重複題目裡的詞),也可能很低(換了話題
              但講得很認真)。這件事屬於 K 那一項(語意重複偵測),
              跟那一項一起做才有意義。

  NONSENSE    沒有任何校準資料。黃金集十份全是認真回答。

先放一個永遠不會觸發的標籤,比沒有這個標籤更糟——它看起來像做過了。
要加的時候連同判定一起加。
"""

MIN_SCORABLE_CHARS = 10
"""短於這個字數視為沒有內容。這個門檻只擋「幾乎等於沒講話」。

【為什麼訂得這麼低——第一版訂 30,是錯的】
第一版想用長度判斷「這是不是認真的回答」,寫測試時被這兩句打掉:

    「我用 SQL 重寫查詢」        11 字,**是真的回答**,只是很薄
    「喔好啊這樣啊對啊我覺得還好」 14 字,是廢話

長度分不開這兩個,而且方向還相反。想用長度判斷內容品質是搞錯了判準。

所以這個常數退回它唯一撐得住的用途:**偵測幾乎沒講話**。
薄而真的回答會通過,然後在各維度上拿它應得的低分——那是對的,
薄回答本來就該被評成薄。

【那 A(亂答拿 90)怎麼辦】
A 的根因不在這裡。流暢度量的是每百字的填充詞密度,14 個字裡觀察到
0 個填充詞,對一個每百字 3.0 的正常講者來說完全是預期結果——
那個比率在這個長度下沒有解析度。那是樣本量問題,不是內容判斷問題,
所以修在 `score_fluency` 的 `MIN_FLUENCY_CHARS`,不修在這裡。

【要真的分辨廢話與薄回答】
需要看有沒有實詞、有沒有具體對象,也就是語意。那跟 OFF_TOPIC / NONSENSE
是同一個前提,見 `LABELS` 的註解。在那之前不要靠調高這個值去逼近它。
"""

ADMITS_UNKNOWN_MAX_CHARS = 20
"""「不知道」的字面命中只在這個長度以內才算數。

【為什麼要長度上限】
`_ADMITS_UNKNOWN` 是子字串比對。這個回答會命中「不確定」:

    「我不確定當時的確切數字,但那一季的回訪率從 18% 拉到 26%。」(40 字)

那是一個有內容、有量化成果的好回答。沒有上限的話它會被打到最低錨點,
而且錯得很難發現——分數低看起來只是「這題答得不好」。
第一版把上限訂在 60,寫測試時就是被這一句抓到的。

【為什麼訂得這麼緊】
真正的「我答不出來」短:

    「不知道。」                  4 字
    「我不知道耶。」               6 字
    「這個我真的沒想過,抱歉。」     12 字

兩種錯的代價不對稱。誤判成答不出來 → 一個好回答被打到 45,使用者看不出
哪裡錯了。漏掉一個長的答不出來 → 照常評分,也就是今天的行為,沒有變糟。
所以門檻往漏掉那一側偏。

【已知會漏掉的】
「我其實沒有做過這方面的專案,所以沒辦法給你具體的例子,真的很抱歉。」
32 字,是誠實的答不出來,但會被判成 SCORABLE 照常評分。要收掉這一類
得看語意而不是長度,跟 OFF_TOPIC 同一個前提。

【三處共用這個上限】
原本只有這裡設上限,`turn_emotion` 與 `turn_decision.forced_decision` 用同一組
字面而不設界,理由是「情緒標錯的代價小」。前端 10/29 的「自我貶低」錄音推翻了這個
判斷:第 1 題「我講得不太好……但我也不確定做得對不對,應該沒有很厲害」
(59 字)命中「不確定」,情緒被程式直接標成卡住,而且蓋過模型的判讀——
那一題要測的正是過度自責。同一句在面試中也會讓決策直接換題,
「他明確表示答不出來」,而他其實講了一段。

所以判斷收成一支 `admitted_phrase`,三處都呼叫它。長的誠實承認仍然交給模型:
情緒那邊模型還是可以標卡住(要抄得到原文),只是程式不再替它下判斷。
"""


def admitted_phrase(answer: str) -> str | None:
    """他有沒有明確說答不出來。有的話回命中的字面,沒有回 None。

    字面命中而且整段夠短才算。理由見 `ADMITS_UNKNOWN_MAX_CHARS`。
    """
    text = (answer or "").strip()
    if len(text) > ADMITS_UNKNOWN_MAX_CHARS:
        return None
    return next((k for k in _ADMITS_UNKNOWN if k in text), None)

LOWEST_ANCHOR = LEVEL_TO_SCORE[1]
"""最低錨點。取協作四維 BARS 的等第 1。

不自己寫一個 45,是因為兩邊是同一個刻度:`collab_rubric.LEVEL_TO_SCORE`
的註解寫明它「對齊 subScores 的區間語意(85 以上強項、70–84 正常、
55–69 有缺口、55 以下幾乎沒表現)」。同一個常數寫兩次,調一邊就會悄悄分岔。
"""


@dataclass(frozen=True)
class Verdict:
    """一輪的可評性判定。"""

    label: str
    reason: str

    @property
    def scorable(self) -> bool:
        """進不進評分。`ADMITS_UNKNOWN` 也算進得去,只是分數固定。"""
        return self.label in (SCORABLE, ADMITS_UNKNOWN)


def classify(answer: str) -> Verdict:
    """判斷單一輪的回答可不可以評。純程式,不呼叫 LLM。

    順序有意義:先看長度再看字面。「不知道。」同時符合兩個條件,
    判成 `ADMITS_UNKNOWN` 才對——他有回應,只是答不出來。
    判成 `TOO_SHORT` 會讓他的報告少一項,而不是拿到一個誠實的低分。
    """
    text = (answer or "").strip()
    hit = admitted_phrase(text)

    if hit:
        return Verdict(ADMITS_UNKNOWN, f"明確表示答不出來(「{hit}」)")
    if len(text) < MIN_SCORABLE_CHARS:
        return Verdict(TOO_SHORT, f"回答只有 {len(text)} 字,不足以評分")
    return Verdict(SCORABLE, "")


@dataclass(frozen=True)
class Scorability:
    """整場的可評性。由逐輪判定彙整。"""

    verdicts: tuple[Verdict, ...]

    @property
    def scorable_turns(self) -> int:
        return sum(1 for v in self.verdicts if v.label == SCORABLE)

    @property
    def admits_turns(self) -> int:
        return sum(1 for v in self.verdicts if v.label == ADMITS_UNKNOWN)

    @property
    def scored(self) -> bool:
        """這場能不能產生分數。"""
        return self.scorable_turns > 0 or self.admits_turns > 0

    @property
    def anchored(self) -> bool:
        """是不是只剩「不知道」——那就是最低錨點,不是不評分。"""
        return self.scorable_turns == 0 and self.admits_turns > 0

    def reason(self) -> str:
        """給 notices 與 basis 用的一句話。可評時回空字串。"""
        if self.scorable_turns:
            return ""
        if self.admits_turns:
            return f"{self.admits_turns} 題都明確表示答不出來,此項為最低錨點"
        if not self.verdicts:
            return "這場沒有任何回答"
        n = len(self.verdicts)
        return f"{n} 題的回答都太短,沒有可以評分的內容"


def assess(answers: Sequence[str]) -> Scorability:
    """整場的可評性判定。

    【為什麼是整場彙整,不是逐項算分】
    兩個公式維度吃的是整份逐字稿的 `TextStats`,不是逐輪的。四題裡有一題
    亂答,整份統計幾乎不動,那一題本來就被稀釋掉了——那不是 A/B/C 要解決的。

    A/B/C 是**整場都沒有可評內容**的情況:亂答一通、全部答不知道、
    或根本沒講話。所以這裡只需要整場這一層。

    逐輪分數要等逐輪 TextStats 做出來才有意義,那是另一件事。
    """
    return Scorability(tuple(classify(a) for a in answers))
