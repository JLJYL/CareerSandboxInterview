"""從段落時間戳算出講話的時間特徵:停頓多長、講多快。

【為什麼要這個】
報告裡「長停頓」這個訊號原本是推論的:逐字稿轉完之後停頓資訊就不在文字裡了,
所以判讀只能從句子長短去猜。而 `app/prompts/interview_report.py` 開頭那句
「沒有音訊,所以不知道語速、音量、語調」是對的——音訊從來沒有離開手機。

但停頓與語速其實不需要音訊,只需要**每一段的開始與結束時點**。
前端的 STT 回呼本來就知道每段何時結束,記下來送出即可。
音量變異才真的需要波形,那是另一件事(見檔尾)。

【不確定就說不確定】
這個檔的每一個函式在資料不足或不一致的時候回 `unavailable`,不回估計值。
理由是 `transcript.py` 那條已經寫過的:過度宣稱比不宣稱糟。
「平均每段 76 字」被當成精確句長用過一次,而它其實是「她停頓前講了多長」。

所以 `SpeechTiming.source` 一定要被下游讀:它說的是這些數字能不能用,
而「沒有時間資料」與「量到停頓是 0」在數值上長得一樣。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TIMING_SOURCES: tuple[str, ...] = ("timestamps", "unavailable")
TIMESTAMPS, UNAVAILABLE = TIMING_SOURCES
"""`SpeechTiming.source` 的值域。

    timestamps   每段的開始與結束時點都有,而且自洽
    unavailable  缺一邊、對不起來、或這一輪是打字的

之後若接了音訊特徵,那會是第三個值(`audio`),而不是把 timestamps 的意思
改掉——兩者量到的東西不同,混成一個值就分不出這一輪的停頓是算出來的
還是測出來的。
"""

MIN_SEGMENT_MS = 100
"""短於這個長度的段落視為時間戳有問題,整輪判為不可用。

100ms 講不完一個字。出現這種值表示前端的時點對不上段落,
而對不上的時候算出來的停頓會是負的或離譜的大。
"""

GLUED_SEGMENTS = "首尾相接"
"""段與段之間全部是 0 毫秒時,整輪判為不可用。這個常數只是給那條規則一個名字。

【為什麼】
這支只算段與段之間的空隙(`start[i+1] - end[i]`),段內的停頓看不到。
前端回報的實測:App 測的五題裡四題的 Whisper segments 是整秒、而且一段的
結束剛好等於下一段的開始。那樣的資料算出來每個空隙都是 0,而原本的程式
照樣標 `timestamps`,`describe_pauses` 就產出

    這一題的段落之間共停頓 2 次,最長一次 0.0 秒,合計 0.0 秒(實測值)

一句自相矛盾的話,而且標著「實測值」。情緒判讀拿它當錨點,等於被告知
「這個人沒有停頓」——專門錄來測長停頓的那幾份會被讀成最流暢的。

這正是本檔開頭那句話講的:「沒有時間資料」與「量到停頓是 0」在數值上
長得一樣。首尾全部相接的時點不是量到 0,是時點本身沒有記錄靜音。

【只擋「全部」是 0,不擋「有一個」是 0】
兩句話之間真的可以接得很緊,一個 0 不代表什麼。全部都是 0 才是
時點沒在量靜音的特徵。

【真正的修法不在這裡】
要量得到停頓,時點得換成字級:Whisper 的 `timestamp_granularities` 加
`word`(要搭 `verbose_json`,而且會多一點延遲)。字與字之間的空隙才是
停頓。那是轉錄端的改動,這裡只負責在資料不夠的時候不要假裝夠。
"""

MAX_REASONABLE_PAUSE_MS = 60_000
"""長於這個長度的停頓視為不是停頓。

一分鐘沒出聲不是在想,是中斷(接電話、App 切到背景、引擎掛掉)。
把它當成停頓會讓這一輪的最長停頓變成一個沒有意義的大數字。
超過就整輪判為不可用,不是夾到上限——夾到上限等於宣稱「他停了 60 秒」。
"""


@dataclass(frozen=True)
class SpeechTiming:
    """一輪回答的時間特徵。`source` 說這些數字能不能用。"""

    source: str = UNAVAILABLE

    speaking_ms: int = 0
    """實際在講話的總時長,不含段落之間的停頓。"""

    pauses_ms: tuple[int, ...] = ()
    """段落之間的停頓,毫秒,照發生順序。"""

    chars: int = 0
    """這一輪講了幾個字。語速的分子。"""

    @property
    def usable(self) -> bool:
        return self.source != UNAVAILABLE

    @property
    def longest_pause_ms(self) -> int:
        return max(self.pauses_ms) if self.pauses_ms else 0

    @property
    def total_pause_ms(self) -> int:
        return sum(self.pauses_ms)

    @property
    def chars_per_second(self) -> float:
        """語速:字數除以實際講話時間,不含停頓。

        含停頓的話算出來的是「字數除以總時長」,那個數字同時反映講多快
        與停多久,兩個訊號混在一起之後哪一個都讀不出來。
        """
        return self.chars / (self.speaking_ms / 1000) if self.speaking_ms else 0.0


def measure(
    *,
    starts_ms: Sequence[int],
    ends_ms: Sequence[int],
    segments: Sequence[str],
    input_mode: str = "unknown",
) -> SpeechTiming:
    """算這一輪的時間特徵。資料不足或不一致就回 unavailable。

    【為什麼打字的一律不可用】
    打字沒有「講話時長」。硬算的話字數除以打字時間會變成打字速度,
    而那跟猶豫沒有關係——打字慢可能是在想,也可能是手機鍵盤難打。
    表達流暢度踩過同一個坑:打字的答案標點乾淨就拿到高流暢度。

    【為什麼 starts 可以比 segments 多一筆】
    `TurnDTO.segment_starts_ms` 的合約寫明可能多 1:使用者按下結束時,
    系統正要開始聽新的一段但那一段沒有內容。所以配對時以 ends 的筆數為準,
    多出來的 start 丟掉,不視為錯誤。
    """
    if input_mode != "voice":
        return SpeechTiming()
    n = min(len(starts_ms), len(ends_ms), len(segments))
    if n == 0:
        return SpeechTiming()

    spans = [(int(starts_ms[i]), int(ends_ms[i])) for i in range(n)]
    if any(e - s < MIN_SEGMENT_MS for s, e in spans):
        return SpeechTiming()
    # 段落必須照時間排好且不重疊。重疊表示時點不是同一個時間軸上的,
    # 那時候相減出來的停頓沒有意義。
    if any(spans[i + 1][0] < spans[i][1] for i in range(n - 1)):
        return SpeechTiming()

    pauses = tuple(spans[i + 1][0] - spans[i][1] for i in range(n - 1))
    if any(p > MAX_REASONABLE_PAUSE_MS for p in pauses):
        return SpeechTiming()
    # 段與段全部首尾相接 = 時點沒有記錄靜音,不是他一次都沒停。
    # 見 `GLUED_SEGMENTS` 的說明——不擋的話情緒那一側會收到
    # 「停頓 2 次,最長 0.0 秒(實測值)」,而那句話是錯的。
    if pauses and all(p == 0 for p in pauses):
        return SpeechTiming()

    return SpeechTiming(
        source=TIMESTAMPS,
        speaking_ms=sum(e - s for s, e in spans),
        pauses_ms=pauses,
        chars=sum(len(x) for x in segments[:n]),
    )


def describe_pauses(timing: SpeechTiming) -> str:
    """把停頓寫成一行給 prompt 當錨點。不可用時回空字串。

    【為什麼只給停頓,不給語速】
    文件那張判讀依據表裡有「長停頓」,沒有語速。而這一輪實測過:
    prompt 多列一條會改變行為(gap 的選項說明把追問率從 0.500 推到 0.567)。
    所以語速算出來放在 `SpeechTiming` 上備用,但不進 prompt——
    設計沒有在用的訊號進了 prompt 只是噪音。

    【為什麼是錨點而不是程式判定】
    「停多久算長」沒有校準資料。`score_fluency` 敢用程式判定是因為
    那幾條帶是校準過的;這裡沒有,所以只給數字讓模型對照,
    不自己訂一條門檻然後假裝它有依據。
    """
    if not timing.usable or not timing.pauses_ms:
        return ""
    longest = timing.longest_pause_ms / 1000
    total = timing.total_pause_ms / 1000
    # 0 毫秒的空隙是兩段接得很緊,不是一次停頓。算進「停頓 N 次」的話
    # 會出現「停頓 2 次、最長 0.0 秒」這種自相矛盾的句子。
    real = sum(1 for p in timing.pauses_ms if p > 0)
    return (
        f"這一題的段落之間共停頓 {real} 次,"
        f"最長一次 {longest:.1f} 秒,合計 {total:.1f} 秒(實測值)"
    )


# ---------------------------------------------------------------------------
# 之後要加音量變異的時候
# ---------------------------------------------------------------------------
#
# 音量變異需要波形,所以需要音訊離開手機:新端點、librosa 或 openSMILE、
# 錄音權限與流量。那是比這個檔大一個量級的改動。
#
# 接的方式是在 `TIMING_SOURCES` 加一個 `audio`,在 `SpeechTiming` 加欄位,
# 而不是改 `timestamps` 的意思——兩者量到的東西不同,混成一個值之後
# 分不出這一輪的停頓是從時間戳算的還是從波形測的。
#
# 要不要做應該等這一輪量完:音量變異不在文件的判讀依據表裡,
# 它是一個新訊號,而現在連既有四類準不準都還沒有數字。
