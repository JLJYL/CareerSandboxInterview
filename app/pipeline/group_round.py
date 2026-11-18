"""群面「一題一輪」:主持方出一題,點名請每位依序回答,全部答完才換下一題。

【跟原本的自由討論差在哪】
原本的群面是小組討論:每輪由派發決定誰接話,同儕互相反駁,主持人只在
討論卡住時開口。11/15 改成輪流回答——每一題每個人只講一次,全部講完主持方
才出下一題。後答的人可以接前面的人的說法。

【誰先答:主持方點名】11/18 改
主持方出題時最後說「請 X 先說」,之後每位答完就點下一位:「謝謝。下一位請
AI-親切。」順序每題隨機,使用者也排在裡面。點名句是程式套的,不經過模型——
名字和順序不能錯,也省一次呼叫。主持方不講「第幾題」,模型自己加了也會拿掉。
11/15 到 11/18 早上的版本沒有點名,AI 依各自的反應速度搶著答。

【搶答保留】
還沒被點到,使用者也可以先開口:在下一則(點名句或 AI 的回答)顯示之前按下
錄音,那一則就丟掉,使用者排到前面。他答完之後,主持方點的是原本順序裡還沒
答的第一位。誰先開口不列入計分。

【等多久】
每一則都等使用者讀完上一則才出現,一個字 0.12 秒(大約一秒八個字)。被點到的
AI 另外停一下,強勢 0.5 秒、沉默 2 秒;主持方點名或出題前停一秒。上一則是使用者
自己的回答時不用讀,主持方只停一秒。兩位 AI 之間大約 8 到 9 秒。

調過兩次。11/15 只有反應時間,11/17 實測一則接一則,使用者還沒讀完上一則,下一則
就出來了。11/18 早上改成講話速度(一字 0.22 秒)加 1.5 到 6.5 秒的停頓,再加上主持方
點名,11/18 下午實測第一位 AI 要等二十幾秒、兩位 AI 之間十幾秒,太慢。

【每個人答的不一樣】
11/17 實測 AI 應徵者答的是同一個答案換句話說。人格寫的是怎麼回應別人,沒有寫
怎麼答題,所以同一題大家都收斂到最穩的那個答案。現在每一題由程式分給每位
一個切入角度,互不重複,其中一位固定唱反調;生成之後再比對這一題前面的
回答,太像就重生成一次。規則寫在 prompt 裡只會被換句話說繞過去,所以分配
和比對都放在程式。

【這支服務照舊不存狀態】
第幾題、誰答過了、下一個點誰,全部從 groupSays 推回來:最後一則主持方出題的
話就是目前的題目,點名句不算(靠固定句型認出來,見 `is_cue_line`);它後面的
應徵者發言就是這一題的回答。計時只有 App 做得到,所以這裡只給「下一則是什麼、
要等多久」,使用者有沒有搶先開口由 App 判斷。

【同一題裡順序要穩定】
App 每換一則就呼叫一次。順序用 (場次, 第幾題) 雜湊出來:同一題算幾次都一樣,
換一題就不同。不用題目的字,因為下一題由誰先答要在題目生成之前決定——點名
就寫在題目最後。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from app.prompts.interview_personas import personas_for
from app.schemas.interview import UtteranceDTO

ROUND = "round"
DISCUSSION = "discussion"
GROUP_FLOWS = (ROUND, DISCUSSION)

USER_SLOT = "你"
"""順序裡代表使用者的那一格,也是點名時對他的稱呼。群面逐字稿裡使用者的 speaker 也是「你」。"""

REACTION_BASE_MS: dict[str, int] = {
    "peer_assertive": 500,
    "peer_logic": 1000,
    "peer_friendly": 1500,
    "peer_quiet": 2000,
}
"""每位 AI 應徵者被點到之後、開口之前的停頓,毫秒。點名句的閱讀時間另外加,見 `speak_ms`。

鍵是 persona id,不是顯示名稱——顯示名稱會隨調校改,id 不會。
11/18 早上這張表決定誰先答,值是 1.5 到 6.5 秒;改成主持方點名之後只剩停頓長短,
被點到名的人不會再想五六秒才開口,那段時間只是空等,所以縮成 0.5 到 2 秒。
強勢的人被點到馬上開口,沉默的人想一下。
"""

REACTION_DEFAULT_MS = 1200
"""不在上表裡的角色(之後新增的 persona)用這個值。"""

REACTION_JITTER_MS = 500
REACTION_MIN_MS = 300
"""反應時間的下限。"""

HOST_PAUSE_MS = 1000
"""主持方在別人講完之後、開口點名或出題之前的停頓。"""

SPEAK_MS_PER_CHAR = 120
"""使用者讀上一則,一個字要多久。大約一秒八個字。

App 上的發言是文字,使用者讀完上一則、想一下,才可能搶在下一位前面開口。
11/18 早上用講話速度(0.22 秒),加上主持方點名之後每一則前面都多一段等待,
實測太慢,改成閱讀速度:四十字的回答大約五秒,點名句大約一秒。
函式名稱 `speak_ms` 沒改,值是讀完的時間。
"""

SPEAK_MAX_MS = 8000
"""閱讀時間的上限。很長的一則(主持方的長題目、使用者的長回答)不讓等待拉得太長。"""

_UNSPOKEN = re.compile(r"[\s,，。、?？!！:：;；「」『』()（）…—\-~～\"'“”‘’]")
"""空白和標點不念出來,不算字數。"""


def speak_ms(text: str) -> int:
    """讀完這一則要多久,毫秒。標點和空白不算。"""
    chars = len(_UNSPOKEN.sub("", text or ""))
    return min(SPEAK_MAX_MS, chars * SPEAK_MS_PER_CHAR)


def _hash_frac(*parts: str) -> float:
    """把幾個字串雜湊成 [0, 1] 的數。同樣的輸入永遠得到同樣的值。"""
    digest = hashlib.md5("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def reaction_ms(persona_id: str, question: str, seed: str = "") -> int:
    """這位 AI 在這一題被點到之後的停頓。同一組 (seed, 題目, 角色) 永遠算出同一個值。"""
    base = REACTION_BASE_MS.get(persona_id, REACTION_DEFAULT_MS)
    jitter = round((_hash_frac(seed, question, persona_id) * 2 - 1) * REACTION_JITTER_MS)
    return max(REACTION_MIN_MS, base + jitter)


# ---------------------------------------------------------------------------
# 點名
# ---------------------------------------------------------------------------

CUE_NEXT = ("謝謝。下一位請 {who}。", "好,接下來請 {who}。")
CUE_LAST = "謝謝。最後請 {who}。"
CUE_USER_NEXT = "謝謝。接下來請你。"
CUE_USER_LAST = "謝謝。最後請你。"
"""點名句。換人時輪流用兩種講法,最後一位用「最後」。

句型要固定:App 會把點名句原封不動記進 groupSays,下一輪要靠句型認出來,
才不會被當成新的一題。改句型就要一起改 `_CUE_RE`,測試會比對兩邊。
"""

_NAME = r"[^\s。,，][^。,，]{0,15}"
_CUE_RE = re.compile(
    rf"^(?:謝謝[。.]下一位請 |好[,，]接下來請 |謝謝[。.]最後請 ){_NAME}[。.]$"
    r"|^謝謝[。.](?:接下來|最後)請你[。.]$"
)
_DESIGNATION_RE = re.compile(rf"(?:請你先說|請 {_NAME} 先說)[。.]$")


def cue_line(who: str, *, last: bool = False, turn: int = 0) -> str:
    """主持方點 `who` 回答的那一句。`turn` 是這一題已經答了幾位,用來換講法。"""
    if who == USER_SLOT:
        return CUE_USER_LAST if last else CUE_USER_NEXT
    if last:
        return CUE_LAST.format(who=who)
    return CUE_NEXT[turn % len(CUE_NEXT)].format(who=who)


def is_cue_line(text: str) -> bool:
    """這一句是不是主持方的點名句。點名句不是題目,它後面的回答還算同一題。"""
    return bool(_CUE_RE.match((text or "").strip()))


def designation(who: str) -> str:
    """出題時接在題目後面的那一句:誰先答。"""
    return "請你先說。" if who == USER_SLOT else f"請 {who} 先說。"


_NUMBERING = re.compile(
    r"(?<![^。,，!！?？:：\s])"
    r"(?:我們)?(?:來看|進入|來到|接下來是|再來是|接著是)?\s*"
    r"(?:第\s*[一二三四五六七八九十兩0-9０-９]+\s*(?:題|個問題)|下一題|下一個問題|最後一題|最後一個問題)"
    r"\s*[:：,，。、]?\s*"
)
NUMBERING_SCAN = 30
"""「第幾題」只在開頭這幾個字裡找,而且要在句首或標點後面。
「跟第一題比」的「第一題」前面是「跟」,不動。"""


def strip_question_number(text: str) -> str:
    """拿掉「第二題:」「下一題,」「我們來看第三題。」這類編號,題目本身不動。

    使用者要求主考官直接講問題。prompt 也有寫,這裡是模型沒照做時的保險。
    """
    t = (text or "").strip()
    m = _NUMBERING.search(t)
    if not m or m.start() >= NUMBERING_SCAN:
        return t
    head, tail = t[: m.start()].rstrip(), t[m.end() :].lstrip()
    return (head + tail).lstrip(",，。、:： ").strip()


def strip_designation(text: str) -> str:
    """拿掉題目最後的「請 X 先說。」。"""
    return _DESIGNATION_RE.sub("", (text or "").strip()).strip()


def clean_question(text: str) -> str:
    """題目本身:沒有編號、沒有點名。給 prompt、比對重複、雜湊用。"""
    return strip_designation(strip_question_number(text))


def with_designation(question: str, who: str) -> str:
    """題目加上點名,畫面上看到的那一句。"""
    q = clean_question(question)
    if q and q[-1] not in "。?？!！.":
        q += "。"
    return q + designation(who)


def round_order(peers: Sequence[str], index: int, seed: str = "") -> tuple[str, ...]:
    """第 `index` 題主持方點名的順序,使用者用 `USER_SLOT` 表示。

    每題隨機,使用者的位置也隨機;同一組 (seed, 第幾題, 人) 永遠同一個順序。
    """
    slots = list(dict.fromkeys(peers)) + [USER_SLOT]
    return tuple(sorted(slots, key=lambda s: (_hash_frac(seed, str(index), "order", s), s)))


# ---------------------------------------------------------------------------
# 這一題的現況
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundState:
    """從 groupSays 推回來的這一題現況。"""

    question: str
    """這一題,去掉編號和點名。找不到主持方出題時用開場題。"""
    asker: str
    """出這一題的主持方,也是這一題負責點名的人。"""
    index: int
    """第幾題,從 1 起算。開場題是第 1 題。點名句不算題。"""
    answers: tuple[UtteranceDTO, ...]
    """這一題的回答,依發言順序,含使用者。"""
    answered: tuple[str, ...]
    """這一題已經答過的 AI 應徵者,依發言順序。"""
    user_answered: bool
    peers: tuple[str, ...]
    """這一場的 AI 應徵者,依 persona 宣告順序。"""
    peer_ids: tuple[tuple[str, str], ...]
    """(顯示名稱, persona id),算反應時間用。"""
    hosts: tuple[str, ...]
    """主持方:一位主考官,或三位主管。"""
    asked: tuple[str, ...]
    """本場主持方出過的題目,依序,去掉編號和點名。"""
    last_line: str = ""
    """畫面上最後一則發言,算「讀完上一則要多久」用。"""
    last_kind: str = ""
    """最後一則是什麼:"question" 主持方出題 / "cue" 點名 / "peer" AI 回答 /
    "user" 使用者回答 / "" 什麼都還沒有。決定下一則是回答還是點名。"""


@dataclass(frozen=True)
class RoundStep:
    """這一輪該發生的事。"""

    kind: str
    """"peer" 被點到的 AI 回答 / "await_user" 輪到使用者 /
    "cue" 主持方點下一位 / "host" 主持方出下一題。"""
    speaker: str = ""
    delay_ms: int = 0
    """App 從上一則發言出現起要等多久才顯示這一則。"""
    position: int = 0
    """回答的是這一題第幾個,從 1 起算。"""
    target: str = ""
    """cue:被點到的人。host:下一題第一個答的人。使用者是 `USER_SLOT`。"""
    last: bool = False
    """cue:被點到的是不是這一題最後一位。"""


def round_state(
    group_says: Sequence[UtteranceDTO],
    *,
    group_interviewers: int = 1,
    group_size: int = 4,
    fallback_question: str = "",
) -> RoundState:
    """把 groupSays 讀成這一題的現況。

    App 會把開場題記進 groupTranscript,所以正常情況下第一則就是主持方。
    沒有任何主持方出題的話(舊版 App 沒記開場題),退回 `fallback_question`,
    也就是 askedQuestions 的第一筆,所有發言都算這一題的回答。
    """
    personas = personas_for("group", group_interviewers, group_size)
    hosts = tuple(p.display_name for p in personas if p.role != "peer" and p.display_name)
    peer_specs = [p for p in personas if p.role == "peer" and p.display_name]
    peers = tuple(p.display_name for p in peer_specs)
    peer_ids = tuple((p.display_name, p.id) for p in peer_specs)

    says = [u for u in group_says if (u.content or "").strip()]

    def kind(u: UtteranceDTO) -> str:
        if u.is_user:
            return "user"
        if u.speaker in hosts:
            return "cue" if is_cue_line(u.content) else "question"
        return "peer"

    kinds = [kind(u) for u in says]
    q_at = [i for i, k in enumerate(kinds) if k == "question"]
    if q_at:
        last = q_at[-1]
        question, asker = clean_question(says[last].content), says[last].speaker
        after = says[last + 1 :]
        asked = tuple(clean_question(says[i].content) for i in q_at)
        index = len(q_at)
    else:
        question = clean_question(fallback_question)
        asker = hosts[0] if hosts else ""
        after = says
        asked = (question,) if question else ()
        index = 1

    answers = tuple(u for u in after if u.is_user or u.speaker in peers)
    answered = tuple(
        dict.fromkeys(u.speaker for u in answers if not u.is_user and u.speaker in peers)
    )
    return RoundState(
        question=question,
        asker=asker,
        index=index,
        answers=answers,
        answered=answered,
        user_answered=any(u.is_user for u in answers),
        peers=peers,
        peer_ids=peer_ids,
        hosts=hosts,
        asked=asked,
        last_line=(says[-1].content.strip() if says else question),
        last_kind=(kinds[-1] if kinds else ""),
    )


def pending(state: RoundState, seed: str = "") -> list[str]:
    """這一題還沒答的人,照主持方點名的順序。使用者是 `USER_SLOT`。"""
    return [
        s
        for s in round_order(state.peers, state.index, seed)
        if not (state.user_answered if s == USER_SLOT else s in state.answered)
    ]


def next_host(state: RoundState) -> str:
    """出下一題的主持方。三位主管時輪流,開場是第一位,第二題換第二位。"""
    if not state.hosts:
        return ""
    return state.hosts[state.index % len(state.hosts)]


def next_step(state: RoundState, seed: str = "") -> RoundStep:
    """這一輪該發生什麼。

    上一則是主持方(出題或點名) → 被點到的那位回答:AI 等點名讀完再加自己的
    停頓;點到使用者就等他。上一則是回答 → 還有人沒答,主持方點順序裡還沒答的
    第一位;全員答完,主持方出下一題。使用者搶先開口時,上一則就是他的回答,
    所以主持方接著點的正好是被他搶先的那一位。
    """
    todo = pending(state, seed)
    position = len(state.answers) + 1
    if state.last_kind in ("question", "cue", "") and todo:
        who = todo[0]
        if who == USER_SLOT:
            return RoundStep("await_user", position=position, target=USER_SLOT)
        pid = dict(state.peer_ids).get(who, who)
        delay = speak_ms(state.last_line) + reaction_ms(pid, state.question, seed)
        return RoundStep("peer", who, delay, position=position, target=who)
    pause = HOST_PAUSE_MS + (0 if state.last_kind == "user" else speak_ms(state.last_line))
    if todo:
        return RoundStep(
            "cue",
            state.asker or next_host(state),
            pause,
            position=position,
            target=todo[0],
            last=len(todo) == 1,
        )
    first = round_order(state.peers, state.index + 1, seed)[0]
    return RoundStep("host", next_host(state), pause, target=first)


# ---------------------------------------------------------------------------
# 每個人答的不一樣
# ---------------------------------------------------------------------------


ANGLES: dict[str, str] = {
    "結論先行": "開口就講你的選擇,再補一個最關鍵的理由。",
    "講第一步": "不講大方向,講你第一件會做的事:找誰、做什麼、多久內做完。",
    "看數字": "用一個能量化的指標做決定,講出你會看哪個數字、到多少就怎麼做。",
    "看風險": "先講這件事最壞會怎樣,再講你怎麼預防或補救。",
    "看成本": "從時間、預算或人力的限制出發,講你願意犧牲什麼、保住什麼。",
    "從使用者看": "站在客戶或使用者那一邊,講他們真正在意什麼,再推回你的做法。",
    "從團隊看": "從分工和溝通出發,講你會讓誰參與、怎麼讓大家對齊。",
    "點出前提": "指出這一題大家容易忽略的前提或限制,講它不成立時你會怎麼做。",
    "舉經驗": "用一個你做過或看過的具體例子撐你的答案:一句講例子,一句講結論。",
}
"""一題裡分給每位 AI 應徵者的切入角度。名稱進 notices,說明進 prompt。"""

DISSENT = "唱反調"
DISSENT_FIRST = "你是第一個回答的人:選一個多數人直覺不會選的答案,講出你的理由。"
DISSENT_LATER = (
    "你的答案要跟前面的人不一樣:直接講你不同意誰的哪一點,再給你的版本和理由。"
    "選不同的選項,或同一個方向但做法相反。"
)
"""每一題固定一位唱反調。前面還沒人答時改成選冷門的答案,沒有對象可以反對。"""

PERSONA_ANGLES: dict[str, tuple[str, ...]] = {
    "peer_assertive": ("結論先行", "講第一步", "看成本"),
    "peer_logic": ("看數字", "看風險", "點出前提"),
    "peer_friendly": ("從使用者看", "從團隊看", "看成本"),
    "peer_quiet": ("點出前提", "舉經驗", "看風險"),
}
"""每位人格偏好的角度。每一題從清單裡輪一個,撞到別人已經分到的就往下挑。

鍵是 persona id。不在表裡的角色(之後新增的)從全部角度裡挑。
唱反調不在任何人的清單裡,另外指定,確保每一題剛好一位。
"""


def round_angles(state: RoundState, seed: str = "") -> dict[str, str]:
    """這一題每位 AI 應徵者的切入角度,{顯示名稱: 角度名稱}。

    同一組 (seed, 題目) 永遠分出同一組角度:App 每換一位就呼叫一次,
    分配不能在兩次呼叫之間變。換一題就換一組。

    【怎麼分】
    兩位以上時先用雜湊挑一位唱反調。其餘的人依雜湊排出的順序挑:
    自己的偏好清單轉一個這一題專屬的位移,第一個還沒被挑走的就是他的;
    清單全被挑走時從全部角度裡補。所以同一題裡不會有兩個人拿到同一個角度。
    """
    ids = dict(state.peer_ids)
    q = state.question
    peers = list(state.peers)
    if not peers:
        return {}
    out: dict[str, str] = {}
    if len(peers) >= 2:
        out[peers[int(_hash_frac(seed, q, "dissent") * len(peers)) % len(peers)]] = DISSENT
    rest = sorted(
        (p for p in peers if p not in out),
        key=lambda n: (_hash_frac(seed, q, "order", ids.get(n, n)), n),
    )
    pool = sorted(ANGLES, key=lambda a: _hash_frac(seed, q, "pool", a))
    for name in rest:
        pid = ids.get(name, name)
        prefs = PERSONA_ANGLES.get(pid) or tuple(pool)
        shift = int(_hash_frac(seed, q, "shift", pid) * len(prefs)) % len(prefs)
        rotated = prefs[shift:] + prefs[:shift]
        taken = set(out.values())
        out[name] = next(
            (a for a in rotated if a not in taken),
            next((a for a in pool if a not in taken), rotated[0]),
        )
    return out


SIMILAR_THRESHOLD = 0.15
SIMILAR_MIN_SHARED = 3
"""兩則回答重疊到這個程度就算換句話說。

【比法】
兩個字一組(bigram)的重疊,除以較短那則的組數;題目裡出現過的組先拿掉,
不然大家都會用到題目的字,「先處理」「預算」這種詞會讓每一對都像。

【校準】11/18 用手寫的十四對,題目的字拿掉之後:

    換句話說(6 對)            0.19 到 0.39,共同組數 5 到 7
    同題不同角度(6 對)        0.00 到 0.06,共同組數 0 到 1
    整合前面兩位的回答(2 對)  0.00 到 0.04

門檻 0.15、至少 3 組,兩邊都有餘裕。最低的那一對換句話說是
「我會先跟主管確認優先順序…」對「我會先找主管討論哪件事比較重要…」,
用字幾乎全換了,只剩 0.19——字面比對抓得到的大概就到這裡。

誤判的代價是多一次生成;漏判是使用者又聽到一次同樣的答案。
等有真實對話之後要用同一場的配對重新校準。
"""

_FILLER = set("的了嗎呢吧啊喔我你他她們這那就也都很會要是在有和跟把被讓再")
"""兩個字都是這些虛字的組不算,「我會」「就是」這種組每一則都有。"""


def _bigrams(text: str) -> set[str]:
    t = _UNSPOKEN.sub("", text or "")
    return {
        t[i : i + 2]
        for i in range(len(t) - 1)
        if not (t[i] in _FILLER and t[i + 1] in _FILLER)
    }


def answer_overlap(text: str, other: str, question: str = "") -> tuple[int, float]:
    """(共同組數, 重疊比例)。題目的字不算。"""
    skip = _bigrams(question)
    a, b = _bigrams(text) - skip, _bigrams(other) - skip
    if not a or not b:
        return 0, 0.0
    shared = len(a & b)
    return shared, shared / min(len(a), len(b))


def most_similar(
    text: str, answers: Sequence[UtteranceDTO], question: str = ""
) -> tuple[UtteranceDTO | None, float]:
    """這一題前面的回答裡跟 `text` 最像的那一則,和它的重疊比例。

    共同組數不到 `SIMILAR_MIN_SHARED` 的不算——短句兩三組重疊,比例就會衝很高。
    """
    best: UtteranceDTO | None = None
    best_ratio = 0.0
    for u in answers:
        shared, ratio = answer_overlap(text, u.content or "", question)
        if shared >= SIMILAR_MIN_SHARED and ratio > best_ratio:
            best, best_ratio = u, ratio
    return best, best_ratio


def too_similar(
    text: str, answers: Sequence[UtteranceDTO], question: str = ""
) -> UtteranceDTO | None:
    """跟這一題前面哪一則回答太像。沒有就 None。使用者的回答也算在內。"""
    best, ratio = most_similar(text, answers, question)
    return best if best is not None and ratio >= SIMILAR_THRESHOLD else None
