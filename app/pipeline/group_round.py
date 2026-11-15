"""群面「一題一輪」:主持方出一題,每位應徵者各答一次,全部答完才換下一題。

【跟原本的自由討論差在哪】
原本的群面是小組討論:每輪由派發決定誰接話,同儕互相反駁,主持人只在
討論卡住時開口。11/15 改成輪流回答——每一題每個人只講一次,講完就輪到
下一位,全部講完主持方才出下一題。後答的人可以接前面的人的說法。

【誰先答】
不做搶答按鈕。每位 AI 應徵者有自己的反應速度:強勢的快、邏輯居中、親切
的慢、沉默的最慢,每一題再加一點隨機,所以順序不會每題都一樣。使用者只要
在某位 AI 開口前開始回答,就排在他前面。這個「誰先開口」不列入計分。

使用者答完之後,這一題剩下的 AI 不再模擬快慢,直接依序答完——等待時間是
為了留空檔給使用者,他答過了就沒有理由再讓他等。AI 都答完而使用者還沒答,
就等他答完才換題。

【這支服務照舊不存狀態】
這一題是第幾題、誰答過了,全部從 groupSays 推回來:最後一則主持方的話就是
目前的題目,它後面的發言就是這一題的回答。計時只有 App 做得到,所以這裡只
給「下一位是誰、要等多久」,等不等、使用者有沒有搶先開口由 App 判斷。

【同一題裡順序要穩定】
App 每換一位就呼叫一次。反應時間如果每次呼叫都重抽,同一題的順序會在兩次
呼叫之間跳來跳去。所以隨機的部分用 (場次, 題目, 角色) 雜湊出來:同一題算幾次
都一樣,換一題就不同。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from app.prompts.interview_personas import personas_for
from app.schemas.interview import UtteranceDTO

ROUND = "round"
DISCUSSION = "discussion"
GROUP_FLOWS = (ROUND, DISCUSSION)

REACTION_BASE_MS: dict[str, int] = {
    "peer_assertive": 1500,
    "peer_logic": 3500,
    "peer_friendly": 5000,
    "peer_quiet": 6500,
}
"""每位 AI 應徵者從上一則發言出現到他開口的基本等待,毫秒。

鍵是 persona id,不是顯示名稱——顯示名稱會隨調校改,id 不會。
相鄰兩位差 1.5 到 2 秒,加上 ±1.2 秒的隨機,相鄰的兩位有時會換順序,
但強勢排在沉默後面的情況不會發生。
"""

REACTION_DEFAULT_MS = 4000
"""不在上表裡的角色(之後新增的 persona)用這個值,排在中間。"""

REACTION_JITTER_MS = 1200
REACTION_MIN_MS = 800
"""等待的下限。太短的話使用者看完上一則還來不及按錄音,搶先開口就做不到。"""


def reaction_ms(persona_id: str, question: str, seed: str = "") -> int:
    """這位 AI 在這一題的反應時間。同一組 (seed, 題目, 角色) 永遠算出同一個值。"""
    base = REACTION_BASE_MS.get(persona_id, REACTION_DEFAULT_MS)
    digest = hashlib.md5(f"{seed}|{question}|{persona_id}".encode("utf-8")).digest()
    frac = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    jitter = round((frac * 2 - 1) * REACTION_JITTER_MS)
    return max(REACTION_MIN_MS, base + jitter)


@dataclass(frozen=True)
class RoundState:
    """從 groupSays 推回來的這一題現況。"""

    question: str
    """這一題:最後一則主持方的發言。找不到時用開場題。"""
    asker: str
    """出這一題的主持方。"""
    index: int
    """第幾題,從 1 起算。開場題是第 1 題。"""
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
    """本場主持方出過的題目,依序。"""


@dataclass(frozen=True)
class RoundStep:
    """這一輪該發生的事。"""

    kind: str
    """"peer" 一位 AI 應徵者回答 / "await_user" 等使用者 / "host" 主持方出下一題。"""
    speaker: str = ""
    delay_ms: int = 0
    """App 從上一則發言出現起要等多久才顯示這一則。使用者答過之後一律是 0。"""
    position: int = 0
    """這一位是這一題第幾個回答的,從 1 起算。只有 peer 有值。"""


def round_state(
    group_says: Sequence[UtteranceDTO],
    *,
    group_interviewers: int = 1,
    group_size: int = 4,
    fallback_question: str = "",
) -> RoundState:
    """把 groupSays 讀成這一題的現況。

    App 會把開場題記進 groupTranscript,所以正常情況下第一則就是主持方。
    沒有任何主持方發言的話(舊版 App 沒記開場題),退回 `fallback_question`,
    也就是 askedQuestions 的第一筆,所有發言都算這一題的回答。
    """
    personas = personas_for("group", group_interviewers, group_size)
    hosts = tuple(p.display_name for p in personas if p.role != "peer" and p.display_name)
    peer_specs = [p for p in personas if p.role == "peer" and p.display_name]
    peers = tuple(p.display_name for p in peer_specs)
    peer_ids = tuple((p.display_name, p.id) for p in peer_specs)

    says = [u for u in group_says if (u.content or "").strip()]
    host_at = [i for i, u in enumerate(says) if not u.is_user and u.speaker in hosts]
    if host_at:
        last = host_at[-1]
        question, asker = says[last].content.strip(), says[last].speaker
        after = says[last + 1 :]
        asked = tuple(says[i].content.strip() for i in host_at)
        index = len(host_at)
    else:
        question = (fallback_question or "").strip()
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
    )


def peer_order(state: RoundState, seed: str = "") -> list[str]:
    """這一題還沒答的 AI 應徵者,依反應時間由快到慢。同分時照 persona 順序。"""
    ids = dict(state.peer_ids)
    pending = [p for p in state.peers if p not in state.answered]
    return sorted(
        pending,
        key=lambda n: (reaction_ms(ids.get(n, n), state.question, seed), state.peers.index(n)),
    )


def next_host(state: RoundState) -> str:
    """出下一題的主持方。三位主管時輪流,開場是第一位,第二題換第二位。"""
    if not state.hosts:
        return ""
    return state.hosts[state.index % len(state.hosts)]


def next_step(state: RoundState, seed: str = "") -> RoundStep:
    """這一輪該發生什麼。

    還有 AI 沒答 → 最快的那位回答;使用者還沒答,他要等自己的反應時間,
    答過了就不等。AI 都答完而使用者還沒 → 等使用者。全員答完 → 主持方出下一題。
    """
    order = peer_order(state, seed)
    if order:
        nxt = order[0]
        ids = dict(state.peer_ids)
        delay = 0 if state.user_answered else reaction_ms(ids.get(nxt, nxt), state.question, seed)
        return RoundStep("peer", nxt, delay, position=len(state.answers) + 1)
    if not state.user_answered:
        return RoundStep("await_user")
    return RoundStep("host", next_host(state), 0)
