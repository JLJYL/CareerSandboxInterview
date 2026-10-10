"""AgentState 三層的生命週期測試。

這三條就是 Qi et al.(2026)講的 persona drift 在我們這邊的防線:

    Long   整場不變
    Mid    只增不減
    Short  不跨輪累積

第一條是會抓到「AI-邏輯 退化成禮貌面試官」的那一條。

【為什麼測型別不測行為】
這一版只宣告形狀,三層都還是空的,沒有行為可測。
但形狀本身就是規則 —— 這裡驗的是「違反規則的寫法做不出來」。
"""

from __future__ import annotations

import dataclasses

import pytest

from app.pipeline.agent_state import (
    AGENT_KINDS,
    EMPTY_AFFECT,
    MEMORY_KINDS,
    Affect,
    AgentState,
    Identity,
    MemoryEntry,
    SessionMemory,
)
from app.prompts.interview_personas import GROUP_PERSONAS_SOLO


def _peer() -> AgentState:
    persona = next(p for p in GROUP_PERSONAS_SOLO if p.role == "peer")
    return AgentState(agent_id=persona.id, kind="peer", identity=persona)


def _entry(turn: int, kind: str = "said", text: str = "我提議先做使用者訪談") -> MemoryEntry:
    return MemoryEntry(turn_idx=turn, kind=kind, text=text)


# ---------------------------------------------------------------------------
# Long —— 整場不變
# ---------------------------------------------------------------------------


def test_persona_unchanged_across_turns():
    """跑完五輪之後,identity 還是同一個物件。

    這是 persona drift 的防線:身分層只要有任何一條路徑可以被換掉,
    就會有人在某一輪「順手」改它。
    """
    state = _peer()
    original = state.identity
    for i in range(5):
        state = state.advanced(
            remembered=(_entry(i),), affect=Affect("insistent", "被跳過")
        )
    assert state.identity is original


def test_advanced_has_no_identity_parameter():
    """advanced() 不接受 identity —— 規則靠介面保證,不靠自律。"""
    params = AgentState.advanced.__kwdefaults__ or {}
    assert "identity" not in params
    with pytest.raises(TypeError):
        _peer().advanced(identity=None)  # type: ignore[call-arg]


def test_state_is_frozen():
    state = _peer()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.identity = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Mid —— 只增不減
# ---------------------------------------------------------------------------


def test_memory_is_monotonic():
    """每一輪之後,先前的記憶逐筆還在,而且順序不變。"""
    state = _peer()
    seen: list[MemoryEntry] = []
    for i in range(4):
        e = _entry(i)
        seen.append(e)
        state = state.advanced(remembered=(e,))
        assert list(state.memory.entries) == seen


def test_memory_has_no_removal_entry_point():
    """SessionMemory 沒有任何刪除或覆寫的公開方法。"""
    public = {n for n in dir(SessionMemory) if not n.startswith("_")}
    assert not public & {"remove", "pop", "clear", "replace", "set", "update"}


def test_ignored_count_reads_only_ignored_entries():
    state = _peer()
    state = state.advanced(remembered=(_entry(0, "said"), _entry(0, "ignored")))
    state = state.advanced(remembered=(_entry(1, "ignored"), _entry(1, "conflict")))
    assert state.memory.ignored_count() == 2
    assert len(state.memory.entries) == 4


def test_memory_kinds_are_the_three_we_declared():
    """次數不在裡面 —— 那個用 spoken_by 數就有,而且數得比模型準。"""
    assert MEMORY_KINDS == ("said", "ignored", "conflict")


# ---------------------------------------------------------------------------
# Short —— 不跨輪累積
# ---------------------------------------------------------------------------


def test_affect_not_carried_over():
    """這一輪沒有推導出狀態,上一輪的就不該還在。"""
    state = _peer().advanced(affect=Affect("insistent", "被跳過兩次"))
    assert state.affect.label == "insistent"

    state = state.advanced(remembered=(_entry(2),))
    assert state.affect.is_empty()
    assert state.affect == EMPTY_AFFECT


def test_empty_affect_is_not_a_label():
    """空的 label 代表「這一輪沒推導出東西」,不是某個預設狀態。"""
    assert EMPTY_AFFECT.is_empty()
    assert EMPTY_AFFECT.label == ""
    assert EMPTY_AFFECT.note == ""


# ---------------------------------------------------------------------------
# 形狀
# ---------------------------------------------------------------------------


def test_persona_spec_satisfies_identity():
    """現有的 PersonaSpec 直接就能當身分層,不必改它。"""
    persona = GROUP_PERSONAS_SOLO[0]
    assert isinstance(persona, Identity)


def test_three_layers_start_empty():
    """只宣告形狀的這一版:Mid 與 Short 都是空的,行為不變。"""
    state = _peer()
    assert state.memory.entries == ()
    assert state.affect.is_empty()
    assert state.kind in AGENT_KINDS
