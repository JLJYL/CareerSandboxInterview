"""決策接進 next_turn 之後的行為。

【為什麼要獨立一支】
`test_turn_decision.py` 測的是決定本身,`test_interview_live.py` 測的是
生成那一段(它不傳 decision,維持舊行為)。這裡測的是**兩者接起來**
之後才出現的事:

    is_follow_up 從猜變成決定
    決定會進到生成端的 user 訊息
    輪次上限比決定大

【一個差點漏掉的】
decision 接上之後,`test_api` 的假 LLM 因為沒有決策分支,每一輪都走降級
(source=fallback、action=NEXT),`is_follow_up` 從 True 變成 False——
而 602 個測試全綠,沒有一個發現。假 LLM 少一條分支就會量到降級路徑,
所以這裡明確測 source。
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.interview_live import next_turn
from app.pipeline.interviewer_spec import derive_specs
from app.pipeline.turn_decision import CLARIFY, FOLLOW_UP, NEXT, TurnDecision

ANSWER = "我把每週三小時的人工整理改成自動產出。"
GEN = {"nextQuestion": "那個時間是怎麼量的?", "topic": "成果", "speaker": ""}


def gen_llm(system: str, user: str) -> str:
    """只負責生成。決策是另外傳進去的,不經過這裡。"""
    gen_llm.seen.append(user)
    return json.dumps(GEN, ensure_ascii=False)


gen_llm.seen = []  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _clear():
    gen_llm.seen = []  # type: ignore[attr-defined]


def spec():
    return derive_specs("single", jd=None, difficulty="中等")[0]


async def turn(decision, **kw):
    return await next_turn(
        mode=kw.pop("mode", "single"),
        answer=kw.pop("answer", ANSWER),
        follow_up_idx=kw.pop("follow_up_idx", 1),
        asked_questions=kw.pop("asked_questions", ["資料分析"]),
        fallback=kw.pop("fallback", ["備援題一"]),
        llm=gen_llm,
        decision=decision,
        **kw,
    )


# ---------------------------------------------------------------------------
# is_follow_up 從猜變成決定
# ---------------------------------------------------------------------------


async def test_no_decision_keeps_the_old_guess():
    """不傳 decision 時維持舊行為,既有測試才測得到生成那一段。"""
    r = await turn(None)
    assert r.is_follow_up is True


@pytest.mark.parametrize(
    "action,expected",
    [(FOLLOW_UP, True), (CLARIFY, True), (NEXT, False)],
)
async def test_is_follow_up_comes_from_the_decision(action, expected):
    r = await turn(TurnDecision(action))
    assert r.is_follow_up is expected


async def test_next_runs_the_duplicate_check():
    """決定是 NEXT 時新主問題要過重複偵測——一對一以前永遠跳過這一關。"""
    r = await turn(
        TurnDecision(NEXT),
        asked_questions=["那個時間是怎麼量的?"],
        fallback=["換個方向:你最近學了什麼?"],
    )
    assert r.next_question == "換個方向:你最近學了什麼?"
    assert any("重複" in n for n in r.notices)


async def test_follow_up_skips_the_duplicate_check():
    """追問本來就該留在同一個話題上,查重複會誤殺。"""
    r = await turn(TurnDecision(FOLLOW_UP), asked_questions=["那個時間是怎麼量的?"])
    assert r.next_question == GEN["nextQuestion"]
    assert not any("重複" in n for n in r.notices)


# ---------------------------------------------------------------------------
# 決定要傳到生成端
# ---------------------------------------------------------------------------


async def test_decision_reaches_the_generator():
    """不講的話決定說 NEXT 而模型生出追問,欄位就跟實際內容不符。"""
    await turn(TurnDecision(FOLLOW_UP, gap="缺結果"))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "【這一輪要做什麼】" in sent
    assert "不要換題" in sent


async def test_the_gap_does_not_reach_the_generator():
    """只傳動作,不傳 gap。

    1d 實測:A3 那筆答案講了結果只是沒有數字,20/20 次 gap 都寫「結果段」。
    照著生成的追問會去問他剛剛講過的東西。生成端看得到完整的回答,
    不需要一個會指錯的標籤。
    """
    await turn(TurnDecision(FOLLOW_UP, gap="缺結果"))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "缺結果" not in sent
    assert "缺的是" not in sent


async def test_clarify_tells_the_generator_not_to_apologise():
    await turn(TurnDecision(CLARIFY))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "不要為系統的問題道歉" in sent


async def test_no_decision_adds_nothing_to_the_prompt():
    await turn(None)
    assert "【這一輪要做什麼】" not in gen_llm.seen[-1]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 輪次上限比決定大
# ---------------------------------------------------------------------------


async def test_turn_cap_overrides_the_decision():
    """到上限的那一句就是最後一個主問題,語意上必然是 NEXT。

    決定說 FOLLOW_UP 也一樣——輪次是確定性事實,比判斷大。
    """
    r = await turn(TurnDecision(FOLLOW_UP), follow_up_idx=4)
    assert r.is_follow_up is False
    assert r.should_advance is True
    assert "【這一輪要做什麼】" not in gen_llm.seen[-1]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 降級看得見
# ---------------------------------------------------------------------------


async def test_source_is_recorded_in_notices():
    """code / llm / fallback 要看得出來,不然會把降級當成真正的決定。"""
    r = await turn(TurnDecision(NEXT, source="fallback"))
    assert any("fallback" in n for n in r.notices)
    r = await turn(TurnDecision(FOLLOW_UP, source="code"))
    assert any("code" in n for n in r.notices)
