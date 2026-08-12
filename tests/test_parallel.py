"""並行工具與追問規則的測試。

不打真 LLM,用 asyncio.sleep 模擬等待。
執行:pytest tests/test_parallel.py -v
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.pipeline.parallel import BlockFailure, gather_blocks
from app.prompts.probe_rules import (
    MAX_FOLLOW_UP_PER_QUESTION,
    PROBE_PROHIBITIONS,
    compose_probe_rules,
)


async def _ok(value: str, delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return value


async def _boom() -> str:
    await asyncio.sleep(0.01)
    raise RuntimeError("模型回了不合法的 JSON")


async def _slow() -> str:
    await asyncio.sleep(5.0)
    return "太慢了"


# ---------------------------------------------------------------------------
# 並行:基本行為
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_succeed() -> None:
    results, notices = await gather_blocks(
        {"a": _ok("A"), "b": _ok("B"), "c": _ok("C")}
    )
    assert results == {"a": "A", "b": "B", "c": "C"}
    assert notices == []


@pytest.mark.asyncio
async def test_empty_input() -> None:
    assert await gather_blocks({}) == ({}, [])


@pytest.mark.asyncio
async def test_actually_parallel() -> None:
    """五個各 0.2 秒的工作,並行總時間應遠小於序列的 1.0 秒。"""
    t = time.perf_counter()
    results, _ = await gather_blocks(
        {f"b{i}": _ok(str(i), delay=0.2) for i in range(5)}
    )
    elapsed = time.perf_counter() - t
    assert len(results) == 5
    assert elapsed < 0.5, f"看起來是序列執行,耗時 {elapsed:.2f} 秒"


# ---------------------------------------------------------------------------
# 並行:失敗隔離(這是相對參考來源最重要的改進)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failure_does_not_kill_the_rest() -> None:
    """一個區塊失敗時,其餘四個的結果必須保留。

    參考來源的寫法會讓整包 gather 失敗,五個成功的結果全部丟掉。
    """
    results, notices = await gather_blocks(
        {
            "face_dimensions": _ok("三維"),
            "sub_scores": _ok("六項"),
            "star_parts": _boom(),
            "question_feedbacks": _ok("逐題"),
            "missing_points": _ok("漏講"),
        }
    )
    assert "star_parts" not in results
    assert len(results) == 4
    assert len(notices) == 1
    assert "star_parts" in notices[0]
    assert "RuntimeError" in notices[0]


@pytest.mark.asyncio
async def test_all_fail_returns_empty_not_exception() -> None:
    """全部失敗也不可以拋例外,要回空結果加 notices。"""
    results, notices = await gather_blocks({"a": _boom(), "b": _boom()})
    assert results == {}
    assert len(notices) == 2


# ---------------------------------------------------------------------------
# 並行:逐工作計時
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_block_times_out_alone() -> None:
    """慢的那個單獨逾時,其餘正常回來。

    參考來源把 timeout 包在 gather 外層,一個慢就取消全部。
    """
    results, notices = await gather_blocks(
        {"fast_a": _ok("A"), "slow": _slow(), "fast_b": _ok("B")},
        timeout=0.3,
    )
    assert results == {"fast_a": "A", "fast_b": "B"}
    assert len(notices) == 1
    assert "逾時" in notices[0]


@pytest.mark.asyncio
async def test_timeout_clock_starts_after_semaphore() -> None:
    """排隊等號誌的時間不可以算進逾時。

    否則批次執行時,後面排隊的工作會全部誤判逾時。
    """
    results, notices = await gather_blocks(
        {f"b{i}": _ok(str(i), delay=0.2) for i in range(6)},
        timeout=0.5,
        max_concurrency=2,  # 六個工作分三批,總耗時約 0.6 秒
    )
    assert len(results) == 6, f"不應有逾時,實際 notices={notices}"


# ---------------------------------------------------------------------------
# 並行:速率限制
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_is_enforced() -> None:
    """同時執行數不得超過上限,否則批次評測會撞 OpenAI 速率限制。"""
    active = 0
    peak = 0

    async def counted() -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "ok"

    await gather_blocks({f"b{i}": counted() for i in range(10)}, max_concurrency=3)
    assert peak <= 3, f"同時數衝到 {peak},號誌沒生效"


# ---------------------------------------------------------------------------
# BlockFailure
# ---------------------------------------------------------------------------


def test_block_failure_equality() -> None:
    assert BlockFailure("a", "逾時") == BlockFailure("a", "逾時")
    assert BlockFailure("a", "逾時") != BlockFailure("b", "逾時")


# ---------------------------------------------------------------------------
# 追問規則
# ---------------------------------------------------------------------------


def test_asked_topics_appear_in_rules() -> None:
    """已問領域必須實際帶進 prompt。

    只寫「不得重複」而不給清單,模型會自行想像哪些問過了,結果照樣重複。
    """
    text = compose_probe_rules(["資料分析", "團隊衝突"])
    assert "資料分析" in text
    assert "團隊衝突" in text


def test_rules_work_with_no_asked_topics() -> None:
    text = compose_probe_rules([])
    assert "尚未涵蓋" in text


def test_prohibitions_are_patterns_not_examples() -> None:
    """規則必須寫成模式。列舉例句會讓模型只避開那幾句。"""
    assert "「" not in PROBE_PROHIBITIONS or "例如" not in PROBE_PROHIBITIONS


def test_follow_up_cap_is_one() -> None:
    """對齊前端 followUpIdx 的控制邏輯。"""
    assert MAX_FOLLOW_UP_PER_QUESTION == 1
