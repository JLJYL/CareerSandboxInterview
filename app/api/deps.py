"""服務層的相依組裝:昂貴元件在啟動時建立一次,請求期間共用。

【為什麼要有這一支】
TranscriptAnalyzer 會載入 bge-m3(約 2.3GB)。每個請求建立新實例的話,
第一次呼叫就會逾時,而且記憶體會爆。合約明文要求它是 app 啟動時的 singleton。

成員 A 已經把 near_misses / residuals 改成回傳值而不是實例狀態,
所以共用實例是安全的——早期版本兩支執行緒交錯呼叫時,
A 的請求會讀到 B 的結果。

【兩個非預設參數,漏掉會靜默降級】
    engine_filler_policy="partial"
        對應 STT 探測的實測結論:Android 移除非詞彙填充音(嗯、呃),
        保留詞彙型(那個、就是、然後)。不帶的話 filler_reliability 一律回
        "unknown",表達流暢度就永遠走備援訊號——那是整個模組唯一乾淨的
        量化訊號,白白丟掉很可惜。

    LLM 的兩個模型
        一般用 gpt-4o-mini;starParts 的 fromAnswer 需要跨行原文引用,
        mini 做不到,要用 gpt-4o。

【LLM 注入而不是在 pipeline 裡 import】
pipeline 全部收 (system, user) -> str 的可呼叫物件。
所以整個專案只有這一支 import SDK,測試不會意外打到 API。
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from functools import lru_cache
from typing import Any

LLMCall = Callable[[str, str], str]

ROOT = pathlib.Path(__file__).resolve().parents[2]
VOCAB_PATH = pathlib.Path(os.getenv("VOCAB_PATH", ROOT / "fixtures" / "vocab" / "skills_v1.json"))
STT_ALIASES_PATH = pathlib.Path(
    os.getenv("STT_ALIASES_PATH", ROOT / "data" / "stt_confusions.v1.json")
)

# 啟動時建立,請求期間共用。由 lifespan 填入。
_analyzer: Any | None = None
_gap: Any | None = None
_collab: Any | None = None


def load_vocab() -> list:
    raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return raw.get("skills", raw) if isinstance(raw, dict) else raw


def load_stt_aliases() -> dict[str, str]:
    """讀 STT_TERM_PROBE 的安全別名表。

    只回傳 ``safe_aliases``——碰撞別名(colliding_aliases)不能無條件套用,
    要靠履歷做條件判定,見 GapComputer.resolve_confusions,不在這裡處理。

    檔案是照 Android 內建語音辨識實測的(見檔案 device 欄位)。改接 Whisper
    API 後轉寫形會不同,這張表屆時要整份換掉,但讀取路徑不必再動——
    新表就位後直接覆蓋 STT_ALIASES_PATH 指向的檔案即可。

    找不到檔案時回空 dict 並靜默降級,不擋啟動:沒有別名表只是
    退回掃全詞彙表比對,不是致命錯誤。
    """
    try:
        raw = json.loads(STT_ALIASES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(raw.get("safe_aliases", {}))


def build_components() -> tuple[Any, Any, Any]:
    """建立三個共用元件。啟動時呼叫一次。

    語意段預設關閉(enable_semantic=False)。開啟它需要 embedding 模型,
    而校準還沒做完——W2 的實驗顯示打開之後候選池會多出很多池外的東西。
    要開啟時由環境變數控制,不要改這裡的預設值。
    """
    from app.contracts.interview_protocols import FakeCollabObserver
    from app.pipeline.gap import GapComputer
    from app.pipeline.transcript import TranscriptAnalyzer

    analyzer = TranscriptAnalyzer(
        load_vocab(),
        enable_semantic=os.getenv("ENABLE_SEMANTIC", "0") == "1",
        engine_filler_policy="partial",
        stt_aliases=load_stt_aliases(),
    )
    gap = GapComputer(analyzer)

    # CollabObserver:成員 A 已交真實作,但 BARS rubric 還沒到,
    # level 一律 None。這裡優先用真實作,沒有才退回 Fake。
    try:
        from app.pipeline.collab import CollabObserver  # type: ignore

        collab: Any = CollabObserver()
    except ImportError:
        collab = FakeCollabObserver()

    return analyzer, gap, collab


def set_components(analyzer: Any, gap: Any, collab: Any) -> None:
    global _analyzer, _gap, _collab
    _analyzer, _gap, _collab = analyzer, gap, collab


def get_analyzer() -> Any:
    if _analyzer is None:
        raise RuntimeError("TranscriptAnalyzer 尚未初始化,lifespan 沒有跑到")
    return _analyzer


def get_gap() -> Any:
    if _gap is None:
        raise RuntimeError("GapComputer 尚未初始化,lifespan 沒有跑到")
    return _gap


def get_collab() -> Any:
    if _collab is None:
        raise RuntimeError("CollabObserver 尚未初始化,lifespan 沒有跑到")
    return _collab


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _client() -> Any:
    from openai import OpenAI

    return OpenAI()


def make_llm(model: str, temperature: float = 0.0) -> LLMCall:
    """回傳 (system, user) -> str。

    temperature 依用途分:
        0.0  抽取、評分、報告——要可重現。同一份逐字稿跑兩次結果要一樣,
             否則使用者重新整理就看到不同分數。
        0.4  出題——全 0 會讓追問變得很制式,每次都是同一句。
    """

    def call(system: str, user: str) -> str:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    return call


def get_llm() -> LLMCall:
    """一般用途。評分與報告要可重現,所以 temperature=0。"""
    return make_llm(os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 0.0)


def get_llm_verbatim() -> LLMCall:
    """starParts 專用。跨行原文引用 mini 做不到。"""
    return make_llm(os.getenv("OPENAI_MODEL_VERBATIM", "gpt-4o"), 0.0)


def get_llm_live() -> LLMCall:
    """出題專用。要有變化,不然每場面試問一樣的話。"""
    return make_llm(os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 0.4)
