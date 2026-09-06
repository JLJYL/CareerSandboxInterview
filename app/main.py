"""面試模組的 FastAPI 應用。

    uvicorn app.main:app --reload --port 8001

【埠號】
前端的 ApiClient.BASE_URL 目前指向 8000(資料庫組的服務)。
本服務獨立部署,預設 8001。前端需要另建一組 Retrofit 實例——見交接清單。

【啟動時做什麼】
lifespan 建立三個共用元件。TranscriptAnalyzer 會載入 bge-m3(約 2.3GB),
每個請求建立新實例會逾時,所以必須在這裡做一次。

【為什麼是 deps.build_components() 而不是頂端 import】
    from app.api.deps import build_components   ← 不要這樣寫

測試用 monkeypatch 換掉 deps 模組上的屬性來避免真的載 bge-m3。
頂端匯入會在匯入當下把原始函式綁死在這個模組的名稱空間裡,patch 換不到;
而且模組只匯入一次,第一個測試的綁定會留到後面全部。

這是成員 A 把分支合進 feat/a-semantic 時發現的——
我原本寫成頂端匯入,測試沒紅是因為 fixture 剛好同時 patch 了
deps 與 routes 兩個模組,繞過去了。她的寫法才對。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import deps
from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("載入共用元件…")
    analyzer, gap, collab = deps.build_components()
    deps.set_components(analyzer, gap, collab)
    log.info("共用元件就緒:%s", type(collab).__name__)
    yield
    log.info("關閉")


app = FastAPI(
    title="CareerSandbox Interview",
    description="面試模擬:A1 一對一、A2 主管 panel、A3 群面、A4 報告、A5 漏講點",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """給前端與部署檢查用。

    不做深度檢查——回 200 只代表 app 起來了。
    元件是否就緒由第一個真實請求驗證,那時 get_analyzer() 會明確報錯,
    而不是在這裡回一個似是而非的 degraded 狀態。
    """
    return {"status": "ok", "service": "interview"}
