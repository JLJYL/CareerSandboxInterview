"""面試模組的 FastAPI 應用。

    uvicorn app.main:app --reload --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import build_components, set_components
from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("載入共用元件…")
    analyzer, gap, collab = build_components()
    set_components(analyzer, gap, collab)
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
    return {"status": "ok", "service": "interview"}
