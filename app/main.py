"""FastAPI 应用：会话/分叉/追加/预测/关闭/占用统计的 HTTP 接口 + 单页网页。

追加有两种入口：
- `POST /api/sessions/{id}/append`：同步执行，容量不足直接 409（原有行为）；
- `POST /api/sessions/{id}/queue` 与 `POST /api/queue`：追加任务队列，按会话
  公平轮转、入队后可取消、容量不足等待回收后重试。

队列只存在于进程内存，服务重启后未执行任务全部丢弃。
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .blocks import CapacityError
from .engine import Engine, NotFoundError, ValidationError
from .queue import AppendQueue, QueueClosedError, QueueFullError

STATIC = Path(__file__).parent / "static" / "index.html"


class CreateSessionReq(BaseModel):
    tokens: list[int]


class ForkReq(BaseModel):
    branch_id: str
    position: int
    name: str | None = None


class AppendReq(BaseModel):
    branch_id: str
    tokens: list[int]


class EnqueueReq(AppendReq):
    session_id: str


def _engine_from_env() -> Engine:
    return Engine.from_config(
        vocab_size=int(os.getenv("VOCAB_SIZE", "64")),
        d_model=int(os.getenv("D_MODEL", "32")),
        block_size=int(os.getenv("BLOCK_SIZE", "4")),
        num_blocks=int(os.getenv("NUM_BLOCKS", "32")),
        seed=int(os.getenv("SEED", "1234")),
    )


def _queue_from_env(engine: Engine) -> AppendQueue:
    return AppendQueue(engine,
                       max_queued=int(os.getenv("MAX_QUEUED", "100")),
                       max_history=int(os.getenv("MAX_HISTORY", "200")))


def create_app(engine: Engine | None = None,
               queue: AppendQueue | None = None) -> FastAPI:
    eng = engine or _engine_from_env()
    q = queue or _queue_from_env(eng)
    # 由本应用创建的队列随应用生命周期启停；外部注入的队列由调用方负责。
    owns_queue = queue is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if owns_queue:
            q.start()
        try:
            yield
        finally:
            if owns_queue:
                q.stop()

    app = FastAPI(title="KV-Cache 分叉对比", lifespan=lifespan)
    app.state.engine = eng
    app.state.queue = q

    @app.exception_handler(NotFoundError)
    async def _404(_: Request, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValidationError)
    async def _400(_: Request, exc: ValidationError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(CapacityError)
    async def _409(_: Request, exc: CapacityError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(QueueFullError)
    async def _429(_: Request, exc: QueueFullError):
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.exception_handler(QueueClosedError)
    async def _503(_: Request, exc: QueueClosedError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC)

    @app.post("/api/sessions", status_code=201)
    def create_session(req: CreateSessionReq):
        return eng.create_session(req.tokens)

    @app.get("/api/sessions")
    def list_sessions():
        return eng.list_sessions()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return eng.get_session(session_id)

    @app.delete("/api/sessions/{session_id}")
    def close_session(session_id: str):
        return eng.close_session(session_id)

    @app.post("/api/sessions/{session_id}/fork", status_code=201)
    def fork(session_id: str, req: ForkReq):
        return eng.fork(session_id, req.branch_id, req.position, req.name)

    @app.post("/api/sessions/{session_id}/append")
    def append(session_id: str, req: AppendReq):
        return eng.append(session_id, req.branch_id, req.tokens)

    @app.get("/api/sessions/{session_id}/branches/{branch_id}/predict")
    def predict(session_id: str, branch_id: str):
        return eng.predict(session_id, branch_id)

    @app.delete("/api/sessions/{session_id}/branches/{branch_id}")
    def close_branch(session_id: str, branch_id: str):
        return eng.close_branch(session_id, branch_id)

    @app.get("/api/stats")
    def stats():
        return eng.stats()

    # ---------- 追加任务队列 ----------

    @app.post("/api/sessions/{session_id}/queue", status_code=202)
    def enqueue(session_id: str, req: AppendReq):
        return q.submit(session_id, req.branch_id, req.tokens)

    @app.post("/api/queue", status_code=202)
    def enqueue_any(req: EnqueueReq):
        return q.submit(req.session_id, req.branch_id, req.tokens)

    @app.get("/api/queue")
    def list_jobs(session_id: str | None = None):
        return q.list(session_id)

    @app.get("/api/queue/status")
    def queue_status():
        return q.snapshot()

    @app.get("/api/queue/{job_id}")
    def get_job(job_id: str):
        return q.get(job_id)

    @app.delete("/api/queue/{job_id}")
    def cancel_job(job_id: str):
        return q.cancel(job_id)

    return app


app = create_app()
