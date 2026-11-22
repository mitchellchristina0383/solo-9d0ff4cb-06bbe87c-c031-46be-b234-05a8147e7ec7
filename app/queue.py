"""追加任务队列：按会话公平轮转、同会话串行、入队后可取消、容量不足就地等待。

调度规则：
- 每个会话最多一条任务处于 `running`；同一会话的任务严格按提交顺序执行；
- 多会话之间轮转（round-robin）：一次执行只整体取走一个会话的队首任务，
  保证空闲容量被各会话轮流使用，而不是被先到的会话垄断；
- 任务在真正拿到 KV 块（引擎在锁内调用取消边界）之前都可以取消；
  跨过边界后不可取消，由引擎保证整体成功或整体回滚；
- 容量不足时任务留在 `queued`（状态里记录 `waiting_capacity`），
  不消耗重试次数；块被回收后由结束事件立即唤醒重试；
- 队列只在内存中：进程重启即丢弃全部未执行任务。
"""
from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from .blocks import CapacityError
from .engine import (AppendCancelled, Engine, NotFoundError, ValidationError)


class QueueFullError(Exception):
    """排队中的任务数超过上限，拒绝新任务。"""


class QueueClosedError(Exception):
    """队列已关闭（服务正在停止），不再接受新任务。"""


@dataclass
class Job:
    id: str
    session_id: str
    branch_id: str
    branch_name: str
    tokens: list[int]
    status: str = "queued"  # queued | running | succeeded | cancelled | failed
    detail: str | None = None
    waiting_capacity: bool = False
    started: bool = False          # 已跨过取消边界（引擎即将分配 KV 块）
    cancel_requested: bool = False
    attempts: int = 0
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def is_cancelable(self) -> bool:
        """还没真正开始执行（尚未取到 KV 块）的任务可以取消。"""
        return not self.started and self.status in ("queued", "running")


class AppendQueue:
    def __init__(self, engine: Engine, max_queued: int = 100,
                 max_history: int = 200) -> None:
        self.engine = engine
        self.max_queued = int(max_queued)
        self.max_history = int(max_history)
        self._cond = threading.Condition(threading.Lock())
        self._queues: dict[str, deque[str]] = {}   # 会话 -> 排队中的任务 id
        self._jobs: dict[str, Job] = {}
        self._finished: deque[str] = deque()       # 已完成任务的展示顺序
        self._order: deque[str] = deque()          # 会话轮转顺序
        self._closed = False
        self._seq = itertools.count(1)
        self._thread: threading.Thread | None = None
        # 块被回收 -> 立即唤醒等待容量的调度循环
        self.engine.add_release_listener(self._on_blocks_released)

    def _on_blocks_released(self) -> None:
        # 由引擎在持有引擎锁时回调；这里只取队列自己的条件变量锁，顺序安全。
        with self._cond:
            self._cond.notify_all()

    # ---------- 生命周期 ----------

    def start(self) -> None:
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._thread = threading.Thread(
                target=self._run, name="append-queue", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            th = self._thread
        if th is not None:
            th.join(timeout=timeout)

    def discard_all(self) -> None:
        """把全部排队任务标记为取消并清空（测试用；进程重启时内存直接丢失）。"""
        with self._cond:
            for q in list(self._queues.values()):
                for jid in list(q):
                    job = self._jobs.get(jid)
                    if job is not None and job.status == "queued":
                        self._settle(job, "cancelled", detail="服务重启，任务未执行")
            self._queues.clear()
            self._order.clear()
            self._cond.notify_all()

    # ---------- 提交 / 取消 / 查询 ----------

    def submit(self, session_id: str, branch_id: str, tokens: list[int]) -> dict:
        meta = self.engine.validate_append(session_id, branch_id, tokens)
        with self._cond:
            if self._closed:
                raise QueueClosedError("队列已关闭，不再接受新任务")
            queued = sum(1 for j in self._jobs.values() if j.status == "queued")
            if queued >= self.max_queued:
                raise QueueFullError(f"排队任务已达上限 {self.max_queued}")
            job = Job(
                id=f"q{next(self._seq):04d}-{uuid.uuid4().hex[:4]}",
                session_id=session_id, branch_id=branch_id,
                branch_name=meta["branch_name"], tokens=list(tokens))
            self._jobs[job.id] = job
            if session_id not in self._queues:
                self._queues[session_id] = deque()
            # 会话只在首次出现时进入轮转队尾；同一会话的连续提交不改变其位置，
            # 否则多任务会话会被排到后面，破坏跨会话公平性。
            if session_id not in self._order:
                self._order.append(session_id)
            self._queues[session_id].append(job.id)
            self._cond.notify_all()
            return self._job_dict(job)

    def cancel(self, job_id: str) -> dict:
        with self._cond:
            job = self._get_job(job_id)
            if job.status == "cancelled":
                return self._job_dict(job)
            if job.status in ("succeeded", "failed"):
                raise ValidationError(
                    f"任务已处于 {job.status}，只能在开始执行前取消")
            if job.started:
                # 已跨过取消边界（引擎已/即将分配 KV 块），必须整体执行完
                raise ValidationError(
                    "任务已开始执行（已分配 KV 块），不能再取消")
            # 从排队结构里摘掉；若调度线程刚取出并置为 running，则由
            # 引擎在锁内的取消边界（_begin_append）兜底，这里只登记取消意图。
            job.cancel_requested = True
            q = self._queues.get(job.session_id)
            if q is not None and job_id in q:
                q.remove(job_id)
                if not q:
                    del self._queues[job.session_id]
                    try:
                        self._order.remove(job.session_id)
                    except ValueError:
                        pass
            if job.status == "queued":
                self._settle(job, "cancelled", detail="用户取消")
            else:
                job.detail = "取消请求已受理，将在分配 KV 块前生效"
            self._cond.notify_all()
            return self._job_dict(job)

    def get(self, job_id: str) -> dict:
        with self._cond:
            return self._job_dict(self._get_job(job_id))

    def list(self, session_id: str | None = None) -> dict:
        with self._cond:
            jobs = [self._job_dict(j)
                    for j in self._jobs.values()
                    if session_id is None or j.session_id == session_id]
            jobs.sort(key=lambda d: d["created_at"], reverse=True)
            counts: dict[str, int] = {}
            for j in self._jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
            running = [j.session_id for j in self._jobs.values()
                       if j.status == "running"]
            return {"jobs": jobs, "counts": counts,
                    "running_sessions": running,
                    "rotation": list(self._order)}

    def snapshot(self) -> dict:
        """调度器内部状态的调试快照：同时执行数、各会话排队数。"""
        with self._cond:
            return {
                "running": sum(1 for j in self._jobs.values()
                               if j.status == "running"),
                "running_jobs": [j.id for j in self._jobs.values()
                                 if j.status == "running"],
                "queued_per_session": {s: len(q) for s, q in self._queues.items()},
                "rotation": list(self._order),
                "closed": self._closed,
            }

    # ---------- 内部 ----------

    def _get_job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise NotFoundError(f"任务不存在: {job_id}")

    def _job_dict(self, job: Job) -> dict:
        return {
            "id": job.id,
            "session_id": job.session_id,
            "branch_id": job.branch_id,
            "branch_name": job.branch_name,
            "tokens": list(job.tokens),
            "status": job.status,
            "detail": job.detail,
            "waiting_capacity": job.waiting_capacity,
            "started": job.started,
            "attempts": job.attempts,
            "result": job.result,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }

    def _settle(self, job: Job, status: str, detail: str | None = None,
                result: dict | None = None) -> None:
        job.status = status
        job.detail = detail
        job.result = result
        job.finished_at = time.time()
        job.waiting_capacity = False
        self._finished.append(job.id)
        while len(self._finished) > self.max_history:
            old = self._finished.popleft()
            self._jobs.pop(old, None)
        self._cond.notify_all()

    def _next_session(self) -> str | None:
        """轮转取下一个有排队任务的会话（调用方持锁）。"""
        while self._order:
            sid = self._order.popleft()
            q = self._queues.get(sid)
            if q:
                self._order.append(sid)  # 轮转：用完排到队尾
                return sid
            self._queues.pop(sid, None)
        return None

    def _take_next(self) -> Job | None:
        """按会话轮转取一个可执行任务并标记 running；同会话已有 running 时整队跳过。

        调用方必须已持有条件变量锁。
        """
        # 正在执行的会话本轮不参与；每取走一个任务，该会话（若还有排队）排到队尾。
        busy = {j.session_id for j in self._jobs.values()
                if j.status == "running"}
        for _ in range(len(self._order) + len(busy)):
            sid = self._next_session()
            if sid is None:
                return None
            if sid in busy:
                continue
            job = self._pop_runnable(sid)
            if job is None:
                continue
            job.status = "running"
            job.started_at = time.time()
            job.waiting_capacity = False
            job.detail = None
            job.attempts += 1
            return job
        return None

    def _pop_runnable(self, sid: str) -> Job | None:
        """从会话队首取出下一条未取消任务，顺带清理空会话（调用方持锁）。"""
        job = None
        q = self._queues.get(sid)
        while q:
            candidate = self._jobs.get(q.popleft())
            if candidate is not None and candidate.status == "queued":
                job = candidate
                break
            # 已取消/已完成的任务不应留在队列里，跳过继续取
        if not q:
            self._queues.pop(sid, None)
            try:
                self._order.remove(sid)
            except ValueError:
                pass
        return job

    def _requeue(self, job: Job, detail: str) -> None:
        """容量不足：放回该会话队首，标记 waiting_capacity；块回收后自动重试。"""
        with self._cond:
            job.status = "queued"
            job.started_at = None
            job.attempts -= 1  # 本次未真正执行，不计入执行次数
            job.waiting_capacity = True
            job.detail = detail
            if job.cancel_requested:
                # 取消请求与执行并发到达：本次未分配块，直接按取消处理
                self._settle(job, "cancelled", detail="用户取消（未分配 KV 块）")
                return
            self._queues.setdefault(job.session_id, deque()).appendleft(job.id)
            if job.session_id not in self._order:
                self._order.append(job.session_id)
            self._cond.notify_all()

    def _begin_append(self, job_id: str) -> bool:
        """引擎锁内的取消边界：原子地判断取消请求并标记“已开始”。

        返回 True 表示已被取消（引擎不得分配块）；返回 False 表示本次追加
        正式跨过取消边界，此后 cancel 会拒绝。
        """
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.cancel_requested:
                return True
            job.started = True
            job.detail = None
            self._cond.notify_all()
            return False

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._closed:
                    return  # 停机：未执行任务随进程一起丢弃
                job = self._take_next()
                if job is None:
                    self._cond.wait(timeout=0.5)
                    continue
            # 从这里开始任务处于 running，但还没拿块；取消请求由引擎在
            # 同一把锁内、分配块之前检查，因此取消不会与“已开始”竞态。
            try:
                result = self.engine.append(
                    job.session_id, job.branch_id, job.tokens,
                    begin=lambda: self._begin_append(job.id))
            except AppendCancelled:
                with self._cond:
                    self._settle(job, "cancelled", detail="用户取消（未分配 KV 块）")
                continue
            except CapacityError as e:
                self._requeue(job, str(e))
                with self._cond:
                    self._cond.wait(timeout=0.2)  # 避免容量持续不足时空转
                continue
            except (NotFoundError, ValidationError) as e:
                with self._cond:
                    self._settle(job, "failed", detail=str(e))
                continue
            except BaseException as e:  # 引擎已回滚；队列记录失败
                with self._cond:
                    self._settle(job, "failed",
                                 detail=f"{type(e).__name__}: {e}")
                continue
            with self._cond:
                self._settle(job, "succeeded",
                             detail=f"追加 {len(job.tokens)} 个词元",
                             result={
                                 "length": result["branch"]["length"],
                                 "prediction": result["branch"]["prediction"],
                             })
