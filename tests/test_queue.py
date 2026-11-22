"""追加任务队列测试：会话内串行、跨会话轮转、取消、容量回收重试、失败与回滚、重启丢弃。"""
import threading
import time

import numpy as np
import pytest

from app.blocks import BlockPool
from app.engine import Engine, NotFoundError, ValidationError
from app.queue import AppendQueue, QueueClosedError, QueueFullError


def make_engine(num_blocks=8, block_size=4, d_model=16, seed=7):
    return Engine.from_config(vocab_size=64, d_model=d_model,
                              block_size=block_size, num_blocks=num_blocks,
                              seed=seed)


def wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def wait_status(q, job_id, statuses, timeout=5.0):
    ok = wait_until(lambda: q.get(job_id)["status"] in statuses, timeout)
    assert ok, f"任务 {job_id} 未在 {timeout}s 内到达 {statuses}，当前 {q.get(job_id)}"
    return q.get(job_id)


def cached_tokens(eng, sid, bid):
    with eng.lock:
        return list(eng.sessions[sid].branches[bid].tokens)


class BlockingAppend:
    """把 Engine.append 包一层闸门，用于稳定地构造 running / queued 状态。"""

    def __init__(self, eng, release: threading.Event | None = None):
        self.eng = eng
        self.gate = release or threading.Event()
        self.entered = threading.Event()
        self._real = eng.append

    def __call__(self, session_id, branch_id, tokens, **kwargs):
        self.entered.set()
        assert self.gate.wait(timeout=5), "测试闸门未被释放"
        return self._real(session_id, branch_id, tokens, **kwargs)


# ---------------- 引擎回滚 ----------------

def test_engine_rolls_back_partial_append_on_failure(monkeypatch):
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6])  # 块0满，块1占2槽
    sid = r["session_id"]
    main = r["branch"]["id"]
    child = eng.fork(sid, main, 5)["branch"]["id"]  # 共享块0、块1（块1未满）
    before = eng.get_session(sid)
    used_before = eng.stats()["used_blocks"]
    assert used_before == 2

    # 写到第 3 个词元时抛异常：此前已完成写时复制与新块分配
    real_write = eng._write_token_kv
    calls = {"n": 0}

    def flaky(branch, token, pos):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("模拟写入故障")
        return real_write(branch, token, pos)

    monkeypatch.setattr(eng, "_write_token_kv", flaky)
    with pytest.raises(RuntimeError):
        eng.append(sid, child, [7, 8, 9, 10, 11, 12])  # 写时复制 + 2 个新块

    after = eng.get_session(sid)
    assert after == before                     # 分支内容、块表、预测完全未变
    st = eng.stats()
    assert st["used_blocks"] == used_before     # 引用计数与空闲栈已回滚
    assert st["free_blocks"] == 6

    # 回滚后池仍可用：再次追加成功，长度与预测正确
    monkeypatch.setattr(eng, "_write_token_kv", real_write)
    r = eng.append(sid, child, [7, 8, 9])
    assert r["branch"]["length"] == 8
    want = int(np.argmax(eng.model.scratch_logits([1, 2, 3, 4, 5, 7, 8, 9])))
    assert r["branch"]["prediction"]["argmax"] == want


def test_pool_state_roundtrip():
    pool = BlockPool(num_blocks=4, block_size=2, d_model=3)
    a = pool.alloc()
    pool.incref(a)
    state = pool.save_state()
    assert pool.num_free == 3
    pool.alloc()
    pool.alloc()
    assert pool.num_free == 1
    pool.restore_state(state)
    assert pool.num_free == 3
    assert pool.refcount[a] == 2


# ---------------- 会话内串行 / 跨会话轮转 ----------------

def test_same_session_jobs_run_one_at_a_time_in_order():
    eng = make_engine(num_blocks=16, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        r = eng.create_session([1, 2, 3, 4])
        sid, bid = r["session_id"], r["branch"]["id"]
        jobs = [q.submit(sid, bid, [10 + i] * 4) for i in range(4)]
        for j in jobs:
            wait_status(q, j["id"], {"succeeded"})
        assert cached_tokens(eng, sid, bid) == [1, 2, 3, 4] + sum(
            ([10 + i] * 4 for i in range(4)), [])
        assert eng.get_session(sid)["branches"][0]["length"] == 20
        assert q.snapshot()["running"] == 0
    finally:
        q.stop()


def test_round_robin_across_sessions():
    eng = make_engine(num_blocks=32, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        sids, bids = [], []
        for i in range(3):
            r = eng.create_session([i, i + 1, i + 2, i + 3])
            sids.append(r["session_id"])
            bids.append(r["branch"]["id"])

        order = []
        gate = threading.Event()
        first_started = threading.Event()
        real_append = eng.append

        def hooked(session_id, branch_id, tokens, **kwargs):
            order.append(session_id)
            if len(order) == 1:
                first_started.set()
                assert gate.wait(timeout=5), "测试闸门未被释放"
            return real_append(session_id, branch_id, tokens, **kwargs)

        eng.append = hooked
        first = q.submit(sids[0], bids[0], [20] * 4)
        assert first_started.wait(timeout=5)
        # 首个任务卡在 running，此时给 3 个会话追加排队（会话 0 共 2 个任务）
        jobs = [first] + [q.submit(sid, bid, [30 + i] * 4)
                          for i, (sid, bid) in enumerate(zip(sids, bids))
                          for _ in range(2)]
        assert q.snapshot()["running"] == 1
        assert q.snapshot()["queued_per_session"] == {
            sids[0]: 2, sids[1]: 2, sids[2]: 2}
        gate.set()
        for j in jobs:
            wait_status(q, j["id"], {"succeeded"}, timeout=10)

        # 会话内串行 + 跨会话轮转。首个任务执行期间 sids[0] 被占用，
        # 调度器只能跳过它，因此前两个相邻执行可能同属 sids[0]；此后
        # 每个会话的 2 个排队任务必须与其他会话交错执行。
        assert len(order) == 7
        assert order[0] == sids[0]
        tail = order[2:]
        for a, b in zip(tail, tail[1:]):
            assert a != b, f"同一会话连续执行，未轮转：{order}"
        assert {s: order.count(s) for s in sids} == {sids[0]: 3,
                                                     sids[1]: 2, sids[2]: 2}
    finally:
        q.stop()


def test_two_sessions_never_run_concurrently():
    eng = make_engine(num_blocks=32, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        sids = [eng.create_session([1, 2, 3, 4]) for _ in range(2)]
        active = {"now": 0, "max": 0}
        lock = threading.Lock()
        real_append = eng.append

        def counting(session_id, branch_id, tokens, **kwargs):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            try:
                time.sleep(0.02)
                return real_append(session_id, branch_id, tokens, **kwargs)
            finally:
                with lock:
                    active["now"] -= 1

        eng.append = counting
        jobs = []
        for _ in range(4):
            for r in sids:
                jobs.append(q.submit(r["session_id"], r["branch"]["id"], [7, 8]))
        for j in jobs:
            wait_status(q, j["id"], {"succeeded"}, timeout=10)
        assert active["max"] == 1
    finally:
        q.stop()


# ---------------- 取消 ----------------

def test_cancel_queued_job_before_execution():
    eng = make_engine(num_blocks=16, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        r = eng.create_session([1, 2, 3, 4])
        sid, bid = r["session_id"], r["branch"]["id"]
        blocked = BlockingAppend(eng)
        eng.append = blocked
        first = q.submit(sid, bid, [5])
        wait_status(q, first["id"], {"running"})
        queued = q.submit(sid, bid, [6, 7, 8, 9])
        assert queued["status"] == "queued"
        out = q.cancel(queued["id"])
        assert out["status"] == "cancelled" and out["detail"] == "用户取消"
        assert q.list()["counts"]["cancelled"] == 1
        blocked.gate.set()
        wait_status(q, first["id"], {"succeeded"})
        time.sleep(0.1)
        assert q.get(queued["id"])["status"] == "cancelled"
        assert cached_tokens(eng, sid, bid) == [1, 2, 3, 4, 5]
    finally:
        q.stop()


def test_cancel_after_blocks_acquired_is_rejected(monkeypatch):
    """引擎已跨过取消边界（开始分配 KV 块）后，取消请求必须被拒绝。"""
    eng = make_engine(num_blocks=16, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        r = eng.create_session([1, 2, 3, 4])
        entered, gate = threading.Event(), threading.Event()
        real_write = eng._write_token_kv

        def slow_write(branch, token, pos):
            entered.set()
            assert gate.wait(timeout=5), "测试闸门未被释放"
            return real_write(branch, token, pos)

        monkeypatch.setattr(eng, "_write_token_kv", slow_write)
        job = q.submit(r["session_id"], r["branch"]["id"], [5, 6])
        assert entered.wait(timeout=5)          # 已拿到块、正在写入
        assert q.get(job["id"])["started"] is True
        with pytest.raises(ValidationError):
            q.cancel(job["id"])
        gate.set()
        wait_status(q, job["id"], {"succeeded"})
        assert cached_tokens(eng, r["session_id"], r["branch"]["id"]) == [1, 2, 3, 4, 5, 6]
    finally:
        q.stop()


def test_cancel_while_waiting_for_engine_lock_is_honored():
    """任务已被调度器取出（running）但还没拿到引擎锁/块，取消仍然生效且不改状态。"""
    eng = make_engine(num_blocks=16, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        r = eng.create_session([1, 2, 3, 4])
        sid, bid = r["session_id"], r["branch"]["id"]
        before = eng.get_session(sid)
        eng.lock.acquire()                      # 卡住引擎，任务停在拿块之前
        try:
            job = q.submit(sid, bid, [5, 6, 7])
            wait_status(q, job["id"], {"running"})
            assert q.get(job["id"])["started"] is False
            out = q.cancel(job["id"])
            # 取消请求已受理；调度线程被引擎锁挡住，尚未走到取消边界
            assert out["status"] == "running" and out["started"] is False
            assert "取消请求已受理" in out["detail"]
        finally:
            eng.lock.release()
        # 引擎拿到锁后、分配块之前发现取消 -> 任务取消且状态完全不变
        wait_status(q, job["id"], {"cancelled"})
        assert eng.get_session(sid) == before   # 一个块都没分配、词元未变
        assert eng.stats()["free_blocks"] == 15
    finally:
        q.stop()


def test_cancel_unknown_job_404_and_closed_queue():
    eng = make_engine()
    q = AppendQueue(eng)
    q.start()
    with pytest.raises(NotFoundError):
        q.cancel("nope")
    q.stop()
    r = eng.create_session([1, 2, 3])
    with pytest.raises(QueueClosedError):
        q.submit(r["session_id"], r["branch"]["id"], [4])


def test_queue_full_rejected():
    eng = make_engine(num_blocks=16, block_size=4)
    q = AppendQueue(eng, max_queued=2)
    q.start()
    try:
        blocked = BlockingAppend(eng)
        eng.append = blocked
        r = eng.create_session([1, 2, 3, 4])
        running = q.submit(r["session_id"], r["branch"]["id"], [5])
        wait_status(q, running["id"], {"running"})
        q.submit(r["session_id"], r["branch"]["id"], [6])
        q.submit(r["session_id"], r["branch"]["id"], [7])
        with pytest.raises(QueueFullError):
            q.submit(r["session_id"], r["branch"]["id"], [8])
        blocked.gate.set()
    finally:
        q.stop()


# ---------------- 容量不足等待、回收后重试 ----------------

def test_capacity_wait_then_retry_after_block_released():
    eng = make_engine(num_blocks=4, block_size=4)   # 总容量 16 词元 / 4 块
    q = AppendQueue(eng)
    q.start()
    try:
        a = eng.create_session([1, 2, 3, 4])          # 1 块
        b = eng.create_session([5, 6, 7, 8])          # 1 块
        donor1 = eng.create_session([9, 10, 11, 12])  # 1 块
        donor2 = eng.create_session([13, 14, 15, 16]) # 1 块，池满
        assert eng.stats()["free_blocks"] == 0
        job_a = q.submit(a["session_id"], a["branch"]["id"], [20] * 4)
        job_b = q.submit(b["session_id"], b["branch"]["id"], [21] * 4)
        jobs = [job_a, job_b]
        # 两个任务都留在队列，标记录等待容量；不消耗执行次数
        assert wait_until(lambda: all(
            q.get(j["id"])["status"] == "queued" and
            q.get(j["id"])["waiting_capacity"] and
            q.get(j["id"])["attempts"] == 0 for j in jobs))
        assert q.snapshot()["queued_per_session"] == {
            a["session_id"]: 1, b["session_id"]: 1}
        assert eng.sessions[a["session_id"]].branches[a["branch"]["id"]].length == 4
        assert eng.stats()["used_blocks"] == 4

        # 回收 1 块 -> 队列自动重试，恰好一个任务成功，另一个继续等待
        eng.close_session(donor1["session_id"])
        assert wait_until(lambda: sum(q.get(j["id"])["status"] == "succeeded"
                                      for j in jobs) == 1)
        statuses = [q.get(j["id"])["status"] for j in jobs]
        assert statuses.count("queued") == 1
        waiting = [q.get(j["id"]) for j in jobs
                   if q.get(j["id"])["status"] == "queued"][0]
        assert waiting["waiting_capacity"]
        assert eng.stats()["used_blocks"] == 4   # 回收的块被成功任务用掉

        # 再回收 1 块 -> 剩余任务也成功
        eng.close_session(donor2["session_id"])
        assert wait_until(lambda: all(q.get(j["id"])["status"] == "succeeded"
                                      for j in jobs))
        sess_a = eng.get_session(a["session_id"])["branches"][0]
        sess_b = eng.get_session(b["session_id"])["branches"][0]
        assert sess_a["length"] == sess_b["length"] == 8
        # 预测来自缓存路径，和从头计算一致
        for branch in (sess_a, sess_b):
            want = int(np.argmax(eng.model.scratch_logits(branch["tokens"])))
            assert branch["prediction"]["argmax"] == want
    finally:
        q.stop()


def test_capacity_wait_then_target_branch_closed():
    eng = make_engine(num_blocks=4, block_size=4)   # 总容量 16 词元 / 4 块
    q = AppendQueue(eng)
    q.start()
    try:
        a = eng.create_session([1, 2, 3, 4])          # 1 块
        donors = [eng.create_session([9 + 4 * i, 10 + 4 * i, 11 + 4 * i,
                                      12 + 4 * i]) for i in range(3)]
        assert eng.stats()["free_blocks"] == 0
        jobs = [q.submit(a["session_id"], a["branch"]["id"], [20 + i] * 4)
                for i in range(2)]
        # 两个任务都在排队；队首被调度过（容量不足），不消耗执行次数
        assert wait_until(lambda: q.get(jobs[0]["id"])["waiting_capacity"])
        assert all(q.get(j["id"])["status"] == "queued" for j in jobs)
        assert q.get(jobs[0]["id"])["attempts"] == 0
        assert eng.sessions[a["session_id"]].branches[a["branch"]["id"]].length == 4

        # 回收 1 块 -> 恰好一个任务成功；另一个仍等待（下一次追加还需要新块）
        eng.close_session(donors[0]["session_id"])
        assert wait_until(lambda: any(q.get(j["id"])["status"] == "succeeded"
                                      for j in jobs))
        left = [q.get(j["id"]) for j in jobs
                if q.get(j["id"])["status"] == "queued"]
        assert len(left) == 1 and left[0]["waiting_capacity"]

        # 关闭目标分支 -> 排队任务不会写回已释放的分支，而是记录失败
        eng.close_branch(a["session_id"], a["branch"]["id"])
        out = wait_status(q, left[0]["id"], {"failed", "succeeded"})
        assert out["status"] == "failed"
        assert "分支不存在" in out["detail"]
    finally:
        q.stop()


def test_queued_job_fails_cleanly_when_session_deleted():
    eng = make_engine(num_blocks=3, block_size=4)   # 满池造成排队
    q = AppendQueue(eng)
    q.start()
    try:
        holder = eng.create_session([1, 2, 3, 4])        # 1 块，剩 2 块空闲
        target = eng.create_session([9, 10, 11, 12])     # 1 块，剩 1 块空闲
        assert eng.stats()["free_blocks"] == 1
        # 目标是 4 词元整块 -> 追加 5 个词元需要 2 个新块，容量不足只能排队
        job = q.submit(target["session_id"], target["branch"]["id"], [20] * 5)
        assert wait_until(lambda: q.get(job["id"])["waiting_capacity"])
        eng.close_session(target["session_id"])   # 占块减少的同时目标也消失
        out = wait_status(q, job["id"], {"failed", "succeeded"})
        assert out["status"] == "failed"
        assert "会话不存在" in out["detail"]
        # 已回收的块仍可被其他会话的追加使用
        eng.append(holder["session_id"], holder["branch"]["id"], [13, 14, 15, 16])
        assert eng.stats()["used_blocks"] == 2
    finally:
        q.stop()


def test_submit_validates_immediately():
    eng = make_engine()
    q = AppendQueue(eng)
    q.start()
    try:
        r = eng.create_session([1, 2, 3])
        with pytest.raises(NotFoundError):
            q.submit(r["session_id"], "nope", [4])
        with pytest.raises(ValidationError):
            q.submit(r["session_id"], r["branch"]["id"], [])
        with pytest.raises(ValidationError):
            q.submit(r["session_id"], r["branch"]["id"], [999])
        assert q.list()["jobs"] == []
    finally:
        q.stop()


# ---------------- 重启丢弃 ----------------

def test_discard_all_drops_unexecuted_jobs():
    eng = make_engine(num_blocks=3, block_size=4)
    q = AppendQueue(eng)
    q.start()
    try:
        holder = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])  # 2 块
        eng.create_session([9, 10, 11, 12])                   # 池满
        jobs = [q.submit(holder["session_id"], holder["branch"]["id"], [1])
                for _ in range(2)]
        assert wait_until(lambda: q.get(jobs[0]["id"])["waiting_capacity"])
        assert all(q.get(j["id"])["status"] == "queued" for j in jobs)
        q.discard_all()
        for j in jobs:
            assert q.get(j["id"])["status"] == "cancelled"
            assert "重启" in q.get(j["id"])["detail"]
        assert q.snapshot()["queued_per_session"] == {}
        assert eng.sessions[holder["session_id"]].branches[
            holder["branch"]["id"]].length == 8
    finally:
        q.stop()


def test_stop_discards_unexecuted_and_worker_exits():
    eng = make_engine(num_blocks=3, block_size=4)
    q = AppendQueue(eng)
    q.start()
    holder = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])
    eng.create_session([9, 10, 11, 12])
    jobs = [q.submit(holder["session_id"], holder["branch"]["id"], [1])
            for _ in range(2)]
    assert wait_until(lambda: q.get(jobs[0]["id"])["waiting_capacity"])
    assert all(q.get(j["id"])["status"] == "queued" for j in jobs)
    q.stop(timeout=2)
    assert not q._thread.is_alive()
    # 停机后队列不再执行任何任务
    time.sleep(0.2)
    assert eng.sessions[holder["session_id"]].branches[
        holder["branch"]["id"]].length == 8
    assert all(q.get(j["id"])["status"] == "queued" for j in jobs)
