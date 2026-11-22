"""HTTP 端到端测试：追加任务队列的排队、轮转、取消、容量回收重试与重启丢弃。"""
import threading
import time

import pytest
import requests
import uvicorn

from app.engine import Engine
from app.main import create_app

BASE = "http://127.0.0.1:8472"


@pytest.fixture(scope="module")
def server():
    eng = Engine.from_config(vocab_size=64, d_model=16, block_size=4,
                             num_blocks=4, seed=7)
    app = create_app(eng)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8472,
                                        log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            requests.get(BASE + "/api/stats", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        raise RuntimeError("server 未能启动")
    yield app
    srv.should_exit = True
    th.join(timeout=10)


def api(path, method="GET", **body):
    r = requests.request(method, BASE + path, json=body or None, timeout=10)
    return r.status_code, r.json()


def wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def new_session(tokens):
    code, r = api("/api/sessions", "POST", tokens=tokens)
    assert code == 201, r
    return r["session_id"], r["branch"]["id"]


def test_queue_schedule_cancel_and_recycle(server):
    app = server
    q = app.state.queue
    eng = app.state.engine
    # 清空前序状态
    for s in requests.get(BASE + "/api/sessions", timeout=5).json()["sessions"]:
        requests.delete(BASE + f"/api/sessions/{s['session_id']}", timeout=5)
    assert eng.stats()["free_blocks"] == 4

    a, a_branch = new_session([1, 2, 3, 4])
    b, b_branch = new_session([5, 6, 7, 8])
    d1, _ = new_session([9, 10, 11, 12])
    d2, _ = new_session([13, 14, 15, 16])   # 池满

    # 两个会话各提交一条 -> 都应排队并标记等待容量
    code, ja = api(f"/api/sessions/{a}/queue", "POST",
                   branch_id=a_branch, tokens=[20, 21, 22, 23])
    assert code == 202, ja
    code, jb = api(f"/api/sessions/{b}/queue", "POST",
                   branch_id=b_branch, tokens=[24, 25, 26, 27])
    assert code == 202, jb

    code, listing = api("/api/queue")
    assert code == 200
    assert listing["counts"].get("queued") == 2

    # 取消其中一条：状态变为 cancelled，不再执行
    code, out = api(f"/api/queue/{jb['id']}", "DELETE")
    assert code == 200 and out["status"] == "cancelled"
    assert out["detail"] == "用户取消"

    # 等待容量 -> 回收一块后剩余任务自动成功
    assert wait_until(lambda: api(f"/api/queue/{ja['id']}")[1]["waiting_capacity"])
    code, _ = api(f"/api/sessions/{d1}", "DELETE")
    assert code == 200
    assert wait_until(lambda: api(f"/api/queue/{ja['id']}")[1]["status"] == "succeeded")
    job = api(f"/api/queue/{ja['id']}")[1]
    assert job["result"]["length"] == 8
    # 取消的任务始终没有执行：b 分支长度不变
    sess = requests.get(BASE + f"/api/sessions/{b}", timeout=5).json()
    assert [br["length"] for br in sess["branches"]] == [4]
    assert api(f"/api/queue/{jb['id']}")[1]["status"] == "cancelled"
    # 排队快照已清空（除了可能的 running）
    assert q.snapshot()["queued_per_session"] == {}

    # 再提交一条：容量仍被占满（4 块），保持排队
    code, jc = api(f"/api/sessions/{b}/queue", "POST",
                   branch_id=b_branch, tokens=[30, 31, 32, 33])
    assert code == 202
    assert wait_until(lambda: api(f"/api/queue/{jc['id']}")[1]["waiting_capacity"])
    # 回收第二块 -> 自动重试成功
    api(f"/api/sessions/{d2}", "DELETE")
    assert wait_until(lambda: api(f"/api/queue/{jc['id']}")[1]["status"] == "succeeded")

    # 清理
    for sid in (a, b):
        requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)


def test_queue_validation_and_errors(server):
    sid, bid = new_session([1, 2, 3])
    code, body = api(f"/api/sessions/{sid}/queue", "POST", branch_id=bid, tokens=[])
    assert code == 400
    code, body = api(f"/api/sessions/{sid}/queue", "POST", branch_id="nope",
                     tokens=[4])
    assert code == 404
    code, body = api("/api/queue", "POST", session_id=sid, branch_id=bid,
                     tokens=[4])
    assert code == 202
    assert api(f"/api/queue/{body['id']}")[0] == 200
    code, body = api("/api/queue/does-not-exist")
    assert code == 404
    code, body = api("/api/queue/does-not-exist", "DELETE")
    assert code == 404
    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)


def test_restart_discards_unexecuted_jobs(server):
    app = server
    q = app.state.queue
    eng = app.state.engine
    for s in requests.get(BASE + "/api/sessions", timeout=5).json()["sessions"]:
        requests.delete(BASE + f"/api/sessions/{s['session_id']}", timeout=5)
    assert eng.stats()["free_blocks"] == 4
    a, a_branch = new_session([1, 2, 3, 4])
    for i in range(3):
        new_session([10 + i, 20 + i, 30 + i, 40 + i])  # 池满
    code, job = api(f"/api/sessions/{a}/queue", "POST",
                    branch_id=a_branch, tokens=[7, 8, 9, 10])
    assert code == 202
    assert wait_until(lambda: api(f"/api/queue/{job['id']}")[1]["waiting_capacity"])

    # 模拟服务重启：队列内存状态丢弃。真实进程重启时 Job 对象直接消失，
    # 这里调用 discard_all 把排队任务标记为取消后清空。
    q.discard_all()
    out = api(f"/api/queue/{job['id']}")[1]
    assert out["status"] == "cancelled" and "重启" in out["detail"]
    assert q.snapshot()["queued_per_session"] == {}
    # 分支内容没有被改动
    sess = requests.get(BASE + f"/api/sessions/{a}", timeout=5).json()
    assert sess["branches"][0]["length"] == 4

    for s in requests.get(BASE + "/api/sessions", timeout=5).json()["sessions"]:
        requests.delete(BASE + f"/api/sessions/{s['session_id']}", timeout=5)
