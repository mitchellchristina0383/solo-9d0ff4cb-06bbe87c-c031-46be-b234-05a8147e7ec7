"""会话 / 分支管理与 KV 块调度。

核心性质：
- 分叉只增加引用计数，不分配新块 —— 池满也能分叉；
- 追加前先算清所需块数（含写时复制），空闲不足则整体拒绝，
  分支内容与块占用保持不变（原子性由全局锁保证）；
- 分叉点落在块内时，子分支引用该块但只承认前 position 个槽位，
  追加时对该共享块写时复制，父分支内容不受影响；
- 关闭分支只回收引用计数归零的块。
"""
from __future__ import annotations

import itertools
import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from .blocks import BlockPool, CapacityError
from .model import TinyCausalAttention


class NotFoundError(Exception):
    pass


class ValidationError(Exception):
    pass


class AppendCancelled(Exception):
    """在分配任何 KV 块之前收到取消请求，本次追加未开始。"""


@dataclass
class Branch:
    id: str
    name: str
    tokens: list[int]
    block_ids: list[int]  # 逻辑块表：第 i 个逻辑块 -> 物理块 id

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class Session:
    id: str
    branches: dict[str, Branch] = field(default_factory=dict)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class Engine:
    def __init__(self, model: TinyCausalAttention, pool: BlockPool) -> None:
        self.model = model
        self.pool = pool
        self.lock = threading.RLock()  # 所有状态变更都在锁内完成，保证并发追加的原子性
        self.sessions: dict[str, Session] = {}
        self._branch_seq = itertools.count(1)
        self._release_listeners: list = []

    # 块被回收后通知等待容量的任务队列重试（回调在引擎锁内调用，
    # 监听方不得反向获取引擎锁；任务队列的实现满足这一点）。
    def add_release_listener(self, fn) -> None:
        with self.lock:
            self._release_listeners.append(fn)

    def _notify_released(self) -> None:
        for fn in list(self._release_listeners):
            try:
                fn()
            except Exception:
                pass

    # ---------- 构造 ----------

    @classmethod
    def from_config(cls, vocab_size: int = 64, d_model: int = 32,
                    block_size: int = 4, num_blocks: int = 32,
                    seed: int = 1234) -> "Engine":
        max_positions = block_size * num_blocks  # 单条分支最长不超过池容量
        model = TinyCausalAttention(vocab_size=vocab_size, d_model=d_model,
                                    max_positions=max_positions, seed=seed)
        pool = BlockPool(num_blocks=num_blocks, block_size=block_size,
                         d_model=model.d_model)
        return cls(model, pool)

    # ---------- 校验 ----------

    def _check_tokens(self, tokens: list[int], allow_empty: bool = False) -> None:
        if not tokens and not allow_empty:
            raise ValidationError("词元列表不能为空")
        for t in tokens:
            if not isinstance(t, int) or isinstance(t, bool):
                raise ValidationError(f"词元必须是整数，收到 {t!r}")
            if not 0 <= t < self.model.vocab_size:
                raise ValidationError(
                    f"词元 {t} 超出词表范围 [0, {self.model.vocab_size})")

    def _get_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise NotFoundError(f"会话不存在: {session_id}")

    def _get_branch(self, session: Session, branch_id: str) -> Branch:
        try:
            return session.branches[branch_id]
        except KeyError:
            raise NotFoundError(f"分支不存在: {branch_id}")

    # ---------- 内部工具（须在锁内调用） ----------

    def _write_token_kv(self, branch: Branch, token: int, pos: int) -> None:
        k, v, _ = self.model.kvq(token, pos)
        b = branch.block_ids[pos // self.pool.block_size]
        self.pool.write(b, pos % self.pool.block_size, k, v)

    def _gather_kv(self, branch: Branch) -> tuple[np.ndarray, np.ndarray]:
        """从块中按逻辑顺序收集该分支全部缓存的 K/V —— 预测只依赖缓存，不重算历史。"""
        n = branch.length
        B = self.pool.block_size
        K = np.empty((n, self.model.d_model))
        V = np.empty((n, self.model.d_model))
        for i, b in enumerate(branch.block_ids):
            lo, hi = i * B, min((i + 1) * B, n)
            K[lo:hi] = self.pool.K[b, :hi - lo]
            V[lo:hi] = self.pool.V[b, :hi - lo]
        return K, V

    def _predict_locked(self, branch: Branch) -> dict:
        if branch.length == 0:
            return {"argmax": None, "top": []}
        _, _, q = self.model.kvq(branch.tokens[-1], branch.length - 1)
        K, V = self._gather_kv(branch)
        logits = self.model.logits_from_kv(q, K, V)
        top = self.model.top_k(logits, k=5)
        return {"argmax": top[0]["token"], "top": top}

    def _branch_dict(self, branch: Branch, with_prediction: bool = True) -> dict:
        d = {
            "id": branch.id,
            "name": branch.name,
            "length": branch.length,
            "tokens": list(branch.tokens),
            "block_ids": list(branch.block_ids),
        }
        if with_prediction:
            d["prediction"] = self._predict_locked(branch)
        return d

    # ---------- 会话 ----------

    def create_session(self, tokens: list[int]) -> dict:
        self._check_tokens(tokens)
        with self.lock:
            need = _ceil_div(len(tokens), self.pool.block_size)
            if self.pool.num_free < need:
                raise CapacityError(
                    f"建会话需要 {need} 个块，空闲仅 {self.pool.num_free} 个")
            branch = Branch(id="b" + str(next(self._branch_seq)),
                            name="main", tokens=list(tokens),
                            block_ids=[self.pool.alloc() for _ in range(need)])
            for pos, tok in enumerate(tokens):
                self._write_token_kv(branch, tok, pos)
            session = Session(id=uuid.uuid4().hex[:8])
            session.branches[branch.id] = branch
            self.sessions[session.id] = session
            return {"session_id": session.id,
                    "branch": self._branch_dict(branch)}

    def close_session(self, session_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            for branch in list(session.branches.values()):
                self._release_branch_blocks(branch)
            del self.sessions[session_id]
            self._notify_released()
            return {"closed": session_id}

    def _release_branch_blocks(self, branch: Branch) -> bool:
        if not branch.block_ids:
            return False
        for b in branch.block_ids:
            self.pool.decref(b)
        branch.block_ids = []
        return True

    # ---------- 分叉 ----------

    def fork(self, session_id: str, branch_id: str, position: int,
             name: str | None = None) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            parent = self._get_branch(session, branch_id)
            if not isinstance(position, int) or not 0 <= position <= parent.length:
                raise ValidationError(
                    f"分叉点须为 [0, {parent.length}] 内的整数，收到 {position!r}")
            # 分叉只共享前缀块，不分配新块 —— 池满也能分叉
            n_blocks = _ceil_div(position, self.pool.block_size)
            shared = parent.block_ids[:n_blocks]
            for b in shared:
                self.pool.incref(b)
            child = Branch(
                id="b" + str(next(self._branch_seq)),
                name=name or f"{parent.name}@{position}",
                tokens=parent.tokens[:position],
                block_ids=list(shared),
            )
            session.branches[child.id] = child
            return {"session_id": session_id, "branch": self._branch_dict(child)}

    # ---------- 追加（原子，受总块数限制） ----------

    def validate_append(self, session_id: str, branch_id: str,
                        tokens: list[int]) -> dict:
        """只校验不执行：供任务队列在入队时快速拒绝非法请求。"""
        self._check_tokens(tokens)
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            if branch.length + len(tokens) > self.model.max_positions:
                raise ValidationError(
                    f"追加后长度 {branch.length + len(tokens)} "
                    f"超出位置表上限 {self.model.max_positions}")
            return {"branch_name": branch.name, "length": branch.length}

    def append(self, session_id: str, branch_id: str, tokens: list[int],
               begin=None) -> dict:
        """同步追加。

        `begin` 在锁内、分配任何块之前调用一次；返回 True 表示已在拿块前
        被取消，此时抛 AppendCancelled 且不改变任何状态。调用方应在这里
        原子地完成“检查取消 / 标记已开始”，从而与取消请求互斥。
        """
        self._check_tokens(tokens)
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            self._append_locked(branch, tokens, begin)
            return {"session_id": session_id,
                    "branch": self._branch_dict(branch)}

    def _append_locked(self, branch: Branch, tokens: list[int],
                       begin=None) -> None:
        """在锁内执行一次追加；任何异常都把分支与块池恢复到调用前的状态。

        提交前已检查容量，因此正常路径不会失败；回滚用于防御写入阶段的
        意外异常（例如模型/内存错误），保证“已开始的追加整体成功或整体回滚”。
        """
        B = self.pool.block_size
        L = branch.length
        new_len = L + len(tokens)
        if new_len > self.model.max_positions:
            raise ValidationError(f"追加后长度 {new_len} 超出位置表上限 {self.model.max_positions}")

        cur_blocks = len(branch.block_ids)
        need_blocks = _ceil_div(new_len, B)
        # 末尾块未满且被共享 -> 追加前必须写时复制一块
        cow = (L % B != 0 and cur_blocks > 0
               and self.pool.refcount[branch.block_ids[-1]] > 1)
        extra = (need_blocks - cur_blocks) + (1 if cow else 0)
        if self.pool.num_free < extra:
            # 整体拒绝：分支内容与块占用保持原样
            raise CapacityError(
                f"追加 {len(tokens)} 个词元需要 {extra} 个空闲块，"
                f"当前仅 {self.pool.num_free} 个，本次追加已整体拒绝")

        # 拿块之前的最后一道取消检查：与分配同处锁内，不存在
        # “取消成功但其实已经拿到块”的窗口。
        if begin is not None and begin():
            raise AppendCancelled("任务在分配 KV 块前被取消")

        # 事务快照：块池只有引用计数与空闲栈会变化，K/V 写入都落在新块上，
        # 回滚后这些块重新空闲，旧内容不会被已提交的分支读到。
        pool_state = self.pool.save_state()
        saved_tokens = list(branch.tokens)
        saved_blocks = list(branch.block_ids)
        try:
            # 提交阶段：先写时复制，再补齐新块，最后写入 K/V
            if cow:
                old = branch.block_ids[-1]
                nb = self.pool.alloc()
                self.pool.copy_prefix(old, nb, L % B)
                self.pool.decref(old)
                branch.block_ids[-1] = nb
            while len(branch.block_ids) < need_blocks:
                branch.block_ids.append(self.pool.alloc())
            for i, tok in enumerate(tokens):
                self._write_token_kv(branch, tok, L + i)
                branch.tokens.append(tok)
        except BaseException:
            branch.tokens[:] = saved_tokens
            branch.block_ids[:] = saved_blocks
            self.pool.restore_state(pool_state)
            raise

    # ---------- 查询 ----------

    def predict(self, session_id: str, branch_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            if branch.length == 0:
                raise ValidationError("空分支没有可预测的上一词元")
            return {"session_id": session_id, "branch_id": branch_id,
                    "prediction": self._predict_locked(branch)}

    def get_session(self, session_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            return {
                "session_id": session.id,
                "branches": [self._branch_dict(b) for b in session.branches.values()],
            }

    def list_sessions(self) -> dict:
        with self.lock:
            return {"sessions": [
                {"session_id": s.id,
                 "branches": [{"id": b.id, "name": b.name, "length": b.length}
                              for b in s.branches.values()]}
                for s in self.sessions.values()
            ]}

    # ---------- 关闭 ----------

    def close_branch(self, session_id: str, branch_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            del session.branches[branch_id]
            self._release_branch_blocks(branch)  # 只有引用归零的块才真正回到空闲池
            self._notify_released()
            return {"closed": branch_id}

    # ---------- 统计 ----------

    def stats(self) -> dict:
        with self.lock:
            return {
                "block_size": self.pool.block_size,
                "num_blocks": self.pool.num_blocks,
                "used_blocks": self.pool.num_used,
                "free_blocks": self.pool.num_free,
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "blocks": self.pool.snapshot(),
            }
