"""devpool: a function over a list of items on several devices.

    out = devpool.run(fn, items, plan.stages["msgmodel"].slots, init=load_model, ctx=ctx)

One process per slot (device × instance, ``hw.Slot``); the model loads once: ``init(slot) -> state``
in the worker, then ``fn(state, batch) -> list`` for every batch (result length = batch length).
``init`` and ``fn`` are module-level functions (``spawn`` passes them). Only ``spawn``: the same on
Windows and Linux, no ``fork``.

- Results come in the order of ``items``. With ``weight(item)`` the heavy ones (long audio) go out
  first; without it batches go one after another in the original order (the same batches — the same
  numbers as the single-process path).
- Out of memory (``MemoryError``, "out of memory" from CUDA/onnxruntime) → the batch is split in half
  and retried; a single item without memory on a GPU goes to a CPU slot if there is one, otherwise an error.
- ``init`` failed on a GPU (no CUDA, the provider failed) → the slot moves to CPU with ``ctx.note``.
  ``init`` failed on CPU — a ``PoolError``.
- A worker died (segfault, killed) → its batch is retried on another slot, the slot is restarted
  (``max_restarts``); a batch that killed workers twice — ``PoolError``. No hangs: the parent waits
  on both the pipes and the process "sentinels".
- Ctrl+C (and any exception in the parent) → all workers are stopped (terminate, then kill), the
  exception goes up. Workers ignore SIGINT and exit by themselves if the parent is gone.

An error in ``fn`` — ``PoolError`` with the worker's traceback text (it is not written to the run
log: exception text may contain data).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing.connection import wait

from .hw import Slot, scrub_note

PARENT_POLL = 0.5          # the worker checks whether the parent is alive
TERM_GRACE = 3.0
OOM_MARKERS = ("out of memory", "outofmemory", "cuda_error_out_of_memory", "failed to allocate memory",
               "cudnn_status_alloc_failed", "cublas_status_alloc_failed", "bad_alloc", "hiperroroutofmemory",
               "e_outofmemory", "dxgi_error_device_removed")


class PoolError(RuntimeError):
    pass


class InitFailed(RuntimeError):
    """Raise from ``init`` when the device does not fit (the provider did not load)."""


def is_oom(e: BaseException | str) -> bool:
    if isinstance(e, MemoryError):
        return True
    s = (e if isinstance(e, str) else f"{type(e).__name__}: {e}").lower()
    return any(m in s for m in OOM_MARKERS)


# ── worker ───────────────────────────────────────────────────────────────────────────────────

def _worker(conn, slot: Slot, init, fn) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent handles Ctrl+C
    except (ValueError, OSError):
        pass
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # indices as in nvidia-smi
    if slot.device == "cpu":
        n = str(max(1, slot.threads))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, n)
    parent = mp.parent_process()
    try:
        state = init(slot) if init else None
    except BaseException as e:  # noqa: BLE001
        _send(conn, ("init_failed", f"{type(e).__name__}", traceback.format_exc()))
        return
    _send(conn, ("ready",))
    while True:
        try:
            if not conn.poll(PARENT_POLL):
                if parent is not None and not parent.is_alive():
                    return
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        tid, batch = msg
        try:
            out = fn(state, batch)
            out = list(out)
            if len(out) != len(batch):
                raise PoolError(f"fn returned {len(out)} results for a batch of {len(batch)}")
            _send(conn, ("ok", tid, out))
        except BaseException as e:  # noqa: BLE001
            kind = "oom" if is_oom(e) else "error"
            _send(conn, (kind, tid, traceback.format_exc()))


def _send(conn, msg) -> None:
    try:
        conn.send(msg)
    except (OSError, EOFError, BrokenPipeError):
        pass


# ── parent ───────────────────────────────────────────────────────────────────────────────────

@dataclass
class _Task:
    tid: int
    idx: list[int]
    crashes: int = 0
    oom_cpu: bool = False       # a single item did not fit on the GPU — CPU only


class _W:
    def __init__(self, ctx, slot: Slot, init, fn):
        self.slot = slot
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_worker, args=(child, slot, init, fn), daemon=True,
                                name=f"devpool-{slot.device}-{slot.instance}")
        self.proc.start()
        child.close()
        self.ready = False
        self.task: _Task | None = None
        self.restarts = 0

    def stop(self, kill: bool = False) -> None:
        if self.proc.is_alive():
            if kill:
                self.proc.kill()
            else:
                try:
                    self.conn.send(None)
                except (OSError, EOFError, BrokenPipeError):
                    pass
        try:
            self.conn.close()
        except OSError:
            pass


def _shutdown(workers: list[_W], graceful: bool) -> None:
    for w in workers:
        if graceful:
            w.stop()
        elif w.proc.is_alive():
            w.proc.terminate()
    deadline = time.monotonic() + TERM_GRACE
    for w in workers:
        w.proc.join(max(0.0, deadline - time.monotonic()))
    for w in workers:
        if w.proc.is_alive():
            w.proc.kill()
            w.proc.join(TERM_GRACE)
        try:
            w.conn.close()
        except OSError:
            pass


def _note(ctx, msg: str) -> None:
    if ctx is not None:
        ctx.note("devpool: " + msg)


def _inprocess(fn, items, slot: Slot, init, batch: int, order: list[int], ctx) -> list:
    """One slot in the own process — no child processes, the same OOM and CPU fallback rules."""
    try:
        state = init(slot) if init else None
    except Exception as e:  # noqa: BLE001
        if slot.device == "cpu":
            raise PoolError(f"init on cpu: {type(e).__name__}") from e
        _note(ctx, f"{slot.device} failed to start ({type(e).__name__}) — on CPU")
        slot = replace(slot, device="cpu", gpu_mem_limit_mb=0)
        state = init(slot) if init else None
    out: list = [None] * len(items)
    cpu_state = None
    q = deque((order[i:i + batch], False) for i in range(0, len(order), batch))
    while q:
        idx, on_cpu = q.popleft()
        try:
            if on_cpu:
                if cpu_state is None:
                    cpu_state = (init(replace(slot, device="cpu", gpu_mem_limit_mb=0)) if init else None,)
                res = list(fn(cpu_state[0], [items[i] for i in idx]))
            else:
                res = list(fn(state, [items[i] for i in idx]))
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and len(idx) > 1:
                h = len(idx) // 2
                q.appendleft((idx[h:], on_cpu))
                q.appendleft((idx[:h], on_cpu))
                continue
            if is_oom(e) and not on_cpu and slot.device != "cpu":
                _note(ctx, f"item does not fit on {slot.device} — running on CPU")
                q.appendleft((idx, True))
                continue
            raise PoolError(f"{type(e).__name__} on {slot.device}") from e
        for i, r in zip(idx, res):
            out[i] = r
    return out


def run(fn, items: list, slots: list[Slot], *, init=None, batch_size: int | None = None, weight=None,
        ctx=None, max_restarts: int = 1, inprocess: bool | None = None, on_result=None) -> list:
    """Run ``fn(state, batch)`` over ``items`` on the slots ``slots`` (see the module docstring).

    ``batch_size`` — ``slots[0].batch_size`` by default. ``inprocess`` — None: one slot and items for
    one batch — no child processes. ``on_result(done, total)`` is called in the parent after every
    batch (progress; an exception from it acts like Ctrl+C: everything is stopped)."""
    items = list(items)
    if not items:
        return []
    if not slots:
        slots = [Slot("cpu")]
    batch = max(1, batch_size or slots[0].batch_size or 1)
    order = list(range(len(items)))
    if weight is not None:
        order.sort(key=lambda i: -float(weight(items[i])))
    if inprocess is None:
        inprocess = len(slots) == 1 and len(items) <= batch
    if inprocess:
        return _inprocess(fn, items, slots[0], init, batch, order, ctx)

    mctx = mp.get_context("spawn")
    out: list = [None] * len(items)
    done = 0
    pending: deque[_Task] = deque()
    tid = 0
    for s in range(0, len(order), batch):
        pending.append(_Task(tid, order[s:s + batch]))
        tid += 1
    n_tasks_left = len(pending)
    workers: list[_W] = []
    ok = False
    try:
        for s in slots:
            workers.append(_W(mctx, s, init, fn))
        while n_tasks_left:
            # hand out
            for w in workers:
                if w.ready and w.task is None and w.proc.is_alive():
                    t = _next_task(pending, w)
                    if t is None:
                        continue
                    w.task = t
                    try:
                        w.conn.send((t.tid, [items[i] for i in t.idx]))
                    except (OSError, EOFError, BrokenPipeError):
                        pass               # the sentinel will show the death
            if not any(w.task is not None or not w.ready for w in workers):
                if pending:
                    raise PoolError("no slot can take the remaining batches (out of GPU memory)")
                raise PoolError(f"the pool lost {n_tasks_left} batches — internal error")   # do not hang
            waitables = []
            for w in workers:
                if not w.conn.closed:
                    waitables.append(w.conn)
                waitables.append(w.proc.sentinel)
            ready = wait(waitables, timeout=5.0)
            for w in list(workers):
                if w not in workers:
                    continue                       # replaced while handling a neighbour
                dead = w.proc.sentinel in ready or not w.proc.is_alive()
                # everything the worker managed to send — before handling its death
                while not w.conn.closed and (w.conn in ready or dead) and w.conn.poll(0):
                    try:
                        msg = w.conn.recv()
                    except (EOFError, OSError):
                        w.conn.close()
                        break
                    n_tasks_left, done = _handle(msg, w, workers, pending, out, n_tasks_left, done,
                                                 mctx, init, fn, ctx)
                    if msg[0] == "ok" and on_result is not None:
                        on_result(done, len(items))
                    if w not in workers or not dead:
                        break
                if w in workers and dead and not w.proc.is_alive():
                    _crashed(w, workers, pending, mctx, init, fn, ctx, max_restarts)
        ok = True
        return out
    finally:
        _shutdown(workers, graceful=ok)


def _next_task(pending: deque, w: _W) -> _Task | None:
    for k, t in enumerate(pending):
        if t.oom_cpu and w.slot.device != "cpu":
            continue
        del pending[k]
        return t
    return None


def _handle(msg, w: _W, workers, pending, out, left, done, mctx, init, fn, ctx):
    kind = msg[0]
    if kind == "ready":
        w.ready = True
    elif kind == "init_failed":
        if w.slot.device == "cpu":
            raise PoolError(f"cpu worker failed to start: {msg[1]}\n{msg[2]}")
        _note(ctx, f"{w.slot.device} failed to start ({scrub_note(msg[1])}) — slot moved to CPU")
        idx = workers.index(w)
        w.proc.join(TERM_GRACE)
        w.stop(kill=True)
        nw = _W(mctx, replace(w.slot, device="cpu", gpu_mem_limit_mb=0), init, fn)
        workers[idx] = nw
    elif kind == "ok":
        t = w.task
        w.task = None
        for i, r in zip(t.idx, msg[2]):
            out[i] = r
        left -= 1
        done += len(t.idx)
    elif kind == "oom":
        t = w.task
        w.task = None
        if len(t.idx) > 1:
            h = len(t.idx) // 2
            pending.appendleft(_Task(t.tid * 2 + 1_000_000, t.idx[h:]))
            pending.appendleft(_Task(t.tid * 2 + 1_000_001, t.idx[:h]))
            left += 1
            _note(ctx, f"out of memory on {w.slot.device} — batch split in half")
        elif w.slot.device != "cpu" and any(x.slot.device == "cpu" for x in workers):
            t.oom_cpu = True
            pending.appendleft(t)
            _note(ctx, f"item does not fit on {w.slot.device} — running on CPU")
        else:
            raise PoolError(f"out of memory on {w.slot.device} even for one item\n{msg[2]}")
    elif kind == "error":
        raise PoolError(f"error in worker {w.slot.device}:\n{msg[2]}")
    return left, done


def _crashed(w: _W, workers, pending, mctx, init, fn, ctx, max_restarts) -> None:
    code = w.proc.exitcode
    t = w.task
    w.task = None
    idx = workers.index(w)
    w.proc.join(TERM_GRACE)
    w.stop(kill=True)
    if t is not None:
        t.crashes += 1
        if t.crashes >= 2:
            raise PoolError(f"the batch killed a worker twice (code {code})")
        pending.appendleft(t)
    _note(ctx, f"worker {w.slot.device} died (code {code})")
    if w.restarts < max_restarts:
        nw = _W(mctx, w.slot, init, fn)
        nw.restarts = w.restarts + 1
        workers[idx] = nw
    else:
        workers.pop(idx)
        if not workers:
            raise PoolError("all workers died")
