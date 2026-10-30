"""Robust fork pool of the engine and shared process helpers (moved out of pipeline.py):
the module is light — without detector dictionaries (importing pipeline
takes ~6 s because of anonymizer.detectors.names), so that stdlib stages (L0 of the
audit, tools/audit_zones.py) get run_pool/available_cpus in milliseconds. pipeline.py
re-exports the same names (pipeline.run_pool, pipeline.available_cpus,
pipeline.pdeathsig_kill, pipeline._pool_init …) — callers and tests do not change."""
from __future__ import annotations

import os
import sys


def available_cpus() -> int:
    """Cores available to THIS process (affinity/cgroup), not "all in the machine"."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def register_sigusr1() -> bool:
    """faulthandler: SIGUSR1 → stacks of ALL threads to the real stderr
    (sys.__stderr__ — the tests' redirect_stderr to a StringIO without fileno does not
    break the stage). The handler is a C-level sigaction, inherited by workers
    through fork; safe in them (does not touch the watchdog thread of dump_traceback_
    later, which does not exist after fork). Returns: whether it was registered."""
    import faulthandler
    import signal
    err = sys.__stderr__
    if not hasattr(signal, "SIGUSR1") or err is None:
        return False
    try:
        err.fileno()
        faulthandler.register(signal.SIGUSR1, file=err, all_threads=True)
    except (AttributeError, OSError, RuntimeError, ValueError, TypeError):
        return False
    return True


_PR_SET_PDEATHSIG = 1


def pdeathsig_kill() -> bool:
    """prctl(PR_SET_PDEATHSIG, SIGKILL) for the current process (Linux): the kernel sends SIGKILL when
    the parent thread (the one that forked) dies. Not Linux / no libc / prctl failed — False,
    silently. A shared helper of the worker fork pools: the engine pool (run_pool) and the batch/write pool
    conveyor/group.py; the caller then checks os.getppid() against the parent pid
    taken before fork (the parent died between fork and prctl — no signal comes for an already dead one)."""
    if not sys.platform.startswith("linux") or not hasattr(os, "fork"):
        return False
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) == 0
    except (OSError, AttributeError, ValueError, TypeError):
        return False


def _pool_init(initializer, initargs, parent: int = 0) -> None:
    """Pool worker initializer: SIGUSR1 → stacks (in case the parent
    did not set it — a library call), then the caller's initializer.
    PR_SET_PDEATHSIG SIGKILL — a SIGKILL/OOM of the process that started the pool
    (the stage's main one; a group-sync worker whose batch runs an engine round) takes down
    the workers through the kernel: before, the grandchildren of a group-sync worker stayed orphaned on call_queue.get
    forever (a worker's PDEATHSIG is not inherited by grandchildren). ``parent`` is the pid of the process that started
    the pool (taken before fork): the parent died between fork and prctl — exit at once."""
    pdeathsig_kill()
    if parent and os.getppid() != parent:
        os._exit(0)
    register_sigusr1()
    if initializer is not None:
        initializer(*initargs)


def _feed_new(feed, tasks: list, i, res) -> list:
    """New tasks from the result of task i (feed) — appended to tasks;
    returns their indexes (feed order kept)."""
    new = list(feed(i, res) or ())
    start = len(tasks)
    tasks.extend(new)
    return list(range(start, len(tasks)))


def _run_batch(fn, tasks, idxs, workers, ctx, initializer, initargs,
               results, on_done, feed=None) -> tuple:
    """One ProcessPoolExecutor per queue idxs (indexes into tasks). Submission
    in a WINDOW: ≤ workers+1 tasks in flight — exactly as many as the executor
    really handed out to workers (call_queue = workers + 1), so when a
    worker dies the circle of suspects is small and known. Returns (suspects, rest):
    suspects — tasks in flight at the moment of BrokenProcessPool (in submission
    order), rest — not submitted yet; both empty = the queue is fully closed.
    An exception of fn goes out after the pool stops. ``feed(i, res)`` —
    tasks spawned by a result (parts of a file after it is prepared):
    appended to tasks and put at the HEAD of this pool's queue."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    pending = list(idxs)
    order: dict = {}                     # task index -> submission number
    inflight: dict = {}
    suspects: list = []
    window = workers + 1
    broken = False
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_pool_init,
                             initargs=(initializer, initargs, os.getpid()))
    try:
        while pending or inflight:
            while pending and not broken and len(inflight) < window:
                try:
                    fut = ex.submit(fn, tasks[pending[0]])
                except BrokenProcessPool:
                    broken = True
                    break
                order[pending[0]] = len(order)
                inflight[fut] = pending.pop(0)
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                i = inflight.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    broken = True
                    suspects.append(i)
                    continue
                results[i] = res
                if on_done is not None:
                    on_done(i, res)
                if feed is not None:
                    pending[0:0] = _feed_new(feed, tasks, i, res)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    suspects.sort(key=order.get)
    return suspects, pending


def run_pool(fn, tasks: list, max_workers: int, ctx, initializer=None,
             initargs=(), on_done=None, label: str = "pool",
             name=repr, max_dead=None, feed=None) -> tuple:
    """Robust pool of fork processes: (results, dead), results = {i: fn(tasks[i])}
    for finished tasks, dead = [i, ...] — tasks on which a worker died
    (SIGKILL/OOM) TWICE: in the shared pool and in a single retry; they are not in
    results. Submission order = order of tasks (LPT by the caller). Workers are forked by
    the first submit — a model/Pipeline loaded in the parent BEFORE the call
    is inherited copy-on-write (initializer/initargs — as in executor).
    A worker dies: the suspects (in flight, ≤ workers+1) are rerun ONE
    BY ONE in a single pool — died again = the culprit (dead), passed = a result
    like everyone else's; the rest of the queue continues on a fresh pool. No
    eternal waiting. An exception of fn goes out (fail loudly on system
    errors). on_done(i, result) — progress; name(task) — the name in WARN;
    max_dead (default max(10, 20 % of tasks)) — workers die systematically →
    SystemExit, not endless restarts. ``feed(i, result)`` —
    tasks spawned by the result of task i: appended to ``tasks`` and
    put at the head of the queue (the pool then keeps max_workers, not the length
    of the starting queue — parts appear on the way)."""
    results: dict = {}
    dead: list = []
    queue = list(range(len(tasks)))
    retry: list = []
    if max_dead is None:
        max_dead = max(10, len(tasks) // 5)
    stalls = 0
    while queue or retry:
        if retry:
            i = retry.pop(0)
            suspects, _rest = _run_batch(fn, tasks, [i], 1, ctx, initializer,
                                         initargs, results, on_done)
            if feed is not None and i in results:
                queue[0:0] = _feed_new(feed, tasks, i, results[i])
            if suspects:
                dead.append(i)
                print(f"{label}: WARN worker died (SIGKILL/OOM) on "
                      f"{name(tasks[i])} and in a single retry — file "
                      "skipped (worker_died)", flush=True)
                if len(dead) > max_dead:
                    raise SystemExit(
                        f"{label}: workers die systematically — {len(dead)} "
                        f"tasks in a row kill the worker (SIGKILL/OOM); stopping, "
                        "this is not a \"bad file\" (memory/worker limits?)")
            continue
        before = len(results)
        nw = max_workers if feed is not None else min(max_workers, len(queue))
        suspects, rest = _run_batch(fn, tasks, queue, nw, ctx, initializer,
                                    initargs, results, on_done, feed)
        queue = rest
        if suspects:
            print(f"{label}: WARN worker died (SIGKILL/OOM); in flight were "
                  f"{len(suspects)} files — retrying one by one: "
                  + ", ".join(name(tasks[i]) for i in suspects), flush=True)
            retry.extend(suspects)
        elif queue and len(results) == before:
            # the pool broke without taking a single task (a worker died idle) —
            # three times in a row is not a coincidence
            stalls += 1
            if stalls >= 3:
                raise SystemExit(f"{label}: the pool broke three times with no tasks in "
                                 "flight — the workers do not live (initialization/"
                                 "limits?)")
        else:
            stalls = 0
    return results, dead
