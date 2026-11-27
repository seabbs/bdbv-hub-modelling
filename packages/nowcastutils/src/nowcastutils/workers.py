# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Worker processes on the CPU that share out one job's threads and cores.

A job splits its work into tasks fixed before any worker starts, so its
results are the same however many workers run them (`pooled`). Each
worker is a fresh interpreter (`START_METHOD`) with its share of the
job's threads (`worker_threads`) and, on Linux, of its cores
(`worker_cores`, `pin_worker`). A worker exits when its coordinator does
(`watch_parent`), and a coordinator stopped by SIGTERM stops its workers.
The German scoring units (`nowcastgermany.checkpoint`) and the held-out
evaluation (`nowcastfoundation.finetune.evaluate`) run this way.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Final

__all__ = [
    "PARENT_POLL",
    "START_METHOD",
    "THREAD_VARS",
    "pin_worker",
    "pooled",
    "watch_parent",
    "worker_cores",
    "worker_threads",
]

THREAD_VARS: Final = (
    "TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""The thread settings a job gives (`flowlib.machine.torch_env`), which a
job split over workers shares out between them."""

START_METHOD: Final = "spawn"
"""How a job starts its workers: fresh interpreters, as torch and JAX
hold threads that a forked process would not have."""

PARENT_POLL: Final = 1.0
"""Seconds between a worker's checks that its coordinator is alive."""

PR_SET_PDEATHSIG: Final = 1
"""Linux's `prctl` option to signal a process when its parent exits."""


def worker_threads(threads: int, workers: int) -> int:
    """Return each worker's share of a job's `threads`, at least one."""
    return max(1, threads // workers)


def worker_cores(cores: Sequence[int], index: int, workers: int) -> list[int]:
    """Return the cores of worker `index` of `workers`: its share of `cores`.

    Each worker takes its own slice, so none of their thread pools (XLA's
    sizes itself to every core it may use) crowds another's.
    """
    each = max(1, len(cores) // workers)
    at = (index % max(1, len(cores) // each)) * each
    return list(cores)[at : at + each]


def watch_parent(parent: int) -> None:
    """Exit this worker when its coordinator, process `parent`, has gone.

    A daemon thread polls the parent every `PARENT_POLL` seconds, so a
    coordinator killed outright (SIGKILL) leaves no worker running on.
    Linux also kills the worker as its parent exits (`prctl`).
    """
    if sys.platform == "linux":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(PARENT_POLL)
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def pin_worker(counter: Any, workers: int) -> None:
    """Pin this worker to its share of the job's cores, where it can.

    `counter`, shared by the job's workers, gives each its own index.
    Only where the platform pins processes (Linux).
    """
    from nowcastutils.devices import allowed_cpus

    # Looked up, not named: mypy checks `os` as each platform has it.
    pin = getattr(os, "sched_setaffinity", None)
    if pin is None:
        return
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    pin(0, worker_cores(allowed_cpus(), index, workers))


def _stop(signum: int, _frame: Any) -> None:
    """Stop the coordinator on `signum` as on an error: its workers too."""
    raise SystemExit(128 + signum)


def pooled[R](
    run: Callable[..., R],
    tasks: Iterable[tuple[Any, ...]],
    workers: int,
    initializer: Callable[..., None],
    initargs: Sequence[Any] = (),
    start_method: str = START_METHOD,
) -> Iterator[tuple[tuple[Any, ...], R]]:
    """Yield each task with its result, `run(*task)`, as it is done.

    In any order, over `workers` processes, each with its share of this
    process's threads (`TORCH_THREADS`; `worker_threads`). A worker first runs
    `initializer(*initargs, counter, workers, parent)`, where `counter`
    is shared by the workers (`pin_worker`) and `parent` is this
    process's id (`watch_parent`). On an error, or SIGTERM in the main
    thread, the workers are stopped without finishing their tasks.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = list(tasks)
    threads = int(os.environ.get("TORCH_THREADS") or workers)
    workers = max(1, min(workers, len(tasks)))
    # The workers take their environment from this process's.
    kept = {var: os.environ.get(var) for var in THREAD_VARS}
    for var in THREAD_VARS:
        os.environ[var] = str(worker_threads(threads, workers))
    context = multiprocessing.get_context(start_method)
    counter = context.Value("i", 0)
    main = threading.current_thread() is threading.main_thread()
    before = signal.signal(signal.SIGTERM, _stop) if main else None
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initializer,
            initargs=(*initargs, counter, workers, os.getpid()),
        ) as pool:
            futures = {pool.submit(run, *task): task for task in tasks}
            try:
                for done in as_completed(futures):
                    yield futures[done], done.result()
            except BaseException:
                # Running tasks are not waited for: each may take minutes.
                for proc in list(pool._processes.values()):
                    proc.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        if main:
            signal.signal(signal.SIGTERM, before or signal.SIG_DFL)
        for var, value in kept.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
