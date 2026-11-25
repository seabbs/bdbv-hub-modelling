"""Run a GPU job on a GPU no other job holds.

Snakemake's `gpu` resource has a slot per GPU (flowlib.machine), so it
runs as many GPU jobs at once as there are GPUs but does not say which
GPU each takes. A GPU job runs through this script: it takes the lock of
a free GPU among `machine.gpu_ids` under `$ROOT/.gpu-claims`, sets
`CUDA_VISIBLE_DEVICES` to that GPU and runs the job, which holds the
lock until its last process exits. The locks are this machine's, so jobs
of every Snakemake run on it share the GPUs out. With every GPU held, it
waits for one. On a machine without CUDA it runs the job as it is.

    python3 workflow/flowlib/gpus.py [KEY=VALUE ...] COMMAND [ARG ...]
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig, machine

FOLDER = ".gpu-claims"
"""The GPUs' lock files, under this machine's ROOT."""

WAIT = 5.0
"""Seconds between tries while every GPU is held."""


def take(ids: Sequence[str], folder: Path) -> tuple[str, int] | None:
    """Return the first free GPU of `ids` and its held lock, or None."""
    folder.mkdir(parents=True, exist_ok=True)
    for gpu in ids:
        fd = os.open(folder / f"gpu-{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        return gpu, fd
    return None


def claim(ids: Sequence[str], folder: Path) -> tuple[str, int]:
    """Return a free GPU of `ids` and its held lock, waiting for one."""
    said = False
    while (got := take(ids, folder)) is None:
        if not said:
            print(
                f"every GPU ({','.join(ids)}) is held: waiting", file=sys.stderr
            )
            said = True
        time.sleep(WAIT)
    return got


def main(argv: Sequence[str]) -> None:
    """Claim a GPU, if this machine has CUDA, and run `argv` as `env` does."""
    if not argv:
        sys.exit("usage: gpus.py [KEY=VALUE ...] COMMAND [ARG ...]")
    cfg = localconfig.read()
    ids = machine.gpu_ids(cfg)
    if ids:
        gpu, fd = claim(ids, Path(cfg["ROOT"]) / FOLDER)
        # The job's processes inherit the lock and hold it until they exit.
        os.set_inheritable(fd, True)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.execvp("env", ["env", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
