"""Run a job on as many CPU cores as its threads, cores no other job holds.

A job's thread settings (`machine.torch_env`) cap torch and BLAS, but not
every pool: XLA's CPU client (JAX) starts a thread for each core it may
use and runs on all of them, whatever `XLA_FLAGS` say, so a job of two
threads ran on twelve cores and three such jobs oversubscribed archie.
This script sets the job's CPU affinity to `N` cores before it starts:
the pools it makes size themselves to those cores and run only on them.

    python3 workflow/flowlib/cpus.py N [KEY=VALUE ...] COMMAND [ARG ...]

It takes the locks of `N` free cores among those the process may use in
this node's `claims_dir`, held until the job's last process exits, so
jobs of every Snakemake run on the node keep apart; with fewer free, it
adds others, so the job still runs on `N` cores. Off Linux (no CPU
affinity), and with `N` at or above the cores it may use, it runs the
job as it is.
"""

from __future__ import annotations

import fcntl
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def claims_dir() -> Path:
    """Return the folder of this node's core locks (`CPU_CLAIMS_DIR`)."""
    if os.environ.get("CPU_CLAIMS_DIR"):
        return Path(os.environ["CPU_CLAIMS_DIR"])
    return Path("/tmp") / f"bdbv-cpu-claims-{os.getuid()}"


def take(
    allowed: Sequence[int], n: int, folder: Path
) -> tuple[list[int], list[int]]:
    """Return `n` cores of `allowed` and the locks held on them.

    Free cores first, in order, each locked; with fewer free, the first
    others to make `n`, unlocked. All of `allowed`, with no lock, when
    `n` covers them.
    """
    if n >= len(allowed):
        return list(allowed), []
    folder.mkdir(parents=True, exist_ok=True)
    cores: list[int] = []
    fds: list[int] = []
    for core in allowed:
        if len(cores) == n:
            break
        fd = os.open(
            folder / f"core-{core}.lock", os.O_RDWR | os.O_CREAT, 0o644
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        cores.append(core)
        fds.append(fd)
    cores += [c for c in allowed if c not in cores][: n - len(cores)]
    return sorted(cores), fds


def main(argv: Sequence[str]) -> None:
    """Pin to `argv[0]` cores, if this system can, and run the rest."""
    if len(argv) < 2:
        sys.exit("usage: cpus.py N [KEY=VALUE ...] COMMAND [ARG ...]")
    n, rest = int(argv[0]), list(argv[1:])
    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        cores, fds = take(allowed, max(1, n), claims_dir())
        # The job's processes inherit the locks and hold them until exit.
        for fd in fds:
            os.set_inheritable(fd, True)
        os.sched_setaffinity(0, cores)
    os.execvp("env", ["env", *rest])


if __name__ == "__main__":
    main(sys.argv[1:])
