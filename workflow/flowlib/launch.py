"""Launch a job on a shared machine: its cores, after its memory is free.

Where Snakemake runs jobs on this machine itself (`COMPUTE` single, or
each worker of `ssh`, which runs its jobs that way), two things keep
them from crowding each other out:

- Cores. A job's thread settings (`machine.torch_env`) cap torch and
  BLAS, but not every pool: XLA's CPU client (JAX) starts a thread for
  each core it may use and runs on all of them, whatever `XLA_FLAGS`
  say. The launcher sets the job's CPU affinity to as many cores as its
  threads before it starts, so every pool sizes itself to them. The
  trainers' cores are kept for them (`reserved`): the last
  `TRAINER_CORES` of the cores the process may use for each `gpu` slot,
  as `machine.side_cores` keeps them from the budget. A trainer
  (`--trainer`) takes `TRAINER_CORES` of them; any other job its
  `--threads` of the rest. Each takes the locks of free cores in this
  node's `claims_dir`, held until the job's last process exits; with
  fewer free, it adds others of its set, so it still runs on as many.
- Memory. Snakemake counts each job's `mem_mb` against the budget, but
  not what runs beside it outside the workflow. The launcher waits until
  the machine's available memory covers the job's `--mem` and the
  machine's headroom (`HEADROOM_GB`), logging while it waits.

    python3 workflow/flowlib/launch.py (--threads N | --trainer) \
        --mem MB [KEY=VALUE ...] COMMAND [ARG ...]

Under a cluster's or cloud's scheduler (`shared` false: inside a Slurm
job, or `COMPUTE` other than single and ssh), which gives each job its
cores and memory, it runs the job as it is, as flowlib.gpus does. Off
Linux it cannot pin or read the available memory, and runs it as it is.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig, machine
from nowcastutils.devices import allowed_cpus


def claims_dir() -> Path:
    """Return the folder of this node's core locks (`CPU_CLAIMS_DIR`)."""
    if os.environ.get("CPU_CLAIMS_DIR"):
        return Path(os.environ["CPU_CLAIMS_DIR"])
    return Path("/tmp") / f"bdbv-cpu-claims-{os.getuid()}"


def reserved(allowed: Sequence[int], kept: int) -> list[int]:
    """Return the trainers' cores: the last `kept` of `allowed`, if fewer."""
    if kept <= 0 or kept >= len(allowed):
        return []
    return list(allowed)[-kept:]


def unreserved(allowed: Sequence[int], kept: int) -> list[int]:
    """Return the cores other jobs pin to: `allowed` less `reserved`."""
    held = set(reserved(allowed, kept))
    return [c for c in allowed if c not in held]


def kept_cores(cfg: dict[str, str]) -> tuple[int, int]:
    """Return a trainer's cores and those kept for every trainer.

    `TRAINER_CORES` each, one trainer for each `gpu` slot
    (`machine.gpu_slots`), as `machine.side_cores` counts them.
    """
    each = int(
        cfg.get("TRAINER_CORES") or dict(localconfig.DEFAULTS)["TRAINER_CORES"]
    )
    return each, each * machine.gpu_slots(cfg)


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


def shared(
    cfg: Mapping[str, str], environ: Mapping[str, str] | None = None
) -> bool:
    """Whether jobs share this machine under Snakemake's own budget.

    Not inside a Slurm job, nor under a `COMPUTE` other than single and
    ssh, whose scheduler gives each job its cores and memory (as
    `gpus.assigned` decides for GPUs).
    """
    env = os.environ if environ is None else environ
    if env.get("SLURM_JOB_ID"):
        return False
    return (cfg.get("COMPUTE") or "single") in ("single", "ssh")


def meminfo() -> tuple[int, int]:
    """Return this machine's available and total memory (MB) on Linux."""
    got = {}
    with Path("/proc/meminfo").open() as f:
        for line in f:
            key, value = line.split(":", 1)
            got[key] = int(value.split()[0]) // 1024
    return got["MemAvailable"], got["MemTotal"]


WAIT = 15.0
"""Seconds between looks at the available memory."""

SAY_EVERY = 20
"""Looks between log lines while a job waits."""


def wait_for_memory(
    mem_mb: int,
    headroom_mb: int,
    meminfo: Callable[[], tuple[int, int]] = meminfo,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until the available memory covers `mem_mb` and the headroom.

    A job asking for more than the machine has less the headroom starts
    at once, saying so: it would otherwise wait for ever.
    """
    need = mem_mb + headroom_mb
    free, total = meminfo()
    if need > total:
        print(
            f"job asks for {mem_mb} MB with {headroom_mb} MB headroom, more"
            f" than the machine's {total} MB: starting",
            file=sys.stderr,
        )
        return
    looks = 0
    while free < need:
        if looks % SAY_EVERY == 0:
            print(
                f"waiting for {need} MB free ({free} MB now) before starting",
                file=sys.stderr,
                flush=True,
            )
        looks += 1
        sleep(WAIT)
        free, total = meminfo()


def main(argv: Sequence[str]) -> None:
    """Pin the job, wait for its memory, if this machine is shared; run it."""
    args = list(argv)
    trainer = threads = mem_mb = None
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--trainer":
            trainer = True
        elif flag == "--threads":
            threads = int(args.pop(0))
        elif flag == "--mem":
            mem_mb = int(args.pop(0))
        else:
            sys.exit(f"launch.py: unknown option {flag}")
    if not args:
        sys.exit(
            "usage: launch.py (--threads N | --trainer) --mem MB"
            " [KEY=VALUE ...] COMMAND ..."
        )
    cfg = localconfig.read()
    if shared(cfg) and hasattr(os, "sched_setaffinity"):
        if mem_mb:
            headroom = int(cfg.get("HEADROOM_GB") or 0) * 1024
            wait_for_memory(mem_mb, headroom)
        allowed = allowed_cpus()
        each, kept = kept_cores(cfg)
        if trainer:
            n, mine = each, reserved(allowed, kept) or allowed
        else:
            n, mine = threads or 1, unreserved(allowed, kept)
        if n > 0:
            cores, fds = take(mine, n, claims_dir())
            # The job's processes inherit the locks, held until they exit.
            for fd in fds:
                os.set_inheritable(fd, True)
            os.sched_setaffinity(0, cores)
    os.execvp("env", ["env", *args])


if __name__ == "__main__":
    main(sys.argv[1:])
