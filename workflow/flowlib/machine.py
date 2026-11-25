"""Cores, memory and accelerator of this machine, for the default profile.

Every workflow run uses `workflow/profiles/default`, with the cores and
resources from here: the machine's cores and memory less the headroom
its local config keeps free (`HEADROOM_CORES`, `HEADROOM_GB`; `CORES`
and `MEM_MB` set them outright), and its device, `DEVICE` or else CUDA
when `nvidia-smi` lists a GPU, MPS on Apple silicon, or the CPU.
Nothing about a machine is committed.

    python3 workflow/flowlib/machine.py args     # Snakemake options
    python3 workflow/flowlib/machine.py device   # cuda, mps or cpu
    python3 workflow/flowlib/machine.py budget   # cores, mem_mb, gpu (JSON)

The `gpu` resource has a slot for each CUDA GPU (`gpu_ids`), at least
one: the trainer's turn on whichever device. Each GPU job runs through
`CLAIM` (flowlib.gpus), which gives it a GPU no other job holds.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig


def gpus(path: str | None = None) -> int:
    """Return how many CUDA GPUs `nvidia-smi` lists (0 without it)."""
    smi = shutil.which("nvidia-smi", path=path)
    if not smi:
        return 0
    out = subprocess.run([smi, "-L"], capture_output=True, text=True)
    if out.returncode:
        return 0
    return sum(line.startswith("GPU ") for line in out.stdout.splitlines())


def device(
    cfg: Mapping[str, str],
    path: str | None = None,
    system: str | None = None,
    arch: str | None = None,
) -> str:
    """Return the device to run on: `DEVICE` unless auto, else detected."""
    set_ = cfg.get("DEVICE", "")
    if set_ and set_ != "auto":
        return set_
    if gpus(path):
        return "cuda"
    system = system or platform.system()
    arch = arch or platform.machine()
    if system == "Darwin" and arch == "arm64":
        return "mps"
    return "cpu"


_device = device


CPUS = "python3 workflow/flowlib/cpus.py"
"""The command a forecasting job runs through, with its threads: it pins
the job to that many cores no other job holds (flowlib.cpus), then runs
`KEY=VALUE ... COMMAND` as `env` does."""

CLAIM = "python3 workflow/flowlib/gpus.py"
"""The command a GPU job runs through: it claims a free GPU
(flowlib.gpus), then runs `KEY=VALUE ... COMMAND` as `env` does."""


def gpu_ids(
    cfg: Mapping[str, str],
    path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the CUDA GPUs the workflow's jobs share out.

    `GPUS` (comma-separated ids) if set, else those
    `CUDA_VISIBLE_DEVICES` names, else every one `nvidia-smi` lists.
    """
    env = os.environ if environ is None else environ
    for given in (cfg.get("GPUS", ""), env.get("CUDA_VISIBLE_DEVICES", "")):
        got = [g.strip() for g in given.split(",") if g.strip()]
        if got:
            return got
    return [str(i) for i in range(gpus(path))]


def gpu_slots(cfg: Mapping[str, str], path: str | None = None) -> int:
    """Return the `gpu` resource: one slot per GPU, at least one."""
    return max(1, len(gpu_ids(cfg, path)))


THREAD_VARS = (
    "TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
"""The thread counts torch and BLAS read, all set to a job's threads."""
EIGEN = "--xla_cpu_multi_thread_eigen=false"
"""XLA's flag for a single-threaded Eigen pool on the CPU."""


def torch_env(
    cfg: Mapping[str, str],
    threads: int,
    device: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    """Return the environment of a job running torch here.

    The job runs on `device`, by default this machine's (`device`), with
    `threads` CPU threads, its rule's, for torch and BLAS. JAX reads
    none of those settings: `XLA_FLAGS` makes XLA's Eigen pool
    single-threaded, kept beside any flags already set (as
    `nowcastutils.devices` keeps them). `CHRONOS_BATCH` in the local
    config, if set, is the series in one Chronos-2 forward pass, and
    `SCORE_DATES` the German dates a scoring unit forecasts together, and
    `PRECISION` the precision Chronos-2 forecasts in, `auto` by default.
    """
    env = {"DEVICE": device or _device(cfg, path)}
    env |= dict.fromkeys(THREAD_VARS, str(threads))
    flags = os.environ.get("XLA_FLAGS", "").split()
    if EIGEN not in flags:
        flags.append(EIGEN)
    env["XLA_FLAGS"] = " ".join(flags)
    for key in ("CHRONOS_BATCH", "SCORE_DATES"):
        if cfg.get(key):
            env[key] = str(cfg[key])
    env["PRECISION"] = str(cfg.get("PRECISION") or "auto")
    return env


def shell_env(env: Mapping[str, str]) -> str:
    """Return `env` as shell assignments to put before a command."""
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def machine_mb() -> int:
    """Return this machine's memory in MB."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout) // 2**20


def budget(
    cfg: Mapping[str, str],
    cores: int | None = None,
    total_mb: int | None = None,
) -> dict[str, int]:
    """Return the cores and memory (MB) the workflow may use here."""
    n = budget_cores(cfg, cores)
    if cfg.get("MEM_MB"):
        mem = int(cfg["MEM_MB"])
    else:
        total = machine_mb() if total_mb is None else total_mb
        mem = total - int(cfg.get("HEADROOM_GB") or 0) * 1024
    return {"cores": n, "mem_mb": max(1024, mem)}


def budget_cores(cfg: Mapping[str, str], cores: int | None = None) -> int:
    """Return the cores the workflow may use here (`budget`)."""
    if cfg.get("CORES"):
        n = int(cfg["CORES"])
    else:
        n = (cores or os.cpu_count() or 1) - int(cfg.get("HEADROOM_CORES") or 0)
    return max(1, n)


_DEFAULTS = dict(localconfig.DEFAULTS)

DEFAULT_MB = 2048
"""The memory (MB) a job takes unless its rule says."""


def side_cores(
    cfg: Mapping[str, str], cores: int, path: str | None = None
) -> int | None:
    """Return the cores the jobs beside the trainers share, or None.

    The budget less a trainer's (`TRAINER_CORES`) for each `gpu` slot
    (`gpu_slots`), as each slot trains a run at once, so a trainer whose
    inputs appear mid-run never waits behind CPU jobs for its cores.
    None, keeping nothing, where nothing trains (`TRAINER_CORES=0`) or
    no core would be left.
    """
    each = int(cfg.get("TRAINER_CORES") or _DEFAULTS["TRAINER_CORES"])
    kept = each * gpu_slots(cfg, path)
    if kept <= 0 or cores - kept < 1:
        return None
    return cores - kept


def scoring_threads(
    cfg: Mapping[str, str],
    threads: int,
    cores: int | None = None,
    path: str | None = None,
) -> int:
    """Return the CPU threads of a checkpoint's held-out evaluation.

    The cores the trainers leave (`side_cores`, or the whole budget where
    nothing trains) less one forecasting unit's `threads`, so another job
    still runs beside it; never fewer than `threads`. Step 0 scores every
    held-out block, hours on a unit's few threads.
    """
    total = budget_cores(cfg, cores)
    free = side_cores(cfg, total, path) or total
    return max(threads, free - threads)


def args(
    cfg: Mapping[str, str],
    cores: int | None = None,
    total_mb: int | None = None,
    path: str | None = None,
) -> list[str]:
    """Return the Snakemake options of this machine's cores and resources.

    `gpu` is a slot per GPU, one on a machine without CUDA
    (`gpu_slots`); `sim_slots`
    (`SIM_SLOTS`, default 2) caps simulation batches; `awaiting` lets one
    batch from another host be waited for at a time. Every job but the
    trainer, which claims none, takes its threads of `side_cores`
    (`side_cores`), at most all of them.
    """
    got = budget(cfg, cores, total_mb)
    side = side_cores(cfg, got["cores"], path)
    return [
        "--cores",
        str(got["cores"]),
        "--default-resources",
        f"mem_mb={DEFAULT_MB}",
        *([f"side_cores=min(threads,{side})"] if side else []),
        "--resources",
        *([f"side_cores={side}"] if side else []),
        f"mem_mb={got['mem_mb']}",
        f"gpu={gpu_slots(cfg, path)}",
        f"sim_slots={cfg.get('SIM_SLOTS') or 2}",
        "awaiting=1",
    ]


if __name__ == "__main__":
    cfg = localconfig.read()
    what = sys.argv[1] if len(sys.argv) > 1 else "args"
    if what == "device":
        print(device(cfg))
    elif what == "budget":
        import json

        print(json.dumps({**budget(cfg), "gpu": gpu_slots(cfg)}))
    else:
        print(" ".join(args(cfg)))
