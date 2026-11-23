# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Host devices: the CPUs a process may use, and XLA's device count.

Apart from the CLI, so the fit store can set them without the CLI being
part of every stored fit's key (`evaluation.store.fit_code`).
"""

from __future__ import annotations

import os

__all__ = ["host_devices", "usable_cpus"]


def usable_cpus() -> int:
    """Return the number of CPUs this process may run on.

    The process's CPU affinity where the platform has it (Linux), so a
    `taskset` or a container's CPU set is respected. Elsewhere (macOS,
    Windows) every CPU the host reports, and at least 1.

    Returns
    -------
    int
        The number of usable CPUs.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def host_devices(n: int) -> None:
    """Give XLA at least `n` host devices, so `n` chains run in parallel.

    NumPyro runs chains in parallel only with as many devices as chains.
    This must run before JAX starts, so the CLI calls it before any fit.
    Other flags in `XLA_FLAGS` are kept, and a larger count is kept.

    Parameters
    ----------
    n : int
        The number of chains.
    """
    key = "--xla_force_host_platform_device_count="
    flags = os.environ.get("XLA_FLAGS", "").split()
    have = [int(f[len(key) :]) for f in flags if f.startswith(key)]
    if have and max(have) >= n:
        return
    rest = [f for f in flags if not f.startswith(key)]
    os.environ["XLA_FLAGS"] = " ".join([*rest, f"{key}{n}"])
