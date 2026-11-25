# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Host devices: the CPUs a process may use, and XLA's device count.

Apart from the CLI, so the fit store can set them without the CLI being
part of every stored fit's key (`bdbvhub.store.store.fit_code`).
"""

from __future__ import annotations

import os

__all__ = ["allowed_cpus", "host_devices", "usable_cpus"]


def allowed_cpus() -> list[int]:
    """Return the ids of the CPUs this process may run on, sorted.

    The process's CPU affinity where the platform has it (Linux), so a
    `taskset` or a container's CPU set is respected. Elsewhere (macOS,
    Windows) every CPU the host reports, and at least CPU 0.

    Returns
    -------
    list[int]
        The usable CPU ids, in increasing order.
    """
    # Looked up, not named: mypy checks `os` as each platform has it.
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return sorted(affinity(0))
    return list(range(os.cpu_count() or 1))


def usable_cpus() -> int:
    """Return the number of CPUs this process may run on.

    The count of `allowed_cpus`.

    Returns
    -------
    int
        The number of usable CPUs.
    """
    return len(allowed_cpus())


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
