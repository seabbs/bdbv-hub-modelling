"""The device and threads of every forecasting job, from the config.

Checkpoint scoring (the German units), held-out evaluation, backtests
(German, Ebola), hub forecasts and the paper's runs all forecast on the
workflow config's `forecast.device` with `forecast.threads` CPU threads.
A machine overrides them with `FORECAST_DEVICE`, `FORECAST_THREADS` or
`FORECAST_GPU` in config/local.env. No rule or script picks a device
itself: each forecasting rule puts `Forecast.env` before its command and
asks for `Forecast.gpu` of the `gpu` resource and `Forecast.gpu_share`
of `gpu_share`.

With `device: cuda`, `forecast.gpu` says how a job meets the trainer,
which holds `gpu=1` while it trains: `exclusive` (the default) makes the
job take the GPU too, so it waits for a free one; `share` runs it beside
the trainer on the same GPU, which costs GPU memory and some training
speed but scores as checkpoints land. Shared jobs each take a
`gpu_share` slot, `FORECAST_GPU_JOBS` of them (1 by default,
flowlib.machine), so the GPU's memory holds the trainer and them.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

DEVICES = ("cpu", "cuda", "mps")
"""The devices a forecasting job may name."""

GPU_MODES = ("exclusive", "share")
"""How a CUDA forecasting job meets the trainer's GPU."""

JAX_PLATFORM = {"cpu": "cpu", "cuda": "", "mps": "cpu"}
"""JAX's platforms for each device. With cuda it is empty, so JAX takes
CUDA where jaxlib has it and the CPU where it has not (archie's jaxlib),
while torch (Chronos-2) takes CUDA: `cuda,cpu` would fail on a jaxlib
without CUDA, as JAX starts every platform listed. JAX has no MPS backend.
"""


THREAD_VARS = (
    "TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
"""The thread counts torch and BLAS read."""

EIGEN = "--xla_cpu_multi_thread_eigen=false"
"""XLA's flag for a single-threaded Eigen pool on the CPU."""


@dataclasses.dataclass(frozen=True)
class Forecast:
    """A forecasting job's device, CPU threads and GPU use."""

    device: str
    threads: int
    gpu_mode: str = "exclusive"

    def __post_init__(self) -> None:
        """Refuse a device, thread count or GPU use not allowed."""
        if self.device not in DEVICES:
            raise ValueError(
                f"forecast device {self.device!r} is not one of {DEVICES}"
            )
        if self.threads < 1:
            raise ValueError(f"forecast threads {self.threads} is below 1")
        if self.gpu_mode not in GPU_MODES:
            raise ValueError(
                f"forecast gpu {self.gpu_mode!r} is not one of {GPU_MODES}"
            )

    @property
    def gpu(self) -> int:
        """Return the `gpu` resource a job takes: 1 for exclusive CUDA."""
        return int(self.device == "cuda" and self.gpu_mode == "exclusive")

    @property
    def gpu_share(self) -> int:
        """Return the `gpu_share` slots a job takes: 1 for shared CUDA."""
        return int(self.device == "cuda" and self.gpu_mode == "share")

    def variables(self) -> dict[str, str]:
        """Return the environment a forecasting job runs with.

        `DEVICE` for torch (Chronos-2), `JAX_PLATFORMS` for JAX (the
        renewal and walk models; for cuda, JAX's own choice), the thread
        counts torch and BLAS read, and a single-threaded XLA Eigen pool,
        kept beside flags already set.
        """
        env = {
            "DEVICE": self.device,
            "JAX_PLATFORMS": JAX_PLATFORM[self.device],
        }
        env |= dict.fromkeys(THREAD_VARS, str(self.threads))
        flags = os.environ.get("XLA_FLAGS", "").split()
        if EIGEN not in flags:
            flags.append(EIGEN)
        env["XLA_FLAGS"] = " ".join(flags)
        return env

    @property
    def env(self) -> str:
        """Return `variables` as a shell prefix (`KEY=value ...`)."""
        return " ".join(
            f"{k}='{v}'" if " " in v or not v else f"{k}={v}"
            for k, v in self.variables().items()
        )


def settings(config: Mapping[str, Any], local: Mapping[str, str]) -> Forecast:
    """Return the forecasting jobs' settings on this machine.

    Parameters
    ----------
    config : mapping
        The workflow config, whose `forecast` gives the defaults.
    local : mapping
        This machine's settings (`localconfig.read`), whose
        `FORECAST_DEVICE`, `FORECAST_THREADS` and `FORECAST_GPU`
        override them.

    Returns
    -------
    Forecast
        The device, threads and GPU use.
    """
    fc = config["forecast"]
    got = Forecast(
        device=local.get("FORECAST_DEVICE") or str(fc["device"]),
        threads=int(local.get("FORECAST_THREADS") or fc["threads"]),
        gpu_mode=local.get("FORECAST_GPU") or str(fc.get("gpu", "exclusive")),
    )
    # A cluster job (COMPUTE other than single or ssh) has only the GPU
    # it asks for, and a shared one asks for none.
    compute = local.get("COMPUTE") or "single"
    shared = got.device == "cuda" and got.gpu_mode == "share"
    if shared and compute not in ("single", "ssh"):
        raise ValueError(
            f"COMPUTE={compute}: forecast.gpu must be exclusive for cuda, "
            "as a cluster job gets only the GPU it asks for"
        )
    return got
