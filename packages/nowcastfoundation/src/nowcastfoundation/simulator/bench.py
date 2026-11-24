# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Benchmark the simulator on a fixed, seeded batch.

    python -m nowcastfoundation.simulator.bench GRID N SEED OUT [--weeks W]

Runs `simulate_batch` twice in one process on the same seed: a cold pass,
which compiles, and a warm pass, which reuses what JAX kept. Each pass
records its wall time, simulations per second, the XLA compilations and
their time (JAX's backend compile events), the slowest and median
simulation, and a digest of its counts and latent files. The digests
are the equality check for a change to the simulator: the same grid,
seed and draws give the same digests before and after. Also recorded:
the process's peak resident memory, the host's load and the CPUs the
process may use, and the JAX backend (set it with
`JAX_PLATFORMS`). ENW_DRAWS sets the observation draws, as the simulate
rule does.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import pandas as pd

from nowcastutils.devices import usable_cpus

from .batch import simulate_batch

COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
"""JAX's event for each XLA compilation, with its duration."""


def digest(path: str | Path) -> str:
    """Return the SHA-256 of a file's contents, gunzipped if gzipped."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return hashlib.sha256(data).hexdigest()


@contextmanager
def _compiles() -> Iterator[list[float]]:
    """Collect the duration of each XLA compilation while open."""
    seen: list[float] = []

    def listen(event: str, duration_secs: float, **_kw: str | int) -> None:
        if event == COMPILE_EVENT:
            seen.append(duration_secs)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        yield seen
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)


def _pass(
    grid_file: str | Path, seed: int, n: int, draws: int, weeks: int | None
) -> dict[str, Any]:
    """Time one batch and describe it."""
    with tempfile.TemporaryDirectory() as tmp, _compiles() as seen:
        t0 = time.perf_counter()
        files = simulate_batch(
            grid_file, seed, n, tmp, draws=draws, weeks=weeks
        )
        seconds = time.perf_counter() - t0
        sims = pd.read_csv(files["settings"])["seconds"]
        return {
            "seconds": seconds,
            "sims_per_second": n / seconds,
            "compiles": len(seen),
            "compile_seconds": float(sum(seen)),
            "slowest_sim_seconds": float(sims.max()),
            "median_sim_seconds": float(sims.median()),
            "counts_sha256": digest(files["counts"]),
            "latent_sha256": digest(files["latent"]),
        }


def benchmark(
    grid_file: str | Path,
    seed: int,
    n: int,
    *,
    draws: int = 1,
    weeks: int | None = None,
) -> dict[str, Any]:
    """Run a seeded batch cold then warm and describe both passes.

    Parameters
    ----------
    grid_file : str or pathlib.Path
        The grid config.
    seed : int
        The batch's seed.
    n : int
        Simulations.
    draws : int
        Observation draws per simulation.
    weeks : int, optional
        Fix every series' length.

    Returns
    -------
    dict
        The settings, `cold` and `warm` passes (see the module
        docstring), `peak_rss_mb`, `load_1min`, `cpus`, the JAX
        `backend` and the versions.
    """
    cold = _pass(grid_file, seed, n, draws, weeks)
    warm = _pass(grid_file, seed, n, draws, weeks)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "n": n,
        "seed": seed,
        "draws": draws,
        "weeks": weeks,
        "backend": jax.default_backend(),
        "cold": cold,
        "warm": warm,
        "peak_rss_mb": peak,
        "load_1min": os.getloadavg()[0],
        "cpus": usable_cpus(),
        **{
            f"{name}_version": metadata.version(name)
            for name in ("nowcastrenewal", "jax", "numpyro", "pyrenew")
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m nowcastfoundation.simulator.bench"
    )
    parser.add_argument("grid", help="the grid config file")
    parser.add_argument("n", type=int, help="simulations")
    parser.add_argument("seed", type=int, help="the batch's seed")
    parser.add_argument("out", help="the JSON file written")
    parser.add_argument("--weeks", type=int, help="fix series' length")
    args = parser.parse_args(argv)
    got = benchmark(
        args.grid,
        args.seed,
        args.n,
        draws=int(os.environ.get("ENW_DRAWS", "1")),
        weeks=args.weeks,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(got, indent=2) + "\n")


if __name__ == "__main__":
    main()
