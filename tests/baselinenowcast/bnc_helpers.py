"""Cases for the baselinenowcast tests, and the pinned R package on them.

`read` returns what R baselinenowcast 0.2.2 (the commit CI's `r-live`
job installs) computes for a case, run live through
`tests/model/r_live.py` on `r/cases.R`, once a case per process. A test
that reads one skips without R and the package, and fails in `r-live`.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any, NamedTuple

import numpy as np
from model.r_live import run_r

CASES = ["toy", "example", "neg", "struct2", "hub"]
SMALL = ["toy", "example", "neg", "struct2"]
NA = np.nan

TOY = np.array(
    [
        [30, 20, 10, 5],
        [28, 22, 9, 6],
        [35, 18, 12, 4],
        [40, 25, 11, 7],
        [33, 21, 10, NA],
        [29, 19, NA, NA],
        [38, NA, NA, NA],
        [36, NA, NA, NA],
    ]
)
"""A small triangle with complete rows to estimate from."""

NEG = np.array(
    [
        [20, 12, -2, 4, 1],
        [25, 10, 6, -3, 0],
        [18, 14, 5, 2, -1],
        [30, 0, 8, 3, 2],
        [22, 16, -4, 5, NA],
        [27, 11, 7, NA, NA],
        [24, 13, NA, NA, NA],
        [31, NA, NA, NA, NA],
        [26, NA, NA, NA, NA],
    ]
)
"""Negative cells and zeros, for `preprocess_negative_values`."""

ZEROS = np.array(
    [
        [5, 10, 15, 15, 10, 5],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, NA, NA, NA],
        [3, 8, NA, NA, NA, NA],
        [0, NA, NA, NA, NA, NA],
        [1, NA, NA, NA, NA, NA],
    ]
)
"""Young rows with 0 or 1 reports, for `apply_delay`'s +1."""

ZEROS_PMF = np.array([0.05, 0.15, 0.3, 0.3, 0.15, 0.05])

TRIANGLES = {"toy": TOY, "neg": NEG}
"""Triangles set here; R builds the others (`r/cases.R`)."""


class Settings(NamedTuple):
    """Rows for the delay and retrospective nowcasts, `k` and draws."""

    n_hist: int
    n_retro: int
    k: int
    draws: int


SETTINGS = {
    "toy": Settings(5, 3, 2, 4000),
    "example": Settings(5, 2, 2, 4000),
    "neg": Settings(6, 3, 2, 4000),
    "struct2": Settings(7, 4, 2, 4000),
    "hub": Settings(126, 126, 7, 2000),
}


class Recon(NamedTuple):
    """The daily triangle of the snapshot reconstruction check."""

    max_delay: int = 10
    n_ref: int = 40
    n_use: int = 25


RECON = Recon()


def _nb_fits() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)

    def nbinom(mu: float, size: float, n: int) -> np.ndarray:
        return rng.negative_binomial(size, size / (size + mu), n).astype(float)

    fits = [
        ([5, 9, 3, 12, 7, 6], [6, 8, 4, 10, 7, 7]),
        ([0, 0, 1, 0, 2], [0.5, 1, 1, 0.2, 2]),
        (nbinom(40, 5, 30), np.full(30, 40.0)),
        (nbinom(400, 20, 30), np.full(30, 400.0)),
        ([0, 0, 0], [0, 0, 0]),
        (rng.poisson(100, 40).astype(float), np.full(40, 100.0)),
        ([3, 80, 1, 40], [30, 30, 30, 30]),
    ]
    return [(np.asarray(x, float), np.asarray(m, float)) for x, m in fits]


NB_FITS = _nb_fits()
"""Counts and means `fit_nb` fits a size to."""

NB_SAMPLE = np.array(
    [[0, 1], [0.5, 0.3], [5, 2], [50, 10], [500, 100]], dtype=float
)
"""Mean and size of each `rnbinom` the sampler is compared with."""

NB_DRAWS = 20000


def _json(x: np.ndarray) -> Any:
    """A matrix as nested lists, with None for NaN."""
    return [
        [None if math.isnan(v) else float(v) for v in row]
        for row in np.asarray(x, float)
    ]


def _inputs(case: str, samples: bool) -> dict[str, Any]:
    if case == "zeros":
        return {"case": case, "tri": _json(ZEROS), "pmf": ZEROS_PMF.tolist()}
    if case == "nb":
        return {
            "case": case,
            "x": [x.tolist() for x, _ in NB_FITS],
            "mu_fit": [m.tolist() for _, m in NB_FITS],
            "mu": NB_SAMPLE[:, 0].tolist(),
            "size": NB_SAMPLE[:, 1].tolist(),
            "n": NB_DRAWS,
        }
    if case == "recon":
        return {"case": case, **RECON._asdict()}
    out = {"case": case, "samples": samples, **SETTINGS[case]._asdict()}
    if case in TRIANGLES:
        out["tri"] = _json(TRIANGLES[case])
    return out


@cache
def live(case: str, samples: bool = False) -> dict[str, np.ndarray]:
    """Run R baselinenowcast on a case; the samples are a separate run."""
    out = run_r(
        'source("tests/baselinenowcast/r/cases.R")\nbnc_run(input)',
        _inputs(case, samples),
        packages=["baselinenowcast", "zoo"],
        na_as_null=True,
    )
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def read(case: str, name: str) -> np.ndarray:
    """R's matrix `name` for `case`, with NaN for NA."""
    return live(case, name.startswith("samples_"))[name]


def read_vec(case: str, name: str) -> np.ndarray:
    """R's vector `name` for `case`."""
    return read(case, name).ravel()


class Case(NamedTuple):
    """A triangle with its settings."""

    name: str
    values: np.ndarray
    mask: np.ndarray
    n_hist: int
    n_retro: int
    k: int
    draws: int


def load_case(name: str) -> Case:
    """A case's triangle, from R where it is not set here."""
    tri = TRIANGLES[name] if name in TRIANGLES else read(name, "tri")
    mask = ~np.isnan(tri)
    return Case(name, np.nan_to_num(tri), mask, *SETTINGS[name])


def from_na(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a matrix with NaN into values (0 under NaN) and mask."""
    m = ~np.isnan(x)
    return np.nan_to_num(x), m


def first_delays(x: Any) -> Any:
    """The counts in the first half of the delays (R's `first_delays`)."""
    return x[:, : math.ceil(x.shape[1] / 2)].sum(axis=1)
