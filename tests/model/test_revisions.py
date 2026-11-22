"""Revisions of reported counts, against the simulator engine's, live.

`tests/model/r/revision_cases.R` revises counts with a copy of the R
engine's `revise_counts` and returns its draws; the port
(`reporting.revisions.Revisions`) takes the same draws and must give the
same revised counts.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from nowcastrenewal.reporting.revisions import Revisions, revise

from .r_live import run_r

G, T, D = 3, 30, 8
VALUES = {
    "revise_over": [0.05, 0.2, 0.1],
    "revise_move": [0.1, 0.3, 0.05],
    "revise_keep": [3.0, 14.0, 6.6],
}


def _counts() -> np.ndarray:
    rng = np.random.default_rng(7)
    y = rng.poisson(4.0, (G, T, D)).astype(float)
    t = np.arange(T)[:, None] + np.arange(D)[None, :]
    y[:, t > T - 1] = 0
    # The first reference day and the last report day have reports, so the
    # engine's first and last dates are the triangle's.
    y[:, 0, 0] = 3
    y[:, T - D, D - 1] = 2
    return y


COUNTS = _counts()


def _engine() -> dict[str, Any]:
    g, t, d = np.nonzero(COUNTS)
    cells = {
        "g": g.tolist(),
        "t": t.tolist(),
        "d": d.tolist(),
        "count": COUNTS[g, t, d].tolist(),
    }
    return run_r(
        'source("tests/model/r/revision_cases.R"); revision_case(input)',
        {"cells": cells, "values": VALUES, "seed": 42},
        packages=["data.table"],
    )


def _dense(rows: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros((G, T, T))
    for row in rows:
        out[row["g"], row["t"], row["r"]] += row["count"]
    return out


def test_revisions_match_the_engine() -> None:
    engine = _engine()
    g, t, d = np.nonzero(COUNTS)
    sites = {}
    for name in ("errors", "moved", "later", "extra"):
        x = np.zeros((G, T, D))
        x[g, t, d] = engine["draws"][name]
        sites[f"revisions::{name}"] = x
    rev = Revisions(
        "revisions",
        VALUES["revise_over"],
        VALUES["revise_move"],
        VALUES["revise_keep"],
    )
    got = np.asarray(handlers.substitute(rev, data=sites)(counts=COUNTS))
    np.testing.assert_allclose(got, _dense(engine["out"]), atol=1e-12)


def test_revisions_leave_the_final_counts() -> None:
    rev = Revisions("revisions", 0.3, 0.3, 5.0)
    with handlers.seed(rng_seed=1):
        out = np.asarray(rev(counts=COUNTS))
    # By the last report day each reference day has its unrevised total,
    # and cumulative counts by report day never go below zero.
    np.testing.assert_allclose(out.sum(axis=2), COUNTS.sum(axis=2))
    assert np.all(np.cumsum(out, axis=2) >= -1e-9)


def test_no_revisions_is_the_triangle_by_report_day() -> None:
    zero = jnp.zeros_like(COUNTS)
    out = np.asarray(
        revise(COUNTS, zero, zero, jnp.ones_like(COUNTS), [1.0] * G)
    )
    want = np.zeros((G, T, T))
    for dd in range(D):
        tt = np.arange(T - dd)
        want[:, tt, tt + dd] = COUNTS[:, tt, dd]
    np.testing.assert_allclose(out, want)


def test_ci_runs_the_revision_checks() -> None:
    import yaml

    from .r_live import ROOT

    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    runs = " ".join(
        s.get("run", "") for s in workflow["jobs"]["r-live"]["steps"]
    )
    assert "tests/model/test_revisions.py" in runs
