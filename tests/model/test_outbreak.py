"""The simulator's joint model (`simulation.outbreak`).

The latent step and the reporting are replayed against their R
references in their own tests; here the joint model is checked to run
them as one numpyro model: one trace holds both, the reporting reads the
latent step's expected onsets, and a key gives the same outbreak again.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator.draws import (
    draw_design,
    draw_values,
    generator,
    read_grid,
)
from nowcastrenewal.reporting.expectation import ReportingExpectation
from nowcastrenewal.reporting.simulate import COUNT_COLUMNS, reporting_frame
from nowcastrenewal.simulation.outbreak import outbreak_model, simulate_outbreak

GRID = read_grid(Path(__file__).parents[2] / "config/finetune/grid.yaml")


def _sim(seed: int) -> tuple[pd.DataFrame, dict, dict, int]:
    """A design and values the latent step runs, as the batch draws them."""
    for k in range(50):
        design = draw_design(GRID, generator(seed, k, 1), weeks=8)
        if int(design["n_places"]) * int(design["n_ages"]) > 6:
            continue
        values = draw_values(GRID, design, generator(seed, k, 2))
        strata = pd.DataFrame(sb._strata(design))
        n_days = sb._n_days(values)
        try:
            with jax.enable_x64(True), handlers.seed(rng_seed=0):
                outbreak_model(n_days, strata, design, values)
        except ValueError:
            continue
        return strata, design, values, n_days
    raise RuntimeError("no simulation ran")


@pytest.fixture(scope="module")
def sim() -> tuple[pd.DataFrame, dict, dict, int]:
    return _sim(818)


def test_one_trace_holds_the_latent_step_and_the_reporting(sim) -> None:
    strata, design, values, n_days = sim
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=1),
    ):
        latent = outbreak_model(n_days, strata, design, values, draws=2)
    names = set(tr)
    assert "latent::r" in names
    assert "expected::log_expected" in names
    assert {"draw_0::reports::counts", "draw_1::reports::counts"} <= names
    # The reporting reads the latent step's expected onsets: its expected
    # reports are those onsets through its own sites.
    frame = reporting_frame(
        values, n_days, list(strata["place"]), list(strata["age"])
    )
    model = ReportingExpectation(
        "expected", frame, design, values, int(values["max_delay"])
    )
    sites = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    with jax.enable_x64(True):
        log_lambda = jnp.log(jnp.maximum(jnp.asarray(latent.onsets), 1e-8)).T
        again = handlers.substitute(model, data=sites)(log_lambda=log_lambda)
    np.testing.assert_allclose(
        np.asarray(again),
        np.asarray(tr["expected::log_expected"]["value"]),
        rtol=1e-12,
    )


def test_a_key_gives_the_same_outbreak(sim) -> None:
    strata, design, values, n_days = sim
    a = simulate_outbreak(7, n_days, strata, design, values, draws=2)
    b = simulate_outbreak(7, n_days, strata, design, values, draws=2)
    c = simulate_outbreak(8, n_days, strata, design, values, draws=2)
    np.testing.assert_array_equal(a[0].onsets, b[0].onsets)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert list(a[1].columns) == list(COUNT_COLUMNS)
    assert set(a[1]["draw"]) <= {1, 2}
    assert not a[1].equals(c[1])
    assert a[2]["stratum_location"] == b[2]["stratum_location"]
