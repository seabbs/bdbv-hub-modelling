"""A simulated outbreak with every part of the delay's and ascertainment's
structure across strata and time, through the joint model
(`simulation.outbreak`)."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from test_outbreak import GRID

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator.draws import (
    draw_design,
    draw_values,
    generator,
)
from nowcastrenewal.simulation.outbreak import outbreak_model, simulate_outbreak

STRUCTURE = {
    "delay_strata_noise": "both",
    "delay_change": "walk",
    "delay_change_strata": "independent",
    "delay_change_switch": "diverge",
    "delay_change_combine": "combined",
    "delay_counts": "strata_time",
    "bursts": "yes",
    "ascertainment_change": "step",
    "ascertainment_change_strata": "related",
    "ascertainment_change_switch": "converge",
    "ascertainment_counts": "strata",
    "ascertainment_noise": "all",
    "delay_jitter": "yes",
    "revisions": "yes",
}


@pytest.fixture(scope="module")
def sim() -> tuple[pd.DataFrame, dict, dict, int]:
    for k in range(200):
        design = draw_design(GRID, generator(919, k, 1), weeks=8)
        n_s = int(design["n_places"]) * int(design["n_ages"])
        if not 2 <= n_s <= 6 or design["strata_link"] == "independent":
            continue
        design.update(STRUCTURE)
        values = draw_values(GRID, design, generator(919, k, 2))
        if not values["burst_strata"]:
            continue
        return (
            pd.DataFrame(sb._strata(design)),
            design,
            values,
            sb._n_days(values),
        )
    raise RuntimeError("no design")


def test_every_part_runs_in_the_joint_model(sim) -> None:
    strata, design, values, n_days = sim
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=1),
    ):
        outbreak_model(n_days, strata, design, values)
    names = set(tr)
    for site in (
        "expected::reference::strata_location",
        "expected::reference::strata_jitter_location",
        "expected::reference::change::walk",
        "expected::reference::counts::strata_jitter_slope",
        "expected::ascertainment::jitter_ascertainment",
        "expected::ascertainment::counts::strata_slope",
        "draw_0::bursts::errors",
        "draw_0::revisions::counts",
    ):
        assert site in names, site
    assert np.all(np.isfinite(tr["expected::log_expected"]["value"][..., 0]))


def test_the_outbreak_is_reproducible_with_withdrawals(sim) -> None:
    strata, design, values, n_days = sim
    a = simulate_outbreak(3, n_days, strata, design, values)
    b = simulate_outbreak(3, n_days, strata, design, values)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert len(a[1]) > 0
    assert (a[1]["count"] < 0).any()
