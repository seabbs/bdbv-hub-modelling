"""The grid's draws of the reporting delay's structure across strata and
time, of ascertainment that changes over time and of over-reporting
bursts (`nowcastfoundation.simulator.draws`)."""

from __future__ import annotations

import numpy as np
import pytest
from test_draws import GRID_FILE

from nowcastfoundation.simulator import draws


@pytest.fixture(scope="module")
def grid():
    return draws.read_grid(GRID_FILE)


@pytest.fixture(scope="module")
def sims(grid):
    out = []
    for i in range(1_500):
        d = draws.draw_design(grid, draws.generator(46, i, 1))
        v = draws.draw_values(grid, d, draws.generator(46, i, 2))
        out.append((d, v))
    return out


def _share(sims, factor, level):
    return np.mean([d[factor] == level for d, _ in sims])


def _weight(grid, factor, level):
    w = grid["factors"][factor]
    return w[level] / sum(w.values())


def test_new_factors_have_a_layer_and_keep_todays_level(grid):
    keep = {"delay_strata_noise": "none"}
    for factor, level in keep.items():
        assert factor in grid["factor_layers"]
        assert level in grid["factors"][factor]


def test_stratum_noise_layers_follow_their_weights(grid, sims):
    for level in ("none", "strata", "strata_time", "both"):
        want = _weight(grid, "delay_strata_noise", level)
        assert abs(_share(sims, "delay_strata_noise", level) - want) < 0.05
    for _, v in sims:
        assert v["delay_strata_sd"] > 0
        assert v["delay_strata_jitter_sd"] > 0
