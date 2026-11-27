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
    keep = {
        "delay_strata_noise": "none",
        "delay_change_strata": "shared",
        "delay_change_switch": "none",
        "delay_change_combine": "single",
        "delay_counts": "none",
        "bursts": "none",
    }
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


def test_strata_draw_their_own_delay_changes(grid, sims):
    kinds = set(grid["factors"]["delay_change"])
    seen = set()
    for d, v in sims:
        n_s = d["n_places"] * d["n_ages"]
        own = v["own_change"]
        assert len(own) == n_s
        for key in ("own_drift", "own_step_at", "own_slow_weeks"):
            assert len(v[key]) == n_s
        first, second = d["delay_change"], v["delay_change_second"]
        if d["delay_change_combine"] == "single" or first == "static":
            assert second == "none"
        else:
            assert second in kinds - {"static", first}
        shared = first if second == "none" else f"{first}+{second}"
        if d["delay_change_strata"] != "independent":
            assert set(own) == {shared}
        for k in own:
            parts = k.split("+")
            assert set(parts) <= kinds
            assert len(set(parts)) == len(parts)
            seen.add(len(parts))
        assert 0.3 <= v["delay_change_rho"] <= 0.9
        assert 0 <= v["delay_switch_at"] <= 1
        assert v["delay_switch_days"] > 0
    assert seen == {1, 2}
    for factor in (
        "delay_change_strata",
        "delay_change_switch",
        "delay_change_combine",
    ):
        for level in grid["factors"][factor]:
            want = _weight(grid, factor, level)
            assert abs(_share(sims, factor, level) - want) < 0.05


def test_delay_counts_slopes_take_either_sign(grid, sims):
    longer, mixed = [], 0
    for d, v in sims:
        sign = v["delay_counts_sign"]
        assert len(sign) == d["n_places"] * d["n_ages"]
        assert set(sign) <= {1, -1}
        if d["delay_counts"] in ("none", "shared"):
            assert len(set(sign)) == 1
        mixed += len(set(sign)) > 1
        longer += sign
        assert v["delay_counts_slope"] > 0
        assert v["delay_counts_sd"] > 0
    share = grid["shares"]["delay_counts_longer"]
    assert abs(np.mean(np.asarray(longer) > 0) - share) < 0.05
    assert mixed > 0
    for level in grid["factors"]["delay_counts"]:
        want = _weight(grid, "delay_counts", level)
        assert abs(_share(sims, "delay_counts", level) - want) < 0.05


def test_bursts_are_rare_short_and_mostly_one_stratum(grid, sims):
    with_bursts = [(d, v) for d, v in sims if d["bursts"] == "yes"]
    assert abs(len(with_bursts) / len(sims) - 0.05) < 0.02
    one = 0
    for d, v in sims:
        n = len(v["burst_strata"])
        for key in ("burst_at", "burst_days", "burst_mult", "burst_keep"):
            assert len(v[key]) == n
        if d["bursts"] == "none":
            assert n == 0
            continue
        n_a = d["n_ages"]
        quiet = set(v["quiet_places"])
        assert n >= 1
        assert all((s - 1) // n_a + 1 not in quiet for s in v["burst_strata"])
        assert all(1 <= k <= 7 for k in v["burst_keep"])
        assert all(k >= 2 for k in v["burst_days"])
        one += n == 1
    assert one / len(with_bursts) > 0.6
    # The first report over the final count: about 1.5 to 7 (5-95%).
    mult = np.asarray(
        [
            draws.draw_setting(grid, "burst_mult", draws.generator(1, i))
            for i in range(4_000)
        ]
    )
    lo, hi = np.quantile(mult, [0.05, 0.95])
    assert 1.3 < lo < 1.8
    assert 6 < hi < 8


def test_over_reporting_rates_reach_bremen_like_values(grid):
    rate = draws.draw_setting(grid, "revise_over", draws.generator(2), 50_000)
    assert 0.005 < np.median(rate) < 0.03
    assert 0.005 < np.mean(rate > 0.5) < 0.03
    assert np.mean(rate > 1.5) > 0
