"""The foundation model's lag count, from the data's reporting delay."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.zones import ZoneData, hub_zones
from nowcasteval.backtest import BacktestConfig
from nowcastfoundation.forecaster import (
    LAG_GUARD,
    LAG_QUANTILE,
    MAX_LAG,
    HubTriangle,
    lag_count,
)
from nowcastutils.base import ForecastTask

ROOT_CONFIG = Path(__file__).parents[3] / "config" / "backtest.toml"
D = 112
"""The lags the test triangles hold."""


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> ForecastTask:
    """The task test_foundation.py forecasts, zones included."""
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


def _triangle(last: int, share, final: float = 100.0) -> HubTriangle:
    """One place's daily triangle: `share(t, d)` of `final` at lag d."""
    n = last + 8
    partial = np.array(
        [[final * share(t, d) for d in range(D)] for t in range(n)]
    )[None]
    times = [str(t) for t in range(n)]
    return HubTriangle(times, np.full((1, n), final), partial, last)


def test_the_lag_count_is_the_delay_s_98_point_in_steps():
    assert LAG_QUANTILE == 0.98
    assert LAG_GUARD == 1.5
    # A tenth more of each count at each lag: complete at lag 9, so
    # lags 0 to 9, ten of them, include it.
    tri = _triangle(200, lambda _t, d: min(1.0, (d + 1) / 10))
    assert lag_count(tri, "daily") == 10


def test_only_steps_complete_at_the_cut_off_inform_it():
    last = 300

    def share(t, d):
        # Recent steps have not yet seen their delays and report slowly;
        # steps seen for twice the delay are complete at lag 4.
        if t > last - 10:
            return min(1.0, (d + 1) / 60)
        return min(1.0, (d + 1) / 5)

    assert lag_count(_triangle(last, share), "daily") == 5


def test_a_lag_no_release_fell_on_carries_the_lag_before():
    # Weekly releases of daily data: each step is seen at every 7th lag,
    # so a step is seen complete at the first release from lag 9, by lag
    # 15 for every step.
    def share(t, d):
        return min(1.0, (d + 1) / 10) if (t + d) % 7 == 0 else np.nan

    assert lag_count(_triangle(200, share), "daily") == 16


def test_no_evidence_takes_as_many_lags_as_the_history_allows():
    tri = _triangle(200, lambda _t, _d: np.nan)
    assert lag_count(tri, "daily") == D
    # With two steps of context before the longest lag.
    short = _triangle(30, lambda _t, _d: np.nan)
    assert lag_count(short, "daily") == 29


def test_a_backfilled_first_release_is_not_delay_evidence():
    # The first release comes long after the first onsets (BDBV's first
    # daily release, 2026-08-18, holds onsets from February): their gap
    # to it is not a reporting delay. Steps from the first release on are
    # complete at lag 4, so 5 lags.
    last = 300
    first_release = 174

    def share(t, d):
        if t + d < first_release:
            return np.nan
        return min(1.0, (d + 1) / 5)

    tri = _triangle(last, share)
    assert lag_count(tri, "daily") == 5


def test_no_informative_step_takes_the_release_history():
    # Releases began 20 days before the cut-off and report slowly: no step
    # has been seen long enough to read the delay, so the block takes the
    # release history, every day from the first release to the cut-off.
    last = 300
    first_release = last - 20

    def share(t, d):
        return np.nan if t + d < first_release else min(1.0, (d + 1) / 60)

    tri = _triangle(last, share)
    assert lag_count(tri, "daily") == last - first_release + 1 == 21


def test_releases_from_day_one_are_unchanged():
    # Germany and the simulated systems release from the first onset day:
    # every complete step is evidence, as before.
    tri = _triangle(200, lambda _t, d: min(1.0, (d + 1) / 10))
    assert lag_count(tri, "daily") == 10


def test_the_bdbv_daily_block_takes_its_release_history(task: ForecastTask):
    # BDBV's daily releases start on 2026-08-18, after the complete
    # window's onsets, so the block takes every day from the first
    # release to the cut-off: neither the 112-lag cap (the context then
    # ended four months back) nor a fixed 28.
    model = get_model("foundation_zeroshot")
    rows, places, weeks = model._rows(task)
    tri = model._triangle(rows, places, weeks[-1])
    cut = dt.date.fromisoformat(tri.times[tri.last])
    assert model.lag_count(task) == (cut - dt.date(2026, 8, 18)).days + 1


def test_a_delay_not_reached_in_the_window_takes_the_history():
    tri = _triangle(200, lambda _t, d: min(0.9, (d + 1) / 10))
    assert lag_count(tri, "daily") == D


def test_the_estimate_is_capped_by_the_history():
    tri = _triangle(8, lambda _t, d: min(1.0, (d + 1) / 10))
    assert lag_count(tri, "daily") == 7


def _lags_seen(model, task: ForecastTask) -> set[str]:
    store: list = []

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        store.append(groups)
        return [
            np.zeros((n, np.atleast_2d(g["target"]).shape[0], horizon))
            for g in groups
        ]

    dataclasses.replace(model, sample=sample).forecast(task)
    names = store[0][0]["future_covariates"]
    return {name.rsplit("_lag_", 1)[1] for name in names}


@pytest.mark.parametrize("source", ["daily", "weekly"])
def test_a_fixed_lag_count_is_used_as_given(source, task: ForecastTask):
    # The SI compares the estimate with the fixed cap of 28 days or 4 weeks.
    model = dataclasses.replace(
        get_model("foundation_zeroshot"), source=source, lags=MAX_LAG[source]
    )
    assert len(_lags_seen(model, task)) == MAX_LAG[source]


def test_the_ragged_block_takes_the_estimated_lag_count(task: ForecastTask):
    model = get_model("foundation_zeroshot")
    assert model.lags is None
    got = len(_lags_seen(model, task))
    assert got == model.lag_count(task)
    assert 1 <= got <= D


def test_a_grid_that_ends_before_the_cut_off_is_read_up_to_its_end():
    # Targets all before the cut-off: the triangle's grid ends earlier
    # than its last release, so only the steps on the grid are read.
    tri = _triangle(200, lambda _t, d: min(1.0, (d + 1) / 10))
    short = HubTriangle(
        tri.times[:150], tri.final[:, :150], tri.partial[:, :150], tri.last
    )
    assert lag_count(short, "daily") == 10


def test_the_bdbv_study_config_sets_d_for_every_model(task: ForecastTask):
    # Sam: for BDBV, D = 40 days in config/backtest.toml, read by the
    # foundation models and the random walk alike, in place of each
    # model's own rule or default.
    from nowcastbaselines.randomwalk import RandomWalk
    from nowcasteval.backtest import load_config

    config = load_config(ROOT_CONFIG)
    assert config.max_delay == 40
    study = dataclasses.replace(task, max_delay=config.max_delay)
    # Lags 0 to D, as baselinenowcast's max_delay.
    assert get_model("foundation_zeroshot").lag_count(study) == 41
    walk = get_model("randomwalknowcast")
    assert isinstance(walk, RandomWalk)
    assert walk.settings_for(study).max_delay == 40
    # Without it, each model keeps its own.
    assert walk.settings_for(task).max_delay == walk.settings.max_delay


def test_a_step_counts_once_seen_for_one_and_a_half_d():
    # Releases began 8 days before the cut-off; steps settle at lag 4, so
    # D = 5. A step 8 days old has been seen for 1.5D = 7.5 lags, enough;
    # under a 2D guard none would be, and D would be the history (9).
    last = 300
    first_release = last - 8

    def share(t, d):
        return np.nan if t + d < first_release else min(1.0, (d + 1) / 5)

    assert lag_count(_triangle(last, share), "daily") == 5
