"""randomwalknowcast's reference times: hub weeks, 7-day sums and days.

A task whose targets are every day gets the 7-day sums ending each day,
from blocks of 7 days ending on each weekday; Saturdays are the hub weeks.
With `error_days` 1 its observation model is fitted on daily counts and
the 7-day sums are sums of daily draws. `daily_quantiles` gives each day's
count.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.zones import ZoneData, hub_zones
from nowcasteval.backtest import BacktestConfig

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def hub_task(target, delays, daily, daily_zones, health_zones, tasks_json):
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


@pytest.fixture(scope="module")
def day_task(hub_task):
    ends = pd.to_datetime(list(hub_task.target_end_dates))
    days = pd.date_range(ends.min(), ends.max(), freq="D")
    return dataclasses.replace(
        hub_task, target_end_dates=tuple(d.date().isoformat() for d in days)
    )


@pytest.fixture(scope="module")
def hub(hub_task):
    return get_model("randomwalknowcast").forecast(hub_task)


@pytest.fixture(scope="module")
def rolling(day_task):
    return get_model("randomwalknowcast").forecast(day_task)


def test_every_day_has_its_seven_day_sum(rolling, day_task) -> None:
    got = set(rolling["target_end_date"].astype(str))
    assert got == set(day_task.target_end_dates)
    assert np.isfinite(rolling["value"]).all()
    assert (rolling["value"] >= 0).all()
    # A sum ending between two Saturdays lies near theirs, not at 0.
    mean = rolling.groupby("target_end_date")["value"].mean()
    mean.index = pd.to_datetime(mean.index)
    sat = mean.loc[mean.index.weekday == 5]
    between = mean.loc[mean.index.weekday != 5]
    assert len(between) > 0
    for day, v in between.items():
        before = sat.loc[:day]
        after = sat.loc[day:]
        if before.empty or after.empty:
            continue
        lo, hi = sorted((before.iloc[-1], after.iloc[0]))
        if lo > 0.5:
            assert 0.3 * lo <= v <= 3 * hi, (day, v, lo, hi)


def test_the_sums_ending_saturday_are_the_hub_weeks(
    hub, rolling, hub_task
) -> None:
    # To the cut-off, the sums ending Saturday are the hub weeks' draws.
    # After it the sums walk in daily steps and the hub weeks in weekly
    # ones, with the same spread a week, so they agree in distribution.
    keys = ["location", "target_end_date", "sample"]
    cut = pd.Timestamp(hub_task.cutoff)
    sat = rolling.loc[
        pd.to_datetime(rolling["target_end_date"]).dt.weekday == 5
    ].copy()
    a = hub.copy()
    for f in (a, sat):
        f["target_end_date"] = pd.to_datetime(f["target_end_date"])
    upto_a = a.loc[a["target_end_date"] <= cut].sort_values(keys)
    upto_b = sat.loc[sat["target_end_date"] <= cut].sort_values(keys)
    pd.testing.assert_frame_equal(
        upto_a.reset_index(drop=True), upto_b.reset_index(drop=True)
    )
    later = ["location", "target_end_date"]
    ma = a.loc[a["target_end_date"] > cut].groupby(later)["value"].median()
    mb = sat.loc[sat["target_end_date"] > cut].groupby(later)["value"].median()
    assert len(ma) > 0
    big = ma > 20
    rel = (mb[big] - ma[big]).abs() / ma[big]
    assert (rel < 0.25).all(), rel.max()


def test_daily_errors_give_seven_day_sums_of_daily_draws(day_task) -> None:
    model = dataclasses.replace(get_model("randomwalknowcast"), error_days=1)
    out = model.forecast(day_task)
    assert set(out["target_end_date"].astype(str)) == set(
        day_task.target_end_dates
    )
    assert np.isfinite(out["value"]).all()
    assert (out["value"] >= 0).all()


def test_daily_quantiles_give_each_day_s_count(day_task) -> None:
    levels = (0.05, 0.5, 0.95)
    q = get_model("randomwalknowcast").daily_quantiles(day_task, levels)
    assert list(q.columns) == ["location", "date", "quantile", "value"]
    assert set(q["date"].astype(str)) == set(day_task.target_end_dates)
    wide = q.pivot_table(
        index=["location", "date"], columns="quantile", values="value"
    )
    assert (wide[0.05] <= wide[0.5]).all()
    assert (wide[0.5] <= wide[0.95]).all()
