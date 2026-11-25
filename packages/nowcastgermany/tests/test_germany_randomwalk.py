"""The walks' German 7-day sums ahead of a mid-week nowcast date.

The 7-day sums after the cut-off walk on from each sample's sum ending
on the cut-off in daily steps, the target's own resolution, so they
carry no weekday cycle. The daily counts walk in daily steps too, and
include the country as the sum of the states. `randomwalk` returns the
sums to its start as reported and walks on in daily steps.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import (
    RandomWalk,
    RandomWalkNowcast,
    _step_sigma,
)
from nowcastgermany.data import nowcast_tasks, read_triangle
from nowcastutils.base import NATIONAL, ForecastTask

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)
DATE = "2021-10-13"
"""A Wednesday nowcast date."""


@pytest.fixture(scope="module")
def task() -> ForecastTask:
    triangle = read_triangle(FIXTURE)
    return nowcast_tasks(triangle, DATE, 200, 1, horizons=range(-3, 22))[0]


@pytest.fixture(scope="module")
def draws(task: ForecastTask) -> pd.DataFrame:
    out = RandomWalkNowcast().forecast(task)
    out["target_end_date"] = pd.to_datetime(out["target_end_date"])
    return out


def test_sums_ahead_walk_in_daily_steps(draws: pd.DataFrame) -> None:
    # A walk at daily steps has independent steps from day to day, so two
    # days' changes of log(1 + sum) are not positively correlated (each
    # day's Poisson count makes them negatively so); sums interpolated
    # between weekly steps would change alike every day of a week.
    cut = pd.Timestamp(DATE)
    wide = draws.pivot_table(
        index=["location", "sample"],
        columns="target_end_date",
        values="value",
    )
    logs = np.log1p(wide.loc["CD-BY"])
    for h in (1, 3, 8):
        days = [cut + dt.timedelta(days=h + i) for i in range(3)]
        first = logs[days[1]] - logs[days[0]]
        second = logs[days[2]] - logs[days[1]]
        assert first.std() > 0
        assert np.corrcoef(first, second)[0, 1] < 0.5, h


def test_daily_counts_include_the_country(task: ForecastTask) -> None:
    # The country is the sum of the states by sample, as for the sums.
    q = RandomWalkNowcast().daily_quantiles(task, (0.5,))
    assert NATIONAL in set(q["location"])
    wide = q.pivot_table(index="date", columns="location", values="value")
    states = [c for c in wide.columns if c != NATIONAL]
    assert (wide[NATIONAL] >= wide[states].max(axis=1)).all()


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
def test_median_sums_ahead_have_no_weekday_cycle(
    task: ForecastTask, model: type[RandomWalk]
) -> None:
    # The medians of the sums after the cut-off, averaged by weekday, are
    # about level: a weekday cycle, or a sum left at 0, would make some
    # weekdays' sums higher.
    cut = pd.Timestamp(DATE)
    out = model().forecast(task)
    ahead = out.loc[pd.to_datetime(out["target_end_date"]) > cut]
    assert set(ahead["location"]) == {"CD-BE", "CD-BY"}
    med = ahead.groupby(["location", "target_end_date"])["value"].median()
    for loc, own in med.groupby(level="location"):
        days = pd.to_datetime(own.index.get_level_values("target_end_date"))
        by_day = own.groupby((days - cut).days % 7).mean()
        assert by_day.std() / by_day.mean() < 0.08, loc


def test_randomwalk_sums_to_its_start_are_as_reported() -> None:
    triangle = read_triangle(FIXTURE)
    task = nowcast_tasks(triangle, DATE, 50, 1, horizons=range(-28, 22))[0]
    out = RandomWalk().forecast(task)
    daily = task.daily_history
    assert daily is not None
    latest = daily.loc[daily["vintage"] == daily["vintage"].max()]
    fixed = 0
    for (loc, day), cell in out.groupby(["location", "target_end_date"]):
        assert cell["value"].median() > 0, (loc, day)
        if cell["value"].nunique() > 1:
            continue
        end = pd.Timestamp(day)
        days = pd.to_datetime(latest["onset_date"])
        week = latest.loc[
            (latest["location"] == loc)
            & (days > end - pd.Timedelta(days=7))
            & (days <= end)
        ]
        assert cell["value"].iloc[0] == week["observed"].sum(), (loc, day)
        fixed += 1
    assert fixed > 0


def test_daily_counts_walk_with_the_sums_spread(
    task: ForecastTask, draws: pd.DataFrame
) -> None:
    # The walk's spread is one estimate, scaled to the step: a day's walk
    # spreads as the 7-day sums' does, not by the weekday swings of the
    # daily counts.
    day = (pd.Timestamp(DATE) + dt.timedelta(days=21)).date().isoformat()
    q = RandomWalkNowcast().daily_quantiles(task, (0.1, 0.9))
    q = q.loc[(q["location"] == "CD-BY") & (q["date"].astype(str) == day)]
    daily = np.log1p(q["value"].to_numpy())
    sums = draws.loc[
        (draws["location"] == "CD-BY")
        & (draws["target_end_date"] == pd.Timestamp(day)),
        "value",
    ]
    weekly = np.log1p(sums.quantile([0.1, 0.9]).to_numpy())
    assert np.ptp(daily) < 1.5 * np.ptp(weekly)


def test_daily_walk_steps_into_each_weekday_with_its_own_spread(
    task: ForecastTask,
) -> None:
    # The daily walk's step into each day takes that day's weekday's
    # spread, from the daily nowcast's changes into that weekday.
    model = RandomWalkNowcast()
    model._start(task)
    origin, _ = model._block_draws(task, 1, weekdays=True)
    assert origin.step_sigma is not None
    assert origin.step_sigma.shape == (len(origin.snap.locations), 7)
    cut = origin.snap.cut_week(origin.o)
    first = np.full(len(origin.snap.locations), cut)
    steps = _step_sigma(origin, first, 14)
    day = pd.Timestamp(DATE) + pd.to_timedelta(np.arange(1, 15), unit="D")
    want = origin.step_sigma[:, day.weekday]
    np.testing.assert_allclose(steps, want)
