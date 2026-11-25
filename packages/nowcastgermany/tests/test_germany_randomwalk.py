"""The walks' German 7-day sums ahead of a mid-week nowcast date.

The 7-day sums after the cut-off walk on from each sample's sum ending
on the cut-off in daily steps, the target's own resolution. The daily
counts walk in daily steps too, each step with its weekday's mean, so
their median repeats the week from the cut-off, and include the country
as the sum of the states. `randomwalk` returns the sums to its start as
reported and walks on in daily steps.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalk, RandomWalkNowcast
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


def _median_interval(
    q: pd.DataFrame, loc: str, day: str, n: int
) -> tuple[float, float, float]:
    """Return a median and its 95% order-statistic interval from `n` draws."""
    own = q.loc[(q["location"] == loc) & (q["date"].astype(str) == day)]
    at = own.set_index("quantile")["value"]
    half = 1.96 * 0.5 / np.sqrt(n)
    return at[0.5], at[round(0.5 - half, 4)], at[round(0.5 + half, 4)]


def test_daily_medians_repeat_the_week_from_the_cut_off(
    task: ForecastTask,
) -> None:
    # The walk's steps into each weekday have that weekday's mean, which
    # sum to 0 over the week: the median follows the weekday pattern and
    # repeats every 7 days, within the medians' Monte Carlo error.
    n = 2000
    big = dataclasses.replace(task, n_samples=n)
    half = round(1.96 * 0.5 / np.sqrt(n), 4)
    q = RandomWalkNowcast().daily_quantiles(big, (0.5 - half, 0.5, 0.5 + half))
    cut = pd.Timestamp(DATE)
    for loc in ("CD-BE", "CD-BY"):
        week = [
            _median_interval(
                q, loc, (cut + dt.timedelta(days=h)).date().isoformat(), n
            )
            for h in range(1, 8)
        ]
        # The week's medians follow the weekday pattern, not one level.
        mids = [m for m, _, _ in week]
        width = max(hi - lo for _, lo, hi in week)
        assert max(mids) - min(mids) > 2 * width, loc
        for h in range(1, 8):
            a = (cut + dt.timedelta(days=h)).date().isoformat()
            b = (cut + dt.timedelta(days=h + 7)).date().isoformat()
            ma, lo_a, hi_a = _median_interval(q, loc, a, n)
            mb, lo_b, hi_b = _median_interval(q, loc, b, n)
            assert abs(ma - mb) <= (hi_a - lo_a) + (hi_b - lo_b), (loc, h)
