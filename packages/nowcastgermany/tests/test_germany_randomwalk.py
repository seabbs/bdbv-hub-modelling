"""randomwalknowcast's German 7-day sums ahead of a mid-week nowcast date.

The 7-day sums after the cut-off walk on from each sample's sum ending
on the cut-off in daily steps, the target's own resolution, so they
carry no weekday cycle. The daily counts walk in daily steps too, and
include the country as the sum of the states.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalkNowcast
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


def test_median_sums_ahead_have_no_weekday_cycle(
    draws: pd.DataFrame,
) -> None:
    # The medians of the sums after the cut-off, averaged by weekday, are
    # about level: a weekday cycle would make some weekdays' sums higher.
    cut = pd.Timestamp(DATE)
    ahead = draws.loc[draws["target_end_date"] > cut]
    assert set(ahead["location"]) == {"CD-BE", "CD-BY"}
    med = ahead.groupby(["location", "target_end_date"])["value"].median()
    for loc, own in med.groupby(level="location"):
        days = own.index.get_level_values("target_end_date")
        by_day = own.groupby((days - cut).days % 7).mean()
        assert by_day.std() / by_day.mean() < 0.08, loc


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
