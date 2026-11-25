"""randomwalknowcast's German 7-day sums ahead of a mid-week nowcast date.

The sums ending after the cut-off come from one random walk of the sums
ending on the cut-off's weekday, so they carry no weekday cycle: each
lies between the sums ending on that weekday either side of it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalkNowcast
from nowcastgermany.data import nowcast_tasks, read_triangle

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
def draws() -> pd.DataFrame:
    triangle = read_triangle(FIXTURE)
    task = nowcast_tasks(triangle, DATE, 200, 1, horizons=range(-3, 22))[0]
    out = RandomWalkNowcast().forecast(task)
    out["target_end_date"] = pd.to_datetime(out["target_end_date"])
    return out


def test_sums_ahead_lie_between_the_cutoff_weekdays(
    draws: pd.DataFrame,
) -> None:
    cut = pd.Timestamp(DATE)
    wide = draws.pivot_table(
        index=["location", "sample"],
        columns="target_end_date",
        values="value",
    )
    for h in range(1, 22):
        if h % 7 == 0:
            continue
        day = cut + dt.timedelta(days=h)
        before = cut + dt.timedelta(days=7 * (h // 7))
        after = before + dt.timedelta(days=7)
        if after not in wide.columns:
            continue
        lo = np.minimum(wide[before], wide[after])
        hi = np.maximum(wide[before], wide[after])
        assert ((wide[day] >= lo) & (wide[day] <= hi)).all(), day


def test_median_sums_ahead_have_no_weekday_cycle(
    draws: pd.DataFrame,
) -> None:
    cut = pd.Timestamp(DATE)
    ahead = draws.loc[
        (draws["location"] == "DE") & (draws["target_end_date"] >= cut)
    ]
    med = ahead.groupby("target_end_date")["value"].median()
    # Between two sums ending on the cut-off weekday the medians move one
    # way only.
    for k in range(3):
        week = med.iloc[7 * k : 7 * k + 8].to_numpy()
        if len(week) < 8:
            break
        steps = np.diff(week)
        assert (steps >= -1).all() or (steps <= 1).all(), week
