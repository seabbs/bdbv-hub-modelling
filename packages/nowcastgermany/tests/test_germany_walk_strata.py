"""Both walks on the German tasks, whatever places they hold."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalk, RandomWalkNowcast
from nowcastgermany.data import ALL_AGES, nowcast_tasks, read_triangle
from nowcastutils.base import NATIONAL, SAMPLE_COLUMNS

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)
DATE = "2021-10-16"
HORIZONS = range(-21, 22, 7)


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
def test_states_ages_and_germany_in_one_task(
    triangle: pd.DataFrame, model: type[RandomWalk]
) -> None:
    states, ages = nowcast_tasks(triangle, DATE, 50, 1, horizons=HORIZONS)
    task = dataclasses.replace(
        states,
        locations=(NATIONAL, *states.locations, *ages.locations),
        daily_history=pd.concat(
            [states.daily_history, ages.daily_history]
        ).drop_duplicates(),
    )
    out = model().forecast(task)
    assert set(out["location"]) == set(task.locations)
    cells = out.groupby(["location", "target_end_date"]).size()
    assert len(cells) == len(task.locations) * len(task.target_end_dates)
    assert (cells == 50).all()


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
def test_data_with_no_age_groups(
    triangle: pd.DataFrame, model: type[RandomWalk]
) -> None:
    rows = triangle.loc[triangle["age_group"] == ALL_AGES]
    states, ages = nowcast_tasks(rows, DATE, 50, 1, horizons=HORIZONS)
    assert ages.locations == ()
    out = model().forecast(states)
    assert set(out["location"]) == set(states.locations)
    empty = model().forecast(ages)
    assert empty.empty
    assert list(empty.columns) == list(SAMPLE_COLUMNS)
