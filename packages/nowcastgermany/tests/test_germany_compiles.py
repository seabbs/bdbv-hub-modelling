"""randomwalknowcast on consecutive German dates reuses its programs.

Each German nowcast date brings a release and a day more. The arrays
of the walks, draws, delay sums and size fits are padded to a few sizes
(`nowcastutils.shapes.bucket`), so a later date runs the programs an
earlier one compiled. The draws are those of the unpadded arrays, pinned
here on the fixture's dates: the 7-day sums and the daily counts.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import jax.monitoring
import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalkNowcast
from nowcastgermany.data import nowcast_tasks, read_triangle
from nowcastutils.base import ForecastTask

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)
GOLDEN = Path(__file__).parent / "fixtures" / "randomwalknowcast-golden.csv"
DATES = ("2021-10-17", "2021-10-18")
"""Consecutive nowcast dates, a Sunday and a Monday, whose arrays pad
to the same sizes."""
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

_COMPILES = [0]


def _count(event: str, _duration: float, **_: object) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count)


@contextmanager
def compiles() -> Iterator[list[int]]:
    """Count the programs JAX compiles in the block."""
    out = [0]
    before = _COMPILES[0]
    try:
        yield out
    finally:
        out[0] = _COMPILES[0] - before


@pytest.fixture(scope="module")
def tasks() -> list[ForecastTask]:
    triangle = read_triangle(FIXTURE)
    return [
        nowcast_tasks(triangle, d, 200, 1, horizons=range(-7, 8))[0]
        for d in DATES
    ]


def _quantiles(task: ForecastTask) -> pd.DataFrame:
    """The 7-day sums' and daily counts' quantiles, by place and day."""
    model = RandomWalkNowcast()
    draws = model.forecast(task)
    sums = (
        draws.groupby(["location", "target_end_date"])["value"]
        .quantile(np.asarray(LEVELS))
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "quantile", "target_end_date": "date"})
    )
    sums["target"] = "7day"
    daily = model.daily_quantiles(task, LEVELS)
    daily["target"] = "daily"
    out = pd.concat([sums, daily], ignore_index=True)
    out["date"] = out["date"].astype(str)
    out.insert(0, "origin", task.vintage)
    return out.sort_values(
        ["origin", "target", "location", "date", "quantile"]
    ).reset_index(drop=True)


def test_a_later_date_runs_the_programs_compiled_for_the_last(
    tasks: list[ForecastTask],
) -> None:
    first = _quantiles(tasks[0])
    with compiles() as n:
        second = _quantiles(tasks[1])
    got = pd.concat([first, second], ignore_index=True)
    if os.environ.get("WRITE_GOLDEN"):
        # Rewrite the pin only on purpose, from the code it pins.
        GOLDEN.parent.mkdir(exist_ok=True)
        got.to_csv(GOLDEN, index=False)
        pytest.skip("golden written")
    assert n[0] == 0
    # The padding leaves every draw as it was.
    want = pd.read_csv(GOLDEN, dtype={"date": str, "origin": str})
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9)
