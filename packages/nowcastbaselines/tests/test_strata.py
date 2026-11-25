"""Both walks forecast any stratification of the task's places.

The places are read from their codes (`nowcastutils.hierarchy`): one
series, places under a root, ages under a root, places by ages, or
deeper, with or without rows for the aggregates.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.randomwalk import RandomWalk, RandomWalkNowcast
from nowcastutils.base import SAMPLE_COLUMNS, ForecastTask
from nowcastutils.hierarchy import parent

LAST = dt.date(2026, 10, 3)
"""The latest release, a Saturday."""

N_SAMPLES = 40


def _daily(leaves: tuple[str, ...], aggregates: bool) -> pd.DataFrame:
    """Return weekly releases of daily onsets of `leaves`.

    Each leaf's rate walks on the log scale, and each case is reported
    after a delay of up to 9 days. With `aggregates`, every ancestor of a
    leaf has rows too, the sum of its leaves.
    """
    rng = np.random.default_rng(1)
    days = pd.date_range(LAST - dt.timedelta(days=119), LAST).date
    log_rate = np.log(10.0) + np.cumsum(
        rng.normal(0.0, 0.1, (len(leaves), days.size)), axis=1
    )
    cases = rng.poisson(np.exp(log_rate))
    delay = [
        [rng.integers(0, 10, n) for n in row] for row in cases
    ]  # fmt: skip
    releases = [LAST - dt.timedelta(days=7 * k) for k in range(8)][::-1]
    rows = []
    for release in releases:
        for i, leaf in enumerate(leaves):
            for t, day in enumerate(days[days <= release]):
                seen = int((delay[i][t] <= (release - day).days).sum())
                if seen:
                    rows.append((release, leaf, day, seen))
    out = pd.DataFrame(
        rows, columns=["vintage", "location", "onset_date", "observed"]
    )
    if aggregates:
        frames = [out]
        for leaf in leaves:
            code = parent(leaf)
            while code is not None:
                frames.append(
                    out.loc[out["location"] == leaf].assign(location=code)
                )
                code = parent(code)
        out = (
            pd.concat(frames)
            .groupby(["vintage", "location", "onset_date"], as_index=False)
            .sum()
        )
    for col in ("vintage", "onset_date"):
        out[col] = out[col].astype(str)
    return out.assign(imputed=0, cutoff=out["vintage"])


def _task(
    leaves: tuple[str, ...], locations: tuple[str, ...], aggregates: bool
) -> ForecastTask:
    ends = tuple(
        (LAST + dt.timedelta(weeks=h)).isoformat() for h in range(-3, 4)
    )
    return ForecastTask(
        vintage=LAST.isoformat(),
        reference_date=LAST,
        kind="real",
        cutoff=LAST.isoformat(),
        target=pd.DataFrame(),
        delays=None,
        locations=locations,
        target_end_dates=ends,
        n_samples=N_SAMPLES,
        seed=1,
        daily_history=_daily(leaves, aggregates),
    )


ZONES = ("XX-A-1", "XX-A-2", "XX-B-1")
PLACES_BY_AGES = ("XX-A-A0", "XX-A-A1", "XX-B-A0", "XX-B-A1")
SHAPES = {
    "single series": (("XX",), ("XX",)),
    "places": (("XX-A", "XX-B", "XX-C"), ("XX-A", "XX-B", "XX-C")),
    "places and the total": (("XX-A", "XX-B"), ("XX", "XX-A", "XX-B")),
    "ages": (("XX-A0", "XX-A1", "XX-A2"), ("XX-A0", "XX-A1", "XX-A2")),
    "places by ages": (PLACES_BY_AGES, PLACES_BY_AGES),
    "zones": (ZONES, ZONES),
    "every level": (ZONES, ("XX", "XX-A", "XX-B", *ZONES)),
    "four levels": (
        ("XX-A-1-a", "XX-A-1-b", "XX-A-2-a"),
        ("XX-A", "XX-A-1", "XX-A-1-a", "XX-A-1-b", "XX-A-2-a"),
    ),
}


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
@pytest.mark.parametrize("aggregates", [True, False])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_shape_gives_samples_for_each_place(
    model: type[RandomWalk], aggregates: bool, shape: str
) -> None:
    leaves, locations = SHAPES[shape]
    task = _task(leaves, locations, aggregates)
    out = model().forecast(task)
    assert list(out.columns) == list(SAMPLE_COLUMNS)
    assert set(out["location"]) == set(locations)
    cells = out.groupby(["location", "target_end_date"]).size()
    assert len(cells) == len(locations) * len(task.target_end_dates)
    assert (cells == N_SAMPLES).all()
    assert (out["value"] >= 0).all()
    # Each place is forecast at its own level: the latest week ahead is
    # not all zeros.
    ahead = out.loc[out["target_end_date"] == task.target_end_dates[-1]]
    assert (ahead.groupby("location")["value"].sum() > 0).all()


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
def test_a_missing_aggregate_is_forecast_as_one_given_its_rows(
    model: type[RandomWalk],
) -> None:
    leaves, locations = SHAPES["every level"]
    with_rows = model().forecast(_task(leaves, locations, True))
    without = model().forecast(_task(leaves, locations, False))
    pd.testing.assert_frame_equal(with_rows, without)


@pytest.mark.parametrize("model", [RandomWalk, RandomWalkNowcast])
def test_a_task_with_no_places_gives_no_samples(
    model: type[RandomWalk],
) -> None:
    task = _task(("XX-A",), (), True)
    out = model().forecast(task)
    assert out.empty
    assert list(out.columns) == list(SAMPLE_COLUMNS)
