"""A first release that is a cumulative dump is left out of model inputs."""

import dataclasses
import datetime as dt
import logging

import pandas as pd

from nowcastutils.base import ForecastTask
from nowcastutils.releases import (
    cumulative_first_release,
    without_cumulative_first_release,
)


def _daily(cutoffs: list[str]) -> pd.DataFrame:
    """Each release holds every onset day from 1 April to its cut-off."""
    rows = []
    for c in cutoffs:
        for day in pd.date_range("2026-04-01", c, freq="D"):
            rows.append(
                {
                    "vintage": c,
                    "location": "CD",
                    "onset_date": day.strftime("%Y-%m-%d"),
                    "observed": 1,
                    "imputed": 0,
                    "cutoff": c,
                }
            )
    return pd.DataFrame(rows)


def _weekly(cutoffs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": "real",
            "vintage": cutoffs,
            "location": "CD",
            "target_end_date": cutoffs,
            "total": 1,
            "cutoff": cutoffs,
        }
    )


GAP = ["2026-08-18", "2026-10-07", "2026-10-14"]
WEEKLY = ["2026-09-30", "2026-10-07", "2026-10-14"]


def test_a_first_release_more_than_d_before_the_next_is_cumulative() -> None:
    assert cumulative_first_release(_daily(GAP), 42) == "2026-08-18"
    # Within D of the next release it is part of the reporting.
    assert cumulative_first_release(_daily(WEEKLY), 42) is None
    assert cumulative_first_release(_daily(GAP), 60) is None


def test_a_lone_release_or_no_d_is_never_cumulative() -> None:
    assert cumulative_first_release(_daily(GAP[:1]), 42) is None
    assert cumulative_first_release(_daily(GAP), None) is None


def _task(cutoffs: list[str], max_delay: int | None = 42) -> ForecastTask:
    daily = _daily(cutoffs)
    return ForecastTask(
        vintage=cutoffs[-1],
        reference_date=dt.date(2026, 10, 17),
        kind="real",
        cutoff=cutoffs[-1],
        target=_weekly(cutoffs[-1:]),
        delays=None,
        locations=("CD",),
        target_end_dates=("2026-10-17",),
        target_history=_weekly(cutoffs),
        daily_history=daily,
        zone_history=_weekly(cutoffs),
        zone_daily_history=daily,
        max_delay=max_delay,
    )


def test_the_task_reads_its_releases_from_the_next_one(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nowcastutils.releases"):
        task = without_cumulative_first_release(_task(GAP))
    for name in (
        "target_history",
        "daily_history",
        "zone_history",
        "zone_daily_history",
    ):
        got = getattr(task, name)
        assert sorted(got["vintage"].unique()) == GAP[1:], name
    assert "2026-08-18" in caplog.text


def test_tasks_without_a_cumulative_first_release_are_unchanged() -> None:
    task = _task(WEEKLY)
    assert without_cumulative_first_release(task) is task
    task = _task(GAP, max_delay=None)
    assert without_cumulative_first_release(task) is task
    # At the first release itself there is nothing after it.
    task = _task(GAP[:1])
    assert without_cumulative_first_release(task) is task


def test_the_first_release_s_own_task_keeps_it() -> None:
    task = dataclasses.replace(_task(GAP[:1]))
    assert sorted(task.daily_history["vintage"].unique()) == GAP[:1]
