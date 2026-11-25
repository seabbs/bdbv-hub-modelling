"""The foundation model's option of weekly pseudo releases."""

import datetime as dt
import itertools

import pandas as pd
import pytest

from nowcastfoundation.forecaster import FROM_DATA, FoundationModel
from nowcastutils.base import ForecastTask, list_models


def _day(s: str, k: int = 0) -> str:
    return (dt.date.fromisoformat(s) + dt.timedelta(days=k)).isoformat()


def _task(delays: pd.DataFrame | None) -> ForecastTask:
    """One release 50 days before a second, both of CD and CD-AA."""
    rows = []
    for cutoff in ("2026-01-10", "2026-03-01"):
        for place in ("CD", "CD-AA"):
            for k in range(60):
                rows.append(
                    {
                        "vintage": cutoff,
                        "location": place,
                        "onset_date": _day(cutoff, -k),
                        "observed": 10.0,
                        "imputed": 0.0,
                        "cutoff": cutoff,
                    }
                )
    daily = pd.DataFrame(rows)
    return ForecastTask(
        vintage="2026-03-01",
        reference_date=dt.date(2026, 3, 7),
        kind="real",
        cutoff="2026-03-01",
        target=pd.DataFrame(),
        delays=delays,
        locations=("CD-AA",),
        target_end_dates=("2026-02-28", "2026-03-07"),
        daily_history=daily,
    )


DELAYS = pd.DataFrame(
    {
        "vintage": "2026-03-01",
        "location": "CD-AA",
        "onset_week": [_day("2025-11-01", 7 * k) for k in range(20)],
        "delay": 3,
        "cutoff": "2026-03-01",
    }
)


def _vintages(model: FoundationModel, task: ForecastTask) -> list[str]:
    rows, _, _ = model.for_task(task)._rows(task)
    return sorted(rows["vintage"].unique())


def test_pseudo_weeks_fill_the_release_history_weekly() -> None:
    task = _task(DELAYS)
    plain = FoundationModel("f", FROM_DATA, "ragged")
    pseudo = FoundationModel("f", FROM_DATA, "ragged", pseudo_weeks=True)
    assert _vintages(plain, task) == ["2026-01-10", "2026-03-01"]
    got = _vintages(pseudo, task)
    assert {"2026-01-10", "2026-03-01"} <= set(got)
    assert "2026-02-22" in got
    days = [dt.date.fromisoformat(v) for v in got]
    assert max((b - a).days for a, b in itertools.pairwise(days)) < 14


def test_pseudo_weeks_need_the_delay_line() -> None:
    task = _task(None)
    pseudo = FoundationModel("f", FROM_DATA, "ragged", pseudo_weeks=True)
    assert _vintages(pseudo, task) == ["2026-01-10", "2026-03-01"]


def test_pseudo_weeks_are_daily_only() -> None:
    with pytest.raises(ValueError, match="pseudo_weeks"):
        FoundationModel("f", "weekly", "ragged", pseudo_weeks=True)


@pytest.mark.parametrize(
    "name", ["foundation_zeroshot_pseudoweeks", "foundation_pseudoweeks"]
)
def test_the_pseudo_week_models_are_registered(name) -> None:
    specs = {s.name: s for s in list_models()}
    assert specs[name].paper_only
    assert specs[name].factory.keywords["pseudo_weeks"]
