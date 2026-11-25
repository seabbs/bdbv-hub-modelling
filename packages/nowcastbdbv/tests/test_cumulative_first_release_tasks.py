"""BDBV tasks leave out a cumulative first release."""

import pandas as pd

from nowcastbdbv.backtest import make_tasks
from nowcasteval.backtest import BacktestConfig


def _target(cutoffs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": "real",
            "vintage": cutoffs,
            "location": "CD-IT",
            "location_name": "Ituri",
            "target_end_date": cutoffs,
            "observed": 1,
            "imputed": 0,
            "total": 1,
            "cutoff": cutoffs,
        }
    )


def _daily(cutoffs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vintage": cutoffs,
            "location": "CD-IT",
            "onset_date": "2026-04-20",
            "observed": 1,
            "imputed": 0,
            "cutoff": cutoffs,
        }
    )


def test_tasks_after_a_cumulative_first_release_leave_it_out() -> None:
    cuts = ["2026-08-18", "2026-10-09", "2026-10-17"]
    tasks = make_tasks(
        _target(cuts),
        None,
        BacktestConfig(max_delay=42),
        daily=_daily(cuts),
        places=pd.DataFrame(columns=["location"]),
    )
    by = {t.vintage: t for t in tasks}
    assert sorted(by["2026-08-18"].daily_history["vintage"]) == cuts[:1]
    for v in cuts[1:]:
        assert "2026-08-18" not in set(by[v].daily_history["vintage"])
        assert "2026-08-18" not in set(by[v].target_history["vintage"])
