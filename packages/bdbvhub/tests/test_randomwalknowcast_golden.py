"""Today's randomwalknowcast on the fixture task, pinned at its seed.

The single random walk nowcast moves the chain ladder's orchestration
into the baselinenowcast port's modules; this pins its BDBV output so
the move changes nothing: every place's and week's quantiles of the
draws, from the fixture task's own seed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.zones import ZoneData, hub_zones
from nowcasteval.backtest import BacktestConfig

GOLDEN = Path(__file__).parent / "fixtures" / "randomwalknowcast-golden.csv"
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _task(target, delays, daily, daily_zones, health_zones, tasks_json):
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


def quantiles(task) -> pd.DataFrame:
    """The forecast's quantiles by place and week, as pinned."""
    draws = get_model("randomwalknowcast").forecast(task)
    q = (
        draws.groupby(["location", "target_end_date"])["value"]
        .quantile(np.asarray(LEVELS))
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "quantile"})
    )
    q["target_end_date"] = q["target_end_date"].astype(str)
    return q.sort_values(["location", "target_end_date", "quantile"])


@pytest.mark.slow
def test_randomwalknowcast_is_unchanged_on_the_fixture_task(
    target, delays, daily, daily_zones, health_zones, tasks_json
):
    got = quantiles(
        _task(target, delays, daily, daily_zones, health_zones, tasks_json)
    ).reset_index(drop=True)
    if os.environ.get("WRITE_GOLDEN"):
        # Rewrite the pin only on purpose, from the code it pins.
        got.to_csv(GOLDEN, index=False)
        pytest.skip("golden written")
    want = pd.read_csv(GOLDEN, dtype={"target_end_date": str})
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9)
