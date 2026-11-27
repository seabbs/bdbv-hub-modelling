"""German tasks carry the truth's delay, so Chronos-2 has fixed lags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nowcastfoundation.forecaster import FoundationModel
from nowcastgermany import evaluation as de
from nowcastgermany.data import nowcast_tasks, read_triangle
from nowcastutils.base import get_model

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)
DATE = "2021-10-16"
HORIZONS = range(-21, 22, 7)
D = 40


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


def test_tasks_carry_the_max_delay(triangle: pd.DataFrame) -> None:
    tasks = nowcast_tasks(triangle, DATE, 50, 1, HORIZONS, max_delay=D)
    assert [t.max_delay for t in tasks] == [D, D]
    # Without one, the lags stay the data's own.
    plain = nowcast_tasks(triangle, DATE, 50, 1, HORIZONS)
    assert [t.max_delay for t in plain] == [None, None]


@pytest.mark.parametrize(
    "name", ["foundation_zeroshot", "foundation_zeroshot_complete"]
)
def test_chronos_has_lags_0_to_d(triangle: pd.DataFrame, name: str) -> None:
    model = get_model(name)
    assert isinstance(model, FoundationModel)
    for task in nowcast_tasks(triangle, DATE, 50, 1, HORIZONS, max_delay=D):
        run = model.for_task(task)
        assert run.source == "daily"
        assert run.lags == D + 1


def test_germany_forecasts_give_tasks_the_max_delay(
    triangle: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    real = de.nowcast_tasks

    def record(*args, **kwargs):
        tasks = real(*args, **kwargs)
        seen.extend(t.max_delay for t in tasks)
        return tasks

    monkeypatch.setattr(de, "nowcast_tasks", record)
    de.germany_forecasts(["randomwalk"], triangle, [DATE], 20, 1, max_delay=D)
    assert seen
    assert set(seen) == {D}


def test_the_task_store_keeps_max_delays_apart(
    triangle: pd.DataFrame, tmp_path: Path
) -> None:
    # Tasks stored without a max delay are not read back for one.
    plain = de._tasks(triangle, DATE, 20, 1, tmp_path, HORIZONS)
    fixed = de._tasks(triangle, DATE, 20, 1, tmp_path, HORIZONS, max_delay=D)
    assert {t.max_delay for t in plain} == {None}
    assert {t.max_delay for t in fixed} == {D}
