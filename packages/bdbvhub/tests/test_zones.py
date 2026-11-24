from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from bdbvhub.submission.hub import to_hub_table, write_submission
from bdbvhub.submission.validate import Hub, validate_submission
from nowcastbaselines.randomwalk import RandomWalk, RandomWalkNowcast
from nowcastbaselines.zones import (
    check_zone_coherence,
    walks_own_zones,
    zone_forecast,
    zone_task,
    zones_for_model,
)
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.tidy import load_health_zones
from nowcastbdbv.data.zones import hub_zones
from nowcasteval.backtest import BacktestConfig
from nowcastutils.base import NATIONAL, ForecastTask
from nowcastutils.hierarchy import province_of
from nowcastutils.snapshots import weekly_counts

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "fixtures" / "data"
SNAPSHOT = ROOT / "packages/bdbvhub/tests/hub_validation/fixtures/hub-snapshot"
HUB = Hub.load(SNAPSHOT)
"""The pinned hub config."""
OPEN_VINTAGE = "2026-10-24"
"""A vintage whose reference date, 2026-10-31, is open in `HUB`."""


@pytest.fixture(scope="module")
def zones() -> tuple[str, ...]:
    return hub_zones(DATA / "hub-config" / "tasks.json")


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-10-17")


@pytest.fixture(scope="module")
def counts(daily_zones: pd.DataFrame) -> pd.DataFrame:
    return weekly_counts(daily_zones, "2026-10-17")


@pytest.fixture(scope="module")
def open_task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, OPEN_VINTAGE)


@pytest.fixture(scope="module")
def open_counts(daily_zones: pd.DataFrame) -> pd.DataFrame:
    return weekly_counts(daily_zones, OPEN_VINTAGE)


def _hub_failures(path: Path) -> list[str]:
    """The hub's checks a written submission fails."""
    rep = validate_submission(
        path,
        HUB,
        metadata_dirs=[ROOT / "packages" / "bdbvsubmission" / "model-metadata"],
        skip_window=True,
    )
    return [f"{c.check}: {c.message}" for c in rep.failed]


def _first_zone(prov: pd.DataFrame, zones: tuple[str, ...]) -> pd.DataFrame:
    """Zone samples that sum to `prov`: each province's first zone."""
    first: dict[str, str] = {}
    for zone in zones:
        first.setdefault(province_of(zone), zone)
    own = prov.loc[prov["location"].isin(first)]
    return own.assign(
        location=own["location"].map(first),
        value=own["value"].clip(lower=0).round(),
    ).reset_index(drop=True)


def test_hub_zones_are_the_519_zone_codes(zones: tuple[str, ...]) -> None:
    assert len(zones) == 519
    assert all(len(z.split("-")) == 3 for z in zones)
    assert province_of("CD-IT-10") == "CD-IT"


def test_weekly_zone_counts_match_the_hub_time_series(
    daily_zones: pd.DataFrame,
) -> None:
    hz = load_health_zones(DATA)
    for vintage in hz["vintage"].unique():
        got = weekly_counts(daily_zones, str(vintage))
        want = hz.loc[hz["vintage"] == vintage]
        both = got.merge(
            want, on=["location", "target_end_date"], how="outer"
        ).fillna(0)
        assert (both["total_x"] == both["total_y"]).all()
        assert set(got.columns) == {"location", "target_end_date", "total"}


def test_zone_task_is_a_zero_filled_randomwalk_task(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
) -> None:
    zt = zone_task(task, counts, zones, pd.concat([daily, daily_zones]))
    assert set(zt.locations) == {
        z for z in zones if province_of(z) in task.locations
    }
    one = zt.target.loc[zt.target["location"] == "CD-IT-10"]
    assert one["target_end_date"].is_monotonic_increasing
    assert one["target_end_date"].max() <= task.vintage
    gaps = pd.to_datetime(one["target_end_date"]).diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all()
    # The zone task carries snapshots of the country, provinces and zones.
    assert zt.daily_history is not None
    assert {"CD", "CD-IT", "CD-IT-10"} <= set(zt.daily_history["location"])
    assert zt.daily_history["vintage"].max() == task.vintage
    assert zone_task(task, counts, zones).daily_history is None


def _zone_model_inputs(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    model: RandomWalk,
    n_provinces: int = 2,
) -> tuple[ForecastTask, pd.DataFrame, pd.DataFrame]:
    sub = dataclasses.replace(task, locations=task.locations[:n_provinces])
    prov = model.forecast(sub)
    got = zone_forecast(model, sub, counts, zones, daily_zones)
    return sub, prov, got


@pytest.mark.parametrize("model", [RandomWalk(), RandomWalkNowcast()])
@pytest.mark.slow
def test_zone_walk_models_are_their_own_zone_draws(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    model: RandomWalk,
) -> None:
    sub, _, got = _zone_model_inputs(task, counts, zones, daily_zones, model)
    history = sub.daily_history
    daily = (
        daily_zones if history is None else pd.concat([history, daily_zones])
    )
    raw = model.forecast(zone_task(sub, counts, zones, daily))
    if "as_of" in raw:
        raw = raw.loc[raw["as_of"].isna()].drop(columns="as_of")
    want = raw.loc[raw["location"] != NATIONAL, list(got.columns)]
    # No province sample is apportioned down: each zone is its own model.
    keys = ["location", "target_end_date", "sample"]
    pd.testing.assert_frame_equal(
        got.sort_values(keys).reset_index(drop=True),
        want.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    assert set(got["location"]) == {
        z for z in zones if province_of(z) in sub.locations
    }


@pytest.mark.slow
def test_zone_hub_table_validates_with_the_zone_models_own_draws(
    open_task: ForecastTask,
    open_counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model = RandomWalkNowcast()
    prov = model.forecast(open_task)
    got = zone_forecast(model, open_task, open_counts, zones, daily_zones)
    ref = open_task.reference_date
    table = to_hub_table(
        prov, ref, zones=got, forecast_date=ref, zones_sum=False
    )
    keys = ["location", "output_type_id", "target_end_date"]
    zone_rows = table.loc[table["location"].str.count("-") == 2]
    # The zone rows are the zone model's draws, not a split of provinces.
    mine = got.assign(
        output_type_id=got["sample"].astype("int64"),
        value=got["value"].clip(lower=0).round(),
    ).set_index(keys)["value"]
    assert (
        zone_rows.set_index(keys)["value"]
        == mine.reindex(zone_rows.set_index(keys).index)
    ).all()
    # National is still the sum of the provinces.
    prov_rows = table.loc[table["location"].str.len() == 5]
    national = prov_rows.groupby(keys[1:])["value"].sum()
    cd = table.loc[table["location"] == NATIONAL].set_index(keys[1:])["value"]
    assert (cd == national.reindex(cd.index)).all()
    path = write_submission(table, tmp_path, "seabbs-rwnowcast")
    assert _hub_failures(path) == []


def test_hub_table_checks_zone_sums_unless_told_not_to(
    open_task: ForecastTask, zones: tuple[str, ...]
) -> None:
    prov = RandomWalk().forecast(open_task)
    alloc = _first_zone(prov, zones)
    assert check_zone_coherence(prov, alloc) == []
    # One more case in one zone than its province has.
    alloc.loc[alloc.index[0], "value"] += 1
    assert check_zone_coherence(prov, alloc)
    ref = open_task.reference_date
    # A summing model's zones must sum to its provinces.
    with pytest.raises(ValueError, match="zone"):
        to_hub_table(prov, ref, zones=alloc, forecast_date=ref)
    # The random walks' own zone draws need not.
    table = to_hub_table(
        prov, ref, zones=alloc, forecast_date=ref, zones_sum=False
    )
    assert set(alloc["location"]) <= set(table["location"])


def test_zones_need_not_sum_for_the_walks_and_the_foundation_model() -> None:
    from bdbvhub.submission.hub import zones_need_not_sum

    # The walks run each zone on its own data; the foundation model takes
    # its zones straight from its block. Neither sums them to provinces.
    for name in (
        "randomwalk",
        "randomwalknowcast",
        "foundation_zeroshot",
        "foundation",
        "foundation_zeroshot_complete",
    ):
        assert zones_need_not_sum(name), name
    for name in ("renewal", "renewalsp", None, "not-a-model"):
        assert not zones_need_not_sum(name), name


def test_only_the_random_walks_forecast_their_own_zones() -> None:
    assert walks_own_zones("randomwalk")
    assert walks_own_zones("randomwalknowcast")
    assert not walks_own_zones("renewal")
    assert not walks_own_zones(None)
    assert not walks_own_zones("not-a-model")


@pytest.mark.slow
def test_walk_models_take_their_own_zone_draws(
    open_task: ForecastTask,
    open_counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
) -> None:
    model = RandomWalkNowcast()
    prov = model.forecast(open_task)
    got = zones_for_model(
        "randomwalknowcast", open_task, prov, open_counts, zones, daily_zones
    )
    want = zone_forecast(model, open_task, open_counts, zones, daily_zones)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want.reset_index(drop=True)
    )


def test_samples_without_zones_get_none(
    open_task: ForecastTask,
    open_counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
) -> None:
    # A model's zones come from its own forecast, so samples with none
    # get none here.
    prov = RandomWalk().forecast(open_task)
    got = zones_for_model(
        "renewal", open_task, prov, open_counts, zones, daily_zones
    )
    assert got.empty
