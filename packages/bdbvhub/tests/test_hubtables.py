from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from bdbvhub.reporting.hubtables import (
    HUB_QUANTILE_LEVELS,
    hub_quantiles,
    hub_tables,
    write_hub_tables,
)
from nowcastbaselines.zones import zone_forecast
from nowcastbdbv import backtest as bdbv_bt
from nowcastbdbv.data.tidy import load_daily_zones
from nowcastbdbv.data.zones import hub_zones
from nowcasteval import backtest as bt
from nowcastutils.base import NATIONAL
from nowcastutils.hierarchy import is_zone, province_of
from nowcastutils.snapshots import weekly_counts

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "fixtures" / "data"


def test_hub_quantiles_give_the_23_levels() -> None:
    table = pd.DataFrame(
        {
            "reference_date": "2026-11-07",
            "target": "wk inc confirmed cases",
            "location": "CD",
            "output_type": "sample",
            "output_type_id": np.arange(1, 101),
            "value": np.arange(100.0),
        }
    )
    out = hub_quantiles(table)
    assert len(HUB_QUANTILE_LEVELS) == 23
    assert list(out.columns) == list(table.columns)
    assert set(out["output_type"]) == {"quantile"}
    assert list(out["output_type_id"]) == pytest.approx(
        list(HUB_QUANTILE_LEVELS)
    )
    assert out["value"].is_monotonic_increasing


def test_hub_quantiles_take_other_levels() -> None:
    table = pd.DataFrame(
        {
            "location": "DE",
            "output_type": "sample",
            "output_type_id": np.arange(1, 101),
            "value": np.arange(100.0),
        }
    )
    levels = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    out = hub_quantiles(table, levels)
    assert list(out["output_type_id"]) == pytest.approx(list(levels))
    want = np.quantile(np.arange(100.0), levels)
    assert list(out["value"]) == pytest.approx(list(want))


@pytest.mark.slow
def test_hub_tables_cover_country_provinces_and_zones(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    tmp_path: Path,
) -> None:
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bdbv_bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(
            ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
        ),
        history=real,
        daily=daily,
    )[0]
    forecasts = bt.run_forecasts([get_model("randomwalk")], [task])
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    daily_zones = load_daily_zones(DATA)
    # The walk's zones are its stored zone run's (`store.stored_zones`).
    stored = zone_forecast(
        get_model("randomwalk"), task, weekly_counts(daily_zones, latest),
        zones, daily_zones,
    ).assign(model="randomwalk", vintage=latest)  # fmt: skip
    tables = hub_tables(forecasts, task, daily_zones, zones, stored)
    table = tables["randomwalk"]
    locations = set(table["location"])
    assert NATIONAL in locations
    assert locations & set(zones)
    assert set(table["output_type_id"]) == set(range(1, 101))
    # National is at least the province sum, sample by sample.
    keys = ["target_end_date", "output_type_id"]
    provinces = (
        table.loc[table["location"].isin(set(task.locations))]
        .groupby(keys)["value"]
        .sum()
    )
    national = table.loc[table["location"] == NATIONAL].set_index(keys)
    assert (national["value"] >= provinces.loc[national.index]).all()
    written = write_hub_tables(tables, tmp_path)
    assert sorted(p.name for p in written) == [
        "randomwalk-quantiles.csv.gz",
        "randomwalk-samples.csv.gz",
    ]


def test_hub_tables_take_a_zone_fit_s_own_zones(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bdbv_bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(
            ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
        ),
        history=real,
        daily=daily,
    )[0]
    forecasts = bt.run_forecasts([get_model("randomwalk")], [task])
    forecasts = forecasts.assign(model="renewal")
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    zone = next(z for z in zones if z.startswith(task.locations[0] + "-"))
    # The zone fit puts all of the first province's cases in one zone.
    first = forecasts.loc[
        (forecasts["location"] == task.locations[0]) & forecasts["as_of"].isna()
        if "as_of" in forecasts
        else forecasts["location"] == task.locations[0]
    ]
    fitted = first.assign(location=zone, vintage=task.vintage)
    daily_zones = load_daily_zones(DATA)
    table = hub_tables(forecasts, task, daily_zones, zones, fitted)["renewal"]
    rows = table.loc[table["location"].isin(set(zones))]
    assert set(rows["location"]) == {zone}
    mine = table.loc[table["location"] == task.locations[0]]
    keys = ["target_end_date", "output_type_id"]
    pd.testing.assert_series_equal(
        rows.set_index(keys)["value"].sort_index(),
        mine.set_index(keys)["value"].sort_index(),
        check_names=False,
    )
    # Without its zone fit, a model's zones are those in its own forecast;
    # these province-only samples have none.
    split = hub_tables(forecasts, task, daily_zones, zones)["renewal"]
    assert not set(split["location"]) & set(zones)


@pytest.mark.slow
def test_a_models_own_zones_stay_apart_from_its_provinces(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bdbv_bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(
            ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
        ),
        history=real,
        daily=daily,
    )[0]
    walk = bt.run_forecasts([get_model("randomwalk")], [task])
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    mine = tuple(z for z in zones if province_of(z) in task.locations)
    provinces = walk.loc[walk["location"].isin(set(task.locations))]
    # A model with zone samples of its own that sum to its provinces:
    # each province's first zone holds it and the others are zero.
    daily_zones = load_daily_zones(DATA)
    first: dict[str, str] = {}
    for zone in mine:
        first.setdefault(province_of(zone), zone)
    final = provinces
    if "as_of" in final:
        final = final.loc[final["as_of"].isna()]
    rows = []
    for zone in mine:
        prov = final.loc[final["location"] == province_of(zone)]
        keep = first[province_of(zone)] == zone
        rows.append(prov.assign(location=zone, value=prov["value"] * keep))
    split = pd.concat(rows, ignore_index=True)
    own = pd.concat([walk, split]).assign(model="own")
    tables = hub_tables(pd.concat([walk, own]), task, daily_zones, zones)
    keys = ["location", "target_end_date", "output_type_id"]

    def places(name: str) -> pd.DataFrame:
        table = tables[name]
        rows = table.loc[~table["location"].map(is_zone)]
        return rows.sort_values(keys).reset_index(drop=True)

    # The country and provinces are as without the zones.
    pd.testing.assert_frame_equal(places("own"), places("randomwalk"))
    # The zones are the model's own, as they are.
    table = tables["own"]
    rows = table.loc[table["location"].map(is_zone)]
    got = rows.set_index(["location", "target_end_date", "output_type_id"])
    want = split.set_index(["location", "target_end_date", "sample"])
    assert np.array_equal(
        got["value"].sort_index().to_numpy(),
        want["value"].sort_index().to_numpy(),
    )


def test_a_model_that_forecasts_the_country_keeps_its_national_rows(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Chronos-2 models forecast the country in their joint block, and
    # their national draws are written as they are, not as the sum of the
    # provinces. Other models' national rows are the provinces' sum.
    from bdbvhub.reporting import hubtables
    from nowcastutils.base import SAMPLE_COLUMNS

    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bdbv_bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(
            ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
        ),
        history=real,
        daily=daily,
    )[0]
    walk = bt.run_forecasts([get_model("randomwalk")], [task])
    national = walk["location"] == NATIONAL
    own = walk.assign(
        model="foundation",
        value=walk["value"].where(~national, 7.0),
    )
    monkeypatch.setattr(
        hubtables,
        "zones_for_model",
        lambda *_a, **_k: pd.DataFrame(columns=list(SAMPLE_COLUMNS)),
    )
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    tables = hub_tables(
        pd.concat([walk, own]), task, load_daily_zones(DATA), zones
    )
    chronos = tables["foundation"]
    assert (chronos.loc[chronos["location"] == NATIONAL, "value"] == 7).all()
    rw = tables["randomwalk"]
    wide = rw.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    provinces = [c for c in wide.columns if c.count("-") == 1]
    assert (wide[NATIONAL] == wide[provinces].sum(axis=1)).all()
