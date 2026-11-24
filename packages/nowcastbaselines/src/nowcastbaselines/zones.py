# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Health-zone forecasts.

The random walk models run at zone level, on the zones' own release
snapshots (`zone_task`), with the zone, then province, then national
pooling of `pooling`.
Their zone samples are the zone model's own draws, and need not sum to
the province samples (`zone_forecast`).

The renewal models fit the zones themselves and sum them to the
provinces, and Chronos-2 forecasts the zones as series of its joint
block beside the provinces and the country.
Both models' zone samples are used as they are (`zones_for_model`).
Chronos-2's need not sum to its province samples
(`submission.hub.zones_need_not_sum`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from nowcastbaselines.randomwalk import RandomWalk
from nowcastutils.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
    get_model,
    list_models,
)
from nowcastutils.dates import as_date
from nowcastutils.hierarchy import is_zone, province_of

__all__ = [
    "check_zone_coherence",
    "walks_own_zones",
    "zone_draws",
    "zone_forecast",
    "zone_task",
    "zones_for_model",
]


def _final(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the final-count rows, without `as_of`."""
    if "as_of" not in samples:
        return samples
    return samples.loc[samples["as_of"].isna()].drop(columns="as_of")


def zone_draws(samples: pd.DataFrame) -> pd.DataFrame:
    """Return a zone task's final-count zone rows, `SAMPLE_COLUMNS` only."""
    final = _final(samples)
    final = final.loc[final["location"] != NATIONAL]
    return final[list(SAMPLE_COLUMNS)].reset_index(drop=True)


def walks_own_zones(name: str | None) -> bool:
    """Return whether model `name` forecasts each zone from its own data.

    Only the random walks without a fit do (`zone_forecast`); their zones
    need not sum to their provinces.
    """
    specs = {s.name: s for s in list_models()}
    if name not in specs or specs[name].fitted:
        return False
    return isinstance(get_model(name), RandomWalk)


def zone_forecast(
    model: ForecastModel,
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
) -> pd.DataFrame:
    """Run a snapshot model at zone level.

    Each zone is forecast from its own release snapshots, with the
    model's own-or-pooled estimates and introductions. No province sample
    is split over the zones, so the zones need not sum to the provinces.

    Parameters
    ----------
    model : ForecastModel
        `randomwalk` or `randomwalknowcast`, any model that reads
        `daily_history`.
    task : ForecastTask
        The province task, with `daily_history` set.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage.
    zones : tuple of str
        Zone codes. Those of the task's provinces are forecast.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).

    Returns
    -------
    pandas.DataFrame
        Columns `SAMPLE_COLUMNS`, one row per zone, week and sample.

    Raises
    ------
    ModelUnavailableError
        As the model, if it cannot forecast the zone task.
    """
    history = task.daily_history
    daily = (
        daily_zones if history is None else pd.concat([history, daily_zones])
    )
    zt = zone_task(task, counts, zones, daily)
    return zone_draws(model.forecast(zt))


def check_zone_coherence(
    provinces: pd.DataFrame, zones: pd.DataFrame
) -> list[str]:
    """Check that zone samples sum to their province samples.

    Parameters
    ----------
    provinces : pandas.DataFrame
        Province samples (`SAMPLE_COLUMNS`). Values are rounded and
        clipped at zero first, as in the hub file.
    zones : pandas.DataFrame
        Zone samples (`SAMPLE_COLUMNS`).

    Returns
    -------
    list of str
        One message per province and week whose zones do not sum to it
        in every sample. Empty when coherent.
    """
    keys = ["location", "target_end_date", "sample"]
    summed = (
        zones.assign(location=zones["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    prov = _final(provinces)
    want = (
        prov.loc[prov["location"].isin(summed.index.unique("location"))]
        .set_index(keys)["value"]
        .clip(lower=0)
        .round()
    )
    gap = summed.reindex(want.index, fill_value=np.nan) - want
    gap = gap.reindex(want.index.union(summed.index))
    bad = gap.loc[gap.isna() | (gap != 0)]
    return [
        f"{loc}, {week}: zone samples do not sum to the province samples"
        for loc, week in sorted({(i[0], i[1]) for i in bad.index})
    ]


def zone_task(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily: pd.DataFrame | None = None,
) -> ForecastTask:
    """Return the task with zones in place of provinces.

    This is how the random walk models run at zone level. Each zone gets
    a weekly series from the first week with zone data to the vintage,
    with zeros filled in.

    Parameters
    ----------
    task : ForecastTask
        A province task.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage.
    zones : tuple of str
        Zone codes. Those of the task's provinces are kept.
    daily : pandas.DataFrame, optional
        Daily onsets by release of the country, provinces and zones.
        The task keeps the rows of its releases for the country, its
        provinces and its zones as `daily_history`. Without it the task
        has none, so the random walk models cannot run.

    Returns
    -------
    ForecastTask
        The same weeks, vintage, delays and seed, with the zones as its
        locations and their counts as its target.
    """
    mine = tuple(z for z in zones if province_of(z) in task.locations)
    first = pd.Timestamp(counts["target_end_date"].min())
    weeks = pd.date_range(first, pd.Timestamp(as_date(task.vintage)), freq="7D")
    grid = pd.MultiIndex.from_product(
        [mine, weeks.date.astype(str)], names=["location", "target_end_date"]
    )
    target = (
        counts.set_index(["location", "target_end_date"])["total"]
        .reindex(grid, fill_value=0)
        .reset_index()
        .assign(vintage=task.vintage)
    )
    history = None
    if daily is not None:
        keep = {NATIONAL, *task.locations, *mine}
        history = daily.loc[
            daily["location"].isin(keep) & (daily["vintage"] <= task.vintage)
        ].reset_index(drop=True)
    return dataclasses.replace(
        task,
        target=target,
        locations=mine,
        mask=(),
        report_by={},
        daily_history=history,
    )


def zones_for_model(
    name: str | None,
    task: ForecastTask,
    samples: pd.DataFrame,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
) -> pd.DataFrame:
    """Return a model's health-zone samples.

    A model's own zone rows in `samples` are used as they are: the
    renewal models fit them and Chronos-2 forecasts them. A random walk
    model with no fit is run on the zones (`zone_forecast`). Other
    samples have no zones.

    Parameters
    ----------
    name : str or None
        The registered model, None for a fit configuration with none.
    task : ForecastTask
        The province task.
    samples : pandas.DataFrame
        The model's samples for `task`: provinces, and zones if it
        forecasts them.
    counts : pandas.DataFrame
        Weekly zone counts at the task's vintage (`weekly_counts`).
    zones : tuple of str
        The hub's zone codes.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).

    Returns
    -------
    pandas.DataFrame
        Zone samples with `SAMPLE_COLUMNS`.
    """
    own = samples["location"].map(is_zone)
    if own.any():
        return samples.loc[own, list(SAMPLE_COLUMNS)].reset_index(drop=True)
    if walks_own_zones(name):
        model = get_model(str(name))
        return zone_forecast(model, task, counts, zones, daily_zones)
    return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
