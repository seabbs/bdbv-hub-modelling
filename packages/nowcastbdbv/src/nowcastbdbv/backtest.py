# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""The BDBV backtest's tasks, from the hub's releases.

`make_tasks` builds a forecast task per release from the hub data, with
the hub's imputation delay and the health zones; `vintage_task` builds
one release's task, `development_tasks` and `lopo_tasks` the development
and leave-one-province-out runs, and `run_development` and `run_lopo` run
them with `nowcasteval.backtest`'s engine.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from nowcastbdbv.data.delays import hub_delay_pmf
from nowcastbdbv.data.tidy import delays_for_vintage
from nowcastbdbv.data.zones import ZoneData, hub_places
from nowcasteval.backtest import (
    BacktestConfig,
    backtest_origins,
    forecast_locations,
    forecast_weeks,
    real_vintages,
    run_forecasts,
    score_tasks,
)
from nowcasteval.nowcast import truth_from_vintage
from nowcastutils.base import ForecastModel, ForecastTask
from nowcastutils.dates import reference_date_for_vintage
from nowcastutils.releases import (
    release_cutoff,
    without_cumulative_first_release,
)

log = logging.getLogger(__name__)

__all__ = [
    "TASK_MIN_HORIZON",
    "development_tasks",
    "lopo_tasks",
    "make_tasks",
    "run_development",
    "run_lopo",
    "vintage_task",
]


def _imputation_delay(vintage: str) -> np.ndarray:
    """Return the hub's imputation delay at a release, read-only.

    Tasks share it, so none may change it in place.
    """
    pmf = hub_delay_pmf(vintage)
    pmf.setflags(write=False)
    return pmf


def make_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    config: BacktestConfig,
    report_by: Sequence[str] = (),
    history: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    zones: ZoneData | None = None,
    places: pd.DataFrame | None = None,
) -> list[ForecastTask]:
    """Build one task per vintage in a target table.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows, possibly of several kinds and vintages.
    delays : pandas.DataFrame or None
        Delay line data with a `vintage` column, and optionally a `kind`
        column. A task gets the rows of its own vintage (and kind), or
        None if there are none.
    config : BacktestConfig
        Horizons, samples and seed.
    report_by : sequence of str
        Truth vintage dates. Each real task asks for the counts as
        reported by those after its own vintage, at their cut-offs, which
        `history` (or `target`) holds.
    history : pandas.DataFrame, optional
        Target rows to take each task's `target_history` from. Defaults
        to `target`.
    daily : pandas.DataFrame, optional
        Daily onsets by release, with `vintage`, `location`, `onset_date`,
        `observed` and `imputed` columns, and optionally `kind`. Each task
        gets the rows of its kind on or before its vintage as
        `daily_history`.
    zones : ZoneData, optional
        The health-zone tables. A real task gets the weekly and daily rows
        on or before its vintage as `zone_history` and
        `zone_daily_history`, and the codes as `zone_codes`.
    places : pandas.DataFrame, optional
        Every task's places (`ForecastTask.places`). Defaults to
        `data.zones.hub_places()` from the default data directory.

    Returns
    -------
    list of ForecastTask
        Tasks ordered by kind and vintage.

    Examples
    --------
    >>> import dataclasses
    >>> from nowcastbdbv.data.tidy import (
    ...     delays_for_vintage, load_daily, load_delays, load_target
    ... )
    >>> from nowcasteval import backtest as bt
    >>> target = load_target()
    >>> real = target.loc[target["kind"] == "real"]
    >>> config = dataclasses.replace(
    ...     bt.load_config("packages/nowcastbdbv/config/backtest.toml"),
    ...     min_horizon=-6,
    ... )
    >>> vintage = "2026-10-17"
    >>> task = bt.make_tasks(
    ...     real.loc[real["vintage"] == vintage],
    ...     delays_for_vintage(load_delays(), vintage)[0],
    ...     config,
    ...     history=real,
    ...     daily=load_daily(),
    ... )[0]
    >>> task.reference_date, task.target_end_dates[-1]
    """
    tasks = []
    source = target if history is None else history
    if places is None:
        places = hub_places()
    imputation = functools.cache(_imputation_delay)
    for (kind, vintage), rows in target.groupby(["kind", "vintage"]):
        mine = source["kind"] == kind
        past_target = source.loc[mine & (source["vintage"] <= vintage)]
        ref = reference_date_for_vintage(str(vintage))
        dly, history = None, None
        if delays is not None:
            mask = delays["vintage"] == vintage
            if "kind" in delays.columns:
                mask &= delays["kind"] == kind
            sel = delays.loc[mask]
            dly = sel if not sel.empty else None
            same = delays["vintage"] <= vintage
            if "kind" in delays.columns:
                same &= delays["kind"] == kind
            past = delays.loc[same]
            history = past if not past.empty else None
        task = ForecastTask(
            vintage=str(vintage),
            reference_date=ref,
            kind=str(kind),
            cutoff=release_cutoff(rows, str(vintage)).isoformat(),
            target=rows.reset_index(drop=True),
            delays=dly,
            delay_history=history,
            target_history=past_target if not past_target.empty else None,
            daily_history=_daily_for(daily, str(kind), str(vintage)),
            **_zone_fields(zones, str(kind), str(vintage)),
            imputation_delay=imputation(str(vintage)),
            places=places,
            locations=forecast_locations(rows),
            target_end_dates=forecast_weeks(ref, config),
            n_samples=config.n_samples,
            seed=config.seed,
            max_delay=config.max_delay,
            report_by={
                d: release_cutoff(source.loc[mine], d).isoformat()
                for d in sorted(report_by)
                if kind != "simulated" and d > str(vintage)
            },
        )
        # A cumulative first release is not part of the reporting.
        tasks.append(without_cumulative_first_release(task))
    return tasks


TASK_MIN_HORIZON = -6
"""First week of a vintage's task, in weeks from the reference date: 7
weeks up to it, which the submission CI's `observed` and `coverage`
checks need. The backtest scores from `[forecast] min_horizon`."""


def vintage_task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    vintage: str,
    daily: pd.DataFrame | None = None,
    extra_weeks: int = 0,
    *,
    zones: ZoneData | None = None,
    places: pd.DataFrame | None = None,
) -> tuple[ForecastTask, str]:
    """Build the task of one real vintage, as the submission fits it.

    The fit store fits the same task (`bdbvhub.store.store.origin_tasks`),
    so the submission and the store give a fit the same key. The weeks run from
    `TASK_MIN_HORIZON`. A vintage with no delays of its own uses the
    latest earlier ones (`delays_for_vintage`).

    Parameters
    ----------
    target, delays : pandas.DataFrame
        The tidy target and delay tables.
    config : BacktestConfig
        Horizons, samples and seed (`[forecast]`).
    vintage : str
        A real vintage.
    daily : pandas.DataFrame, optional
        Daily onsets by release.
    extra_weeks : int
        Weeks to forecast past `[forecast] max_horizon`. The store adds
        one, so its fit also covers the next round's weeks.
    zones : ZoneData, optional
        The health-zone tables (`make_tasks`).
    places : pandas.DataFrame, optional
        The task's places (`make_tasks`).

    Returns
    -------
    tuple of ForecastTask and str
        The task, and a note of the delays that stood in (empty if none).
    """
    config = replace(
        config,
        min_horizon=TASK_MIN_HORIZON,
        max_horizon=config.max_horizon + extra_weeks,
    )
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"] == vintage]
    used, note = delays_for_vintage(delays, vintage)
    task = make_tasks(
        rows,
        used,
        config,
        history=real,
        daily=daily,
        zones=zones,
        places=places,
    )[0]
    return task, note


def _zone_fields(
    zones: ZoneData | None, kind: str, vintage: str
) -> dict[str, Any]:
    """Return a task's zone fields: the releases on or before `vintage`."""
    if zones is None:
        return {}
    return {
        "zone_history": _daily_for(zones.weekly, kind, vintage),
        "zone_daily_history": _daily_for(zones.daily, kind, vintage),
        "zone_codes": zones.codes,
    }


def _daily_for(
    daily: pd.DataFrame | None, kind: str, vintage: str
) -> pd.DataFrame | None:
    """Return the daily releases of a kind on or before a vintage, or None."""
    if daily is None:
        return None
    rows = daily
    if "kind" in rows.columns:
        rows = rows.loc[rows["kind"] == kind]
    elif kind != "real":
        return None
    rows = rows.loc[rows["vintage"] <= vintage]
    return rows.reset_index(drop=True) if not rows.empty else None


def development_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    daily: pd.DataFrame | None = None,
    zones: ZoneData | None = None,
) -> list[ForecastTask]:
    """Build the development tasks: every origin but the latest.

    The origins are `nowcasteval.backtest.backtest_origins`. Earlier
    vintages stay in the later tasks' histories.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    daily : pandas.DataFrame, optional
        Daily onsets by release, with `vintage`, `location`, `onset_date`,
        `observed` and `imputed` columns, and optionally `kind`. Each task
        gets the rows of its kind on or before its vintage as
        `daily_history`.
    zones : ZoneData, optional
        The health-zone tables (`make_tasks`).

    Returns
    -------
    list of ForecastTask
        Tasks for every origin before the latest real vintage.
    """
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    origins = set(backtest_origins(real, config))
    return [
        t
        for t in make_tasks(
            real,
            delays.assign(kind="real"),
            config,
            real_vintages(real),
            daily=daily,
            zones=zones,
        )
        if t.vintage < latest and t.vintage in origins
    ]


def run_development(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    fits: list[pd.DataFrame] | None = None,
    learnt: list[pd.DataFrame] | None = None,
    daily: pd.DataFrame | None = None,
    zones: ZoneData | None = None,
) -> pd.DataFrame:
    """Forecast from every real vintage before the latest.

    The forecasts are scored by
    `nowcasteval.nowcast.release_backtest`.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to compare.
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.
    learnt : list, optional
        Collects learnt delays, as for `run_forecasts`.
    daily : pandas.DataFrame, optional
        Daily onsets by release, as for `development_tasks`.
    zones : ZoneData, optional
        The health-zone tables, as for `development_tasks`.

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts`.
    """
    tasks = development_tasks(target, delays, config, daily=daily, zones=zones)
    return run_forecasts(models, tasks, fits, learnt)


def lopo_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    provinces: Sequence[str] | None = None,
) -> list[ForecastTask]:
    """Build the development tasks with one province masked in each.

    Parameters
    ----------
    target, delays, config
        As for `development_tasks`.
    provinces : sequence of str, optional
        Provinces to mask. Defaults to every province with cases in each
        task.

    Returns
    -------
    list of ForecastTask
        Each with `mask` set to one province.
    """
    return [
        replace(task, mask=(prov,))
        for task in development_tasks(target, delays, config)
        for prov in task.locations
        if provinces is None or prov in provinces
    ]


def run_lopo(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    provinces: Sequence[str] | None = None,
    fits: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one province out: forecast it with its counts masked.

    For each development task and each province with cases, the model
    gets the task with that province in `ForecastTask.mask`
    (`lopo_tasks`).
    Only the masked province's forecast is kept and scored, against the
    latest real vintage.
    Models without `supports_mask = True` cannot honour a mask and are
    skipped.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to check.
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    provinces : sequence of str, optional
        Provinces to mask, one at a time. Defaults to every province with
        cases in each task.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts, with a `masked` column, and the scores. Both empty
        if no model supports a mask.
    """
    able = [m for m in models if getattr(m, "supports_mask", False)]
    for m in models:
        if m not in able:
            log.info("Skipping %s in leave-one-province-out: no mask.", m.name)
    if not able:
        return pd.DataFrame(), pd.DataFrame()
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    frames = []
    for task in lopo_tasks(target, delays, config, provinces=provinces):
        prov = task.mask[0]
        samples = run_forecasts(able, [task], fits)
        kept = samples.loc[samples["location"] == prov].copy()
        kept.insert(4, "masked", prov)
        frames.append(kept)
    forecasts = pd.concat(frames, ignore_index=True)
    scores = score_tasks(
        forecasts,
        truth_from_vintage(real, latest),
        latest,
        config.complete_after_weeks,
    )
    return forecasts, scores
