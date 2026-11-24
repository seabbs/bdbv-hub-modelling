"""The BDBV foundation-model check: Chronos-2 on every hub release, scored.

Each model forecasts every real release of the hub data that has a later
one (`nowcasteval.nowcast.forecast_releases`), alone, into its own folder,
so the reference models are made once and a new fine-tuned checkpoint
reruns only itself. The Chronos-2 models, zero-shot and fine-tuned alike,
build the fine-tuned models' layout: one joint block of the country,
every province and the active zones, and one block per province of the
province and all its zones (`nowcastfoundation.forecaster.active_views`,
the rule in `config/finetune.toml`).

The report (`bdbvhub.reporting.ftcheck_report`) scores the folders
together (`score_releases`) and gives each model's pairwise relative
skill (`nowcasteval.scoring.pairwise_skill`) on the CRPS and the log
CRPS, scaled to the reference, overall and by weeks back, with each
model's forecasts and rounds; each model's calibration by level and
weeks back (`calibration`: medians against truth and against what was
already reported, 90% coverage) and the week-to-week roughness of its
medians against the truth's (`roughness`); the active zones per release;
and the forecasts with their 50% and 90% intervals. Only the hub's
weekly targets at its locations are scored.

- `release_tasks`: the scored tasks of every release.
- `check_model`: a model as the check runs it.
- `forecast`: one model's forecasts, written to a folder.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pandas as pd

from bdbvhub.models import get_model
from bdbvhub.store import store
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcastbdbv.data.zones import load_zone_data
from nowcasteval import backtest as bt
from nowcasteval.nowcast import (
    ReleaseForecasts,
    forecast_releases,
)
from nowcastfoundation.forecaster import (
    FINETUNED_ENV,
    FoundationModel,
    active_zone_rule,
)
from nowcastutils.base import ForecastModel, ForecastTask

__all__ = [
    "check_model",
    "forecast",
    "release_tasks",
]


def release_tasks(
    data: Path | str, config: Path | str = "config/backtest.toml"
) -> tuple[list[ForecastTask], pd.DataFrame, pd.DataFrame, Any]:
    """Return the scored task of every release, the target, daily, config.

    Parameters
    ----------
    data : pathlib.Path or str
        The tidy hub data (`bdbvhub data`).
    config : pathlib.Path or str
        The backtest config.

    Returns
    -------
    tuple
        The tasks, the target table, the daily onsets and the config.
    """
    conf = bt.load_config(config)
    target, delays, daily = (
        load_target(data),
        load_delays(data),
        load_daily(data),
    )
    tasks = store.origin_tasks(
        target, delays, conf, daily, load_zone_data(data)
    ).values()
    return bt.tasks_for_scoring(list(tasks), conf), target, daily, conf


def check_model(name: str, checkpoint: str | None = None) -> ForecastModel:
    """Return a registered model as the check runs it.

    Parameters
    ----------
    name : str
        A registered model.
    checkpoint : str, optional
        The fine-tuned checkpoint (a settings hash) an `_ft` model loads,
        in place of `config/finetune.toml`'s.

    Returns
    -------
    ForecastModel
        The model; a Chronos-2 model with the active-zone rule.
    """
    if checkpoint:
        os.environ[FINETUNED_ENV] = checkpoint
    model = get_model(name)
    if isinstance(model, FoundationModel):
        cases, weeks, recent = active_zone_rule()
        model = dataclasses.replace(
            model,
            active_zone_cases=cases,
            active_zone_weeks=weeks,
            active_zone_recent_weeks=recent,
        )
    return model


def forecast(
    name: str,
    data: Path | str,
    out: Path | str,
    checkpoint: str | None = None,
) -> ReleaseForecasts:
    """Forecast every release with one model and write its folder.

    Also writes `blocks.csv`, the block sizes of a Chronos-2 model at
    each release (`FoundationModel.block_sizes`).

    Parameters
    ----------
    name : str
        A registered model.
    data : pathlib.Path or str
        The tidy hub data.
    out : pathlib.Path or str
        The folder.
    checkpoint : str, optional
        See `check_model`.

    Returns
    -------
    ReleaseForecasts
        The forecasts.
    """
    tasks, target, _, _ = release_tasks(data)
    model = check_model(name, checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model, FoundationModel):
        sizes = [
            model.block_sizes(t).assign(origin=t.vintage)
            for t in tasks
            if t.kind == "real"
        ]
        pd.concat(sizes, ignore_index=True).to_csv(
            out / "blocks.csv", index=False
        )
    run = forecast_releases(tasks, target, [model])
    run.write(out)
    return run
