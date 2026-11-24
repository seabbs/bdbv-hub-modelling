# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Zone backtest: models that read release snapshots, run at zone level.

Each real vintage before the truth vintage is forecast on each zone's
own release snapshots (`zone_task`), for every zone with cases at the
vintage.
The functions and the CLI's `zone-backtest` run the two random walks.
The zone delay, observation model and walk spread are each the zone's own
where it has at least 10 cases a week, else its province's, else
national (`pooling`).
Each model is scored on its zone samples as it runs, the form submitted.
Zones are scored on the zone count each later real vintage reports.
`randomwalk` is scored on its samples as they are, with no delay.
`randomwalknowcast`'s final-count samples are drawn as for provinces
(`nowcasteval.nowcast.draw_reported`) under its own zone
delay at the origin.
The energy score takes every zone and week of an origin and release as
one vector, from the same drawn samples.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from nowcastbaselines.zones import zone_draws, zone_task
from nowcastbdbv.backtest import make_tasks
from nowcasteval.backtest import BacktestConfig, real_vintages
from nowcasteval.nowcast import (
    ForecastParts,
    complete_weeks,
    model_origin_rows,
    reported_samples,
    stack_parts,
)
from nowcasteval.scoring import energy_scores, relative_skill, score_forecasts
from nowcastutils.base import ForecastTask, ModelUnavailableError, get_model
from nowcastutils.snapshots import weekly_counts

log = logging.getLogger(__name__)

__all__ = [
    "ZONE_BASELINE",
    "ZoneForecasts",
    "score_zones",
    "zone_backtest",
    "zone_forecasts",
    "zone_relative_skill",
]

ZONE_BASELINE = "randomwalknowcast"
"""Name of the zone baseline in the scores."""

_KEYS = (
    "model",
    "origin",
    "target",
    "truth_vintage",
    "location",
    "target_end_date",
)
"""Columns that identify one zone forecast scored against one release."""


def _frame(
    samples: pd.DataFrame, task: ForecastTask, name: str
) -> pd.DataFrame:
    """Add the columns `score_tasks` matches on."""
    out = samples.copy()
    ref = pd.Timestamp(task.reference_date)
    out["horizon"] = (pd.to_datetime(out["target_end_date"]) - ref).dt.days // 7
    out.insert(0, "model", name)
    out.insert(1, "kind", task.kind)
    out.insert(2, "vintage", task.vintage)
    out.insert(3, "reference_date", task.reference_date.isoformat())
    return out


def zone_backtest(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    models: Sequence[str] = ("randomwalk", "randomwalknowcast"),
    config: BacktestConfig | None = None,
    origins: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score models run at zone level on later releases.

    Two targets are scored, as for the provinces (`release_backtest`): what
    each later release reported, and the complete weeks of the latest
    release, decided by `complete_weeks` from the zone counts of every
    release, a week with no onset counting as 0.

    Parameters
    ----------
    target : pandas.DataFrame
        The province target table (`load_target`).
    delays : pandas.DataFrame or None
        The delay table (`load_delays`).
    daily : pandas.DataFrame
        The daily table of the country and provinces (`load_daily`).
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    zones : tuple of str
        The hub's zone codes (`hub_zones`). Those with cases at a vintage
        are forecast from it.
    models : sequence of str
        Registered models that read release snapshots.
    config : BacktestConfig, optional
        Horizons, samples and seed. Defaults to `BacktestConfig()`.
    origins : collection of str, optional
        Forecast only from these vintages, so blocks of origins can run
        apart and their outputs be stacked. Every release is still read
        for the truth and the complete weeks. All but the latest if None.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores, `score_forecasts` output with `model`, `origin`,
        `target` (`"reported"` or `"complete"`), `truth_vintage` and
        `horizon` in weeks
        from the reference date, one row per model, origin, later
        vintage, zone and week.
        Then `energy_scores` of every model's zones by `origin`, `target`
        and `truth_vintage`, from the same drawn samples. Then the zone
        forecasts, as submitted, under each model's name: `model`,
        `origin`, `reference_date`, `location`, `target_end_date`,
        `sample` and `value`.
    """
    run = zone_forecasts(
        target, delays, daily, daily_zones, zones, models, config, origins
    )
    return score_zones(target, delays, daily, daily_zones, run, config)


@dataclass(frozen=True)
class ZoneForecasts(ForecastParts):
    """The forecasts `zone_backtest` scores, made apart from scoring.

    Every table has `origin`, so the forecasts of models run apart can be
    stacked (`read`) and scored together (`score_zones`).

    Attributes
    ----------
    runs : pandas.DataFrame
        One row per origin and model that forecast there, in the order
        run: `origin` and `model`.
    samples : pandas.DataFrame
        Zone samples under `key`, the model's name: `key`, `origin`,
        `location`, `target_end_date`, `sample` and `value`.
    delays : pandas.DataFrame
        Each model's delay summary at each origin, if it has one.
    """

    runs: pd.DataFrame
    samples: pd.DataFrame
    delays: pd.DataFrame

    _NAMES = ("runs", "samples", "delays")
    _FILE = "zone-forecast-{}.csv.gz"


def _zone_task(
    real: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    config: BacktestConfig,
    vintage: str,
) -> ForecastTask:
    """Return the province task at an origin."""
    return make_tasks(
        real.loc[real["vintage"] == vintage], delays, config, daily=daily
    )[0]


def zone_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    models: Sequence[str] = ("randomwalk", "randomwalknowcast"),
    config: BacktestConfig | None = None,
    origins: Collection[str] | None = None,
) -> ZoneForecasts:
    """Forecast the zones from every origin, without scoring.

    The forecasting half of `zone_backtest`.

    Parameters
    ----------
    target, delays, daily, daily_zones, zones, models, config, origins
        As for `zone_backtest`.

    Returns
    -------
    ZoneForecasts
        The zone samples and the delay summaries.
    """
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    runs: list[dict[str, str]] = []
    samples, summaries = [], []
    for vintage in vintages[:-1]:
        if origins is not None and vintage not in origins:
            continue
        task = _zone_task(real, delays, daily, config, vintage)
        counts = weekly_counts(daily_zones, vintage)
        active = tuple(z for z in zones if z in set(counts["location"]))
        zt = zone_task(task, counts, active, pd.concat([daily, daily_zones]))
        for name in models:
            model = get_model(name)
            try:
                direct = zone_draws(model.forecast(zt))
                summary = getattr(model, "delay_summary", None)
            except ModelUnavailableError as err:
                log.warning("%s has no zone forecast at %s: %s", name,
                            vintage, err)  # fmt: skip
                continue
            runs.append({"origin": vintage, "model": name})
            samples.append(direct.assign(key=name, origin=vintage))
            if summary is not None:
                summaries.append(summary.assign(origin=vintage))

    return ZoneForecasts(
        pd.DataFrame(runs, columns=["origin", "model"]),
        stack_parts(samples),
        stack_parts(summaries),
    )


def score_zones(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    run: ZoneForecasts,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the forecasts of `zone_forecasts`.

    The scoring half of `zone_backtest`, which it equals given the same
    models and origins.

    Parameters
    ----------
    target, delays, daily, daily_zones, config
        As for `zone_backtest`.
    run : ZoneForecasts
        The forecasts, of models run together or stacked.

    Returns
    -------
    tuple of pandas.DataFrame
        As `zone_backtest`.
    """
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    latest = vintages[-1]
    done = _complete_zone_weeks(daily_zones, daily, vintages)
    latest_counts = weekly_counts(daily_zones, latest)
    runs = run.runs.astype({"origin": str})
    frames, energy, kept = [], [], []
    for vintage in vintages[:-1]:
        here = runs.loc[runs["origin"] == vintage, "model"].astype(str)
        if here.empty:
            continue
        task = _zone_task(real, delays, daily, config, vintage)
        counts = weekly_counts(daily_zones, vintage)
        forecasts: dict[str, pd.DataFrame] = {}
        summaries: dict[str, pd.DataFrame | None] = {}
        for name in here:
            mine = (run.samples["key"] == name) & (
                run.samples["origin"].astype(str) == vintage
            )
            forecasts[name] = (
                run.samples.loc[mine]
                .drop(columns=["key", "origin"])
                .reset_index(drop=True)
            )
            summaries[name] = model_origin_rows(run.delays, name, vintage)
            kept.append(
                forecasts[name].assign(
                    model=name,
                    origin=vintage,
                    reference_date=task.reference_date.isoformat(),
                )
            )
        if not forecasts:
            continue
        reported = counts.rename(columns={"total": "reported"})[
            ["location", "target_end_date", "reported"]
        ]
        # Each model draws from its own stream, as in release_backtest.
        rngs = {name: task.rng(name) for name in forecasts}
        later = [v for v in vintages if v > vintage]
        by_model = {
            key: reported_samples(
                samples,
                summaries[key],
                key,
                vintage,
                later,
                daily,
                reported,
                rngs[key],
            )
            for key, samples in forecasts.items()
        }
        for o2 in later:
            drawn = [
                draws[o2].assign(
                    model=key,
                    origin=vintage,
                    target="reported",
                    truth_vintage=o2,
                )
                for key, draws in by_model.items()
                if o2 in draws
            ]
            if not drawn:
                continue
            both = pd.concat(drawn, ignore_index=True)
            truth = _zone_truth(weekly_counts(daily_zones, o2), both, o2)
            both["horizon"] = _weeks(both, task)
            frames.append(score_forecasts(both, truth, keys=_KEYS))
            energy.append(energy_scores(both, truth, unit=_KEYS[1:4]))
        # The final-count samples on the latest release's complete weeks.
        final = pd.concat(
            [
                samples.assign(
                    model=key,
                    origin=vintage,
                    target="complete",
                    truth_vintage=latest,
                )
                for key, samples in forecasts.items()
            ],
            ignore_index=True,
        )
        truth = _zone_truth(latest_counts, final, latest).merge(done)
        final["horizon"] = _weeks(final, task)
        frames.append(score_forecasts(final, truth, keys=_KEYS))
        energy.append(energy_scores(final, truth, unit=_KEYS[1:4]))
    if not frames:
        # No origin forecast, as in a block of origins every model skips.
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(energy, ignore_index=True),
        pd.concat(kept, ignore_index=True)[
            [
                "model",
                "origin",
                "reference_date",
                "location",
                "target_end_date",
                "sample",
                "value",
            ]
        ],
    )


def _weeks(samples: pd.DataFrame, task: ForecastTask) -> pd.Series:
    """Weeks from the task's reference date to each sample's week."""
    ref = pd.Timestamp(task.reference_date)
    return (pd.to_datetime(samples["target_end_date"]) - ref).dt.days // 7


def zone_relative_skill(
    scores: pd.DataFrame, by: Sequence[str] = ()
) -> pd.DataFrame:
    """Return each model's mean CRPS and log CRPS relative to zone randomwalk.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `zone_backtest`.
    by : sequence of str
        Extra columns to compare within, such as `("horizon",)`.

    Returns
    -------
    pandas.DataFrame
        See `nowcasteval.scoring.relative_skill`.
    """
    return relative_skill(
        scores,
        baseline=ZONE_BASELINE,
        by=by,
        metrics=("crps", "log_crps"),
        match_on=_KEYS[1:],
    )


def _complete_zone_weeks(
    daily_zones: pd.DataFrame, daily: pd.DataFrame, vintages: Sequence[str]
) -> pd.DataFrame:
    """Return the zones and weeks the latest release reports in full.

    `complete_weeks` on the zone counts of every release, over every zone
    and week with an onset in any release, a missing week counting as 0.
    """
    counts = {v: weekly_counts(daily_zones, v) for v in vintages}
    every = pd.concat(counts.values())
    grid = pd.MultiIndex.from_product(
        [sorted(set(every["location"])), sorted(set(every["target_end_date"]))],
        names=["location", "target_end_date"],
    )
    target = pd.concat(
        [
            c.set_index(["location", "target_end_date"])["total"]
            .reindex(grid, fill_value=0)
            .reset_index()
            .assign(kind="real", vintage=v)
            for v, c in counts.items()
        ],
        ignore_index=True,
    )
    weeks = complete_weeks(target, daily, vintages[-1])
    return weeks.loc[weeks["complete"], ["location", "target_end_date"]]


def _zone_truth(
    counts: pd.DataFrame, forecasts: pd.DataFrame, truth_date: str
) -> pd.DataFrame:
    """Zone counts as a truth table over the forecast zones and weeks."""
    weeks = sorted(
        w for w in forecasts["target_end_date"].unique() if w <= truth_date
    )
    grid = pd.MultiIndex.from_product(
        [sorted(forecasts["location"].unique()), weeks],
        names=["location", "target_end_date"],
    )
    return (
        counts.set_index(["location", "target_end_date"])["total"]
        .reindex(grid, fill_value=0)
        .rename("truth")
        .reset_index()
    )
