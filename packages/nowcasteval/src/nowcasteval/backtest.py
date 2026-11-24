# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Rolling-origin backtest over every real vintage.

Each vintage becomes a `ForecastTask`: the data as of that vintage, the
reference date it serves (`reference_date_for_vintage`) and the weeks
from `min_horizon` to `max_horizon` (the BDBV data's are built by
`nowcastbdbv.backtest.make_tasks`).
Every registered model forecasts every task.
Real forecasts are scored by `nowcasteval.nowcast` on the
count each later release reported.

Every real vintage but the latest is backtested.
`truth_changes` shows how much the truth moved between vintages.
"""

from __future__ import annotations

import datetime as dt
import logging
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcasteval.nowcast import TRUTH_LEVEL
from nowcasteval.scoring import energy_scores, score_forecasts
from nowcastutils.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
    MissingSnapshots,
    ModelUnavailableError,
    add_national,
    check_samples,
)
from nowcastutils.dates import as_date

__all__ = [
    "BacktestConfig",
    "check_distinct",
    "energy_tasks",
    "for_scoring",
    "forecast_locations",
    "forecast_weeks",
    "joint_order",
    "load_config",
    "merge_by_model",
    "real_vintages",
    "run_forecasts",
    "score_tasks",
    "tasks_for_scoring",
    "truth_changes",
]

log = logging.getLogger(__name__)

FORECAST_COLUMNS = (
    "model",
    "kind",
    "vintage",
    "reference_date",
    *SAMPLE_COLUMNS,
    "as_of",
    "horizon",
)
"""Columns of `run_forecasts` output, which is empty if no model could run."""

_FORECAST_DTYPES = {
    "sample": "int64",
    "value": "float64",
    "as_of": "float64",
    "horizon": "int64",
}


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings, usually read from `config/backtest.toml`.

    Attributes
    ----------
    min_horizon, max_horizon : int
        Range of horizons in each forecast, in weeks.
    n_samples : int
        Samples per forecast, `[forecast] n_samples`. The fit store, the
        submission and the hub comparison draw this many.
    score_samples : int
        Samples per forecast of the models the backtest runs itself,
        `[samples] backtest` (`for_scoring`).
    seed : int
        Random seed shared by all tasks.
    complete_after_weeks : int
        $K$: a truth week counts as complete if it ended at least this
        many weeks before the truth vintage.
    sampler : dict
        NUTS settings of the fitted models
        (`nowcastbdbv.backtest.apply_sampler`).
    skip_origins : tuple of str
        Real releases that are not backtest origins. They stay in the
        histories of later origins.
    truth_delay : int, optional
        $D$, in days: the release backtest scores each week's complete
        target on the first release whose cut-off is at least $D$ days
        after it. None estimates $D$ from the releases
        (`nowcasteval.nowcast.resolve_truth_delay`).
    truth_level : float
        The share of the national count reported by an estimated $D$.
    """

    min_horizon: int = -4
    max_horizon: int = 3
    n_samples: int = 100
    score_samples: int = 100
    seed: int = 0
    complete_after_weeks: int = 0
    sampler: dict[str, Any] = field(default_factory=dict)
    skip_origins: tuple[str, ...] = ()
    truth_delay: int | None = None
    truth_level: float = TRUTH_LEVEL


def for_scoring(config: BacktestConfig) -> BacktestConfig:
    """Return the config of the models run here, with `score_samples` draws.

    Only `n_samples` changes. The seed is kept, so every model draws from
    the shared seed (common random numbers).

    Parameters
    ----------
    config : BacktestConfig
        The backtest settings.

    Returns
    -------
    BacktestConfig
        `config` with `n_samples` set to `score_samples`.
    """
    return replace(config, n_samples=config.score_samples)


def tasks_for_scoring(
    tasks: Sequence[ForecastTask], config: BacktestConfig
) -> list[ForecastTask]:
    """Return copies of tasks with `score_samples` draws, as `for_scoring`.

    The fit store and the submission key their fits on the tasks they are
    given, so they keep the originals and only the models run here take
    these.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        Tasks, such as `bdbvhub.store.store.origin_tasks` gives.
    config : BacktestConfig
        Gives `score_samples`.

    Returns
    -------
    list of ForecastTask
        The tasks with `n_samples` set to `score_samples`.
    """
    return [replace(t, n_samples=config.score_samples) for t in tasks]


def load_config(path: Path | str = "config/backtest.toml") -> BacktestConfig:
    """Read the backtest settings.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file.

    Returns
    -------
    BacktestConfig
        The settings.
    """
    raw = tomllib.loads(Path(path).read_text())
    fc, sp = raw["forecast"], raw["split"]
    return BacktestConfig(
        min_horizon=int(fc["min_horizon"]),
        max_horizon=int(fc["max_horizon"]),
        n_samples=int(fc["n_samples"]),
        score_samples=int(
            raw.get("samples", {}).get("backtest", fc["n_samples"])
        ),
        seed=int(fc["seed"]),
        complete_after_weeks=int(sp["complete_after_weeks"]),
        sampler=dict(raw["sampler"]),
        skip_origins=tuple(str(d) for d in sp.get("skip_origins", ())),
        truth_delay=_truth_delay(sp.get("truth_delay", "auto")),
        truth_level=_truth_level(sp.get("truth_level", TRUTH_LEVEL)),
    )


def _truth_level(value: object) -> float:
    """Read `truth_level`: a share above 0 and at most 1."""
    if isinstance(value, int | float) and 0 < value <= 1:
        return float(value)
    raise ValueError(f"truth_level must be in (0, 1], not {value!r}.")


def _truth_delay(value: object) -> int | None:
    """Read `truth_delay`: a number of days, or "auto" to estimate it."""
    if value == "auto":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(
        f'truth_delay must be "auto" or a number of days, not {value!r}.'
    )


def forecast_weeks(
    reference_date: dt.date | str, config: BacktestConfig
) -> tuple[str, ...]:
    """Return the weeks a forecast covers.

    Parameters
    ----------
    reference_date : datetime.date or str
        The reference date.
    config : BacktestConfig
        Gives the horizon range.

    Returns
    -------
    tuple of str
        ISO Saturdays from `min_horizon` to `max_horizon`.
    """
    ref = as_date(reference_date)
    return tuple(
        (ref + dt.timedelta(weeks=h)).isoformat()
        for h in range(config.min_horizon, config.max_horizon + 1)
    )


def forecast_locations(vintage_rows: pd.DataFrame) -> tuple[str, ...]:
    """Return the provinces a vintage lists, with or without cases.

    Parameters
    ----------
    vintage_rows : pandas.DataFrame
        Target rows of one vintage.

    Returns
    -------
    tuple of str
        Province codes, sorted.
    """
    prov = vintage_rows.loc[vintage_rows["location"] != NATIONAL]
    return tuple(sorted(set(prov["location"])))


def _fit_rows(model: ForecastModel, task: ForecastTask) -> pd.DataFrame:
    """Return the model's summary of each fit it made for the task."""
    rows = getattr(model, "fit_diagnostics", None) or []
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.insert(0, "model", model.name)
    table.insert(1, "kind", task.kind)
    table.insert(2, "vintage", task.vintage)
    table.insert(3, "masked", "-".join(task.mask))
    return table


def check_distinct(forecasts: pd.DataFrame) -> None:
    """Stop if two models gave the same samples for a task.

    A model that silently ran another would give the same samples, as
    the daily random walk once did when it had no snapshots.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts` with `model`, `kind`, `vintage` and
        `value`.

    Raises
    ------
    ValueError
        If two models have identical values for the same task.
    """
    for key, rows in forecasts.groupby(["kind", "vintage"]):
        seen: dict[str, np.ndarray] = {}
        for name, grp in rows.groupby("model"):
            values = grp.sort_values(["location", "target_end_date", "sample"])[
                "value"
            ].to_numpy()
            for other, prior in seen.items():
                if prior.shape == values.shape and np.array_equal(
                    prior, values
                ):
                    raise ValueError(
                        f"{name} and {other} gave identical samples for {key}."
                    )
            seen[str(name)] = values


def run_forecasts(
    models: Sequence[ForecastModel],
    tasks: Iterable[ForecastTask],
    fits: list[pd.DataFrame] | None = None,
    learnt: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run every model on every task, adding the national total.

    A model that raises `ModelUnavailableError` for a task, as the random
    walk models do when no pair of releases is within the delay window,
    has no rows for that task.
    A model with `needs_snapshots` set must be given the task's release
    snapshots, and the run stops if it is not.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to run.
    tasks : iterable of ForecastTask
        Tasks to run them on.
    fits : list, optional
        Collects a table of each model's fit summaries per task, for
        models that set `fit_diagnostics`.
    learnt : list, optional
        Collects each fit's learnt delay per task, for models that set
        `delay_summary`.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `kind`, `vintage`, `reference_date`, `horizon`,
        `location`, `target_end_date`, `sample`, `value` and `as_of`
        (missing for final counts).

    Examples
    --------
    >>> from nowcasteval.backtest import run_forecasts
    >>> from bdbvhub.models import get_model
    >>> models = [get_model("randomwalk"), get_model("randomwalknowcast")]
    >>> forecasts = run_forecasts(models, [task])  # e.g. nowcastbdbv make_tasks
    >>> forecasts.groupby("model")["sample"].nunique()
    """
    frames = []
    for task in tasks:
        for model in models:
            if getattr(model, "needs_snapshots", False) and (
                task.daily_history is None or task.daily_history.empty
            ):
                raise MissingSnapshots(
                    f"{model.name} needs daily release snapshots, and the "
                    f"{task.kind} task for {task.vintage} has none."
                )
            if hasattr(model, "fit_diagnostics"):
                model.fit_diagnostics = []
            try:
                samples = model.forecast(task)
            except ModelUnavailableError as err:
                log.warning(
                    "%s skipped %s %s: %s", model.name, task.kind,
                    task.vintage, err,
                )  # fmt: skip
                continue
            table = _fit_rows(model, task)
            if fits is not None and not table.empty:
                fits.append(table)
            delay = getattr(model, "delay_summary", None)
            if learnt is not None and delay is not None:
                learnt.append(delay)
            check_samples(samples, task)
            if "as_of" not in samples:
                samples["as_of"] = np.nan
            samples = add_national(samples)
            ref = pd.Timestamp(task.reference_date)
            samples["horizon"] = (
                pd.to_datetime(samples["target_end_date"]) - ref
            ).dt.days // 7
            samples.insert(0, "model", model.name)
            samples.insert(1, "kind", task.kind)
            samples.insert(2, "vintage", task.vintage)
            samples.insert(3, "reference_date", task.reference_date.isoformat())
            frames.append(samples)
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLUMNS).astype(_FORECAST_DTYPES)
    return pd.concat(frames, ignore_index=True)


def merge_by_model(path: Path | str, new: pd.DataFrame) -> pd.DataFrame:
    """Merge a run's rows into a results file, keeping other models' rows.

    The fitted variants are slow, so their rows come from the fit store
    (`bdbvhub.store.store`). A run replaces the rows of the
    models it ran and keeps the rest, so a run of no model keeps them all.

    Parameters
    ----------
    path : pathlib.Path or str
        A results file with a `model` column. It need not exist.
    new : pandas.DataFrame
        The run's rows.

    Returns
    -------
    pandas.DataFrame
        The kept rows of other models, then `new`.
    """
    path = Path(path)
    if not path.exists():
        return new
    # An empty table still has a gzip header, so the size cannot tell.
    try:
        old = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return new
    # A run of no model replaces nothing.
    if new.empty:
        return old
    if "model" not in old or "model" not in new:
        return new
    kept = old.loc[~old["model"].isin(set(new["model"]))]
    if kept.empty:
        return new
    return pd.concat([kept, new], ignore_index=True)


def score_tasks(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    truth_date: str,
    complete_after_weeks: int = 0,
) -> pd.DataFrame:
    """Score forecasts on every week against one truth vintage.

    Forecasts made at or after `truth_date` are dropped.
    Every week the truth covers is scored, as reported at `truth_date`.
    Final-count rows (no `as_of`) are scored `"as reported"`.
    Rows with `as_of` equal to `truth_date` predict the counts as reported
    then, and are scored `"vintage-matched"`. Rows for other dates are
    left out.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    truth_date : str
        The date of the truth data.
    complete_after_weeks : int
        $K$: weeks that ended at least this many weeks before `truth_date`
        are flagged `complete`.

    Returns
    -------
    pandas.DataFrame
        Output of `nowcasteval.scoring.score_forecasts`, with
        `truth_vintage`, `scoring` and `complete` columns.
    """
    keep = forecasts["vintage"] < truth_date
    as_of = (
        forecasts["as_of"]
        if "as_of" in forecasts
        else pd.Series(np.nan, index=forecasts.index)
    )
    frames = []
    for scoring, rows in (
        ("as reported", as_of.isna()),
        ("vintage-matched", as_of == truth_date),
    ):
        sel = forecasts.loc[keep & rows].drop(columns="as_of", errors="ignore")
        if sel.empty:
            continue
        scores = score_forecasts(sel, truth)
        scores["scoring"] = scoring
        frames.append(scores)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["truth_vintage"] = truth_date
    age = pd.Timestamp(truth_date) - pd.to_datetime(out["target_end_date"])
    out["complete"] = age.dt.days >= 7 * complete_after_weeks
    return out


def energy_tasks(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    truth_date: str,
    complete_after_weeks: int = 0,
) -> pd.DataFrame:
    """Score the final-count forecasts of `score_tasks` with the energy score.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    truth_date : str
        The date of the truth data. Forecasts made at or after it are
        dropped.
    complete_after_weeks : int
        $K$: only weeks that ended at least this many weeks before
        `truth_date` are scored, the `complete` weeks of `score_tasks`.

    Returns
    -------
    pandas.DataFrame
        `nowcasteval.scoring.energy_scores` by `kind`,
        `vintage` and `reference_date`, with `truth_vintage`.
    """
    keep = forecasts["vintage"] < truth_date
    if "as_of" in forecasts:
        keep &= forecasts["as_of"].isna()
    age = pd.Timestamp(truth_date) - pd.to_datetime(truth["target_end_date"])
    done = truth.loc[age.dt.days >= 7 * complete_after_weeks]
    out = energy_scores(
        forecasts.loc[keep], done, unit=("kind", "vintage", "reference_date")
    )
    out["truth_vintage"] = truth_date
    return out


def real_vintages(target: pd.DataFrame) -> list[str]:
    """List the real vintages in a target table, oldest first.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows.

    Returns
    -------
    list of str
        The vintage dates.
    """
    return sorted(set(target.loc[target["kind"] == "real", "vintage"]))


def truth_changes(
    target: pd.DataFrame, locations: Sequence[str] | None = None
) -> pd.DataFrame:
    """Show each week's truth in every real vintage and its change since.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows with every real vintage.
    locations : sequence of str, optional
        Locations to keep. All if None.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date`, `truth_vintage`, `truth`
        and `change`, this vintage's value minus the latest one. A
        negative change means the week has been revised upwards since.
        Sorted by location, week and vintage.
    """
    real = target.loc[target["kind"] == "real"]
    if locations is not None:
        real = real.loc[real["location"].isin(list(locations))]
    out = real[["location", "target_end_date", "vintage", "total"]].rename(
        columns={"vintage": "truth_vintage", "total": "truth"}
    )
    latest = out.loc[out["truth_vintage"] == out["truth_vintage"].max()]
    out = out.merge(
        latest[["location", "target_end_date", "truth"]],
        on=["location", "target_end_date"],
        suffixes=("", "_latest"),
    )
    out["change"] = out["truth"] - out.pop("truth_latest")
    return out.sort_values(
        ["location", "target_end_date", "truth_vintage"]
    ).reset_index(drop=True)


def joint_order(forecasts: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Put forecasts stacked by model in the order one run makes them.

    `run_forecasts` runs every model on a task before the next task. Rows
    are sorted by vintage, then by model in the order of `models`, and
    otherwise kept in order.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts with `vintage` and `model`.
    models : sequence of str
        The models in the order one run takes them.

    Returns
    -------
    pandas.DataFrame
        The rows reordered, with a fresh index.
    """
    if forecasts.empty:
        return forecasts.reset_index(drop=True)
    rank = forecasts["model"].map({m: i for i, m in enumerate(models)})
    order = np.lexsort((rank.to_numpy(), forecasts["vintage"].astype(str)))
    return forecasts.iloc[order].reset_index(drop=True)
