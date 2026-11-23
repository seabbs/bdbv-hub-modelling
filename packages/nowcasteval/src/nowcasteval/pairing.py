# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How much pairing the walk models' draws matters (the sensitivity page).

`randomwalk` and `randomwalknowcast` pair their province draws within
each week inside their forecast (`nowcastbaselines.copula.pair_walks`). Here
each forecasts a task once and gives its draws under both pairings
(`VARIANTS`): none (`off`) and the rolling $\rho_p$ it uses (`rolling`),
from its unpaired draws (`RandomWalk.unpaired`). Pairing only reorders
draws, so per-cell scores are equal across variants and only sums and
joint scores move.

`variant_scores` scores each variant of one forecast by level (national,
provinces or states or age groups, zones): per cell and on summed targets
(the country, which is the sum of the provinces; each place's total over
its weeks; the zones of a province each week), by sample CRPS and the
coverage of the 50% and 90% central intervals, and by the energy score of
each level's places and weeks as one vector. `summarise_pairing` gives the
mean of each score by study, level, model, variant and target.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from nowcastbaselines.randomwalk import RandomWalk
from nowcasteval.scoring import coverage_sample, crps_sample, energy_sample
from nowcastutils.base import (
    NATIONAL,
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
)

__all__ = [
    "CANDIDATE_COLUMNS",
    "MODELS",
    "RHO_COLUMNS",
    "SCORE_COLUMNS",
    "VARIANTS",
    "bdbv_candidates",
    "best_candidate",
    "summarise_pairing",
    "variant_forecasts",
    "variant_scores",
]

MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their draws."""

VARIANTS: Final = ("off", "rolling")
"""The pairings compared: none, and the rolling $\rho_p$ the models use."""

SCORE_COLUMNS: Final = (
    "study",
    "level",
    "model",
    "variant",
    "origin",
    "target",
    "metric",
    "value",
    "n",
)
"""Columns of `variant_scores`."""

RHO_COLUMNS: Final = (
    "model",
    "origin",
    "level",
    "n_scores",
    "rho_provinces",
    "n_pairs",
    "fallback",
)
"""Columns of the rho_p rows (`nowcastgermany.pairing.germany_sensitivity`)."""

CANDIDATE_COLUMNS: Final = (*SCORE_COLUMNS, "rho_provinces", "fallback")
"""Columns of `bdbv_candidates`: `variant` is the candidate's label."""

_CELL = ["location", "target_end_date"]


def variant_forecasts(
    name: str, task: ForecastTask
) -> tuple[dict[str, pd.DataFrame], dict[str, object] | None]:
    """Forecast a task once and return it under both pairings.

    Parameters
    ----------
    name : str
        A walk model (`MODELS`).
    task : ForecastTask
        The task.

    Returns
    -------
    dict
        Each variant's samples, by label (`VARIANTS`), with the country
        added as the sum of the provinces where the task has provinces.
        Empty where the model cannot forecast the task.
    dict or None
        The model's copula record (`RandomWalk.copula_fit`: `model`,
        `origin`, `n_scores`, `rho_provinces`, `n_pairs`, `fallback`),
        None where it paired nothing.
    """
    model: RandomWalk = dataclasses.replace(
        cast("RandomWalk", get_model(name)), keep_variants=True
    )
    try:
        paired = model.forecast(task)
    except ModelUnavailableError:
        return {}, None
    out = {}
    for label, drawn in zip(VARIANTS, (model.unpaired(), paired), strict=True):
        zones = drawn["location"].str.count("-") >= 2
        out[label] = drawn if zones.all() else add_national(drawn)
    return out, model.copula_fit


def _level(
    location: pd.Series, levels: dict[str, str], national: str
) -> pd.Series:
    """Each location's level: national, a named level, or zones."""
    zone = location.str.count("-") >= 2
    out = location.map(levels).fillna("provinces")
    out[location == national] = "national"
    out[zone] = "zones"
    return out


def _sums(
    rows: pd.DataFrame, by: list[str], target: str
) -> list[dict[str, object]]:
    """CRPS and coverage of per-sample sums of `rows` over `by` groups."""
    found: list[dict[str, object]] = []
    for _, grp in rows.groupby(by):
        draws = grp.groupby("sample")["value"].sum().to_numpy()
        obs = float(grp.drop_duplicates(_CELL)["truth"].sum())
        for metric, value in (
            ("crps", crps_sample(obs, draws)),
            ("cover50", float(coverage_sample(obs, draws, 0.5))),
            ("cover90", float(coverage_sample(obs, draws, 0.9))),
        ):
            found.append({"target": target, "metric": metric, "value": value})
    return found


def variant_scores(
    samples: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    study: str,
    model: str,
    variant: str,
    origin: str,
    levels: dict[str, str] | None = None,
    national: str = NATIONAL,
) -> pd.DataFrame:
    """Score one variant of one forecast by level.

    Parameters
    ----------
    samples : pandas.DataFrame
        `location`, `target_end_date`, `sample` and `value`.
    truth : pandas.DataFrame
        `location`, `target_end_date` and `truth`. Only cells with a truth
        are scored.
    study, model, variant, origin : str
        Labels.
    levels : dict, optional
        A level for places that are neither the country nor zones (such as
        age groups); `provinces` by default.
    national : str
        The country's code.

    Returns
    -------
    pandas.DataFrame
        `SCORE_COLUMNS`: the mean of each score over the forecast's cells
        or sums, and their number `n`.
    """
    rows = samples.merge(truth[[*_CELL, "truth"]], on=_CELL)
    if rows.empty:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    rows = rows.assign(level=_level(rows["location"], levels or {}, national))
    found: list[pd.DataFrame] = []
    for level, grp in rows.groupby("level"):
        scored: list[dict[str, object]] = [
            *_sums(grp, _CELL, "cell"),
            *_sums(grp, ["location"], "sum_weeks"),
        ]
        if level == "zones":
            province = grp["location"].str.rsplit("-", n=1).str[0]
            scored += _sums(
                grp.assign(province=province),
                ["province", "target_end_date"],
                "sum_zones",
            )
            groups = list(grp.assign(province=province).groupby("province"))
        else:
            groups = [(level, grp)]
        for _, own in groups:
            wide = own.pivot(index="sample", columns=_CELL, values="value")
            obs = (
                own.drop_duplicates(_CELL)
                .set_index(_CELL)["truth"]
                .reindex(wide.columns)
                .to_numpy()
            )
            x = wide.to_numpy()
            scored += [
                {"target": "joint", "metric": "energy",
                 "value": energy_sample(obs, x)},
                {"target": "joint", "metric": "log_energy",
                 "value": energy_sample(np.log1p(obs), np.log1p(x))},
            ]  # fmt: skip
        frame = pd.DataFrame(scored)
        summary = frame.groupby(["target", "metric"], as_index=False).agg(
            value=("value", "mean"), n=("value", "size")
        )
        found.append(summary.assign(level=level))
    out = pd.concat(found, ignore_index=True).assign(
        study=study, model=model, variant=variant, origin=origin
    )
    return out[list(SCORE_COLUMNS)]


def summarise_pairing(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the mean of each score by study, level, model and variant.

    Parameters
    ----------
    scores : pandas.DataFrame
        `variant_scores` rows of many origins.

    Returns
    -------
    pandas.DataFrame
        `study`, `level`, `model`, `variant`, `target`, `metric`, `value`
        (the mean over origins, weighted by each origin's `n`), `n` and
        `relative` (the value over the `off` variant's).
    """
    keys = ["study", "level", "model", "variant", "target", "metric"]
    weighted = scores.assign(total=scores["value"] * scores["n"])
    out = weighted.groupby(keys, as_index=False).agg(
        total=("total", "sum"), n=("n", "sum")
    )
    out["value"] = out["total"] / out["n"]
    base = out.loc[out["variant"] == "off"].set_index(
        [k for k in keys if k != "variant"]
    )["value"]
    idx = pd.MultiIndex.from_frame(out[[k for k in keys if k != "variant"]])
    out["relative"] = out["value"].to_numpy() / base.reindex(idx).to_numpy()
    return out[[*keys, "value", "n", "relative"]]


def bdbv_sensitivity(
    data: str,
    config_path: str = "config/backtest.toml",
    origins: list[str] | None = None,
) -> pd.DataFrame:
    """Score both pairings on the BDBV release backtest's origins.

    Each real origin (less `skip_origins` and the latest) is forecast for
    the provinces, with the country their sum. Zones are their own
    forecasts and are not paired, so they are not scored here. Truth is
    each week's count in the first release the backtest's truth delay
    after it.

    Parameters
    ----------
    data : str
        The data directory.
    config_path : str
        The backtest config.
    origins : list of str, optional
        Only these origins (a block of a chunked run).

    Returns
    -------
    pandas.DataFrame
        `variant_scores` rows of every model, variant, origin and level.
    """
    from bdbvhub.data.tidy import load_daily, load_delays, load_target
    from bdbvhub.store.store import origin_tasks
    from nowcasteval.backtest import load_config, real_vintages
    from nowcasteval.nowcast import resolve_truth_delay, rolling_truth

    config = load_config(config_path)
    target, delays = load_target(data), load_delays(data)
    daily = load_daily(data)
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    truth = rolling_truth(real, resolve_truth_delay(config, target))
    tasks = origin_tasks(target, delays, config, daily)
    frames = []
    for origin in vintages[:-1]:
        if origin in config.skip_origins or origin not in tasks:
            continue
        if origins is not None and origin not in origins:
            continue
        for name in MODELS:
            drawn, _ = variant_forecasts(name, tasks[origin])
            for label, s in drawn.items():
                frames.append(
                    variant_scores(
                        s,
                        truth,
                        study="bdbv",
                        model=name,
                        variant=label,
                        origin=origin,
                    )
                )
    if not frames:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def bdbv_candidates(
    candidates: Sequence[Any],
    data: str,
    config_path: str = "config/backtest.toml",
    origins: list[str] | None = None,
) -> pd.DataFrame:
    """Score each candidate copula setting on the BDBV release backtest.

    As `bdbv_sensitivity`, with each origin's draws paired under every
    candidate setting (`RandomWalk.paired_with`). The draws are the
    forecast's own, so only their pairing differs.

    Parameters
    ----------
    candidates : sequence of CopulaSettings
        The settings, `nowcastbaselines.copula.CANDIDATES`.
    data : str
        The data directory.
    config_path : str
        The backtest config.
    origins : list of str, optional
        Only these origins (a block of a chunked run).

    Returns
    -------
    pandas.DataFrame
        `variant_scores` rows by candidate (`variant`, its label), with
        the candidate's `rho_provinces` and `fallback` at that origin.
    """
    from bdbvhub.data.tidy import load_daily, load_delays, load_target
    from bdbvhub.store.store import origin_tasks
    from nowcasteval.backtest import load_config, real_vintages
    from nowcasteval.nowcast import resolve_truth_delay, rolling_truth

    config = load_config(config_path)
    target, delays = load_target(data), load_delays(data)
    real = target.loc[target["kind"] == "real"]
    truth = rolling_truth(real, resolve_truth_delay(config, target))
    tasks = origin_tasks(target, delays, config, load_daily(data))
    frames = []
    for origin in real_vintages(real)[:-1]:
        if origin in config.skip_origins or origin not in tasks:
            continue
        if origins is not None and origin not in origins:
            continue
        for name in MODELS:
            model: RandomWalk = dataclasses.replace(
                cast("RandomWalk", get_model(name)), keep_variants=True
            )
            try:
                model.forecast(tasks[origin])
            except ModelUnavailableError:
                continue
            for candidate in candidates:
                drawn, fit = model.paired_with(candidate)
                scores = variant_scores(
                    add_national(drawn),
                    truth,
                    study="bdbv",
                    model=name,
                    variant=candidate.label,
                    origin=origin,
                )
                fit = fit or {"rho_provinces": np.nan, "fallback": False}
                frames.append(
                    scores.assign(
                        rho_provinces=float(cast(float, fit["rho_provinces"])),
                        fallback=bool(fit["fallback"]),
                    )
                )
    if not frames:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def best_candidate(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank the candidates by their mean provinces energy score.

    The energy score of each origin's provinces and weeks as one vector is
    the joint score the pairing is for. Each model's scores are divided by
    its own mean over candidates, so both models count alike.

    Parameters
    ----------
    scores : pandas.DataFrame
        `bdbv_candidates`.

    Returns
    -------
    pandas.DataFrame
        One row per candidate (`variant`): `energy` (the mean relative
        energy score), `national_crps` (the same for the country, the sum
        of the provinces), and `fallback` (the share of origins that fell
        back to the prior), best first.
    """
    out = {}
    for metric, level, target in (
        ("energy", "provinces", "joint"),
        ("crps", "national", "cell"),
    ):
        rows = scores.loc[
            (scores["metric"] == metric)
            & (scores["level"] == level)
            & (scores["target"] == target)
        ]
        by = rows.groupby(["model", "variant", "origin"])["value"].mean()
        rel = by / by.groupby(["model", "origin"]).transform("mean")
        out[f"{'energy' if metric == 'energy' else 'national_crps'}"] = (
            rel.groupby("variant").mean()
        )
    table = pd.DataFrame(out)
    table["fallback"] = scores.groupby("variant")["fallback"].mean()
    return table.sort_values("energy").reset_index()
