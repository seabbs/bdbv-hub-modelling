# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""The pairing sensitivity of the walk models on the BDBV backtest.

`bdbv_sensitivity` scores both pairings (`nowcastbaselines.pairing`) on
the BDBV release backtest's origins, and `bdbv_candidates` each candidate
copula setting, with `nowcasteval.pairing`'s scores. The German study is
`nowcastgermany.pairing`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from bdbvhub.store.store import origin_tasks
from nowcastbaselines.pairing import MODELS, variant_forecasts
from nowcastbaselines.randomwalk import RandomWalk
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcasteval.backtest import load_config, real_vintages
from nowcasteval.nowcast import resolve_truth_delay, rolling_truth
from nowcasteval.pairing import CANDIDATE_COLUMNS, SCORE_COLUMNS, variant_scores
from nowcastutils.base import ModelUnavailableError, add_national, get_model

__all__ = ["bdbv_candidates", "bdbv_sensitivity"]


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
