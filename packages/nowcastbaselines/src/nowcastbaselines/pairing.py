# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The walk models' forecasts under each pairing of their draws.

`randomwalk` and `randomwalknowcast` pair their province draws within
each week inside their forecast (`nowcastbaselines.copula.pair_walks`).
`variant_forecasts` forecasts a task once and gives its draws under both
pairings (`VARIANTS`): none (`off`) and the rolling $\rho_p$ it uses
(`rolling`), from its unpaired draws (`RandomWalk.unpaired`).
`nowcasteval.pairing` scores them.
"""

from __future__ import annotations

import dataclasses
from typing import Final, cast

import pandas as pd

from nowcastbaselines.randomwalk import RandomWalk
from nowcastutils.base import (
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
)

__all__ = ["MODELS", "VARIANTS", "variant_forecasts"]

MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their draws."""

VARIANTS: Final = ("off", "rolling")
"""The pairings compared: none, and the rolling $\rho_p$ the models use."""


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
