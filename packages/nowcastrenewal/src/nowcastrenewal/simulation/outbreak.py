# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A simulated outbreak as one numpyro model.

`outbreak_model` runs a simulation's layers in one model from the shared
components: the latent infections and expected reference events
(`latent.simulate.simulate_latent`), then the reporting delay and its
effects, the report days, the observation noise and the revisions
(`reporting.simulate.reporting_model`). The grid's draws (design and
values) enter as fixed values; every random quantity is a numpyro site,
so the model is sampled forward under one seed and could be conditioned
on reported counts (`"draw_{k}::reports::counts"`).

`simulate_outbreak` samples it and returns the latent run, the counts in
the simulator's long format and the values the reporting adds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
import pandas as pd
from numpyro import handlers

from ..components.pmfs import Delay, generation_time_pmf
from ..latent.rt import KINDS
from ..latent.simulate import Latent, simulate_latent
from ..reporting.simulate import reporting_frame, reporting_model, tabulate

__all__ = ["outbreak_model", "simulate_outbreak"]


def outbreak_model(
    n_days: int,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    draws: int = 1,
    kinds: Sequence[str] = KINDS,
    incubation_cv: float = 0.5,
    depletion: str = "linear",
) -> Latent:
    """Run a simulated outbreak's layers as one numpyro model.

    Parameters
    ----------
    n_days : int
        Days of the reporting frame, the generation time's span included.
    strata : pandas.DataFrame
        The strata (`place`, `age`, `.group`), places first then ages.
    design, values : mapping
        The simulation's design and values.
    draws : int
        Observation draws.
    kinds, incubation_cv, depletion
        As `latent.simulate.simulate_latent`.

    Returns
    -------
    Latent
        The latent run; the reporting is in the trace's sites.
    """
    gt = generation_time_pmf(
        Delay.from_mean_cv(
            design["gt_family"], values["gt_mean"], values["gt_cv"]
        )
    )
    latent = simulate_latent(
        n_days - len(gt),
        gt,
        strata,
        design,
        values,
        kinds,
        incubation_cv,
        depletion,
    )
    frame = reporting_frame(
        values, n_days, list(strata["place"]), list(strata["age"])
    )
    reporting_model(frame, design, values, latent.onsets, draws)
    return latent


def simulate_outbreak(
    key: int,
    n_days: int,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    draws: int = 1,
    kinds: Sequence[str] = KINDS,
    incubation_cv: float = 0.5,
    depletion: str = "linear",
) -> tuple[Latent, pd.DataFrame, dict[str, Any]]:
    """Sample a simulated outbreak forward (`outbreak_model`).

    Parameters
    ----------
    key : int
        The simulation's seed.
    n_days, strata, design, values, draws, kinds, incubation_cv, depletion
        As `outbreak_model`.

    Returns
    -------
    tuple
        The latent run, the counts (`reporting.simulate.COUNT_COLUMNS`)
        and the values the reporting adds.
    """
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=jax.random.key(int(key))),
    ):
        latent = outbreak_model(
            n_days,
            strata,
            design,
            values,
            draws,
            kinds,
            incubation_cv,
            depletion,
        )
    counts, added = tabulate(
        tr,
        design,
        values,
        np.asarray(latent.onsets),
        list(strata["place"]),
        list(strata["age"]),
        draws,
    )
    return latent, counts, added
