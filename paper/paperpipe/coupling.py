"""How Chronos-2's Ebola trajectories score under each coupling.

The Ebola hub takes trajectories, so Chronos-2's submission joins its
cells' quantiles with a Gaussian copula. `coupling_rows` turns the paper
run's Ebola forecasts into each cell's quantiles, read from its samples,
with the cell's truth and the release that reported it, in the layout of
`nowcasteval.coupling`. `ebola_coupling` scores them with the copula,
with every cell drawn independently and with one level per trajectory
(comonotone), and summarises each coupling against the copula. A forecast
is dated by its round's close, so the copula's fit uses only the truth
reported by then.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from nowcasteval.coupling import coupling_scores, summary

__all__ = [
    "COUPLINGS",
    "COUPLING_MODELS",
    "LEVELS",
    "coupling_rows",
    "ebola_coupling",
]

COUPLING_MODELS: Final = ("foundation",)
"""The models scored: the fine-tuned Chronos-2 the hub was sent."""

COUPLINGS: Final = ("gaussian", "independent", "comonotone")
"""The copula Chronos-2 submits, then the bounds on the dependence."""

LEVELS: Final = (
    0.01,
    0.025,
    *np.round(np.arange(0.05, 0.951, 0.05), 2),
    0.975,
    0.99,
)
"""The quantile levels each cell's samples are read at."""

CLOSE_DAYS: Final = 4
"""Days from a round's close to its reference date (`ebola.round_close`)."""


def _vintages(observations: pd.DataFrame, delays: pd.Series) -> pd.DataFrame:
    """Return the release each target's truth is taken from.

    The first release on or after `target_date + D`, as the Ebola truth
    is taken (`paperfigs.truth.rolling_truth` with `first_after`).
    """
    obs = observations.assign(
        due=observations["target_date"]
        + pd.to_timedelta(observations["location"].map(delays), unit="D")
    )
    obs = obs.loc[obs["as_of_date"] >= obs["due"]]
    return (
        obs.groupby(["location", "target_date"], as_index=False)["as_of_date"]
        .min()
        .rename(columns={"as_of_date": "truth_vintage"})
    )


def coupling_rows(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    delays: pd.Series,
    models: Sequence[str] = COUPLING_MODELS,
) -> pd.DataFrame:
    """Return each cell's quantiles with its truth, for the coupling.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        The Ebola forecasts in the paper's schema, sample forecasts.
    truth : pandas.DataFrame
        The rolling truth (`paperfigs.truth.rolling_truth`).
    observations : pandas.DataFrame
        Every release's counts, which date each truth.
    delays : pandas.Series
        D by location, in days.
    models : sequence of str
        The models kept.

    Returns
    -------
    pandas.DataFrame
        `model`, `origin` (the round's close), `location`,
        `target_end_date`, `horizon`, `quantile`, `value`, `truth` and
        `truth_vintage`, the dates as ISO strings, for cells with a truth.
    """
    fc = forecasts.loc[forecasts["model"].isin(models)]
    keys = ["model", "location", "origin_date", "target_date", "horizon"]
    q = (
        fc.groupby(keys)["value"]
        .quantile(list(LEVELS))
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )
    q = q.merge(
        truth[["location", "target_date", "truth"]],
        on=["location", "target_date"],
        how="inner",
    ).dropna(subset=["truth"])
    q = q.merge(
        _vintages(observations, delays),
        on=["location", "target_date"],
        how="inner",
    )
    close = q["origin_date"] - pd.Timedelta(days=CLOSE_DAYS)
    return pd.DataFrame(
        {
            "model": q["model"],
            "origin": close.dt.strftime("%Y-%m-%d"),
            "location": q["location"].astype(str),
            "target_end_date": q["target_date"].dt.strftime("%Y-%m-%d"),
            "horizon": q["horizon"].astype(int),
            "quantile": q["quantile"].astype(float),
            "value": q["value"].astype(float),
            "truth": q["truth"].astype(float),
            "truth_vintage": q["truth_vintage"].dt.strftime("%Y-%m-%d"),
        }
    )


def ebola_coupling(
    rows: pd.DataFrame, n_samples: int = 200, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the trajectories of each coupling and summarise them.

    Returns `coupling_scores` by model, origin and coupling, and its
    `summary` with every coupling's ratio to the copula (`gaussian`).
    """
    scores = coupling_scores(
        rows, couplings=COUPLINGS, n_samples=n_samples, seed=seed
    )
    return scores, summary(scores, reference="gaussian")
