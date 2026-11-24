# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Scores of the walk models' draws under each pairing (the sensitivity page).

Pairing only reorders draws (`nowcastbaselines.pairing.variant_forecasts`
gives them), so per-cell scores are equal across variants and only sums
and joint scores move.

`variant_scores` scores each variant of one forecast by level (national,
provinces or states or age groups, zones): per cell and on summed targets
(the country, which is the sum of the provinces; each place's total over
its weeks; the zones of a province each week), by sample CRPS and the
coverage of the 50% and 90% central intervals, and by the energy score of
each level's places and weeks as one vector. `summarise_pairing` gives the
mean of each score by study, level, model, variant and target.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from nowcasteval.scoring import coverage_sample, crps_sample, energy_sample
from nowcastutils.base import (
    NATIONAL,
)

__all__ = [
    "CANDIDATE_COLUMNS",
    "RHO_COLUMNS",
    "SCORE_COLUMNS",
    "best_candidate",
    "summarise_pairing",
    "variant_scores",
]

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
"""Columns of `bdbvhub.pairing.bdbv_candidates`: `variant` is the
candidate's label."""

_CELL = ["location", "target_end_date"]


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
