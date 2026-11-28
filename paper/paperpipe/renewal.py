"""The renewal model variants compared on the Ebola hub rounds.

The renewal models were fitted only for submission, so their forecasts
are the submission workflow's stored fits, each from the release before
its round closed, which the paper run's Ebola table reads with the other
models (`paperdata.ebola.backtest_samples`). `renewal_variants` compares
each pair of `RENEWAL_PAIRS` on the forecasts both made, by WIS on the
natural and log scales, for the country, provinces and zones apart.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from paperfigs.scores import relative_scores
from paperpipe.outputs import _level as level_of

__all__ = ["LEVELS", "RENEWAL_PAIRS", "renewal_variants"]

RENEWAL_PAIRS: Final = {
    "bvd": ("renewalbvd", "renewal"),
    "pooling": ("renewal", "renewalsp"),
}
"""Each comparison: the model and the model it is relative to. `bvd` is
what stabilising with BVDOutbreakSize adds, and `pooling` what pooling
and transmission between places add over fitting each zone on its own."""

LEVELS: Final = ("national", "provinces", "zones")
"""The Ebola levels, each compared on its own."""

COLUMNS: Final = (
    "comparison",
    "model",
    "baseline",
    "level",
    "rel_score",
    "rel_log_score",
    "n",
)


def renewal_variants(scores: pd.DataFrame) -> pd.DataFrame:
    """Return each renewal pair's relative WIS by level.

    Parameters
    ----------
    scores : pandas.DataFrame
        The Ebola scores (`paperfigs.scores.score_forecasts`), every model.

    Returns
    -------
    pandas.DataFrame
        One row per comparison and level in `LEVELS`: `model`,
        `baseline`, `rel_score` and `rel_log_score` (the model's mean WIS
        over the baseline's on the forecasts both made) and `n`, those
        forecasts. A pair with no shared forecast at a level has no row.
    """
    scores = scores.assign(level=level_of(scores))
    out = []
    for name, (model, baseline) in RENEWAL_PAIRS.items():
        pair = scores.loc[scores["model"].isin([model, baseline])]
        if baseline not in set(pair["model"]):
            continue
        rel = relative_scores(pair, by=["level"], baseline=baseline)
        rel = rel.loc[(rel["model"] == model) & rel["level"].isin(LEVELS)]
        out.append(rel.assign(comparison=name, baseline=baseline))
    if not out:
        return pd.DataFrame(columns=list(COLUMNS))
    table = pd.concat(out, ignore_index=True)
    table["_level"] = table["level"].map(LEVELS.index)
    table = table.sort_values(["comparison", "_level"])
    return table[list(COLUMNS)].reset_index(drop=True)
