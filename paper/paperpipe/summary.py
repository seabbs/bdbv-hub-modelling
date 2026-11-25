"""The numbers the paper quotes, as Quarto metadata.

`study_summary` reads each study's run notes and main-text score table and
returns one entry per study: where its data came from, its baseline, the
best model with its relative scores, and each of `QUOTED_MODELS` against
`randomwalknowcast`. `write_summary` writes them as
YAML under `results`, so `main.qmd` and `si.qmd` quote them with
`{{< meta results.<study>.<field> >}}`. Where a study has a split, the
numbers are from its held-out half.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from nowcastfoundation.forecaster import checkpoint_spec

from paperfigs.periods import PERIODS
from paperpipe.meta import DUMMY_LABEL, baseline_of
from paperpipe.outputs import REFERENCE

__all__ = ["QUOTED_MODELS", "finetuned_spec", "study_summary", "write_summary"]


SCOPES: dict[str, tuple[str, str]] = {
    "germany": ("development half", "held-out half"),
    "ebola": ("release backtest before the scored rounds", "scored rounds"),
    "synthetic": ("development simulations", "held-out simulations"),
}
"""Each study's development and scored parts, as the text names them."""

QUOTED_MODELS: tuple[str, ...] = (
    "foundation_zeroshot_complete",
    "foundation_zeroshot",
    "foundation",
    "foundation_zeroshot_twostep",
    "foundation_twostep",
)
"""Models whose scores against `randomwalknowcast` the text quotes."""


def study_summary(
    meta: Mapping[str, Any], scores: pd.DataFrame, mode: str | None = None
) -> dict:
    """Return the quoted numbers of one study.

    Parameters
    ----------
    meta : mapping
        The study's run notes (`paperpipe.meta`).
    scores : pandas.DataFrame
        The study's main-text score table (`paperpipe.outputs`).
    mode : str, optional
        The run's mode, which names the synthetic study's part.

    Returns
    -------
    dict
        `source`, `label` (empty for real data), `note` (what is
        provisional), `baseline`, `n_models`, `best_model`,
        `best_rel_score`, `best_rel_log_score` and, where the baseline is
        another model, `best_rel_score_randomwalknowcast`, and `status`,
        which says when the scored part has no scores yet and is empty
        otherwise, and `scope`, the part the numbers come from (`SCOPES`),
        and `models`, each of `QUOTED_MODELS` relative to
        `randomwalknowcast`, or `pending` while it has no scores.
    """
    baseline = baseline_of(meta)
    dev, scored = SCOPES.get(meta["study"], ("", ""))
    # The synthetic study has no split: a paper run is all held out.
    if meta["study"] == "synthetic":
        scope = scored if mode == "paper" else dev
    else:
        scope = scored if "period" in scores.columns else dev
    if "period" in scores.columns:
        scores = scores.loc[scores["period"] == PERIODS[-1]]
    others = scores.loc[scores["model"] != baseline].dropna(
        subset=["rel_score"]
    )
    best = others.loc[others["rel_score"].idxmin()] if len(others) else None

    def value(col: str) -> float | None:
        if best is None or col not in best or pd.isna(best[col]):
            return None
        return round(float(best[col]), 2)

    return {
        # The paper shows a study's results only when its run has it.
        "shown": True,
        "source": meta["source"],
        "label": DUMMY_LABEL if meta["source"] == "dummy" else "",
        "note": meta.get("note", ""),
        "baseline": baseline,
        "n_models": int(scores["model"].nunique()),
        "best_model": "" if best is None else str(best["model"]),
        "best_rel_score": value("rel_score"),
        "best_rel_log_score": value("rel_log_score"),
        "best_rel_score_randomwalknowcast": value(
            "rel_score_randomwalknowcast"
        ),
        "scope": scope,
        "models": _against_reference(scores, baseline),
        "status": (
            ""
            if len(scores)
            else "There are no scored forecasts yet, as no forecast of "
            "the scored part has its truth reported."
        ),
    }


def _against_reference(
    scores: pd.DataFrame, baseline: str
) -> dict[str, float | str]:
    """Return each quoted model's score relative to `REFERENCE`."""
    col = "rel_score" if baseline == REFERENCE else f"rel_score_{REFERENCE}"
    rel = (
        scores.set_index("model")[col].dropna()
        if col in scores.columns
        else pd.Series(dtype=float)
    )
    return {
        m: round(float(rel[m]), 2) if m in rel.index else "pending"
        for m in QUOTED_MODELS
    }


def finetuned_spec(record: Path | str) -> str:
    """Return the spec of the checkpoint a release record names.

    The record (`published.json`) sits in the checkpoint's segment folder
    beside its `settings.json`, whose spec (`checkpoint_spec`) the `_ft`
    models loaded.
    """
    settings = Path(record).with_name("settings.json")
    return checkpoint_spec(json.loads(settings.read_text()))


def write_summary(
    path: Path,
    mode: str,
    studies: Mapping[str, dict],
    finetuned: str | None = None,
    realdata_after: str | None = None,
) -> None:
    """Write every study's numbers as Quarto metadata under `results`.

    `realdata_after` is the last BDBV release the real-data model trained
    on, quoted by the SI.
    `finetuned` is the fine-tuned checkpoint the `_ft` models loaded, or
    `pending` while the final run has published none.
    """
    pin = finetuned or "pending"
    doc = {
        "results": {
            "mode": mode,
            "finetuned": pin,
            "realdata_after": realdata_after or "pending",
            **studies,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
