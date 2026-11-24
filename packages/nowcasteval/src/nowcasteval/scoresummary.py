# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Per-forecast scores for the in-browser summaries.

The page embeds one row per model, origin, release, place and week, and
`docs/javascripts/summarise.js` summarises a selection of them as
`relative_crps` does: the mean of each metric over each model's own
forecasts and, for relative metrics, its pairwise relative skill
(`nowcasteval.scoring.pairwise_skill`). The skill is computed here, once,
as each model's geometric mean of ratios (`skill`) for every selection
the page offers: each round or all rounds, by scale or by place, overall
or at each horizon. The page divides one model's by the chosen baseline's,
which only rescales them. A model with fewer forecasts than the baseline
is marked partial (`mark_partial`). A new metric is a score column and a
label.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nowcastutils.base import NATIONAL

__all__ = [
    "ENERGY_SUMMARY",
    "SUMMARY_METRICS",
    "mark_partial",
    "skill_key",
    "summary_html",
    "summary_rows",
    "targets_attr",
]

SUMMARY_METRICS: tuple[tuple[str, str, float | None, bool], ...] = (
    ("crps", "CRPS", None, True),
    ("log_crps", "CRPS on the log scale", None, True),
    ("coverage_50", "50% coverage", 0.5, False),
    ("coverage_90", "90% coverage", 0.9, False),
    ("bias", "Bias", 0.0, False),
)
"""Score column, label, reference line and whether it is also shown
relative to the baseline, whose reference is then 1."""

ENERGY_SUMMARY: tuple[tuple[str, str], ...] = (
    ("energy", "Energy score"),
    ("log_energy", "Energy score on the log scale"),
)
"""Energy score columns and labels. There is one score per model, origin,
release and pool, so they show in the overall view only."""


def mark_partial(
    table: pd.DataFrame, by: Sequence[str], baseline: str
) -> pd.DataFrame:
    """Mark the rows of models that lack some of the baseline's cells.

    Relative scores are on the forecasts the baseline also made, so `n`
    of a model with every one of the baseline's cells is the baseline's,
    and those rows can be read side by side. A model with fewer is
    partial: its scores are on other forecasts and are read apart.

    Parameters
    ----------
    table : pandas.DataFrame
        `relative_crps` or `relative_energy` output, with `model`, the `by`
        columns and `n` (or `n_energy`).
    by : sequence of str
        Columns of a row other than `model`.
    baseline : str
        The baseline of `table`.

    Returns
    -------
    pandas.DataFrame
        `table` with `cells`, the baseline's n in the row's group, and
        `partial`, whether the model has fewer.
    """
    n = "n" if "n" in table.columns else "n_energy"
    by = [c for c in by if c in table.columns]
    base = table.loc[table["model"] == baseline, [*by, n]]
    base = base.rename(columns={n: "cells"})
    out = (
        table.merge(base, on=by, how="left")
        if by
        else table.assign(cells=base["cells"].iloc[0] if len(base) else None)
    )
    out["partial"] = (out[n] < out["cells"]).fillna(False).astype(bool)
    return out


_POOLS = {
    "national": "National",
    "provinces": "Provinces",
    "zones": "Health zones",
}


def _pool(location: str) -> str:
    if location == NATIONAL:
        return "national"
    return "provinces" if location.count("-") == 1 else "zones"


def summary_rows(
    scores: pd.DataFrame,
    models: Sequence[str],
    colours: Mapping[str, tuple[str, str, str]],
    names: Mapping[str, str],
    baseline: str,
    metrics: Sequence[tuple[str, str, float | None, bool]] = SUMMARY_METRICS,
    default_on: Sequence[str] | None = None,
    energy: pd.DataFrame | None = None,
    pool_of: Callable[[str], str] = _pool,
    pool_labels: Mapping[str, str] = _POOLS,
) -> dict[str, object]:
    """Return the per-forecast rows and their labels for one target.

    Parameters
    ----------
    scores : pandas.DataFrame
        One row per forecast scored against one release, with `model`,
        `origin`, `truth_vintage`, `location`, `target_end_date`,
        `horizon` (weeks from the reference date) and the metric columns.
    models : sequence of str
        Models in order. Rows of other models are left out.
    colours : mapping of str to (str, str, str)
        Model to its light colour, dark colour and dash.
    names : mapping of str to str
        Place names by id. Zones not named show their code.
    baseline : str
        The model relative metrics divide by.
    metrics : sequence of (str, str, float or None, bool)
        As `SUMMARY_METRICS`.
    default_on : sequence of str, optional
        Models ticked at first. Defaults to all.
    energy : pandas.DataFrame, optional
        Energy scores (`energy_scores`) with `model`, `origin`,
        `truth_vintage`, `scale` and the `ENERGY_SUMMARY` columns.

    Returns
    -------
    dict
        `models` (name, light and dark colour, dash, ticked at first),
        `baseline` (its index), `metrics` (key, label, ref, relative),
        `origins`, `releases`, `weeks`, `pools` (id and label, in order),
        `places` (id, name, pool) and `rows`: model, origin, release,
        place and week indices, then each metric, `horizons`, each row's
        horizon, and `skill` (`_skill`). With `energy`, also `energy`:
        `metrics` (key and label) and `rows` of model, origin, release and
        pool indices, then each energy score.
    """
    shown = [m for m in models if m in set(scores["model"])]
    s = scores.loc[scores["model"].isin(shown)]
    origins = sorted(set(s["origin"].astype(str)))
    releases = sorted(set(s["truth_vintage"].astype(str)))
    weeks = sorted(set(s["target_end_date"].astype(str)))
    pools = [p for p in pool_labels if p in set(s["location"].map(pool_of))]
    places = sorted(
        set(s["location"]),
        key=lambda x: (pools.index(pool_of(x)), names.get(x, x)),
    )
    on = set(shown if default_on is None else default_on)
    keys = [m[0] for m in metrics]
    idx = {
        "model": {m: i for i, m in enumerate(shown)},
        "origin": {o: i for i, o in enumerate(origins)},
        "truth_vintage": {r: i for i, r in enumerate(releases)},
        "location": {p: i for i, p in enumerate(places)},
        "target_end_date": {w: i for i, w in enumerate(weeks)},
    }
    cols = s[[*idx, *keys]].astype(dict.fromkeys(idx, str))
    horizons = [int(h) for h in s["horizon"]]
    rows = [
        [
            *(idx[c][v] for c, v in zip(idx, r[: len(idx)], strict=True)),
            *(round(float(v), 6) for v in r[len(idx) :]),
        ]
        for r in cols.itertuples(index=False)
    ]
    out: dict[str, object] = {
        "models": [
            {
                "name": m,
                "light": colours[m][0],
                "dark": colours[m][1],
                "dash": colours[m][2],
                "on": m in on,
            }
            for m in shown
        ],
        "baseline": shown.index(baseline),
        "metrics": [
            {"key": k, "label": label, "ref": ref, "relative": rel}
            for k, label, ref, rel in metrics
        ],
        "origins": origins,
        "releases": releases,
        "weeks": weeks,
        "pools": [{"id": p, "label": pool_labels[p]} for p in pools],
        "places": [
            {"id": p, "name": names.get(p, p), "pool": pool_of(p)}
            for p in places
        ],
        "rows": rows,
        "horizons": horizons,
        "skill": _skill(
            s,
            shown,
            baseline,
            idx,
            pools,
            pool_of,
            [k for k, _, _, rel in metrics if rel],
        ),
    }
    if energy is not None:
        out["energy"] = _energy_rows(energy, idx, pools)
    return out


SKILL_MATCH = ("origin", "truth_vintage", "location", "target_end_date")
"""What makes two models' scores the same forecast in a summary."""


def skill_key(round_: int | None, select: str, horizon: int | None) -> str:
    """Return the key of one selection's skill in `summary_rows`.

    `round_` is an origin index or None for every round; `select` is
    `p<pool index>` or `x<place index>`; `horizon` a horizon or None for
    every horizon.
    """
    r = "all" if round_ is None else str(round_)
    h = "all" if horizon is None else str(horizon)
    return f"{r}|{select}|{h}"


def _skill(
    scores: pd.DataFrame,
    shown: Sequence[str],
    baseline: str,
    idx: Mapping[str, Mapping[str, int]],
    pools: Sequence[str],
    pool_of: Callable[[str], str],
    keys: Sequence[str],
) -> dict[str, dict[str, list[float | None]]]:
    """Return each model's pairwise skill for every selection of the page.

    One entry per `skill_key`: each round and all rounds, by scale (and at
    each horizon) and by place. Each holds, per relative metric, the
    models' `skill_<metric>` in `shown` order, None where a model has none.
    """
    if not keys:
        return {}
    # Imported here: nowcasteval.scoring needs scoringrules, which the
    # page modules do not otherwise import.
    from nowcasteval.scoring import pairwise_skill

    t = scores.assign(
        _round=scores["origin"].astype(str).map(idx["origin"]),
        _pool=scores["location"]
        .map(pool_of)
        .map({p: i for i, p in enumerate(pools)}),
        _place=scores["location"].astype(str).map(idx["location"]),
    )
    at = {m: i for i, m in enumerate(shown)}
    out: dict[str, dict[str, list[float | None]]] = {}
    for rounds in (None, "_round"):
        for select, hz in (
            ("_pool", None),
            ("_pool", "horizon"),
            ("_place", None),
        ):
            by = [c for c in (rounds, select, hz) if c]
            skill = pairwise_skill(
                t,
                baseline,
                by=by,
                metrics=keys,
                match_on=SKILL_MATCH,
                rounds="origin",
            )
            tag = "p" if select == "_pool" else "x"
            for row in skill.to_dict("records"):
                key = skill_key(
                    int(row["_round"]) if rounds else None,
                    f"{tag}{int(row[select])}",
                    int(row["horizon"]) if hz else None,
                )
                entry = out.setdefault(
                    key, {k: [None] * len(shown) for k in keys}
                )
                for k in keys:
                    v = row[f"skill_{k}"]
                    if np.isfinite(v):
                        entry[k][at[row["model"]]] = round(float(v), 10)
    return out


def _energy_rows(
    energy: pd.DataFrame,
    idx: Mapping[str, Mapping[str, int]],
    pools: Sequence[str],
) -> dict[str, object]:
    """Index the energy scores into the summary's models, rounds and pools."""
    e = energy.loc[
        energy["model"].isin(set(idx["model"]))
        & energy["origin"].astype(str).isin(set(idx["origin"]))
        & energy["truth_vintage"].astype(str).isin(set(idx["truth_vintage"]))
        & energy["scale"].isin(set(pools))
    ]
    keys = [k for k, _ in ENERGY_SUMMARY]
    return {
        "metrics": [{"key": k, "label": label} for k, label in ENERGY_SUMMARY],
        "rows": [
            [
                idx["model"][m],
                idx["origin"][str(o)],
                idx["truth_vintage"][str(r)],
                pools.index(p),
                *(round(float(v), 6) for v in vals),
            ]
            for m, o, r, p, *vals in e[
                ["model", "origin", "truth_vintage", "scale", *keys]
            ].itertuples(index=False)
        ],
    }


def targets_attr(srcs: Mapping[str, str]) -> str:
    """Return the `data-targets` attribute: each target's summary URL.

    The targets are `"latest"` (scored on the latest release) and
    `"complete"` (scored on complete weeks), in the order given; the first
    is shown first and `toggles.js` offers a picker when there are two.
    """
    return f" data-targets='{json.dumps(dict(srcs))}'"


def summary_html(
    srcs: Mapping[str, str],
    title: str,
    view: str = "overall",
    static: str | None = None,
    map_src: str | None = None,
) -> str:
    """Return the page element `docs/javascripts/scoresummary.js` fills.

    Parameters
    ----------
    srcs : mapping of str to str
        URL of each target's `summary_rows` JSON, relative to the page
        (`targets_attr`).
    title : str
        Accessible name of the summary.
    view : str
        `"overall"` for one group of bars for the chosen scale, `"place"`
        for one group per place and a place picker, or `"horizon"` for a
        line per model by horizon.
    static : str, optional
        Path of a static SVG, relative to the page, shown without
        JavaScript. Its `-dark` twin is shown in the dark theme.
    map_src : str, optional
        URL of the health zone GeoJSON, relative to the page. The place
        view then offers a map of the metric shown.
    """
    fallback = "The interactive summary needs JavaScript."
    if static is not None:
        dark = static.removesuffix(".svg") + "-dark.svg"
        fallback = (
            f'<img src="{static}#only-light" alt="{title}">'
            f'<img src="{dark}#only-dark" alt="{title}">'
        )
    shown = "" if map_src is None else f' data-map="{map_src}"'
    return (
        f'<div class="scoresummary"{targets_attr(srcs)} data-view="{view}"'
        f"{shown} "
        f'role="group" aria-label="{title}">\n'
        f"<noscript>{fallback}</noscript>\n"
        "</div>\n"
    )
