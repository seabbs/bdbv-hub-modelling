"""SI tables and figures from results kept on evidence branches.

The copula and the fine-tuning runs write
their results as rows to evidence branches (see
`workflow/config.yaml`, block `evidence`). This module turns those rows
into the SI's tables, figures and quoted numbers, so nothing is typed into
the paper. A result whose rows are not set yet gets a note saying what it
waits on in place of its table, and `None` in place of its numbers.

- `copula_figure`: the rolling copula against no pairing, by study and
  level.
- `rho_medians`: the median rolling rho_p of each walk model.
- `realism_figure`: the simulated outbreaks against the data and
  published values.
- `checkpoint_figure`: relative scores by checkpoint, the German panels of
  each run's progress figure (`paperpipe.progress`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure, SubFigure
from nowcasteval.labels import (
    FINETUNED_OF,
    MODEL_LABELS,
    MODEL_STYLES,
    model_label,
)
from nowcastfoundation.live_checkpoint import (
    LIVE_RUN,
    checkpoint_table,
    newest,
    summary_table,
    tidy,
)
from nowcastfoundation.live_checkpoint import label as checkpoint_label

from paperpipe.outputs import to_markdown

__all__ = [
    "BIAS_SUBSETS",
    "COMPARISONS",
    "EVIDENCE",
    "checkpoint_figure",
    "checkpoint_numbers",
    "checkpoint_rows",
    "compared",
    "comparison_figure",
    "comparison_rows",
    "copula_figure",
    "diseases_table",
    "drop_bdbv",
    "live_numbers",
    "realism_figure",
    "realism_numbers",
    "realism_reference",
    "rho_medians",
    "slots_summary",
    "waiting",
    "write_evidence",
]


def waiting(what: str) -> str:
    """Return the note a table shows until its rows exist."""
    return f"*Not yet available: waits on {what}.*\n"


def _pair(a: pd.Series, b: pd.Series) -> list[str]:
    return [
        "" if pd.isna(x) or pd.isna(y) else f"{x:.2f} / {y:.2f}"
        for x, y in zip(a, b, strict=True)
    ]


CHECKPOINT_RULE: Final = "own-quantile-wis_crps-1000-common"
"""The scoring rule of the checkpoint rows the paper shows: WIS on each
model's own quantiles for Germany."""


def drop_bdbv(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop BDBV (Ebola) score rows, as the Ebola study is held out.

    Rows whose `study` is `ebola` or `bdbv`, whose `set` names BDBV (the
    `compare_*` layout), or the scale-floor test's BDBV rows (those with
    a `source`). Data summaries (`sim_realism`) are not scores and stay.
    """
    keep = pd.Series(True, index=frame.index)
    if "study" in frame:
        keep &= ~frame["study"].astype(str).str.lower().isin(["ebola", "bdbv"])
    if "set" in frame:
        keep &= ~frame["set"].astype(str).str.startswith("BDBV")
    if {"source", "relative_wis"} <= set(frame.columns):
        keep &= frame["source"].isna()
    return frame.loc[keep].reset_index(drop=True)


def checkpoint_rows(scores: pd.DataFrame, run: str = LIVE_RUN) -> pd.DataFrame:
    """Return zero-shot and the live run's checkpoint rows, by level.

    Rows of `CHECKPOINT_RULE` and the hub target on the case studies'
    development data (`live_checkpoint.tidy`), at every level, each for
    its own panel. A row with no relative WIS keeps it empty.
    """
    out = tidy(scores)
    out = out.loc[out["study"] == "germany"]
    out = out.assign(
        rel_wis=out.get("rel_wis", np.nan),
        rel_log_wis=out.get("rel_log_wis", np.nan),
    )
    live = out["run"] == run if "run" in out else False
    keep = (out["checkpoint"] == "zero-shot") | live
    return out.loc[keep].reset_index(drop=True)


VALIDATION_LEVELS: Final = ("national", "intermediate", "leaf")
"""Levels of the trainer's validation loss (study `held_out` in the score
tables), each reported on its own."""

COMPARISONS: Final = {
    "slots": (
        "the slots and no-slots runs' scores by checkpoint",
        ("all biases", "no biases"),
    ),
    "delays": (
        "the full and no-delay runs' scores by checkpoint",
        ("with delays", "without delays"),
    ),
    "realdata": (
        "the real-data and synthetic fine-tunes' scores",
        ("synthetic", "synthetic and German"),
    ),
}
"""Each fine-tuning comparison of the live run with a variant: what it
waits on until the variant is scored, and the two runs' labels. The
variant is named by its workflow run (the paper config's `comparisons`)."""

BIAS_SUBSETS: Final = {
    "sameonly": "same place only",
    "parentonly": "parent and child only",
}
"""The runs with only some of the attention biases, by their name in the
paper config's `comparisons`, and their labels. Each joins the slots
comparison beside the run without biases, rather than being one of its
own."""


def compared(
    name: str, pairs: Mapping[str, str]
) -> tuple[list[str], tuple[str, ...]]:
    """Return a comparison's variant runs and the labels of all its runs.

    The live run's label comes first. The slots comparison also takes
    each attention bias subset that `pairs` names (`BIAS_SUBSETS`).
    """
    labels = COMPARISONS[name][1]
    others = [pairs[name]]
    if name == "slots":
        for key, label in BIAS_SUBSETS.items():
            if key in pairs:
                others.append(pairs[key])
                labels = (*labels, label)
    return others, labels


def comparison_rows(
    scores: pd.DataFrame,
    run: str,
    other: str | Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Return the live run and its variants, by checkpoint.

    Sets are, one per level, the trainer's validation loss and the relative
    scores of each model on the German first half,
    in the layout of `comparison_figure` (`set`, `run`, `outbreaks`,
    `score`). `labels` name `run` and then each run of `other`, one
    variant or several (`compared`). Empty while no variant has a scored
    checkpoint.
    """
    others = [other] if isinstance(other, str) else list(other)
    rows = tidy(scores)
    names = dict(zip((run, *others), labels, strict=True))
    if not rows["run"].isin(others).any():
        return pd.DataFrame(columns=["set", "run", "outbreaks", "score"])
    rows = rows.loc[rows["run"].isin(names) & (rows["outbreaks"] > 0)]
    loss = rows.loc[
        (rows["study"] == "held_out") & rows["level"].isin(VALIDATION_LEVELS)
    ]
    out = [
        pd.DataFrame(
            {
                "set": "Validation loss, " + loss["level"],
                "run": loss["run"].map(names),
                "outbreaks": loss["outbreaks"],
                "score": loss["loss"],
            }
        )
    ]
    case = rows.loc[rows["study"] == "germany"]
    source = case["model"].map(_source_of)
    out.append(
        pd.DataFrame(
            {
                "set": "German " + case["level"] + ", " + source,
                "run": case["run"].map(names),
                "outbreaks": case["outbreaks"],
                "score": case["rel"],
            }
        )
    )
    return pd.concat(out, ignore_index=True).dropna(subset=["score"])


def slots_summary(
    scores: pd.DataFrame,
    run: str,
    without: str | Sequence[str],
    labels: Sequence[str] = COMPARISONS["slots"][1],
) -> pd.DataFrame:
    """Return zero-shot and each attention bias setup at a shared size.

    `without` is the run without biases, or it and the bias subsets
    (`compared`); `labels` name `run` and then each of them. A setup with
    no scored checkpoint has no row. Per study, each run keeps only
    checkpoints up to the largest number of simulations every scored run
    reached, so the rows compare like with like; a study with no shared
    size is blank.
    """
    others = [without] if isinstance(without, str) else list(without)
    names = dict(zip((run, *others), labels, strict=True))
    rows = tidy(scores)
    kept = [r for r in names if r == run or (rows["run"] == r).any()]
    both = rows.loc[rows["run"].isin(kept)]
    keep = pd.Series(True, index=scores.index)
    for study, g in both.groupby("study"):
        sizes = [set(g.loc[g["run"] == r, "outbreaks"]) for r in kept]
        shared = set.intersection(*sizes)
        top = max(shared) if shared else -1
        drop = (
            (scores["study"] == study)
            & scores["run"].isin(kept)
            & (scores["outbreaks"] > top)
        )
        keep &= ~drop
    matched = scores.loc[keep]
    tables = [summary_table(matched, r) for r in kept]
    out = pd.concat(
        [tables[0], *(t.iloc[[1]] for t in tables[1:])], ignore_index=True
    )
    out = out.rename(columns={"Simulations": "Run"})
    out["Run"] = ["Zero-shot", *(names[r] for r in kept)]
    return out


def live_numbers(scores: pd.DataFrame | None, run: str = LIVE_RUN) -> dict:
    """Return each study's newest checkpoint of the live run and its label.

    `label` is the newest over the studies, for captions; it says the
    scores are not yet available before any checkpoint is scored.
    """
    top = {} if scores is None else newest(scores, run)
    out: dict = {
        study: {"checkpoint": ck, "outbreaks": n, "label": checkpoint_label(n)}
        for study, (ck, n) in top.items()
    }
    out["label"] = (
        checkpoint_label(max(n for _, n in top.values()))
        if top
        else "Fine-tuned checkpoint scores are not yet available."
    )
    return out


MIN_CELLS: Final = 5
"""Panels of the checkpoint figure with fewer matched cells are greyed."""


def checkpoint_numbers(scores: pd.DataFrame) -> dict:
    """Return each panel's cells and each model's relative WIS by size.

    Keys are study and level, then `n` (the fewest matched cells of a
    point), `n_origins` where the rows give it, and each model with its
    `rel_wis` by outbreaks (`zero-shot` for 0), to two decimals.
    """
    out: dict = {}
    for (study, level), one in scores.groupby(["study", "level"]):
        node = out.setdefault(study, {}).setdefault(level, {})
        node["n"] = int(one["n"].min())
        if "n_origins" in one:
            node["n_origins"] = int(one["n_origins"].min())
        for model, g in one.groupby("model"):
            node[model] = {
                ("zero-shot" if k == 0 else f"n{int(k)}"): round(float(v), 2)
                for k, v in zip(g["outbreaks"], g["rel_wis"], strict=True)
            }
    return out


PLANNED_CHECKPOINTS: Final = (1000, 2000, 4000, 8400)
"""The final run's checkpoints, which set the axis before any lands."""


def _source_of(model: str) -> str:
    """Return the model's source (daily or weekly), which sets its colour.

    The `foundation` models read the daily releases; scores stored under
    a name with a source keep the source their name says.
    """
    for source in ("daily", "weekly"):
        if f"_{source}_" in model:
            return source
    return "daily" if model.startswith("foundation") else model


def _family(model: str) -> str:
    """Return the model a `_complete` model is the complete-data form of."""
    return model.removesuffix("_complete")


def _family_colours(models: Sequence[str]) -> dict[str, str]:
    """Return a distinct colour for each model family.

    Each keeps its shared colour (`nowcasteval.labels.MODEL_STYLES`)
    unless an earlier family has it, then takes the next free one.
    """
    out: dict[str, str] = {}
    spare = (f"C{i}" for i in range(10))
    for family in sorted({_family(m) for m in models}):
        colour = MODEL_STYLES.get(family, (None,))[0]
        while colour is None or colour in out.values():
            colour = next(spare)
        out[family] = colour
    return out


def checkpoint_figure(
    scores: pd.DataFrame,
    fig: Figure | SubFigure | None = None,
    legend: bool = True,
) -> Figure | SubFigure:
    """Plot each model's relative WIS against the fine-tune's outbreaks.

    One panel per study and level, titled with its n of matched cells and
    greyed when n is under `MIN_CELLS`. Each fine-tuned model is a line
    over its checkpoints on a log axis of outbreaks, one colour per
    model. The models with no fine-tuning are horizontal lines in the
    same colour: zero-shot Chronos-2 on the reporting-triangle layout
    (dashed) and on complete data (dotted, the `_complete` models), with
    `randomwalknowcast` the line at 1. `scores` has one row per study,
    level, model and checkpoint, with `outbreaks` (0 for zero-shot) and
    `rel_wis` against `randomwalknowcast`.
    Drawn on `fig`, a new figure by default, or a subfigure of a run's
    progress figure (`paperpipe.progress`); `legend` False leaves out the
    shared legend.
    """
    sc = scores.loc[scores["model"] != "randomwalknowcast"]
    sizes = sorted(sc.loc[sc["outbreaks"] > 0, "outbreaks"].unique())
    ticks = sizes or list(PLANNED_CHECKPOINTS)
    # One colour per model across panels, so the shared legend holds; a
    # model on complete data takes its reporting-triangle model's colour.
    colours = _family_colours(sc["model"])
    panels = list(sc.groupby(["study", "level"], sort=False).groups)
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    if fig is None:
        fig = plt.figure(figsize=(3.6 * ncol, 2.8 * nrow), layout="constrained")
    axes = fig.subplots(nrow, ncol, squeeze=False).ravel()
    for ax, (study, level) in zip(axes, panels, strict=False):
        one = sc.loc[(sc["study"] == study) & (sc["level"] == level)]
        n = int(one["n"].min())
        thin = n < MIN_CELLS
        k = int(one["n_origins"].min()) if "n_origins" in one else None
        origins = "" if k is None else f", {k} origin{'s' if k != 1 else ''}"
        for model, g in one.groupby("model"):
            colour = "#bbbbbb" if thin else colours[_family(model)]
            tuned = g.loc[g["outbreaks"] > 0].sort_values("outbreaks")
            if not tuned.empty:
                ax.plot(
                    tuned["outbreaks"],
                    tuned["rel_wis"],
                    marker="o",
                    ms=3,
                    color=colour,
                    label=MODEL_LABELS.get(
                        FINETUNED_OF.get(model, ""),
                        f"{model_label(model)}, fine-tuned",
                    ),
                )
            complete = "_complete" in model
            for value in g.loc[g["outbreaks"] == 0, "rel_wis"]:
                ax.axhline(
                    value,
                    color=colour,
                    ls=":" if complete else "--",
                    lw=1.2,
                    label=model_label(model),
                )
        if thin:
            ax.text(
                0.5,
                0.5,
                f"too few cells (n < {MIN_CELLS})",
                transform=ax.transAxes,
                ha="center",
                color="#888888",
            )
        ax.set_xscale("log")
        ax.set_xlim(ticks[0] / 1.5, ticks[-1] * 1.5)
        ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks], fontsize=7)
        ax.minorticks_off()
        reference = model_label("randomwalknowcast")
        ax.axhline(1, color="#888888", ls="-", lw=1, label=reference)
        ax.set_title(f"{study}, {level} (n = {n}{origins})", fontsize=9)
        ax.set_xlabel("Fine-tuning outbreaks")
        ax.set_ylabel("Relative WIS")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    if not legend:
        return fig
    # The legend from the panels drawn in colour, one entry per line.
    shown = [ax for ax in axes if ax.get_visible() and ax.texts == []]
    entries = {}
    for ax in shown or axes[:1]:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, h)
    labels = sorted(entries)
    handles = [entries[label] for label in labels]
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    return fig


def realism_figure(
    sims: pd.DataFrame,
    reference: pd.DataFrame,
    summaries: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
) -> Figure:
    """Plot each summary of the simulated outbreaks against the data.

    Parameters
    ----------
    sims : pandas.DataFrame
        One row per simulated outbreak, one column per summary.
    reference : pandas.DataFrame
        One row per summary (the index), one column per data set (such
        as `BDBV`), its value for that data set; other columns are
        ignored when they start with `sim`.
    summaries : sequence of str, optional
        Summaries to show. Defaults to those in both tables.
    colours : mapping, optional
        Line colour of each data set.

    Returns
    -------
    matplotlib.figure.Figure
        A histogram per summary with a line per data set.
    """
    data = [c for c in reference.columns if not str(c).startswith("sim")]
    names = list(summaries or [s for s in reference.index if s in sims])
    colours = colours or dict(
        zip(data, ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"], strict=False)
    )
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names, strict=False):
        ax.hist(sims[name].dropna(), bins=30, color="#9e9e9e")
        for d in data:
            ax.axvline(reference.loc[name, d], color=colours[d], lw=1.5)
        ax.set_title(name.replace("_", " "), fontsize=9)
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=colours[d], label=d) for d in data]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(data))
    return fig


GERMANY_KINDS: Final = {
    "national": "Germany national",
    "states": "Germany states",
    "ages": "Germany ages",
}
"""The German development half's series, as the realism columns name them."""

REALISM_QUANTILES: Final = (0.05, 0.5, 0.95)
"""The simulations' quantiles of each summary in the realism table."""


def realism_reference(
    germany: pd.DataFrame, sims: pd.DataFrame
) -> pd.DataFrame:
    """Return each summary's German values and simulated quantiles.

    Parameters
    ----------
    germany : pandas.DataFrame
        The German development half's statistics (`kind`, `statistic`,
        `value`), from the workflow's rule germany_dev_stats.
    sims : pandas.DataFrame
        One row per simulated outbreak of the grid's training rounds, one
        column per summary (rule round_summaries).

    Returns
    -------
    pandas.DataFrame
        One row per summary in both, the index; a column per German series
        (`GERMANY_KINDS`) and `sim_q5`, `sim_q50` and `sim_q95`.
    """
    wide = germany.pivot(index="statistic", columns="kind", values="value")
    wide = wide.reindex(columns=list(GERMANY_KINDS)).rename(
        columns=GERMANY_KINDS
    )
    names = [s for s in wide.index if s in sims]
    values = sims[names].apply(pd.to_numeric, errors="coerce")
    out = wide.loc[names]
    for q in REALISM_QUANTILES:
        out[f"sim_q{round(100 * q)}"] = values.quantile(q).to_numpy()
    out.index.name = None
    out.columns.name = None
    return out


def realism_numbers(reference: pd.DataFrame) -> dict:
    """Return each summary's simulated quantiles and data values.

    Keys are the summaries, then the columns with spaces as underscores
    (`sim_q95`, `BDBV`, `Germany_states`), rounded to three figures.
    """
    return {
        str(name): {
            str(col).replace(" ", "_"): float(f"{value:.3g}")
            for col, value in row.items()
            if pd.notna(value)
        }
        for name, row in reference.iterrows()
    }


def diseases_table(diseases: pd.DataFrame) -> pd.DataFrame:
    """Return the simulated ranges against published values, with DOIs."""
    sims = [
        f"{a:.3g} [{b:.3g}, {c:.3g}]"
        for a, b, c in zip(
            diseases["sim_q50"],
            diseases["sim_q5"],
            diseases["sim_q95"],
            strict=True,
        )
    ]
    sources = [
        "; ".join(f"[{d}](https://doi.org/{d})" for d in str(s).split("; "))
        for s in diseases["sources"]
    ]
    return pd.DataFrame(
        {
            "Summary": diseases["summary"],
            "Simulated median [5%, 95%]": sims,
            "Published": diseases["published"],
            "Sources": sources,
        }
    )


EVIDENCE: Final = {
    "si_copula": "the walks' scores, no pairing against rolling rho_p",
    "copula_rho": "the walk models' German rho_p by nowcast date",
    "sim_summaries": "per-outbreak summaries of the final grid's simulations",
    "sim_realism": "the German development half's values",
    "sim_diseases": "the published disease values",
    "checkpoint_scores": "the scores of each checkpoint of the final run",
    "compare_daily": "our models' daily and hub-target scores by checkpoint",
    "block_combinations": ("the realised shares of block layouts and options"),
}
"""Each evidence input and what it waits on until set.

The `compare_*` files share one layout: `set` (a validation source, or a
case study and level), `run`, `outbreaks` (the run's checkpoint) and
`score` (validation loss, or relative WIS or CRPS against
`randomwalknowcast`)."""


NOT_YET: Final = "(not yet available)"
"""What the paper quotes in place of a number whose rows are not set."""

COPULA_MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their province draws by rho_p."""

QUOTED: Final = {
    "rho": {
        m: {"rho_p": NOT_YET, "origins": NOT_YET, "fallback": NOT_YET}
        for m in COPULA_MODELS
    },
    "realism": {
        "zero_share": {"sim_q95": NOT_YET, "Germany_states": NOT_YET},
        "delay_q99": {"sim_q95": NOT_YET, "Germany_states": NOT_YET},
    },
}
"""The numbers the paper quotes, as they read until their rows exist."""


COPULA_STUDIES: Final = {
    "germany": "Germany",
    "synthetic": "Synthetic",
}
"""The copula figure's studies, in its row order, all on development data."""

COPULA_LEVELS: Final = ("national", "provinces", "states")
"""The levels the copula changes: the country, which is the sum of the
provinces (or states), and the provinces themselves. Zones are not
paired."""

COPULA_METRICS: Final = {
    "cell_crps": "CRPS",
    "cell_cover90": "90% coverage",
    "joint_log_energy": "ES (log)",
}
"""The copula figure's scores: CRPS and 90% coverage of each cell, which
at the national level is the sum of the provinces, and the energy score
on log counts of the level's places and weeks as one vector."""


def rho_medians(rho: pd.DataFrame) -> dict:
    """Return each walk model's median rolling rho_p between German states.

    `rho` has the walk models' German rows (`nowcasteval.pairing.
    RHO_COLUMNS`): `model`, `origin`, `level`, `n_scores`,
    `rho_provinces`, `n_pairs` and `fallback`, one per task. Only the
    states tasks count, as rho_p pairs provinces. Tasks that fell back to
    the prior for want of pairs are counted, not in the median.
    """
    rho = rho.loc[rho["level"] == "states"]
    fallback = rho["fallback"].astype(str).str.lower() == "true"
    out: dict = {}
    for model, g in rho.groupby("model", sort=False):
        own = fallback.loc[g.index]
        est = g.loc[~own, "rho_provinces"]
        out[model] = {
            "rho_p": round(float(est.median()), 2) if len(est) else NOT_YET,
            "origins": int(g["origin"].nunique()),
            "fallback": int(own.sum()),
        }
    return out


def copula_figure(rows: pd.DataFrame) -> Figure:
    """Plot each score of the rolling copula relative to no pairing.

    One panel per study and level in `COPULA_LEVELS`, never pooled across
    levels. Each panel gives the ratio rolling / off of every score in
    `COPULA_METRICS`, one marker per model, with 1 no pairing. `rows` are
    `evaluation.pairing.summarise_pairing`'s: `study`, `level`, `model`,
    `variant` (`off` or `rolling`), `target`, `metric`, `value` and `n`.
    """
    rows = rows.assign(metric=rows["target"] + "_" + rows["metric"])
    rows = rows.loc[
        rows["metric"].isin(list(COPULA_METRICS))
        & rows["level"].isin(COPULA_LEVELS)
    ]
    wide = rows.pivot_table(
        index=["study", "level", "model", "metric"],
        columns="variant",
        values="value",
    ).reset_index()
    wide["ratio"] = wide["rolling"] / wide["off"]
    n = rows.groupby(["study", "level"])["n"].min()
    studies = [s for s in COPULA_STUDIES if s in set(wide["study"])]
    levels = {
        s: list(dict.fromkeys(rows.loc[rows["study"] == s, "level"]))
        for s in studies
    }
    ncol = max(len(v) for v in levels.values())
    fig, axes = plt.subplots(
        len(studies),
        ncol,
        figsize=(3.4 * ncol, 2.6 * len(studies)),
        layout="constrained",
        squeeze=False,
    )
    metrics = list(COPULA_METRICS)
    models = sorted(wide["model"].unique())
    markers = dict(zip(models, "osD^v", strict=False))
    for i, study in enumerate(studies):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(levels[study]):
                ax.set_visible(False)
                continue
            level = levels[study][j]
            one = wide.loc[(wide["study"] == study) & (wide["level"] == level)]
            for model, g in one.groupby("model"):
                x = [metrics.index(m) for m in g["metric"]]
                ax.plot(
                    x,
                    g["ratio"],
                    ls="none",
                    marker=markers[model],
                    label=model_label(model),
                )
            ax.axhline(1, color="#888888", lw=1)
            ax.set_xticks(
                range(len(metrics)),
                [COPULA_METRICS[m] for m in metrics],
                fontsize=7,
            )
            ax.set_title(
                f"{COPULA_STUDIES[study]}, {level} "
                f"(n = {int(n.loc[(study, level)])})",
                fontsize=9,
            )
            ax.set_ylabel("Rolling / no pairing")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(models))
    return fig


def comparison_figure(rows: pd.DataFrame) -> Figure:
    """Plot a fine-tuning comparison: each run's score by checkpoint.

    One panel per set, in the order the rows give, one line per run on a
    log axis of outbreaks.
    """
    sets = list(dict.fromkeys(rows["set"]))
    runs = sorted(rows["run"].unique())
    colours = {r: f"C{i}" for i, r in enumerate(runs)}
    ncol = min(3, len(sets))
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.6 * nrow),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, name in zip(axes, sets, strict=False):
        one = rows.loc[rows["set"] == name]
        for run, g in one.groupby("run"):
            g = g.sort_values("outbreaks")
            ax.plot(
                g["outbreaks"],
                g["score"],
                marker="o",
                ms=3,
                color=colours[run],
                label=run,
            )
        ax.set_xscale("log")
        ticks = sorted(one["outbreaks"].unique())
        ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks], fontsize=7)
        ax.minorticks_off()
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Fine-tuning outbreaks")
        ax.set_ylabel("Score")
    for ax in axes[len(sets) :]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(runs))
    return fig


def _read(paths: Mapping[str, Path | None], name: str) -> pd.DataFrame | None:
    path = paths.get(name)
    if path is None:
        return None
    return pd.read_csv(path, index_col=0 if name == "sim_realism" else None)


def write_evidence(
    paths: Mapping[str, Path | None],
    si_dir: Path,
    fig_dir: Path,
    out_yml: Path,
    run: str = LIVE_RUN,
    comparisons: Mapping[str, str] | None = None,
) -> dict:
    """Write every evidence table and figure, and the numbers they quote.

    Parameters
    ----------
    paths : mapping
        `EVIDENCE` name to its file, or None while it is not set.
    si_dir, fig_dir : pathlib.Path
        Folders for the SI tables and figures.
    out_yml : pathlib.Path
        The quoted numbers, as Quarto metadata under `evidence`.
    run : str
        The live fine-tuning run, whose scored checkpoints the curve, the
        tables and the comparisons show as preliminary.
    comparisons : mapping, optional
        `COMPARISONS` name to the variant run compared with `run`; a
        comparison not named, or whose variant has no scored checkpoint,
        waits.

    Returns
    -------
    dict
        The quoted numbers.
    """
    si_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {name: _read(paths, name) for name in EVIDENCE}
    for name, frame in data.items():
        if frame is not None and name != "sim_realism":
            kept = drop_bdbv(frame)
            data[name] = None if kept.empty else kept
    raw = data["checkpoint_scores"]
    pairs = dict(comparisons or {})
    for name in COMPARISONS:
        rows = None
        if raw is not None and name in pairs:
            rows = comparison_rows(raw, run, *compared(name, pairs))
        data[f"compare_{name}"] = None if rows is None or rows.empty else rows
    waits = EVIDENCE | {f"compare_{n}": d for n, (d, _) in COMPARISONS.items()}

    def table(name: str, frame: pd.DataFrame | None, needs: list[str]) -> None:
        # Rows for another study only still leave this one waiting.
        text = (
            to_markdown(frame)
            if frame is not None and not frame.empty
            else waiting(", ".join(waits[n] for n in needs))
        )
        (si_dir / f"{name}.md").write_text(text)

    def figure(name: str, fig: Figure | None, needs: list[str]) -> None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                waiting(waits[needs[0]]).strip("*\n"),
                ha="center",
                va="center",
            )
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    cop = data["si_copula"]
    figure(
        "copula-impact",
        None if cop is None else copula_figure(cop),
        ["si_copula"],
    )
    dis = data["sim_diseases"]
    table(
        "sim-diseases",
        None if dis is None else diseases_table(dis),
        ["sim_diseases"],
    )
    ckpt = None if raw is None else checkpoint_rows(raw, run)
    table(
        "checkpoint-table",
        None if raw is None else checkpoint_table(raw, run),
        ["checkpoint_scores"],
    )
    table(
        "live-summary",
        None if raw is None else summary_table(raw, run),
        ["checkpoint_scores"],
    )
    table(
        "slots-summary",
        None
        if data["compare_slots"] is None
        else slots_summary(raw, run, *compared("slots", pairs)),
        ["compare_slots"],
    )
    table(
        "block-combinations",
        data["block_combinations"],
        ["block_combinations"],
    )
    for name in ("slots", "delays", "realdata", "daily"):
        rows = data[f"compare_{name}"]
        figure(
            f"compare-{name}",
            None if rows is None else comparison_figure(rows),
            [f"compare_{name}"],
        )
    sims, ref = data["sim_summaries"], data["sim_realism"]
    figure(
        "sim-realism",
        None if sims is None or ref is None else realism_figure(sims, ref),
        ["sim_summaries", "sim_realism"],
    )
    rho = data["copula_rho"]
    numbers = {
        "evidence": {
            "rho": QUOTED["rho"] if rho is None else rho_medians(rho),
            "checkpoints": (None if ckpt is None else checkpoint_numbers(ckpt)),
            "live": live_numbers(raw, run),
            "realism": (
                QUOTED["realism"] if ref is None else realism_numbers(ref)
            ),
            "waiting": [waits[n] for n, d in data.items() if d is None],
        }
    }
    out_yml.write_text(yaml.safe_dump(numbers, sort_keys=False))
    return numbers
