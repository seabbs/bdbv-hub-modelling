"""A fine-tuning run's progress over its checkpoints, as one SI figure.

Each run in the paper (`PaperFlow.progress_runs`) has one figure with the
same panels, from that run's own tables (workflow rules run_progress and
horizon_table, `flowlib.progress`), so no run waits on another:

- loss: the trainer's training loss by step, the validation loss by
  source at each evaluated step and by level at the start and at each
  checkpoint, with the checkpoints marked;
- the 7-day check by checkpoint: the 7-day row's median over the 7-day
  sums of the daily medians, the share of those sums in the row's 90%
  interval, and the roughness of its median over the truth's;
- the BDBV check by checkpoint, relative CRPS against
  `randomwalknowcast`, in a paper run only, as the Ebola study is held
  out of development runs;
- German relative WIS by checkpoint and level, on every horizon, the
  nowcasts and the forecasts, as `paperpipe.evidence.checkpoint_figure`
  draws them, with zero-shot Chronos-2 on the reporting-triangle layout
  and on complete data as reference lines.

A panel whose rows do not exist yet says so.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nowcasteval.labels import model_label

from paperpipe.evidence import checkpoint_figure

__all__ = ["HORIZON_TITLES", "progress_figure"]

HORIZON_TITLES: Final = {
    "all": "German WIS, every horizon",
    "nowcast": "German WIS, nowcasts (days 0 and before)",
    "forecast": "German WIS, forecasts (days after)",
}
"""The German rows of the figure, by part of the horizons
(`nowcastgermany.checkpoint.HORIZON_PARTS`)."""

SMOOTH: Final = 50
"""Training loss is shown as a rolling mean over this many logged steps."""

SEVEN_DAY: Final = {
    "ratio": ("7-day / summed daily medians", 1.0),
    "inside90": ("Summed daily in 7-day 90% interval", 0.9),
    "rough_ratio": ("Roughness / truth's", 1.0),
}
"""The 7-day check's measures, with each one's target value."""

BDBV_TUNED: Final = "foundation"
"""The BDBV check's fine-tuned model; the others are references."""


def _waiting(ax: Axes, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#888888",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _log_outbreaks(ax: Axes, sizes: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{t:,.0f}" for t in sizes], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("Fine-tuning outbreaks")


def _loss_panel(ax: Axes, loss: pd.DataFrame) -> None:
    if loss.empty:
        _waiting(ax, "No losses logged yet")
        ax.set_title("Loss", fontsize=9)
        return
    train = loss.loc[loss["kind"] == "training"].sort_values("step")
    if not train.empty:
        ax.plot(
            train["step"],
            train["loss"].rolling(SMOOTH, min_periods=1).mean(),
            color="#555555",
            lw=1,
            label=f"Training (mean of {SMOOTH} logged steps)",
        )
    val = loss.loc[loss["kind"] == "validation"]
    for i, (source, g) in enumerate(val.groupby("series", sort=True)):
        g = g.sort_values("step")
        ax.plot(
            g["step"],
            g["loss"],
            color=f"C{i}",
            lw=1.2,
            label=f"Validation, {source}",
        )
    level = loss.loc[loss["kind"] == "validation by level"]
    markers = dict(zip(sorted(level["series"].unique()), "osD^v", strict=False))
    for name, g in level.groupby("series", sort=True):
        ax.plot(
            g["step"],
            g["loss"],
            ls="none",
            marker=markers[name],
            ms=4,
            color="black",
            label=f"Validation by level, {name}",
        )
    for _, mark in loss.loc[loss["kind"] == "checkpoint"].iterrows():
        ax.axvline(mark["step"], color="#bbbbbb", lw=0.8, ls=":")
        ax.text(
            mark["step"],
            1.0,
            f"{int(mark['outbreaks']):,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6,
            color="#888888",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training step (checkpoints by outbreaks)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss", fontsize=9, pad=12)
    ax.legend(frameon=False, fontsize=6, ncol=2)


def _seven_day_panel(ax: Axes, rows: pd.DataFrame) -> None:
    ax.set_title("7-day check", fontsize=9)
    if rows.empty:
        _waiting(ax, "No checkpoint checked yet")
        return
    rows = rows.sort_values("outbreaks")
    for i, (name, (label, target)) in enumerate(SEVEN_DAY.items()):
        ax.plot(
            rows["outbreaks"],
            rows[name],
            marker="o",
            ms=3,
            color=f"C{i}",
            label=label,
        )
        ax.axhline(target, color=f"C{i}", lw=0.8, ls="--")
    _log_outbreaks(ax, sorted(rows["outbreaks"].unique()))
    ax.set_ylabel("Value (dashed: target)")
    ax.legend(frameon=False, fontsize=6)


def _bdbv_panel(ax: Axes, rows: pd.DataFrame | None) -> None:
    ax.set_title("BDBV check, relative CRPS", fontsize=9)
    if rows is None:
        _waiting(ax, "Held out until the paper run")
        return
    if rows.empty:
        _waiting(ax, "No checkpoint checked yet")
        return
    tuned = rows.loc[rows["model"] == BDBV_TUNED].sort_values("outbreaks")
    ax.plot(
        tuned["outbreaks"],
        tuned["relative_crps"],
        marker="o",
        ms=3,
        color="C0",
        label=f"{model_label(BDBV_TUNED)}, fine-tuned",
    )
    others = rows.loc[rows["model"] != BDBV_TUNED]
    for i, (model, g) in enumerate(others.groupby("model", sort=True)):
        ax.axhline(
            float(np.mean(g["relative_crps"])),
            color=f"C{i + 1}",
            ls="--",
            lw=1,
            label=model_label(model),
        )
    _log_outbreaks(ax, sorted(rows["outbreaks"].unique()))
    ax.set_ylabel("Relative CRPS")
    ax.legend(frameon=False, fontsize=6)


def _by_share(horizons: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Put a run on real data alone on the axis of its share of passes.

    Its checkpoints have no outbreaks, and their labels `n<N>` are
    thousandths of its passes, which become their `outbreaks`. Any other
    run is returned as it is.
    """
    if "checkpoint" not in horizons:
        return horizons, False
    tuned = horizons["checkpoint"].astype(str) != "zero-shot"
    if not tuned.any() or (horizons.loc[tuned, "outbreaks"] > 0).any():
        return horizons, False
    out = horizons.copy()
    out.loc[tuned, "outbreaks"] = (
        out.loc[tuned, "checkpoint"].astype(str).str[1:].astype(int)
    )
    return out, True


def progress_figure(
    loss: pd.DataFrame,
    horizons: pd.DataFrame,
    seven_day: pd.DataFrame,
    bdbv: pd.DataFrame | None = None,
    run: str = "",
) -> Figure:
    """Plot one run's fine-tuning progress over its checkpoints.

    Parameters
    ----------
    loss : pandas.DataFrame
        `flowlib.progress.loss_rows`: `kind`, `series`, `step`,
        `outbreaks` and `loss`.
    horizons : pandas.DataFrame
        `nowcastgermany.checkpoint.horizon_table`: the run's checkpoints
        and zero-shot by `horizons`, level and model, with `outbreaks`,
        `n` and `rel_wis`.
    seven_day : pandas.DataFrame
        `flowlib.progress.seven_day_rows`, by `outbreaks`.
    bdbv : pandas.DataFrame, optional
        `flowlib.progress.bdbv_rows`; None while the Ebola study is held
        out.
    run : str
        The run's name, the figure's title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    parts = [
        p for p in HORIZON_TITLES if p in set(horizons.get("horizons", []))
    ]
    fig = plt.figure(
        figsize=(11, 3.2 * (1 + max(len(parts), 1))), layout="constrained"
    )
    if run:
        fig.suptitle(run, fontsize=11)
    rows = fig.subfigures(
        1 + max(len(parts), 1),
        1,
        height_ratios=[1] * max(len(parts), 1) + [1.15],
    )
    top = rows[0].subplots(1, 3, width_ratios=[1.6, 1, 1])
    _loss_panel(top[0], loss)
    _seven_day_panel(top[1], seven_day)
    _bdbv_panel(top[2], bdbv)
    if not parts:
        ax = rows[1].subplots()
        _waiting(ax, "No German checkpoint scores yet")
        return fig
    horizons, share = _by_share(horizons)
    for i, part in enumerate(parts):
        sub = rows[1 + i]
        one = horizons.loc[horizons["horizons"] == part]
        checkpoint_figure(one, fig=sub, legend=i == len(parts) - 1, share=share)
        sub.suptitle(HORIZON_TITLES[part], fontsize=9)
    return fig
