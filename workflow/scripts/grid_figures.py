"""Figures comparing the per-outbreak summaries of simulated rounds.

Reads `grid_summaries.py outbreaks` tables, one per source, and draws:

- `<out>/compare-ecdf.png`: each numeric summary's ECDF by source, one
  panel per summary, grouped by layer (size, dynamics, R, delays, phase);
- `<out>/compare-shares.png`: the shares of each pattern and phase at the
  cut-offs, by source.

    uv run python workflow/scripts/grid_figures.py <out_dir> \
        <name>=<outbreaks.csv> [<name>=<outbreaks.csv> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
"""Categorical slots 1-4 of the reference palette, in fixed order."""

PANELS = [
    ("infections", "Infections", True),
    ("reported", "Reported cases", True),
    ("peak_weekly", "Peak weekly reported", True),
    ("attack", "Attack rate", False),
    ("max_growth_14d", "Fastest 14-day growth (/day)", False),
    ("min_growth_14d", "Fastest 14-day decline (/day)", False),
    ("half_to_peak", "Half-peak to peak (days)", False),
    ("peak_to_half", "Peak to half-peak (days)", False),
    ("r_q05", "R, 5% over days", False),
    ("r_q95", "R, 95% over days", True),
    ("reff_q05", "Effective R, 5% over days", False),
    ("reff_q95", "Effective R, 95% over days", True),
    ("gt_mean", "Generation time mean (days)", False),
    ("incubation_mean", "Incubation mean (days)", False),
    ("mean_delay", "Mean reporting delay (days)", False),
    ("reported_7d", "Share reported within 7 days", False),
]


def ecdf_panels(sources: dict[str, pd.DataFrame], out: Path) -> None:
    """Draw each summary's ECDF by source."""
    cols = 4
    rows = -(-len(PANELS) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    for ax, (col, title, log) in zip(axes.flat, PANELS, strict=False):
        for (name, d), colour in zip(sources.items(), COLOURS, strict=False):
            v = pd.to_numeric(d.get(col), errors="coerce").dropna()
            if log:
                v = v[v > 0]
            if not len(v):
                continue
            x = np.sort(v.to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(
                x,
                y,
                where="post",
                color=colour,
                lw=2,
                label=f"{name} (n={len(x)})",
            )
        if log:
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(PANELS) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(sources), frameon=False
    )
    fig.supylabel("Share of outbreaks at or below")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def share_panels(sources: dict[str, pd.DataFrame], out: Path) -> None:
    """Draw pattern shares and mean phase shares at cut-offs by source."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    names = list(sources)
    width = 0.8 / len(names)
    patterns = ["wave", "multi-wave", "rising", "falling", "flat", "low"]
    phases = ["phase_rising", "phase_falling", "phase_flat", "phase_low"]
    for i, (name, d) in enumerate(sources.items()):
        share = d["pattern"].value_counts(normalize=True)
        x = np.arange(len(patterns)) + i * width
        a.bar(
            x,
            [share.get(p, 0) for p in patterns],
            width * 0.92,
            color=COLOURS[i],
            label=name,
        )
        means = [pd.to_numeric(d[p], errors="coerce").mean() for p in phases]
        b.bar(
            np.arange(len(phases)) + i * width,
            means,
            width * 0.92,
            color=COLOURS[i],
            label=name,
        )
    centre = width * (len(names) - 1) / 2
    a.set_xticks(np.arange(len(patterns)) + centre, patterns)
    a.set_title("Pattern of expected onsets (share of outbreaks)")
    b.set_xticks(
        np.arange(len(phases)) + centre,
        [p.removeprefix("phase_") for p in phases],
    )
    b.set_title("Phase at the weekly cut-offs (mean share)")
    for ax in (a, b):
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str]) -> None:
    """Draw the figures from the command line."""
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        k: pd.read_csv(v) for k, v in (a.split("=", 1) for a in argv[1:])
    }
    ecdf_panels(sources, out / "compare-ecdf.png")
    share_panels(sources, out / "compare-shares.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
