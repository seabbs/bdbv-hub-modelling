# ruff: noqa: E501 (the page's HTML and figure titles are long lines)
"""Write a review page of what the release layer does to training blocks.

    release_layer_page.py SIMS GRID LAYER BDBV_DAILY GERMANY OUT
        [--sims-n 20] [--workers 4]

Cuts the same simulations twice, as Run A cuts them (its grid, seed,
blocks per simulation, block limit and quiet floor), without and with the
release layer (`finetune.chronos.RELEASE_LAYER`), and sets them beside a
real BDBV block (the hub's daily releases) and German blocks (daily
releases): example blocks, the drawn delays and cadences, each block's
lag-row fill and its lag count.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nowcastfoundation import forecaster as foundation
from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import stream as fs
from nowcastfoundation.finetune import training_block
from nowcastutils.base import NATIONAL

RUN_A = {
    "seed": 1,
    "per_simulation": 32,
    "block_limit": 1000,
    "quiet": 10,
    "loss_days": 7,
}
"""Run A's cut settings (its held-out key)."""

PLAIN = ("daily", "weekly", "3day")
"""Sources whose covariate rows are lag series alone, so a block's fill
and lag count read straight off its rows."""

INK, MUTED, BEFORE, AFTER, REAL = (
    "#1f2328",
    "#6e7781",
    "#8c959f",
    "#0969da",
    "#cf222e",
)


def _init(grid: str, layer: dict | None) -> None:
    fs.init_worker(
        2000,
        ft.SOURCES,
        0.2,
        1,
        1,
        grid,
        RUN_A["loss_days"],
        RUN_A["quiet"],
        RUN_A["block_limit"],
        "ragged",
        layer,
    )


def _cut(args: tuple[str, str, dict | None, str]) -> list[tuple[str, dict]]:
    sims, grid, layer, sim = args
    _init(grid, layer)
    return fs.cut(Path(sims), [sim], RUN_A["seed"], RUN_A["per_simulation"])


def cut_both(sims: Path, grid: Path, layer: dict, ids: list[str], workers: int):
    """Return each simulation's blocks without and with the layer."""
    out = {}
    with ProcessPoolExecutor(workers) as pool:
        for name, lay in (("before", None), ("after", layer)):
            jobs = [(str(sims), str(grid), lay, s) for s in ids]
            out[name] = [p for part in pool.map(_cut, jobs) for p in part]
    return out


def block_rows(block: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return a block's first place's counts row and lag rows to its cut."""
    # The context ends at the last complete step; the steps after it, up
    # to the cut-off, are the nowcast part of the horizon.
    x = block["context"][:, : ft.CONTEXT + block["nowcast"]]
    keep = np.isfinite(x[0])
    n_targets = block["n_targets"]
    lags = block["n_covariates"] // max(n_targets, 1)
    counts = x[0, keep]
    lag_rows = x[n_targets : n_targets + lags][:, keep]
    return counts, lag_rows


def fill(lag_rows: np.ndarray) -> float:
    """Return the share of a block's lag cells with a value."""
    return float(np.isfinite(lag_rows).mean()) if lag_rows.size else np.nan


def table(pairs: list[tuple[str, dict]], part: str) -> pd.DataFrame:
    """Return one row a block: its source, cadence, delay, fill and D."""
    rows = []
    for outbreak, b in pairs:
        if b["source"] not in PLAIN:
            continue
        counts, lag_rows = block_rows(b)
        width = ft.DAYS_PER_STEP[b["source"]]
        cut = dt.date.fromisoformat(b["cut_date"])
        first = dt.date.fromisoformat(b["first_release"])
        rows.append(
            {
                "part": part,
                "outbreak": outbreak,
                "source": b["source"],
                "cut": b["cut"],
                "view": b["view"].get("kind", "whole"),
                "cadence": b["cadence"],
                "start_days": b["release_start_days"],
                "releases": b["releases"],
                "steps": len(counts),
                "D": lag_rows.shape[0],
                "history_D": max(1, (cut - first).days // width),
                "fill": fill(lag_rows),
            }
        )
    return pd.DataFrame(rows)


def bdbv(daily: Path) -> dict:
    """Return the BDBV hub's latest daily block: rows, fill, D, releases."""
    rows = pd.read_csv(daily)
    rows = rows.loc[rows["location"] == NATIONAL]
    last = str(rows["onset_date"].max())
    tri = foundation.hub_triangle(
        rows,
        "daily",
        (NATIONAL,),
        last,
        max_lag=foundation.LAG_WINDOW["daily"][1],
    )
    d = foundation.lag_count(tri, "daily")
    lag_rows = tri.partial[0, : tri.last + 1, :d].T
    return {
        "counts": tri.final[0, : tri.last + 1],
        "lag_rows": lag_rows,
        "fill": fill(lag_rows),
        "filled": int(np.isfinite(lag_rows[0]).sum()),
        "steps": tri.last + 1,
        "D": d,
        "releases": rows["vintage"].nunique(),
        "first_release": str(rows["vintage"].min()),
        "cut": str(rows["vintage"].max()),
    }


def germany(triangle: Path, cuts: list[str]) -> pd.DataFrame:
    """Return the fill and D of German national blocks at `cuts`."""
    from nowcastgermany import finetune_blocks as gb

    as_of = max(cuts)
    counts = gb.german_counts(pd.read_csv(triangle), as_of)
    own = counts.loc[~counts["location"].str.contains("-A")]
    out = []
    for cut in cuts:
        cutoffs = sorted(d for d in own["report_date"].unique() if d <= cut)
        b = training_block(own, "daily", (NATIONAL,), cutoffs, ahead_weeks=0)
        if b is None:
            continue
        # To the cut-off, as `block_rows` reads the simulated blocks.
        x = b["context"][:, : b["start"] + b["nowcast"]]
        lag_rows = x[b["n_targets"] :][:, np.isfinite(x[0])]
        out.append({"cut": cut, "fill": fill(lag_rows), "D": lag_rows.shape[0]})
    return pd.DataFrame(out)


def png(fig) -> str:
    """Return a figure as base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _rows_axis(ax, counts, lag_rows, title):
    grid = np.vstack([counts[None, :], lag_rows])
    shown = np.log1p(np.where(np.isfinite(grid), np.maximum(grid, 0), np.nan))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f6f8fa")
    ax.imshow(shown, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_yticks([0, len(grid) - 1], ["counts", f"lag {len(grid) - 2}"])
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xlabel("step (onset)", fontsize=7, color=MUTED)


def examples(before, after, real, n=3):
    """Return the example figure: blocks before and after, and BDBV."""
    by = {
        (o, b["source"], b["cut"], b["view"].get("kind")): b for o, b in before
    }
    picks = [
        (k, b)
        for o, b in after
        if b["source"] == "daily"
        and b["release_start_days"] > 30
        and b["releases"] > 3
        and (k := (o, b["source"], b["cut"], b["view"].get("kind"))) in by
    ]
    picks = sorted(picks, key=lambda kb: -kb[1]["release_start_days"])[
        :: max(1, len(picks) // n)
    ][:n]
    fig, axes = plt.subplots(
        len(picks) + 1, 2, figsize=(11, 2.1 * (len(picks) + 1))
    )
    for row, (k, b) in zip(axes, picks, strict=False):
        a = by[k]
        ca, la = block_rows(a)
        cb, lb = block_rows(b)
        _rows_axis(
            row[0],
            ca,
            la,
            f"{k[0]} {k[1]} cut {k[2]}: before ({a['cadence']}, {a['releases']} releases, fill {fill(la):.0%})",
        )
        _rows_axis(
            row[1],
            cb,
            lb,
            f"after: releases start {b['release_start_days']} days late, first {b['first_release']} ({b['cadence']}, {b['releases']} releases, fill {fill(lb):.0%})",
        )
    _rows_axis(
        axes[-1][0],
        real["counts"],
        real["lag_rows"],
        f"BDBV hub, CD daily, releases {real['first_release']} to {real['cut']} ({real['releases']} releases, fill {real['fill']:.0%}, D {real['D']})",
    )
    axes[-1][1].axis("off")
    fig.tight_layout()
    return png(fig)


def distributions(t, real, de):
    """Return the distributions figure."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    late = t[(t.part == "after") & (t.start_days > 0)]
    ax[0, 0].hist(late["start_days"], bins=30, color=AFTER)
    ax[0, 0].set_title(
        f"release-start delay of late blocks (days), n={len(late)}",
        fontsize=9,
        loc="left",
    )
    shares = []
    for part in ("before", "after"):
        p = t[t.part == part]
        daily = p[p.source == "daily"]
        shares.append(
            (
                part,
                (p.start_days > 0).mean(),
                (daily.cadence == "weekly").mean(),
            )
        )
    x = np.arange(2)
    for i, (part, s_late, s_weekly) in enumerate(shares):
        ax[0, 1].bar(
            x + 0.38 * i,
            [s_late, s_weekly],
            0.36,
            color=[BEFORE, AFTER][i],
            label=part,
        )
        for xi, v in zip(x + 0.38 * i, (s_late, s_weekly), strict=True):
            ax[0, 1].text(
                xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK
            )
    ax[0, 1].set_xticks(
        x + 0.19, ["blocks starting late", "daily blocks released weekly"]
    )
    ax[0, 1].legend(fontsize=8, frameon=False)
    ax[0, 1].set_title("shares of blocks", fontsize=9, loc="left")
    bins = np.linspace(0, 1, 26)
    for part, colour in (("before", BEFORE), ("after", AFTER)):
        p = t[(t.part == part) & (t.source == "daily")]
        ax[1, 0].hist(
            p["fill"],
            bins=bins,
            histtype="step",
            lw=2,
            color=colour,
            label=f"{part} (daily blocks)",
        )
    ax[1, 0].axvline(
        real["fill"], color=REAL, lw=2, label=f"BDBV {real['fill']:.0%}"
    )
    if len(de):
        ax[1, 0].axvline(
            de["fill"].median(),
            color=INK,
            ls="--",
            lw=1.5,
            label=f"Germany {de['fill'].median():.0%}",
        )
    ax[1, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].set_title(
        "lag-row fill: share of lag cells with a value", fontsize=9, loc="left"
    )
    for part, colour in (("before", BEFORE), ("after", AFTER)):
        p = t[(t.part == part) & (t.source == "daily")]
        ax[1, 1].hist(
            p["D"],
            bins=30,
            histtype="step",
            lw=2,
            color=colour,
            label=f"{part}: rule's D",
        )
    a = t[(t.part == "after") & (t.source == "daily") & (t.start_days > 0)]
    ax[1, 1].hist(
        a["history_D"],
        bins=30,
        histtype="step",
        lw=1.5,
        ls=":",
        color=AFTER,
        label="late blocks: release history",
    )
    ax[1, 1].axvline(real["D"], color=REAL, lw=2, label=f"BDBV D {real['D']}")
    ax[1, 1].legend(fontsize=8, frameon=False)
    ax[1, 1].set_title(
        "lag count D (days), daily blocks", fontsize=9, loc="left"
    )
    for a_ in ax.flat:
        a_.tick_params(labelsize=7, colors=MUTED)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    return png(fig)


def main() -> None:
    """Cut the blocks both ways and write the page."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sims", "grid", "layer", "bdbv_daily", "germany", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--sims-n", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    layer = ft.read_release_layer(a.layer)
    held = (
        (a.sims.parent / "train" / "outbreaks-held-out.txt").read_text().split()
    )
    ids = held[: a.sims_n]
    cuts = cut_both(a.sims, a.grid, layer, ids, a.workers)
    t = pd.concat([table(cuts[k], k) for k in ("before", "after")])
    real = bdbv(a.bdbv_daily)
    de = germany(
        a.germany, ["2021-06-01", "2021-09-01", "2021-12-01", "2022-03-01"]
    )
    ex = examples(cuts["before"], cuts["after"], real)
    dist = distributions(t, real, de)
    summary = t.groupby(["part", "source"]).agg(
        blocks=("fill", "size"),
        late=("start_days", lambda s: (s > 0).mean()),
        weekly=("cadence", lambda s: (s == "weekly").mean()),
        fill=("fill", "median"),
        D=("D", "median"),
    )
    daily = t[t.source == "daily"].assign(late=lambda d: d.start_days > 0)
    breakdown = daily.groupby(["part", "cadence", "late"]).agg(
        blocks=("fill", "size"),
        fill=("fill", "median"),
        D=("D", "median"),
        history_D=("history_D", "median"),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        page(layer, ids, ex, dist, summary, breakdown, real, de, len(t))
    )
    t.to_csv(a.out.with_suffix(".csv"), index=False)
    print("wrote", a.out, file=sys.stderr)


def page(layer, ids, ex, dist, summary, breakdown, real, de, n) -> str:
    """Return the page's HTML."""

    def pct(v):
        return f"{v:.0%}"

    rows = "".join(
        f"<tr><td>{part}</td><td>{src}</td><td>{r.blocks}</td><td>{pct(r.late)}</td>"
        f"<td>{pct(r.weekly)}</td><td>{pct(r.fill)}</td><td>{r.D:.0f}</td></tr>"
        for (part, src), r in summary.iterrows()
    )
    split = "".join(
        f"<tr><td>{part}</td><td>{cad}</td><td>{'yes' if late else 'no'}</td>"
        f"<td>{r.blocks}</td><td>{pct(r.fill)}</td><td>{r.D:.0f}</td>"
        f"<td>{r.history_D:.0f}</td></tr>"
        for (part, cad, late), r in breakdown.iterrows()
    )
    start = layer.get("start", {})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Release layer review</title>
<style>
:root{{--ink:#1f2328;--muted:#6e7781;--bg:#fff;--line:#d0d7de}}
@media (prefers-color-scheme:dark){{:root{{--ink:#e6edf3;--muted:#8d96a0;--bg:#0d1117;--line:#30363d}} img{{background:#fff}}}}
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;color:var(--ink);background:var(--bg);line-height:1.45}}
img{{max-width:100%;height:auto;border:1px solid var(--line)}} td,th{{padding:3px 8px;text-align:right;border-bottom:1px solid var(--line)}}
table{{border-collapse:collapse;font-size:13px}} .muted{{color:var(--muted);font-size:13px}} code{{font-size:12px}}
</style></head><body>
<h1>What the release layer does to training blocks</h1>
<p>The release layer (PR #883, <code>config/finetune/release-layer.yaml</code>) acts when blocks are cut, on the simulations already made.
A share of blocks ({start.get("share", 0):.0%}) has its releases start late: every release before the drawn start day is dropped and the cut's is kept.
The first kept release then holds every onset reported before it, the lag rows start there and the counts row keeps the whole history, as in the BDBV hub's data.
Its cadences set the share of daily-step blocks released weekly ({layer.get("cadences", {}).get("daily", {}).get("weekly", 0.2):.0%}).</p>
<p class="muted">The same {len(ids)} Run A held-out simulations ({ids[0]} to {ids[-1]}), cut as Run A cuts them (grid, seed 1, 32 blocks a simulation, block limit 1,000, quiet floor 10), without and with the layer: {n} blocks of the daily, weekly and 3-day sources (the summed sources have extra rows and are left out of the counts here). Delay drawn from {start.get("delay_days")}.</p>
<h2>1. Example blocks, before and after</h2>
<p>Each panel is one block's country: the top row is the counts (onset day), the rows below are its lag series (lag 0 down to lag D-1). Grey cells have no value. The last panel is the BDBV hub's national daily block.</p>
<img src="data:image/png;base64,{ex}" alt="Example blocks before and after the release layer, and a BDBV block">
<h2>2. Distributions</h2>
<img src="data:image/png;base64,{dist}" alt="Release-start delays, shares of blocks, lag-row fill and lag counts">
<table><tr><th>cut</th><th>source</th><th>blocks</th><th>start late</th><th>released weekly</th><th>median fill</th><th>median D</th></tr>{rows}</table>
<p>Daily blocks by cadence and late start. A late start with weekly releases is the BDBV shape.</p>
<table><tr><th>cut</th><th>cadence</th><th>starts late</th><th>blocks</th><th>median fill</th><th>median D</th><th>median release history (days)</th></tr>{split}</table>
<p>BDBV hub (CD, daily onsets, releases {real["first_release"]} to {real["cut"]}): {real["releases"]} releases, lag 0 row filled on {real["filled"]} of {real["steps"]} days, fill {pct(real["fill"])}, D {real["D"]}.
Germany (national, daily releases, cuts {", ".join(de["cut"]) if len(de) else "none"}): median fill {pct(de["fill"].median()) if len(de) else "n/a"}, median D {de["D"].median() if len(de) else "n/a"}.</p>
<h2>3. Lag count</h2>
<p>D on this page is next's lag rule (<code>forecaster.lag_count</code>), which the blocks are cut with.
Investigate's corrected rule (lead 10:25/10:45, no PR yet) differs where no step is informative: there D is the release history, from the first release to the cut-off, and a backfilled first release is not delay evidence.
The dotted line is that release history for the late blocks, so it is the corrected rule's D for those of them with no informative step.
Once lag-count-history lands this page is rebuilt from it.</p>
<p class="muted">Built by <code>workflow/scripts/release_layer_page.py</code> on branch train-ragged. Per-block table beside this page (release-layer.csv).</p>
</body></html>"""


if __name__ == "__main__":
    main()
