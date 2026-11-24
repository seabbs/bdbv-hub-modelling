# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""One panel per location the forecast checks warned on.

Each panel shows the weekly counts reported at the vintage, the
submitted model's median and 90% interval, last week's submission's and
the random walk nowcast's, from `WEEKS_BEFORE` weeks before the vintage
to the last horizon. Its title is the location and the warnings.

Built in the submission run from the candidate's `checks.json`, its hub
file, the hub's `time-series.csv`, `previous.csv` and the random walk
nowcast candidate's hub file, and kept as `warnings.png` beside the
candidate's other files. The round page and the this-round report show
that stored plot.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nowcastbdbv.data.tidy import TARGET
from nowcasteval.plots import (
    HUB_GREYS,
    legend_below,
    model_colour,
    panel_grid,
    theme,
)

__all__ = [
    "WEEKS_BEFORE",
    "plot_warnings",
    "read_quantiles",
    "warned_locations",
    "warnings_figure",
]

WEEKS_BEFORE: Final = 14
"""Weeks of data shown before the vintage."""

_LEVELS: Final = (0.05, 0.5, 0.95)


def warned_locations(checks: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return each warned location's warnings, named by their check.

    Parameters
    ----------
    checks : mapping
        A `checks.json`, whose checks list each warned location's
        warnings under `warnings`.

    Returns
    -------
    dict
        Location to its warnings, as `<check>: <warning>`, in the order
        the checks ran.
    """
    out: dict[str, list[str]] = {}
    for c in checks.get("checks", []):
        if c.get("status") != "warn":
            continue
        for loc, msgs in (c.get("warnings") or {}).items():
            out.setdefault(loc, []).extend(f"{c['name']}: {m}" for m in msgs)
    return out


def read_quantiles(path: Path | str) -> pd.DataFrame:
    """Return a hub file's sample median and 90% interval.

    Parameters
    ----------
    path : pathlib.Path or str
        A hub file with sample output for the case target.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `date` and the levels 0.05, 0.5 and 0.95.
    """
    table = pd.read_csv(path)
    table = table.loc[
        (table["target"] == TARGET) & (table["output_type"] == "sample")
    ]
    q = (
        table.groupby(["location", "target_end_date"])["value"]
        .quantile(np.array(_LEVELS))
        .unstack()
        .reset_index()
    )
    q["date"] = pd.to_datetime(q["target_end_date"])
    return q.drop(columns="target_end_date")


def _observed(path: Path | str, as_of: str) -> pd.DataFrame:
    """Return the weekly counts reported at `as_of`."""
    ts = pd.read_csv(path)
    ts = ts.loc[(ts["target"] == TARGET) & (ts["as_of"].astype(str) == as_of)]
    return ts.assign(date=pd.to_datetime(ts["target_end_date"]))


def _title(loc: str, msgs: Sequence[str]) -> str:
    """Return the location, then each warning without the location."""
    short = [m.replace(f" {loc}: ", " ").replace(f" {loc} ", " ") for m in msgs]
    return "\n".join([loc, *(textwrap.fill(m, 70) for m in short)])


def plot_warnings(
    warned: Mapping[str, Sequence[str]],
    observed: pd.DataFrame,
    as_of: str,
    forecasts: Mapping[str, tuple[str, pd.DataFrame]],
    weeks_before: int = WEEKS_BEFORE,
    ncol: int = 2,
) -> Figure | None:
    """Plot one panel per warned location.

    Parameters
    ----------
    warned : mapping
        Location to its warnings, as from `warned_locations`.
    observed : pandas.DataFrame
        Counts reported at the vintage: `location`, `date` and
        `observation`.
    as_of : str
        The vintage.
    forecasts : mapping
        Label to its colour and quantiles, as from `read_quantiles`,
        drawn in order.
    weeks_before : int
        Weeks of data shown before the vintage.
    ncol : int
        Panels a row.

    Returns
    -------
    matplotlib.figure.Figure or None
        None when no location was warned on.
    """
    if not warned:
        return None
    start = pd.Timestamp(as_of) - pd.Timedelta(weeks=weeks_before)
    dates = [q["date"] for _, q in forecasts.values() if len(q)]
    end = max((d.max() for d in dates), default=pd.Timestamp(as_of))
    with theme("light") as ink:
        fig, axes = panel_grid(len(warned), ncol, width=6.4, height=3.6)
        for ax, (loc, msgs) in zip(axes, warned.items(), strict=True):
            obs = observed.loc[
                (observed["location"] == loc) & (observed["date"] >= start)
            ].sort_values("date")
            ax.plot(
                obs["date"],
                obs["observation"],
                "o-",
                color=ink["data"],
                ms=3,
                lw=1,
                label=f"reported as of {as_of}",
            )
            for label, (colour, q) in forecasts.items():
                s = q.loc[q["location"] == loc].sort_values("date")
                ax.fill_between(
                    s["date"], s[0.05], s[0.95], color=colour, alpha=0.18, lw=0
                )
                ax.plot(s["date"], s[0.5], color=colour, lw=1.5, label=label)
            pad = pd.Timedelta(days=3)
            ax.set_xlim(
                float(mdates.date2num(start - pad)),
                float(mdates.date2num(end + pad)),
            )
            ax.axvline(
                mdates.date2num(pd.Timestamp(as_of)),
                color=ink["muted"],
                lw=0.8,
                ls=":",
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.set_title(_title(loc, msgs), loc="left", fontsize=8)
            ax.set_ylabel("Weekly cases")
        legend_below(fig, axes[0])
    return fig


def warnings_figure(
    checks: Path | str,
    submission: Path | str,
    observed: Path | str,
    as_of: str,
    previous: Path | str | None = None,
    randomwalk: Path | str | None = None,
    label: str = "submitted",
) -> Figure | None:
    """Plot the warned locations of a candidate from its files.

    Parameters
    ----------
    checks : pathlib.Path or str
        The candidate's `checks.json`.
    submission : pathlib.Path or str
        The candidate's hub file.
    observed : pathlib.Path or str
        The hub's `time-series.csv`.
    as_of : str
        The vintage.
    previous : pathlib.Path or str, optional
        Last week's submission (`previous.csv`).
    randomwalk : pathlib.Path or str, optional
        The random walk nowcast candidate's hub file.
    label : str
        The candidate's name in the key; a model in the palette keeps its
        colour.

    Returns
    -------
    matplotlib.figure.Figure or None
        None when no location was warned on.
    """
    warned = warned_locations(json.loads(Path(checks).read_text()))
    if not warned:
        return None
    walk = "randomwalknowcast"
    forecasts = {
        label: (model_colour(label, [label]), read_quantiles(submission))
    }
    if previous is not None:
        forecasts["last week's submission"] = (
            HUB_GREYS["light"][0],
            read_quantiles(previous),
        )
    if randomwalk is not None and label != walk:
        forecasts["random walk nowcast"] = (
            model_colour(walk, [walk]),
            read_quantiles(randomwalk),
        )
    return plot_warnings(warned, _observed(observed, as_of), as_of, forecasts)
