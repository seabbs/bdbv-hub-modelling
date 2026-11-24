# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures of the BDBV hub data and our submission, for the docs report.

The releases' revisions, cut-off days, imputed onsets, downward updates,
newly missing onsets and delays (`nowcastbdbv.data`), a nowcast on the
hub data and a submission file. The shared figure helpers are
`nowcasteval.plots`'.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Final, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nowcastbdbv.data.delays import delay_summary
from nowcastbdbv.data.tidy import (
    cutoff_window,
    downward_summary,
    imputed_by_release,
    imputed_share,
    load_daily,
    load_delays,
    load_target,
    newly_missing,
    release_updates,
    revision_table,
)
from nowcasteval.plots import (
    MODEL_COLOURS,
    THEMES,
    date_axis,
    draw_forecast,
    incomplete_after,
    legend_below,
    location_names,
    locations_by_size,
    model_colour,
    model_dash,
    panel_grid,
    shade_incomplete,
    theme,
)
from nowcastutils.base import NATIONAL

__all__ = [
    "AGE_BINS",
    "plot_cutoff_days",
    "plot_delays_over_time",
    "plot_downward_updates",
    "plot_imputed_by_release",
    "plot_imputed_share",
    "plot_newly_missing",
    "plot_nowcast",
    "plot_release_counts",
    "plot_revisions",
    "plot_submission",
    "read_submission",
]


def plot_revisions(target: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot how recent weeks fill in, as a share of the latest vintage.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        National share reported by weeks before the vintage.
    """
    rev = revision_table(target.loc[target["location"] == "CD"])
    latest = rev["vintage"].max()
    rev = rev.loc[(rev["vintage"] < latest) & (rev["weeks_before"] <= 10)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    palette = MODEL_COLOURS[mode]
    for i, (vin, r) in enumerate(rev.groupby("vintage")):
        ax.plot(
            r["weeks_before"],
            r["fraction"],
            marker="o",
            ms=4,
            color=palette[i % len(palette)],
            label=str(vin),
        )
    ax.axhline(1, color=THEMES[mode]["muted"], lw=1, ls=":")
    ax.set_xlabel("Weeks from onset week to the vintage's epi week")
    ax.set_ylabel(f"Share of the {latest} count")
    ax.invert_xaxis()
    ax.legend(title="Vintage", title_fontsize=8)
    return fig


def plot_release_counts(
    target: pd.DataFrame,
    locations: Sequence[str] = ("CD", "CD-IT", "CD-NK"),
    weeks: int = 6,
    mode: str = "light",
) -> Figure:
    """Plot each recent onset week's count in every release that has it.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    locations : sequence of str
        Places to show, one panel each.
    weeks : int
        How many of the latest release's onset weeks to follow.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Releases on the x axis, one line per onset week, labelled at its
        end.
    """
    locs = list(locations)
    rev = revision_table(target.loc[target["location"].isin(locs)])
    latest = rev["vintage"].max()
    recent = sorted(
        rev.loc[rev["vintage"] == latest, "target_end_date"].unique()
    )[-weeks:]
    rev = rev.loc[rev["target_end_date"].isin(recent)]
    releases = sorted(rev["vintage"].unique())
    at = {v: i for i, v in enumerate(releases)}
    names = location_names(target)
    palette = MODEL_COLOURS[mode]
    fig, axes = panel_grid(len(locs), 3, width=3.6, height=3.0)
    for ax, loc in zip(axes, locs, strict=True):
        rows = rev.loc[rev["location"] == loc]
        ends = rows.loc[rows["vintage"] == latest].set_index("target_end_date")
        ys = _spread(ends["total"].reindex(recent).to_numpy(float), 0.06)
        for i, week in enumerate(recent):
            r = rows.loc[rows["target_end_date"] == week].sort_values("vintage")
            x = r["vintage"].map(at)
            colour = palette[i % len(palette)]
            ax.plot(x, r["total"], marker="o", ms=3.5, lw=1.5, color=colour)
            ax.annotate(
                pd.Timestamp(week).strftime("%d %b"),
                (x.iloc[-1] + 0.12, ys[i]),
                va="center",
                fontsize=7,
                color=colour,
            )
        ax.set_xticks(
            range(len(releases)),
            [pd.Timestamp(v).strftime("%d %b") for v in releases],
        )
        ax.set_xlim(-0.3, len(releases) - 0.2)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        ax.set_xlabel("Release")
    axes[0].set_ylabel("Confirmed cases in the onset week")
    return fig


def _spread(values: np.ndarray, gap: float) -> np.ndarray:
    """Move label heights apart by at least `gap` of their range."""
    order = np.argsort(values)
    out = values.astype(float).copy()
    step = gap * max(float(np.nanmax(values)), 1.0)
    for a, b in pairwise(order):
        out[b] = max(out[b], out[a] + step)
    return out


def plot_cutoff_days(
    daily: pd.DataFrame,
    vintages: Sequence[str],
    location: str = NATIONAL,
    weeks: int = 4,
    mode: str = "light",
) -> Figure:
    """Plot daily onsets up to each release's cut-off, with weekly totals.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets, as from `load_daily`.
    vintages : sequence of str
        Releases to show, one panel each.
    location : str
        The place.
    weeks : int
        Epi weeks to show before each cut-off, including its own.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Stacked bars of known and imputed onsets by day, alternate epi
        weeks shaded and labelled with the week's total (the hub target),
        and a dotted line at the release date.
    """
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    fig, axes = panel_grid(len(vintages), 2, width=4.8, height=2.8)
    for ax, vin in zip(axes, vintages, strict=True):
        win = cutoff_window(daily, vin, location, weeks)
        x = mdates.datestr2num(win["onset_date"].tolist())
        ax.bar(x, win["observed"], 0.85, color=palette[0], label="Known onset")
        ax.bar(
            x,
            win["imputed"],
            0.85,
            bottom=win["observed"],
            color=palette[1],
            label="Imputed onset",
        )
        top = float(win["total"].max())
        ax.set_ylim(0, top * 1.3)
        for i, (week, r) in enumerate(win.groupby("target_end_date")):
            start = float(mdates.datestr2num(str(week))) - 6.5
            if i % 2 == 0:
                ax.axvspan(start, start + 7, color=ink["grid"], lw=0, zorder=0)
            days = int(r["week_days"].iloc[0])
            note = "" if days == 7 else f"\n{days} of 7 days"
            ax.text(
                start + 3.5,
                top * 1.27,
                f"{int(r['week_total'].iloc[0])}{note}",
                ha="center",
                va="top",
                fontsize=7,
                color=ink["text"],
            )
        cutoff = str(win["onset_date"].iloc[-1])
        release = float(mdates.datestr2num(vin))
        ax.axvline(release, color=ink["muted"], lw=1, ls=":")
        ax.set_xlim(float(x[0]) - 1, max(float(x[-1]), release) + 1)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_title(f"Release {vin}, cut-off {cutoff}", fontsize=9)
    axes[0].set_ylabel("Onsets per day")
    legend_below(fig, axes[0])
    return fig


def plot_imputed_share(
    target: pd.DataFrame,
    locations: Sequence[str] | None = None,
    mode: str = "light",
) -> Figure:
    """Plot the share of cases with an imputed onset by onset week.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    locations : sequence of str, optional
        Locations to show. Defaults to the four largest and the country.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per location, one line per vintage, with point size by
        the week's total.
    """
    share = imputed_share(target)
    locs = list(locations or locations_by_size(target, 5))
    vints = sorted(share["vintage"].unique())
    names = location_names(target)
    palette = MODEL_COLOURS[mode]
    fig, axes = panel_grid(len(locs), 3)
    for ax, loc in zip(axes, locs, strict=True):
        rows = share.loc[(share["location"] == loc) & (share["total"] > 0)]
        for i, vin in enumerate(vints):
            r = rows.loc[rows["vintage"] == vin]
            x = pd.to_datetime(r["target_end_date"])
            col = palette[i % len(palette)]
            ax.plot(x, r["imputed_share"], color=col, lw=1, label=vin)
            ax.scatter(
                x, r["imputed_share"], s=np.sqrt(r["total"]) * 2, color=col
            )
        ax.set_ylim(0, 1)
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        date_axis(ax)
    axes[0].set_ylabel("Imputed share of cases")
    axes[0].legend(title="Vintage", title_fontsize=8)
    return fig


def plot_imputed_by_release(
    target: pd.DataFrame, mode: str = "light"
) -> Figure:
    """Plot the share of all cases with an imputed onset in each release.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One line for the country and one per province with cases.
    """
    share = imputed_by_release(target)
    names = location_names(target)
    palette = MODEL_COLOURS[mode]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for i, loc in enumerate(locations_by_size(target)):
        r = share.loc[share["location"] == loc]
        national = loc == "CD"
        ax.plot(
            pd.to_datetime(r["vintage"]),
            r["imputed_share"],
            marker="o",
            ms=4,
            lw=2.5 if national else 1.2,
            color=THEMES[mode]["data"] if national else palette[(i - 1) % 8],
            label=f"{names.get(loc, loc)} ({loc})",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Release")
    ax.set_ylabel("Share of cases with an imputed onset")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.legend(loc="outside right center", fontsize=8)
    return fig


AGE_BINS: Final = (0, 1, 2, 3, 4, 8, 16)
"""Lower edges of the onset-week age groups for downward updates."""


def plot_downward_updates(
    target: pd.DataFrame, zones: pd.DataFrame, mode: str = "light"
) -> Figure:
    """Plot how often and how much counts fell between releases.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    zones : pandas.DataFrame
        Health zone counts, with columns `vintage`, `cutoff`, `location`,
        `target_end_date` and `total`.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Columns for the country, the provinces and the health zones, and
        rows for the share of place-weeks that fell and the falls as a
        share of all change, by the week's age at the earlier release.
        One line per pair of consecutive releases.
    """
    nat = target["location"] == "CD"
    updates = pd.concat(
        [
            release_updates(target.loc[nat]).assign(scale="Country"),
            release_updates(target.loc[~nat]).assign(scale="Provinces"),
            release_updates(zones).assign(scale="Health zones"),
        ]
    )
    edges = list(AGE_BINS)
    labels = [
        str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}"
        for lo, hi in pairwise(edges)
    ] + [f"{edges[-1]}+"]
    updates["age"] = pd.cut(
        updates["weeks_before"],
        [*edges, np.inf],
        right=False,
        labels=labels,
    )
    updates["pair"] = updates["previous"] + " to " + updates["vintage"]
    summary = downward_summary(updates, ("scale", "pair", "age"))
    pairs = sorted(summary["pair"].unique())
    palette = MODEL_COLOURS[mode]
    scales = ["Country", "Provinces", "Health zones"]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.6, 5.2),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    x = np.arange(len(labels))
    for col, scale in enumerate(scales):
        for i, pair in enumerate(pairs):
            pick = (summary["scale"] == scale) & (summary["pair"] == pair)
            r = summary.loc[pick].set_index("age").reindex(labels)
            for row, metric in enumerate(("share_fell", "downward_share")):
                axes[row][col].plot(
                    x,
                    r[metric],
                    marker="o",
                    ms=4,
                    lw=1.2,
                    color=palette[i % len(palette)],
                    label=pair,
                )
        axes[0][col].set_title(scale)
        axes[1][col].set_xticks(x, labels)
        axes[1][col].set_xlabel("Weeks old at the earlier release")
    axes[0][0].set_ylabel("Share of place-weeks\nthat fell")
    axes[1][0].set_ylabel("Falls as a share\nof all change")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="outside lower center",
        ncol=len(pairs),
        title="Releases compared",
        title_fontsize=8,
    )
    return fig


def plot_newly_missing(target: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot cases added between vintages, split by onset status.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Bars of new observed and new imputed cases per vintage step, by
        province.
    """
    new = newly_missing(target)
    new = new.loc[new["location"] != "CD"]
    keep = new.groupby("location")["new_total"].sum().abs().sort_values()
    locs = list(keep.index[keep > 0][::-1])
    new = new.loc[new["location"].isin(locs)]
    steps = sorted(new["vintage"].unique())
    fig, axes = panel_grid(len(steps), 3, width=3.6, height=2.8)
    palette = MODEL_COLOURS[mode]
    for ax, vin in zip(axes, steps, strict=True):
        r = new.loc[new["vintage"] == vin].set_index("location").reindex(locs)
        x = np.arange(len(locs))
        ax.bar(
            x - 0.2,
            r["new_observed"],
            0.38,
            color=palette[0],
            label="Known onset",
        )
        ax.bar(
            x + 0.2,
            r["new_imputed"],
            0.38,
            color=palette[1],
            label="Imputed onset",
        )
        ax.axhline(0, color=THEMES[mode]["axis"], lw=1)
        ax.set_xticks(x, [loc.removeprefix("CD-") for loc in locs])
        prev = r["previous_vintage"].dropna()
        ax.set_title(f"{prev.iloc[0] if len(prev) else ''} to {vin}")
    axes[0].set_ylabel("Cases added")
    axes[0].legend()
    return fig


def plot_delays_over_time(
    delays: pd.DataFrame, min_cases: int = 5, mode: str = "light"
) -> Figure:
    """Plot the mean and median delay by onset week, with counts.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one or more vintages.
    min_cases : int
        Weeks with fewer cases are not drawn.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Top: mean (line) and median (points) delay by onset week for the
        largest provinces, latest vintage. Bottom: cases per week.
    """
    latest = delays.loc[delays["vintage"] == delays["vintage"].max()]
    summ = delay_summary(latest, by=("location", "onset_week"))
    summ = summ.loc[summ["n"] >= min_cases]
    locs = list(latest["location"].value_counts().index[:4])
    names = location_names(latest)
    palette = MODEL_COLOURS[mode]
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5),
        sharex=True,
        height_ratios=(2, 1),
        layout="constrained",
    )
    for i, loc in enumerate(locs):
        s = summ.loc[summ["location"] == loc]
        x = pd.to_datetime(s["onset_week"])
        col = palette[i]
        ax.plot(x, s["mean"], color=col, label=names.get(loc, loc))
        ax.scatter(x, s["median"], color=col, s=10)
        ax2.plot(x, s["n"], color=col)
    ax.set_ylabel("Delay (days): mean line, median points")
    ax.legend()
    ax2.set_ylabel("Cases")
    ax2.set_yscale("log")
    date_axis(ax2)
    ax.set_title(f"Delay data of the {latest['vintage'].max()} vintage")
    return fig


def plot_nowcast(
    location: str,
    vintage: str | None = None,
    models: Sequence[str] = ("randomwalk",),
    target: pd.DataFrame | None = None,
    delays: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
    weeks_before: int = 16,
    mode: str = "light",
    complete_after_weeks: int = 3,
    data_dir: Path | str | None = None,
    config_path: Path | str = "config/backtest.toml",
    ax: Axes | None = None,
) -> Figure:
    """Plot the nowcast and forecast for one location from one vintage.

    If `forecasts` is not given, the models are run on the vintage.
    This is the figure the submission pull request embeds.

    Parameters
    ----------
    location : str
        Location to show, such as `"CD-IT"` or `"CD"`.
    vintage : str, optional
        Real vintage to forecast from. Defaults to the latest.
    models : sequence of str
        Registered models to show.
    target, delays : pandas.DataFrame, optional
        Tidy tables. Loaded from `data_dir` if not given.
    forecasts : pandas.DataFrame, optional
        Output of `run_forecasts` for this vintage.
    weeks_before : int
        Weeks of data shown before the forecast.
    mode : str
        `"light"` or `"dark"`.
    complete_after_weeks : int
        Weeks after which a week counts as complete.
        Later weeks are shaded and their bars drawn hollow.
    data_dir : pathlib.Path or str, optional
        Directory of the tidy tables.
    config_path : pathlib.Path or str
        Backtest settings, for horizons and samples.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, for a grid of locations. A new figure if None.

    Returns
    -------
    matplotlib.figure.Figure
        Observed, imputed and total cases by onset week, and each model's
        sample trajectories, 50% and 90% intervals and median.
    """
    from nowcasteval.backtest import load_config, make_tasks, run_forecasts
    from nowcastutils.base import get_model

    target = load_target(data_dir) if target is None else target
    real = target.loc[target["kind"] == "real"]
    vintage = vintage or str(real["vintage"].max())
    rows = real.loc[real["vintage"] == vintage]
    if forecasts is None:
        delays = load_delays(data_dir) if delays is None else delays
        config = load_config(config_path)
        daily = load_daily(data_dir)
        tasks = make_tasks(rows, delays, config, daily=daily)
        forecasts = run_forecasts([get_model(m) for m in models], tasks)
    fc = forecasts.loc[
        (forecasts["location"] == location) & (forecasts["vintage"] == vintage)
    ]
    ref = pd.Timestamp(fc["reference_date"].iloc[0])
    data = rows.loc[rows["location"] == location].copy()
    data["date"] = pd.to_datetime(data["target_end_date"])
    start = ref - pd.Timedelta(days=7 * weeks_before)
    data = data.loc[data["date"] >= start]
    ink = THEMES[mode]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    else:
        fig = cast(Figure, ax.get_figure())
    cut = incomplete_after(vintage, complete_after_weeks)
    done = data["date"] <= cut
    for part, hollow in ((data.loc[done], False), (data.loc[~done], True)):
        for col, colour, label, base in (
            ("observed", ink["axis"], "Known onset", None),
            ("imputed", ink["muted"], "Imputed onset", "observed"),
        ):
            ax.bar(
                part["date"],
                part[col],
                width=5,
                bottom=None if base is None else part[base],
                color="none" if hollow else colour,
                edgecolor=colour,
                label=label if not hollow else None,
            )
    for m in models:
        s = fc.loc[fc["model"] == m]
        draw_forecast(
            ax, s, model_colour(m, list(models), mode), model_dash(m), m
        )
    ax.axvline(mdates.date2num(ref), color=ink["muted"], lw=0.8, ls=":")
    shade_incomplete(ax, cut, mode)
    names = location_names(rows)
    ax.set_title(
        f"{names.get(location, location)} ({location}): vintage {vintage}, "
        f"reference date {ref.date()}"
    )
    ax.set_ylabel("Confirmed cases by onset week")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7, loc="upper left")
    return fig


def read_submission(path: Path | str, vintage: str) -> pd.DataFrame:
    """Read a hub submission file as forecasts for `plot_nowcast`.

    Parameters
    ----------
    path : pathlib.Path or str
        A hub file with sample output. Rows of other output types, such
        as the quantiles of the `wk R_t` target, are left out.
    vintage : str
        The data vintage it was made from.

    Returns
    -------
    pandas.DataFrame
        Columns `model` (`"submission"`), `vintage`, `reference_date`,
        `location`, `target_end_date`, `sample` and `value`.
    """
    table = pd.read_csv(path, dtype={"output_type_id": str})
    table = table.loc[table["output_type"] == "sample"]
    return pd.DataFrame(
        {
            "model": "submission",
            "vintage": vintage,
            "reference_date": table["reference_date"].astype(str),
            "location": table["location"].astype(str),
            "target_end_date": table["target_end_date"].astype(str),
            "sample": table["output_type_id"].astype("int64"),
            "value": table["value"].astype(float),
        }
    )


def plot_submission(
    submission: Path | str,
    vintage: str,
    target: pd.DataFrame | None = None,
    n_provinces: int = 3,
    data_dir: Path | str | None = None,
) -> Figure:
    """Plot what was submitted: the national total and the largest provinces.

    Draws the samples in the file, not a re-run of the model, with the
    data of the vintage behind them. Light mode, about 1600 by 1000 pixels
    at 100 dpi, for the submission pull request.

    Parameters
    ----------
    submission : pathlib.Path or str
        The hub file.
    vintage : str
        The data vintage the file was made from.
    target : pandas.DataFrame, optional
        Tidy target table. Loaded from `data_dir` if not given.
    n_provinces : int
        Provinces shown beside the national total, largest median first.
    data_dir : pathlib.Path or str, optional
        Directory of the tidy tables.

    Returns
    -------
    matplotlib.figure.Figure
        A grid with one panel per location.
    """
    from nowcastutils.hierarchy import province_of

    target = load_target(data_dir) if target is None else target
    fc = read_submission(submission, vintage)
    provinces = fc["location"].map(province_of) == fc["location"]
    sizes = (
        fc.loc[provinces & (fc["location"] != "CD")]
        .groupby("location")["value"]
        .median()
        .sort_values(ascending=False)
    )
    locations = [
        loc
        for loc in ["CD", *sizes.index[:n_provinces]]
        if loc in set(fc["location"])
    ]
    ncol = 2
    nrow = max(1, -(-len(locations) // ncol))
    with theme("light"):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(16, 5 * nrow), layout="constrained"
        )
        flat = list(np.atleast_1d(axes).ravel())
        for loc, ax in zip(locations, flat, strict=False):
            plot_nowcast(
                loc,
                vintage=vintage,
                models=("submission",),
                target=target,
                forecasts=fc,
                ax=ax,
            )
        for ax in flat[len(locations) :]:
            ax.set_visible(False)
    return fig
