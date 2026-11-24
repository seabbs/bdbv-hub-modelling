# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures for the data, the forecasts and the scores.

Every function returns a matplotlib `Figure` and takes `theme`, `"light"`
or `"dark"`, so the docs can show both.
Models get a fixed colour and dash from `MODEL_STYLE`, so a model looks
the same on every page and keeps its look when others are added.
The vintage kind is always shown, in a panel title or the legend.

The submission phase can call `plot_nowcast(location=..., vintage=...)`
for the figure it embeds in the hub pull request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nowcastutils.base import NATIONAL

__all__ = [
    "DASHES",
    "MODEL_COLOURS",
    "MODEL_STYLE",
    "THEMES",
    "bar_style",
    "comparison_colours",
    "comparison_label",
    "date_axis",
    "draw_forecast",
    "hub_styles",
    "incomplete_after",
    "legend_below",
    "location_names",
    "locations_by_size",
    "model_colour",
    "model_dash",
    "model_style",
    "panel_grid",
    "plot_delay_distribution",
    "plot_forecasts",
    "plot_hub_round",
    "plot_location_grid",
    "plot_renewal_delay",
    "plot_scores",
    "plot_scores_by_truth",
    "plot_target",
    "point_style",
    "save_light_dark",
    "shade_incomplete",
    "theme",
]

THEMES: Final[dict[str, dict[str, str]]] = {
    "light": {
        "ink": "#0b0b0b",
        "text": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "data": "#0b0b0b",
    },
    "dark": {
        "ink": "#f0efec",
        "text": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "data": "#f0efec",
    },
}
"""Ink colours for each theme. Backgrounds are transparent."""

MODEL_COLOURS: Final[dict[str, tuple[str, ...]]] = {
    "light": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}
"""Categorical colours in fixed order (checked for colour vision deficiency)."""

DASHES: Final[dict[str, str]] = {
    "solid": "",
    "dashed": "6 3",
    "dotted": "1.5 2.5",
    "dashdot": "6 3 1.5 3",
    "longdash": "12 4",
    "dashdotdot": "6 3 1.5 3 1.5 3",
    "shortdash": "3 3",
    "longdashdot": "12 4 1.5 4",
}
"""Line dash patterns as SVG `stroke-dasharray` values, solid first."""

MODEL_STYLE: Final[dict[str, tuple[int, str]]] = {
    "randomwalk": (0, "solid"),
    "randomwalknowcast": (1, "solid"),
    "randomwalknowcast_daily": (1, "dashed"),
    "renewal": (2, "solid"),
    "renewalbvd": (3, "solid"),
    "renewalsp": (4, "solid"),
    "fm_chronos_weekly_complete": (6, "solid"),
    "fm_chronos_weekly_ragged": (6, "dashed"),
    "fm_chronos_daily_complete": (6, "dotted"),
    "fm_chronos_daily_ragged": (6, "dashdot"),
    "fm_chronos_weekly_ragged_ft": (6, "longdash"),
    "fm_chronos_daily_ragged_ft": (6, "dashdotdot"),
    "fm_chronos_weekly_complete_ft": (5, "longdash"),
    "fm_chronos_daily_complete_ft": (5, "dashdotdot"),
    "fm_chronos_daily_complete_2step": (5, "dotted"),
    "fm_chronos_daily_complete_2step_ft": (5, "solid"),
    "fm_chronos_weekly_ragged_real": (6, "shortdash"),
    "fm_chronos_daily_ragged_real": (6, "longdashdot"),
}
"""Each model's place in `MODEL_COLOURS` and its dash in `DASHES`.

A new main model takes the next free colour; a variant takes its parent's
colour with the next dash, since past eight colours some pairs can no
longer be told apart.
"""

_FREE_SLOT: Final = 1 + max(s for s, _ in MODEL_STYLE.values())

HUB_GREYS: Final[dict[str, tuple[str, ...]]] = {
    "light": ("#5f5e5a", "#898781", "#b4b2a9"),
    "dark": ("#c3c2b7", "#898781", "#5f5e5a"),
}
"""Greys for the hub teams beside our models."""

_KIND_MARKERS: Final = {"real": "o", "simulated": "^"}


def model_colour(name: str, models: Sequence[str], mode: str = "light") -> str:
    """Return a model's fixed colour.

    A model in `MODEL_STYLE` has its own colour. Others, such as test
    stand-ins, take the colours after the last one `MODEL_STYLE` uses, in
    their order in `models`.

    Parameters
    ----------
    name : str
        The model.
    models : sequence of str
        All models shown.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    str
        A hex colour.
    """
    palette = MODEL_COLOURS[mode]
    if name in MODEL_STYLE:
        return palette[MODEL_STYLE[name][0]]
    others = [m for m in models if m not in MODEL_STYLE]
    return palette[(_FREE_SLOT + others.index(name)) % len(palette)]


def model_dash(name: str) -> str:
    """Return a model's dash as an SVG `stroke-dasharray`, empty if solid.

    Parameters
    ----------
    name : str
        The model.

    Returns
    -------
    str
        A value of `DASHES`.
    """
    return DASHES[MODEL_STYLE.get(name, (0, "solid"))[1]]


def model_style(name: str, models: Sequence[str]) -> tuple[str, str, str]:
    """Return a model's light colour, dark colour and dash.

    Parameters
    ----------
    name : str
        The model.
    models : sequence of str
        All models shown, as for `model_colour`.

    Returns
    -------
    tuple of str
        Light colour, dark colour and SVG dash.
    """
    return (
        model_colour(name, models, "light"),
        model_colour(name, models, "dark"),
        model_dash(name),
    )


def mpl_dash(dash: str) -> str | tuple[int, tuple[float, ...]]:
    """Turn an SVG dash into a matplotlib line style."""
    return (0, tuple(float(v) for v in dash.split())) if dash else "-"


def bar_style(colour: str, dash: str) -> dict[str, Any]:
    """Bar keywords: filled when solid, hatched in the colour otherwise.

    Bars cannot show a dash, so a variant is hatched to keep it apart
    from its parent.
    """
    if not dash:
        return {"color": colour}
    return {"facecolor": "none", "edgecolor": colour, "hatch": "////"}


def point_style(colour: str, dash: str) -> dict[str, Any]:
    """Marker keywords: filled when solid, hollow otherwise."""
    return {"color": colour, "mfc": "none" if dash else colour}


@contextmanager
def theme(mode: str = "light") -> Iterator[dict[str, str]]:
    """Apply a light or dark theme to figures made inside the block.

    Parameters
    ----------
    mode : str
        `"light"` or `"dark"`.

    Yields
    ------
    dict
        The theme's ink colours.
    """
    ink = THEMES[mode]
    params: dict[str, Any] = {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": ink["axis"],
        "axes.labelcolor": ink["text"],
        "axes.titlecolor": ink["ink"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": ink["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": ink["muted"],
        "ytick.color": ink["muted"],
        "xtick.labelcolor": ink["text"],
        "ytick.labelcolor": ink["text"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": ink["text"],
        "text.color": ink["text"],
        "font.size": 9,
        "lines.linewidth": 2,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(params):  # type: ignore[arg-type]
        yield ink


def save_light_dark(
    make: Callable[..., Figure], path: Path | str
) -> list[Path]:
    """Save a figure in light and dark versions.

    Parameters
    ----------
    make : callable
        Takes the keyword `mode` (`"light"` or `"dark"`) and returns a
        figure.
    path : pathlib.Path or str
        Output path for the light version, such as `figures/x.svg`. The
        dark version gets a `-dark` suffix.

    Returns
    -------
    list of pathlib.Path
        The two paths written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out, out.with_name(f"{out.stem}-dark{out.suffix}")]
    for mode, p in zip(("light", "dark"), paths, strict=True):
        with theme(mode):
            fig = make(mode=mode)
            fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
    return paths


def date_axis(ax: Axes) -> None:
    """Label a date axis by month."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def panel_grid(
    n: int, ncol: int, width: float = 3.2, height: float = 2.4
) -> tuple[Figure, list[Axes]]:
    """Return a figure of `n` panels, `ncol` a row, and its used axes."""
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(width * ncol, height * nrow),
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def location_names(target: pd.DataFrame) -> dict[str, str]:
    """Return each location's name, from `location` and `location_name`.

    Parameters
    ----------
    target : pandas.DataFrame
        Rows with `location` and `location_name`.

    Returns
    -------
    dict of str to str
        Name by location code.
    """
    pairs = target[["location", "location_name"]].drop_duplicates()
    return dict(zip(pairs["location"], pairs["location_name"], strict=False))


def locations_by_size(target: pd.DataFrame, n: int | None = None) -> list[str]:
    """Return the places with cases in the latest vintage, largest first."""
    latest = target.loc[target["vintage"] == target["vintage"].max()]
    totals = (
        latest.groupby("location")["total"].sum().sort_values(ascending=False)
    )
    locs = list(totals.index[totals > 0])
    return locs if n is None else locs[:n]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def plot_target(
    target: pd.DataFrame,
    locations: Sequence[str] | None = None,
    mode: str = "light",  # noqa: ARG001 - shared signature
) -> Figure:
    """Plot weekly totals by onset week for each vintage, by location.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows of one kind.
    locations : sequence of str, optional
        Locations to show. Defaults to all with cases, largest first.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per location.
    """
    locs = list(locations or locations_by_size(target))
    vints = sorted(target["vintage"].unique())
    names = location_names(target)
    fig, axes = panel_grid(len(locs), 3)
    cmap = mpl.colormaps["Blues"]
    shades = np.linspace(0.35, 0.95, len(vints))
    for ax, loc in zip(axes, locs, strict=True):
        rows = target.loc[target["location"] == loc]
        for vin, shade in zip(vints, shades, strict=True):
            r = rows.loc[rows["vintage"] == vin]
            ax.plot(
                pd.to_datetime(r["target_end_date"]),
                r["total"],
                color=cmap(shade),
                lw=1.5,
                label=vin,
            )
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        date_axis(ax)
    axes[0].set_ylabel("Confirmed cases by onset week")
    axes[0].legend(title="Vintage", title_fontsize=8)
    return fig


def plot_delay_distribution(
    delays: pd.DataFrame, max_delay: int = 40, mode: str = "light"
) -> Figure:
    """Plot the onset-to-report delay distribution by province.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one vintage.
    max_delay : int
        Longest delay shown.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per province with at least 10 cases, with the count,
        mean and median in the title.
    """
    counts = delays["location"].value_counts()
    locs = list(counts.index[counts >= 10])
    names = location_names(delays)
    fig, axes = panel_grid(len(locs), 3)
    colour = MODEL_COLOURS[mode][0]
    bins = list(np.arange(-0.5, max_delay + 1.5))
    for ax, loc in zip(axes, locs, strict=True):
        d = delays.loc[delays["location"] == loc, "delay"]
        ax.hist(d.clip(upper=max_delay), bins=bins, density=True, color=colour)
        ax.set_title(
            f"{names.get(loc, loc)}: n {len(d)}, mean {d.mean():.1f}, "
            f"median {d.median():.0f}"
        )
        ax.set_xlim(-1, max_delay + 1)
    axes[0].set_ylabel("Share of cases")
    for ax in axes:
        ax.set_xlabel("Days from onset to report")
    return fig


def plot_renewal_delay(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    names: Mapping[str, str] | None = None,
    mode: str = "light",
) -> Figure:
    """Plot the renewal models' learnt delay at their latest origin.

    Parameters
    ----------
    summary : pandas.DataFrame
        `delay-summary.csv` rows (`nowcastrenewal.forecaster.
        delay_summary` with `model`, `vintage` and `location`).
    baseline : pandas.DataFrame
        `baseline_share` at the same release, with the country as
        `NATIONAL` for the fits pooled over provinces.
    names : mapping, optional
        Location names by code.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        A row per fit, pooled over provinces. Left: the share of onsets
        in the last week reported by each day since onset, to the day
        before the longest delay, with randomwalknowcast's share for the
        same place. Right: the mean delay by onset week. Lines are medians
        and bands 90% intervals.
    """
    names = dict(names or {})
    ink = THEMES[mode]
    latest = summary.groupby("model")["vintage"].transform("max")
    rows = summary.loc[summary["vintage"] == latest]
    models = list(dict.fromkeys(rows["model"]))
    places = list(dict.fromkeys(rows["location"]))
    places.sort(key=lambda p: (p != "shared", names.get(p, p)))
    fig, axes = plt.subplots(
        len(places),
        2,
        figsize=(7.2, 2.1 * len(places)),
        squeeze=False,
        layout="constrained",
    )
    last_day = int(pd.to_numeric(rows["x"], errors="coerce").max())
    for (left, right), place in zip(axes, places, strict=True):
        here = rows.loc[rows["location"] == place]
        fitted = list(dict.fromkeys(here["model"]))
        where = NATIONAL if place == "shared" else place
        title = "Shared delay" if place == "shared" else names.get(place, place)
        left.set_title(f"{title}: {', '.join(fitted)}", fontsize=8, loc="left")
        base = baseline.loc[baseline["location"] == where]
        left.plot(
            base["age"],
            base["share"],
            color=ink["data"],
            ls="--",
            lw=1,
            label="randomwalknowcast",
        )
        lines = []
        for model in fitted:
            fit = here.loc[here["model"] == model]
            colour = model_colour(model, models, mode)
            share = fit.loc[fit["quantity"] == "reported_share"]
            day = pd.to_numeric(share["x"])
            share = share.loc[day < last_day]
            day = day.loc[share.index]
            delay = fit.loc[fit["quantity"] == "mean_delay"]
            week = pd.to_datetime(delay["x"])
            for ax, x, q in ((left, day, share), (right, week, delay)):
                ax.fill_between(x, q["q5"], q["q95"], color=colour, alpha=0.12)
                lines.append((ax, x, q["q50"], colour, model))
        # Later models are drawn first and wider, so a line that coincides
        # with another stays visible under it.
        width = {m: 1.2 + 1.2 * i for i, m in enumerate(fitted)}
        for ax, x, y, colour, model in reversed(lines):
            ax.plot(x, y, color=colour, lw=width[model], label=model)
        left.set_ylim(0, 1)
        left.set_ylabel("Share reported")
        right.set_ylabel("Mean delay (days)")
        date_axis(right)
    axes[-1][0].set_xlabel("Days since onset, last onset week")
    vintage = ", ".join(sorted(rows["vintage"].astype(str).unique()))
    fig.suptitle(f"Fits at {vintage}", fontsize=9)
    handles: dict[str, Any] = {}
    for ax in axes[:, 0]:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, h)
    fig.legend(
        list(handles.values()),
        list(handles),
        loc="outside lower center",
        ncol=len(handles),
        fontsize=7,
        frameon=False,
    )
    return fig


def _bands(samples: pd.DataFrame) -> pd.DataFrame:
    grp = samples.groupby("target_end_date")["value"]
    out = grp.quantile(np.array([0.05, 0.25, 0.5, 0.75, 0.95])).unstack()
    out.index = pd.to_datetime(out.index)
    return out


INCOMPLETE_LABEL = "Still being reported"


def incomplete_after(vintage: object, weeks: int) -> pd.Timestamp:
    """Return the date after which weeks are not completely reported.

    Weeks complete once they ended `weeks` weeks before the vintage.
    The returned date is the middle of the first incomplete week.
    """
    last = pd.Timestamp(str(vintage)) - pd.Timedelta(weeks=weeks)
    return last + pd.Timedelta(days=3)


def shade_incomplete(ax: Axes, start: pd.Timestamp, mode: str) -> None:
    """Shade the weeks still being reported, from `start` to the axis end."""
    right = max(ax.get_xlim()[1], mdates.date2num(start))
    ax.axvspan(
        mdates.date2num(start),
        right + 1,
        color=THEMES[mode]["muted"],
        alpha=0.12,
        lw=0,
        label=INCOMPLETE_LABEL,
        zorder=0,
    )


def _plot_data(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    start: pd.Timestamp,
    colour: str,
    label: str,
    **kwargs: Any,
) -> None:
    """Draw data as a line with filled points, hollow where incomplete."""
    ax.plot(x, y, color=colour, label=label, **kwargs)
    done = x <= start
    ax.plot(
        x[done],
        y[done],
        "o",
        color=colour,
        ms=3,
        zorder=kwargs.get("zorder", 2),
    )
    ax.plot(
        x[~done],
        y[~done],
        "o",
        mfc="none",
        color=colour,
        ms=3,
        zorder=kwargs.get("zorder", 2),
    )


def draw_forecast(
    ax: Axes, samples: pd.DataFrame, colour: str, dash: str, label: str
) -> None:
    """Draw sample trajectories, 50% and 90% ribbons and the median.

    The median takes the model's dash, an SVG `stroke-dasharray`.
    """
    paths = samples.pivot_table(
        index="target_end_date", columns="sample", values="value"
    ).sort_index()
    x = pd.to_datetime(paths.index)
    for s in paths.columns:
        ax.plot(x, paths[s], color=colour, lw=0.4, alpha=0.08, zorder=1)
    q = _bands(samples)
    ax.fill_between(q.index, q[0.05], q[0.95], color=colour, alpha=0.18, lw=0)
    ax.fill_between(q.index, q[0.25], q[0.75], color=colour, alpha=0.35, lw=0)
    ax.plot(
        q.index, q[0.5], color=colour, ls=mpl_dash(dash), lw=1.5, label=label
    )


def _fit_y(ax: Axes, samples: pd.DataFrame, shown: Sequence[float]) -> None:
    """Set the y-axis from the data and the 95% band, not single paths.

    A few sample paths of a walk can reach far above the band, so the
    axis ends at the band's top, with room for at least one case.
    """
    top = max([*shown, float(_bands(samples)[0.95].max()), 1.0])
    ax.set_ylim(0, top * 1.08)


def legend_below(fig: Figure, ax: Axes) -> None:
    """Put the legend of `ax` under the figure, clear of the data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        fontsize=7,
        frameon=False,
    )


def plot_location_grid(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    locations: Sequence[str],
    models: Sequence[str] | None = None,
    truth_vintage: str | None = None,
    max_horizon: int | None = None,
    weeks_before: int = 10,
    ncol: int = 2,
    mode: str = "light",
    complete_after_weeks: int = 3,
) -> Figure:
    """Plot one origin's forecasts as a grid with a panel per location.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts` for one origin.
    target : pandas.DataFrame
        Target rows holding the origin and the truth vintage.
    locations : sequence of str
        Locations to show, one panel each.
    models : sequence of str, optional
        Models in registry order. Defaults to those in `forecasts`.
    truth_vintage : str, optional
        Vintage drawn as later data. Defaults to the latest real vintage.
    max_horizon : int, optional
        Last horizon drawn. `-1` shows the nowcast alone, without the
        reference week or the forecast weeks.
    weeks_before : int
        Weeks of data shown before the reference date.
    ncol : int
        Panels per row.
    mode : str
        `"light"` or `"dark"`.
    complete_after_weeks : int
        Weeks after which a week of the later data counts as complete.

    Returns
    -------
    matplotlib.figure.Figure
        Per panel, the data at the origin (dashed), the later data
        (solid) and each model's trajectories, intervals and median.

    Examples
    --------
    >>> from nowcasteval.plots import (
    ...     plot_location_grid, save_light_dark
    ... )
    >>> fig = plot_location_grid(forecasts, real, ["CD", "CD-IT"])
    >>> save_light_dark(
    ...     lambda mode: plot_location_grid(
    ...         forecasts, real, ["CD"], mode=mode
    ...     ),
    ...     "figures/grid.svg",
    ... )
    """
    fc = forecasts
    if max_horizon is not None:
        fc = fc.loc[fc["horizon"] <= max_horizon]
    models = list(models or fc["model"].unique())
    real = target.loc[target["kind"] == "real"]
    truth_v = truth_vintage or real["vintage"].max()
    vintage = fc["vintage"].iloc[0]
    ref = pd.Timestamp(fc["reference_date"].iloc[0])
    end = pd.Timestamp(fc["target_end_date"].max())
    start = ref - pd.Timedelta(weeks=weeks_before)
    names = location_names(real)
    ink = THEMES[mode]
    cut = incomplete_after(truth_v, complete_after_weeks)
    fig, axes = panel_grid(len(locations), ncol, width=3.6, height=2.5)
    shown_data = [(truth_v, "-", f"Data {truth_v}")]
    if vintage != truth_v:
        shown_data.append((vintage, "--", "Data at the origin"))
    for ax, loc in zip(axes, locations, strict=True):
        top: list[float] = []
        for v, style, lab in shown_data:
            rows = real.loc[(real["vintage"] == v) & (real["location"] == loc)]
            x = pd.to_datetime(rows["target_end_date"])
            sel = (x >= start) & (x <= end)
            top.append(float(rows["total"][sel].max()) if sel.any() else 0.0)
            if style == "-":
                _plot_data(
                    ax,
                    x[sel],
                    rows["total"][sel],
                    cut,
                    ink["data"],
                    lab,
                    ls=style,
                    lw=1.2,
                )
            else:
                ax.plot(
                    x[sel],
                    rows["total"][sel],
                    color=ink["data"],
                    ls=style,
                    lw=1.2,
                    label=lab,
                )
        for m in models:
            s = fc.loc[(fc["location"] == loc) & (fc["model"] == m)]
            if not s.empty:
                draw_forecast(
                    ax, s, model_colour(m, models, mode), model_dash(m), m
                )
                _fit_y(ax, s, top)
        ax.axvline(mdates.date2num(ref), color=ink["muted"], lw=0.8, ls=":")
        shade_incomplete(ax, cut, mode)
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel("Cases by onset week")
    legend_below(fig, axes[0])
    return fig


def plot_forecasts(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    location: str,
    models: Sequence[str] | None = None,
    kind: str = "real",
    truth_vintage: str | None = None,
    weeks_before: int = 10,
    mode: str = "light",
    complete_after_weeks: int = 3,
) -> Figure:
    """Plot forecasts against data at the time and later data, by date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `nowcasteval.backtest.run_forecasts`.
    target : pandas.DataFrame
        Target rows holding each forecast's vintage and the truth vintage.
    location : str
        Location to show.
    models : sequence of str, optional
        Models in registry order. Defaults to those in `forecasts`.
    kind : str
        Vintage kind to show.
    truth_vintage : str, optional
        Vintage drawn as later data. Defaults to the latest real vintage.
    weeks_before : int
        Weeks of data shown before each forecast.
    mode : str
        `"light"` or `"dark"`.
    complete_after_weeks : int
        Weeks after which a week of the later data counts as complete.
        Later weeks are shaded and their points drawn hollow.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per reference date: data at the time (dashed), later
        data (solid) and each model's sample trajectories, 50% and 90%
        intervals and median.

    Examples
    --------
    >>> from nowcasteval.plots import plot_forecasts
    >>> fig = plot_forecasts(forecasts, real, "CD")  # from `run_forecasts`
    >>> fig.savefig("national.png")
    """
    fc = forecasts.loc[
        (forecasts["location"] == location) & (forecasts["kind"] == kind)
    ]
    models = list(models or fc["model"].unique())
    real = target.loc[target["kind"] == "real"]
    truth_v = truth_vintage or real["vintage"].max()
    truth = real.loc[
        (real["vintage"] == truth_v) & (real["location"] == location)
    ]
    refs = sorted(fc["reference_date"].unique())
    ink = THEMES[mode]
    cut = incomplete_after(truth_v, complete_after_weeks)
    fig, axes = panel_grid(len(refs), 4, width=3.0, height=2.3)
    for ax, ref in zip(axes, refs, strict=True):
        top: list[float] = []
        f_ref = fc.loc[fc["reference_date"] == ref]
        vintage = f_ref["vintage"].iloc[0]
        start = pd.Timestamp(ref) - pd.Timedelta(weeks=weeks_before)
        end = pd.Timestamp(f_ref["target_end_date"].max())
        at_time = target.loc[
            (target["kind"] == kind)
            & (target["vintage"] == vintage)
            & (target["location"] == location)
        ]
        shown = [(truth, "-", f"Data {truth_v}")]
        if vintage != truth_v:
            shown.append((at_time, "--", "Data at the time"))
        for rows, style, lab in shown:
            x = pd.to_datetime(rows["target_end_date"])
            sel = (x >= start) & (x <= end)
            top.append(float(rows["total"][sel].max()) if sel.any() else 0.0)
            if style == "-":
                _plot_data(
                    ax,
                    x[sel],
                    rows["total"][sel],
                    cut,
                    ink["data"],
                    lab,
                    ls=style,
                    lw=1.2,
                )
                continue
            ax.plot(
                x[sel],
                rows["total"][sel],
                color=ink["data"],
                ls=style,
                lw=1.2,
                label=lab,
            )
        drawn = False
        for m in models:
            s = f_ref.loc[f_ref["model"] == m]
            if not s.empty:
                draw_forecast(
                    ax, s, model_colour(m, models, mode), model_dash(m), m
                )
                _fit_y(ax, s, top)
                drawn = True
        if not drawn:
            ax.text(
                0.5,
                0.92,
                "No forecast from this origin",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                color=ink["muted"],
            )
        ax.axvline(
            mdates.date2num(pd.Timestamp(ref)),
            color=ink["muted"],
            lw=0.8,
            ls=":",
        )
        shade_incomplete(ax, cut, mode)
        ax.set_title(f"{location}, origin {vintage}")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel("Cases by onset week")
    legend_below(fig, axes[0])
    return fig


def comparison_label(model: str, source: str) -> str:
    """Label a model in the hub comparison.

    Parameters
    ----------
    model : str
        Hub model id or our registry name.
    source : str
        `hub`, `submitted` (our file in the hub, by hub id), or the kind of
        vintage our main build used.

    Returns
    -------
    str
        A hub team's id, `<hub id> (submitted)` or `<model> (main build)`.
    """
    if source == "hub":
        return model
    if source == "submitted":
        return f"{model} (submitted)"
    return f"{model} (main build)"


def comparison_colours(
    ours: Mapping[str, str],
    submitted: Mapping[str, str],
    teams: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    """Return the light colour, dark colour and dash of each comparison series.

    Parameters
    ----------
    ours : mapping of str to str
        Label of each of our main builds to its registry name.
    submitted : mapping of str to str
        Label of each of our submitted series to the registry name of the
        model it submits, whose colour it takes, dashed.
    teams : sequence of str
        Every hub team.

    Returns
    -------
    dict
        Label to its light colour, dark colour and SVG dash: our main
        builds in their `model_style`, our submitted series in their
        model's colour dashed, and the hub teams solid in the greys of
        `hub_styles`.
    """
    from nowcastutils.base import list_models

    registry = [s.name for s in list_models()]
    out = {lab: model_style(m, registry) for lab, m in ours.items()}
    out |= {
        lab: (*model_style(m, registry)[:2], DASHES["dashed"])
        for lab, m in submitted.items()
    }
    grey = {mode: hub_styles(teams, mode) for mode in ("light", "dark")}
    out |= {t: (grey["light"][t][0], grey["dark"][t][0], "") for t in teams}
    return out


def hub_styles(teams: Sequence[str], mode: str) -> dict[str, tuple[str, str]]:
    """Give each hub team a grey and a marker, fixed by its sorted place.

    Parameters
    ----------
    teams : sequence of str
        Hub team names.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    dict of str to tuple of str
        Each team's colour and marker.
    """
    greys = HUB_GREYS[mode]
    markers = "os^Dv"
    return {
        team: (greys[i % len(greys)], markers[i // len(greys) % len(markers)])
        for i, team in enumerate(sorted(teams))
    }


def plot_hub_round(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    reference_date: str,
    colours: Mapping[str, tuple[str, str, str]],
    locations: Sequence[str] = ("CD",),
    complete_after_weeks: int = 3,
    weeks_before: int = 8,
    mode: str = "light",
    shown: Sequence[str] | None = None,
) -> Figure:
    """Plot our and the hub teams' intervals for one round against the data.

    Our series are in the top row, in their `comparison_colours`, and the
    hub teams in the bottom row, in greys with a marker each. Colours and
    markers do not depend on the round.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        Output of `bdbvhub.compare.forecast_quantiles`,
        with every round, so a team keeps its style across rounds.
    target : pandas.DataFrame
        Target rows; the latest real vintage is drawn as data.
    reference_date : str
        The round.
    locations : sequence of str
        Locations to show, one column each, if any model forecast them.
    complete_after_weeks : int
        Weeks after which a week of the data counts as complete.
        Later weeks are shaded and their points drawn hollow.
    weeks_before : int
        Weeks of data shown before the reference date.
    mode : str
        `"light"` or `"dark"`.
    colours : mapping of str to (str, str, str)
        Our series' labels (`comparison_label`) to their light
        colour, dark colour and dash, in the order drawn. Every series of
        ours in the round needs one.
    shown : sequence of str, optional
        Labels of our series to draw. Defaults to all of ours in the
        round.

    Returns
    -------
    matplotlib.figure.Figure
        Per week and model, the median (point), 50% interval (thick line)
        and 90% interval (thin line), side by side, over the latest release,
        with the legend below the panels.
    """
    styles = hub_styles(
        sorted(set(quantiles.loc[quantiles["source"] == "hub", "model"])), mode
    )
    q = quantiles.loc[quantiles["reference_date"].astype(str) == reference_date]
    q = q.assign(
        label=[
            comparison_label(m, s)
            for m, s in zip(q["model"], q["source"], strict=True)
        ]
    )
    hub = q["source"] == "hub"
    if shown is not None:
        q = q.loc[hub | q["label"].isin(list(shown))]
        hub = q["source"] == "hub"
    here = set(q.loc[~hub, "label"])
    pick = 0 if mode == "light" else 1
    rows = [
        {
            lab: (c[pick], "o", c[2])
            for lab, c in colours.items()
            if lab in here
        },
        {lab: (*styles[lab], "") for lab in sorted(set(q.loc[hub, "label"]))},
    ]
    locs = [loc for loc in locations if loc in set(q["location"])]
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    names = location_names(real)
    ink = THEMES[mode]
    cut = incomplete_after(latest, complete_after_weeks)
    n_col = max(len(locs), 1)
    fig, axes = plt.subplots(
        2,
        n_col,
        figsize=(4.0 * n_col, 5.6),
        squeeze=False,
        sharex=True,
        sharey="col",
        layout="constrained",
    )
    ref = pd.Timestamp(reference_date)
    start = ref - pd.Timedelta(weeks=weeks_before)
    for j, loc in enumerate(locs):
        ql = q.loc[q["location"] == loc]
        data = real.loc[(real["vintage"] == latest) & (real["location"] == loc)]
        x = pd.to_datetime(data["target_end_date"])
        end = pd.to_datetime(ql["target_end_date"]).max()
        sel = (x >= start) & (x <= end)
        for i, (ax, style) in enumerate(zip(axes[:, j], rows, strict=True)):
            _plot_data(
                ax,
                x[sel],
                data["total"][sel],
                cut,
                ink["data"],
                f"Data {latest}",
                lw=1.2,
                zorder=3,
            )
            width = 5.0 / max(len(style), 1)
            for k, (lab, (colour, marker, dash)) in enumerate(style.items()):
                m = ql.loc[ql["label"] == lab]
                offset = pd.Timedelta(days=(k - (len(style) - 1) / 2) * width)
                xm = pd.to_datetime(m["target_end_date"]) + offset
                ls = mpl_dash(dash)
                ax.vlines(
                    xm, m["q05"], m["q95"], color=colour, lw=1, alpha=0.7, ls=ls
                )
                ax.vlines(xm, m["q25"], m["q75"], color=colour, lw=3, ls=ls)
                ax.plot(
                    xm,
                    m["q50"],
                    marker,
                    ms=4,
                    ls="",
                    label=lab,
                    **point_style(colour, dash),
                )
            ax.axvline(mdates.date2num(ref), color=ink["muted"], lw=0.8, ls=":")
            shade_incomplete(ax, cut, mode)
            title = ("Ours", "Hub teams")[i]
            ax.set_title(f"{names.get(loc, loc)} ({loc}): {title}", fontsize=9)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Confirmed cases by onset week")
    seen: dict[str, Any] = {}
    for ax in axes[:, 0]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen),
        loc="outside lower center",
        ncol=min(len(seen), 4),
        fontsize=7,
        frameon=False,
    )
    return fig


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


def plot_scores(
    scores: pd.DataFrame,
    by: str = "horizon",
    metric: str = "crps",
    models: Sequence[str] | None = None,
    mode: str = "light",
    colours: Mapping[str, tuple[str, str, str]] | None = None,
    baseline: str | None = None,
) -> Figure:
    """Plot mean scores by horizon or location, one panel per vintage kind.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : str
        `"horizon"` or `"location"`.
    metric : str
        A score column, such as `"crps"`, `"log_crps"`, `"coverage_90"` or
        `"bias"`.
    models : sequence of str, optional
        Models in registry order.
    mode : str
        `"light"` or `"dark"`.
    colours : mapping of str to (str, str, str), optional
        Model to its light colour, dark colour and dash. Defaults to
        `model_style`.

    baseline : str, optional
        The model relative metrics are relative to, named in the axis
        label; needed for a `relative_` metric.

    Returns
    -------
    matplotlib.figure.Figure
        Mean score by `by`, with the number of forecasts in the title.
    """
    models = list(models or scores["model"].unique())
    kinds = [k for k in ("real", "simulated") if k in set(scores["kind"])]
    fig, axes = panel_grid(len(kinds), 3, width=3.6, height=2.6)
    for ax, kind in zip(axes, kinds, strict=True):
        s = scores.loc[scores["kind"] == kind]
        mean = s.groupby(["model", by])[metric].mean().unstack(0)
        if by == "location":
            order = s.groupby("location")["truth"].sum().sort_values()
            mean = mean.reindex(order.index[::-1])
        x = np.arange(len(mean))
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            if m not in mean.columns:
                continue
            col, dash = (
                (colours[m][0 if mode == "light" else 1], colours[m][2])
                if colours
                else (model_colour(m, models, mode), model_dash(m))
            )
            if by == "horizon":
                ax.plot(
                    mean.index,
                    mean[m],
                    color=col,
                    ls=mpl_dash(dash),
                    marker=_KIND_MARKERS[kind],
                    ms=5,
                    label=m,
                )
            else:
                ax.bar(
                    x + i * width,
                    mean[m],
                    width,
                    label=m,
                    **bar_style(col, dash),
                )
        if by != "horizon":
            ax.set_xticks(
                x + width * (len(models) - 1) / 2,
                [str(v).removeprefix("CD-") for v in mean.index],
            )
        if metric.startswith("coverage"):
            level = 0.5 if metric.endswith("50") else 0.9
            ax.axhline(level, color=THEMES[mode]["muted"], lw=1, ls=":")
        if metric == "bias":
            ax.axhline(0, color=THEMES[mode]["muted"], lw=1, ls=":")
        ax.set_title(
            f"{kind}: {s[['vintage']].nunique().iloc[0]} vintages, n {len(s)}"
        )
        ax.set_xlabel(by.capitalize())
    axes[0].set_ylabel(_metric_label(metric, baseline))
    axes[0].legend()
    return fig


def plot_scores_by_truth(
    scores: pd.DataFrame,
    metric: str = "crps",
    models: Sequence[str] | None = None,
    mode: str = "light",
    baseline: str | None = None,
) -> Figure:
    """Plot how each forecast's mean score moves as the truth is revised.

    Parameters
    ----------
    scores : pandas.DataFrame
        Scores with `model`, `vintage`, `truth_vintage` and `metric`.
    metric : str
        A score column.
    models : sequence of str, optional
        Models in registry order, one panel each.
    mode : str
        `"light"` or `"dark"`.

    baseline : str, optional
        The model relative metrics are relative to, named in the axis
        label; needed for a `relative_` metric.

    Returns
    -------
    matplotlib.figure.Figure
        Mean score against the truth vintage, one line per forecast
        vintage.
    """
    models = list(models or scores["model"].unique())
    models = [m for m in models if m in set(scores["model"])]
    fig, axes = panel_grid(len(models), 3, width=3.6, height=2.6)
    vintages = sorted(set(scores["vintage"]))
    for ax, m in zip(axes, models, strict=True):
        s = scores.loc[scores["model"] == m]
        mean = s.groupby(["vintage", "truth_vintage"])[metric].mean()
        for v in vintages:
            if v not in mean.index.get_level_values(0):
                continue
            line = mean.loc[v]
            ax.plot(
                pd.to_datetime(line.index),
                line.to_numpy(),
                color=model_colour(v, vintages, mode),
                marker="o",
                ms=5,
                label=f"Forecast from {v}",
            )
        ax.set_title(m)
        ax.set_xlabel("Truth vintage")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel(_metric_label(metric, baseline))
    axes[0].legend(fontsize=7)
    return fig


_METRIC_LABELS: Final[dict[str, str]] = {
    "crps": "Mean CRPS",
    "log_crps": "Mean CRPS, log scale",
    "coverage_50": "50% interval coverage",
    "coverage_90": "90% interval coverage",
    "bias": "Mean bias",
}


def _metric_label(metric: str, baseline: str | None) -> str:
    """Return a metric's axis label; relative ones name the baseline."""
    if metric.startswith("relative_") and baseline is None:
        msg = f"{metric} needs the baseline it is relative to"
        raise ValueError(msg)
    base = baseline
    return {
        **_METRIC_LABELS,
        "relative_crps": f"CRPS relative to {base}",
        "relative_log_crps": f"CRPS on the log scale relative to {base}",
    }.get(metric, metric)
