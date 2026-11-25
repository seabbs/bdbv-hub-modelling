# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Release cut-offs and the renewal model's report to publication lag.

A release's data cut-off is its latest onset day over all places.
The last day or two before extraction are thin, as new reports take time
to reach the line list. `latest_onset_days` derives the cut-off from the
data, whatever the label, and every tidy table carries it as a `cutoff`
column, which `release_cutoff` reads.
`release_pairs` tabulates, for each pair of consecutive delay files, which
rows of the later file were already in the earlier one, the data the
renewal models learn their report to publication lag from.
`pseudo_releases` adds a weekly release history to daily releases with
gaps, from the latest release and the delay line (`pseudo_cutoffs`).
`cumulative_first_release` finds a first release that is a cumulative dump
rather than part of the reporting, and `without_cumulative_first_release`
leaves it out of a task's releases.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import logging

import numpy as np
import pandas as pd

from nowcastutils.base import ForecastTask
from nowcastutils.dates import as_date, epiweek_end
from nowcastutils.hierarchy import parent

__all__ = [
    "HISTORY_FIELDS",
    "MAX_PAIR_AGE",
    "PSEUDO_STEP",
    "cumulative_first_release",
    "latest_onset_days",
    "pseudo_cutoffs",
    "pseudo_releases",
    "release_cutoff",
    "release_pairs",
    "without_cumulative_first_release",
]

log = logging.getLogger(__name__)


def latest_onset_days(daily: pd.DataFrame) -> pd.Series:
    """Return each release's data cut-off, its latest onset day.

    The cut-off is the latest onset day over all places in the release's
    daily onsets, one date per release.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with `vintage` and `onset_date`.

    Returns
    -------
    pandas.Series
        The cut-off (ISO) by release label, for every release in `daily`.
    """
    onset = pd.to_datetime(daily["onset_date"]).groupby(daily["vintage"]).max()
    out: pd.Series = onset.dt.strftime("%Y-%m-%d").rename("cutoff")
    return out


def release_cutoff(rows: pd.DataFrame, release: str) -> dt.date:
    """Return a release's data cut-off, its latest onset day.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows with `vintage` and `cutoff` columns
        (`nowcastbdbv.data.tidy.load_target`).
    release : str
        The release label.

    Returns
    -------
    datetime.date
        The cut-off.
    """
    cut = rows.loc[rows["vintage"] == release, "cutoff"]
    return as_date(str(cut.iloc[0]))


MAX_PAIR_AGE = 60
"""Oldest report age at the earlier release of a pair kept by
`release_pairs`, in days, the renewal models' maximum delay."""


def release_pairs(
    delays: pd.DataFrame,
    as_of: dt.date | str,
    max_age: int = MAX_PAIR_AGE,
) -> dict[str, np.ndarray] | None:
    r"""Rows of each delay file and whether the previous file had them.

    For each pair of consecutive delay files $k$ and $k + 1$ on or before
    `as_of`, every row cell (location, onset week, delay) of file $k + 1$
    is counted with how many of its rows file $k$ already had. A row with
    onset week ending $W$ and delay $d$ was reported on $W - u + d$ for
    an onset $u = 0..6$ days before $W$, so its report age at a release
    with cut-off $T$ is $T - W + u - d$. Cells that no onset day
    could have reported by $T_k$, or that are older than `max_age` at
    $T_k$ for every onset day, are left out.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows of several releases, with `vintage`, `cutoff`,
        `location`, `onset_week` and `delay`.
    as_of : datetime.date or str
        The forecast date. Later files are not used.
    max_age : int, optional
        Oldest report age kept, in days.

    Returns
    -------
    dict or None
        `before` and `after`, report ages at $T_k$ and $T_{k + 1}$ by
        onset day, `(n, 7)`; `kept` and `total` rows of each cell,
        `(n,)`; and `pair`, the one-hot pair of each cell, `(n, K)`.
        None if no pair of files is available.
    """
    day = as_date(as_of).isoformat()
    vintages = sorted(str(v) for v in delays["vintage"].unique() if v <= day)
    key = ["location", "onset_week", "delay"]
    parts = []
    for a, b in itertools.pairwise(vintages):
        seen = delays.loc[delays["vintage"] == a].groupby(key).size()
        rows = (
            delays.loc[delays["vintage"] == b]
            .groupby(key)
            .size()
            .reset_index(name="total")
        )
        had = seen.reindex(
            pd.MultiIndex.from_frame(rows[key]), fill_value=0
        ).to_numpy()
        # Report day of each row for an onset 0 to 6 days before the end
        # of its onset week, as days since the epoch.
        report = (
            pd.to_datetime(rows["onset_week"]).to_numpy().astype("datetime64[D]")
            .astype(int)[:, None]
            - np.arange(7)[None, :]
            + rows["delay"].to_numpy()[:, None]
        )  # fmt: skip

        def age(release: str, report: np.ndarray = report) -> np.ndarray:
            cut = np.datetime64(release_cutoff(delays, release), "D")
            return np.asarray(cut.astype(int) - report, dtype=int)

        before = age(a)
        keep = (before.max(axis=1) >= 0) & (before.min(axis=1) <= max_age)
        total = rows["total"].to_numpy()
        parts.append(
            {
                "before": before[keep],
                "after": age(b)[keep],
                "kept": np.minimum(had, total)[keep],
                "total": total[keep],
            }
        )
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["kept"] = out["kept"].astype(float)
    out["total"] = out["total"].astype(float)
    sizes = [len(p["total"]) for p in parts]
    out["pair"] = np.repeat(np.eye(len(parts)), sizes, axis=0)
    return out


PSEUDO_STEP = 7
"""Days between pseudo releases (`pseudo_cutoffs`)."""


def pseudo_cutoffs(
    cutoffs: list[str], first_onset: str, step: int = PSEUDO_STEP
) -> list[str]:
    """Return the cut-offs of the pseudo releases a release history needs.

    Every `step` days back from each release while `step` days or more
    remain after the release before it, so no gap between releases is
    `2 * step` days or more, and every `step` days back from the first
    release to `first_onset`. Releases `step` to `2 * step - 1` days apart
    get none between them.

    Parameters
    ----------
    cutoffs : list of str
        The releases' cut-offs, ISO dates.
    first_onset : str
        The earliest onset day, ISO date.
    step : int
        Days between pseudo releases.

    Returns
    -------
    list of str
        The pseudo cut-offs, ISO dates, in order.
    """
    days = sorted({as_date(c) for c in cutoffs})
    start = as_date(first_onset)
    out = []
    for i, cut in enumerate(days):
        floor = start if i == 0 else days[i - 1] + dt.timedelta(days=step)
        c = cut - dt.timedelta(days=step)
        while c >= floor:
            out.append(c)
            c -= dt.timedelta(days=step)
    return [c.isoformat() for c in sorted(out)]


def _delay_cdf(
    delays: pd.DataFrame, groups: list[str | None], weeks: list[str], top: int
) -> np.ndarray:
    """Return the share of cases reported by each delay, per group and week.

    `groups` are delay-line locations, None for every location. A group
    and week with no case takes the group's cases of every week, and a
    group with none every location's.
    """
    d = delays.assign(delay=delays["delay"].clip(lower=0, upper=top))
    width = top + 1

    def cdf(rows: pd.DataFrame) -> np.ndarray | None:
        if rows.empty:
            return None
        n = np.bincount(rows["delay"].to_numpy(dtype=int), minlength=width)
        return np.cumsum(n) / n.sum()

    every = cdf(d)
    assert every is not None
    by_group = {
        g: d if g is None else d.loc[d["location"] == g] for g in set(groups)
    }
    out = np.empty((len(groups), width))
    for i, (g, w) in enumerate(zip(groups, weeks, strict=True)):
        mine = by_group[g]
        f = cdf(mine.loc[mine["onset_week"] == w])
        if f is None:
            f = cdf(mine)
        out[i] = every if f is None else f
    return out


def pseudo_releases(
    daily: pd.DataFrame,
    delays: pd.DataFrame | None,
    step: int = PSEUDO_STEP,
) -> pd.DataFrame:
    r"""Add weekly pseudo releases to daily releases from the delay line.

    A pseudo release with cut-off $c$ (`pseudo_cutoffs`, from the
    releases' cut-offs and the latest release's first onset) gives each
    place and onset day $t \le c$ the latest release's count times the
    share $F(c - t)$ of the delay line's cases of the place's group and
    $t$'s onset week reported within $c - t$ days of onset. A place's
    group is the nearest of it and its parents (`hierarchy.parent`) with
    delay-line cases, else every place; an onset week with no case takes
    the group's other weeks (`_delay_cdf`). Negative delays count as
    zero. Observed and imputed onsets are scaled alike. The releases
    given are kept as they are; zero counts are left out.

    Only the releases and delay line passed in are read, so a forecast
    passing those available at its cut-off uses no later data.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily releases: `vintage`, `location`, `onset_date`, `observed`,
        `imputed` and `cutoff`.
    delays : pandas.DataFrame or None
        The delay line, one row per case: `location`, `onset_week` (ISO
        Saturday) and `delay` (days from onset to report). None or empty
        adds no pseudo release.
    step : int
        Days between pseudo releases.

    Returns
    -------
    pandas.DataFrame
        `daily` with the pseudo releases after it, each labelled by its
        cut-off.
    """
    if delays is None or delays.empty or daily.empty:
        return daily
    latest = daily.loc[daily["vintage"] == daily["vintage"].max()]
    real = daily.drop_duplicates("vintage").sort_values("vintage")
    cuts = pseudo_cutoffs(
        [str(c) for c in real["cutoff"]],
        str(latest["onset_date"].min()),
        step,
    )
    # A pseudo release sorts after the release before it, by label.
    labels = real["vintage"].astype(str).tolist()
    ends = [str(c) for c in real["cutoff"]]
    cuts = [
        c
        for c in cuts
        if c not in labels
        and all(
            lab < c for lab, end in zip(labels, ends, strict=True) if end < c
        )
    ]
    if not cuts:
        return daily
    have = set(delays["location"])

    def group(place: str) -> str | None:
        code: str | None = place
        while code is not None and code not in have:
            code = parent(code)
        return code

    place_group = {p: group(p) for p in latest["location"].unique()}
    week = {
        t: epiweek_end(t).isoformat() for t in latest["onset_date"].unique()
    }
    keys = pd.DataFrame(
        {
            "g": latest["location"].map(place_group),
            "w": latest["onset_date"].map(week),
        }
    )
    key_of, uniq = pd.factorize(
        keys["g"].fillna("\0") + "|" + keys["w"], sort=False
    )
    pairs = [u.split("|") for u in uniq]
    top = int(max(delays["delay"].max(), 0))
    cdf = _delay_cdf(
        delays,
        [None if g == "\0" else g for g, _ in pairs],
        [w for _, w in pairs],
        top,
    )
    onset = pd.to_datetime(latest["onset_date"]).to_numpy("datetime64[D]")
    observed = latest["observed"].to_numpy(dtype=float)
    imputed = latest["imputed"].to_numpy(dtype=float)
    parts = [daily]
    for c in cuts:
        age = (np.datetime64(c, "D") - onset).astype(int)
        keep = age >= 0
        share = cdf[key_of[keep], np.minimum(age[keep], top)]
        obs, imp = observed[keep] * share, imputed[keep] * share
        some = obs + imp > 0
        parts.append(
            pd.DataFrame(
                {
                    "vintage": c,
                    "location": latest["location"].to_numpy()[keep][some],
                    "onset_date": latest["onset_date"].to_numpy()[keep][some],
                    "observed": obs[some],
                    "imputed": imp[some],
                    "cutoff": c,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


HISTORY_FIELDS = (
    "target_history",
    "daily_history",
    "zone_history",
    "zone_daily_history",
    "delay_history",
)
"""The task's release tables, each with a `vintage` column."""


def cumulative_first_release(
    rows: pd.DataFrame | None, max_delay: int | None
) -> str | None:
    """Return the first release if it is a cumulative dump, else None.

    The first release is a cumulative dump when the next release's
    cut-off is more than $D$ (`max_delay`) days after its own. Every case
    it holds is then older than $D$ by the next release, so it shares no
    report within the delay window with any later release: it records
    the counts to date, not how they were reported. How far its onsets
    reach back is no test, as the first release of any data that start
    after the outbreak reaches back to its start, and so does the release
    after it.

    Parameters
    ----------
    rows : pandas.DataFrame or None
        Releases, with `vintage` and `cutoff`.
    max_delay : int or None
        $D$ in days. None finds no dump.

    Returns
    -------
    str or None
        The first release's label, or None if it is not a dump, there is
        no later release or no $D$.
    """
    if max_delay is None or rows is None or rows.empty:
        return None
    cut = rows.drop_duplicates("vintage")
    cut = cut.assign(cutoff=pd.to_datetime(cut["cutoff"])).sort_values("cutoff")
    if len(cut) < 2:
        return None
    gap = (cut["cutoff"].iloc[1] - cut["cutoff"].iloc[0]).days
    return str(cut["vintage"].iloc[0]) if gap > max_delay else None


def without_cumulative_first_release(task: ForecastTask) -> ForecastTask:
    """Return the task without a cumulative first release in its releases.

    The first release of the task's releases (its daily releases, else
    its weekly ones) is left out of every release table
    (`HISTORY_FIELDS`) when it is a cumulative dump
    (`cumulative_first_release`, with the task's `max_delay`), so the
    history starts at the next release. Logged when it is.

    Parameters
    ----------
    task : ForecastTask
        The task.

    Returns
    -------
    ForecastTask
        The task, unchanged if its first release is not a dump.
    """
    daily = task.daily_history
    rows = daily if daily is not None and not daily.empty else None
    first = cumulative_first_release(
        task.target_history if rows is None else rows, task.max_delay
    )
    if first is None:
        return task
    log.info(
        "%s: release %s left out, a cumulative first release more than "
        "%d days before the next",
        task.vintage,
        first,
        task.max_delay,
    )
    tables = {}
    for name in HISTORY_FIELDS:
        table = getattr(task, name)
        if table is not None and "vintage" in table.columns:
            kept = table.loc[table["vintage"].astype(str) != first]
            tables[name] = kept.reset_index(drop=True) if len(kept) else None
    return dataclasses.replace(task, **tables)
