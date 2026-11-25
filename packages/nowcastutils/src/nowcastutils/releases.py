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
`cumulative_first_release` finds a first release that is a cumulative dump
rather than part of the reporting, and `without_cumulative_first_release`
leaves it out of a task's releases.
`origin_releases` gives the releases forecasting starts from: those with
at least `MIN_ORIGIN_RELEASES` usable releases up to them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from nowcastutils.base import ForecastTask
from nowcastutils.dates import as_date

__all__ = [
    "HISTORY_FIELDS",
    "MAX_PAIR_AGE",
    "MIN_ORIGIN_RELEASES",
    "cumulative_first_release",
    "latest_onset_days",
    "origin_releases",
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


def release_cutoffs(
    rows: pd.DataFrame, releases: Sequence[str]
) -> list[dt.date]:
    """Return each release's data cut-off, as `release_cutoff` does.

    One pass over `rows`, where `release_cutoff` compares every row for
    each release.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows with `vintage` and `cutoff` columns.
    releases : sequence of str
        The release labels.

    Returns
    -------
    list of datetime.date
        The cut-offs, in the order of `releases`.

    Raises
    ------
    KeyError
        If a release has no rows.
    """
    first = rows.drop_duplicates("vintage")
    at = {
        str(v): str(c)
        for v, c in zip(first["vintage"], first["cutoff"], strict=True)
    }
    return [as_date(at[str(r)]) for r in releases]


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


MIN_ORIGIN_RELEASES = 2
"""Usable releases up to a release for it to be a forecast origin
(`origin_releases`)."""


def origin_releases(
    rows: pd.DataFrame,
    max_delay: int | None,
    min_releases: int = MIN_ORIGIN_RELEASES,
) -> list[str]:
    """Return the releases forecasting starts from, oldest first.

    The usable releases are every release less a cumulative first
    release (`cumulative_first_release`). A release is an origin when at
    least `min_releases` usable releases are on or before it, so the
    models that learn from pairs of releases have one at every origin.

    Parameters
    ----------
    rows : pandas.DataFrame
        Releases, with `vintage` and `cutoff`.
    max_delay : int or None
        $D$ in days, for `cumulative_first_release`.
    min_releases : int, optional
        Usable releases needed up to an origin.

    Returns
    -------
    list of str
        The origins' labels, in cut-off order. Empty if there are fewer
        than `min_releases` usable releases.
    """
    if rows.empty:
        return []
    cut = rows.drop_duplicates("vintage")
    cut = cut.assign(cutoff=pd.to_datetime(cut["cutoff"])).sort_values("cutoff")
    labels = [str(v) for v in cut["vintage"]]
    first = cumulative_first_release(rows, max_delay)
    usable = [v for v in labels if v != first]
    return usable[min_releases - 1 :]


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
    tables: dict[str, Any] = {}
    for name in HISTORY_FIELDS:
        table = getattr(task, name)
        if table is not None and "vintage" in table.columns:
            kept = table.loc[table["vintage"].astype(str) != first]
            tables[name] = kept.reset_index(drop=True) if len(kept) else None
    return dataclasses.replace(task, **tables)
