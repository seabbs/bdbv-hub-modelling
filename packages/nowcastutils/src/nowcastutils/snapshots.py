# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Daily release snapshots as arrays on the onset clock.

The hub's `data/daily.csv` holds one table of daily onsets for each
release.
A release $r$ has an as-of date $A_r$ and a cut-off $c_r$, its last
report day (`nowcastutils.releases.release_cutoff`).
Its cell for onset day $t$ is observed when $t \le c_r$.
An observed day with no row counts 0.
The age of day $t$ at release $r$ is $c_r - t$, the onset clock.
Counts are observed plus imputed onsets, which sum to the hub's weekly
target.
Hub weeks run Sunday to Saturday and are labelled by the Saturday.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from nowcastutils.dates import as_date, epiweek_end
from nowcastutils.hierarchy import depth
from nowcastutils.hierarchy import parent as parent_of
from nowcastutils.releases import release_cutoff

__all__ = [
    "SnapshotArrays",
    "blocks",
    "snapshot_arrays",
    "week_sum",
    "weekly_counts",
    "weekly_sums",
]


@dataclass(frozen=True)
class SnapshotArrays:
    """Daily release snapshots as arrays.

    Attributes
    ----------
    locations : tuple of str
        The $L$ nodes, parents before children.
    releases : tuple of str
        The $R$ release dates, oldest first.
    days : pandas.DatetimeIndex
        The $T$ onset days, from the first onset to the Saturday of the
        week holding the latest release's cut-off.
    counts : numpy.ndarray
        Observed plus imputed onsets, `[L, R, T]`. Cells a release does
        not observe are 0.
    observed : numpy.ndarray
        Whether a release observes an onset day (`t <= cut`), `[R, T]`.
    cut : numpy.ndarray
        Day index of each release's cut-off, `[R]`.
    week : numpy.ndarray
        Hub week index of each day, `[T]`.
    week_ends : tuple of str
        The $W$ hub weeks, labelled by their Saturday.
    parent : numpy.ndarray
        Index of each node's parent, or -1, `[L]`.
    period : int
        Days in each block of `week`: 7 for weeks, 1 for days
        (`blocks`).
    """

    locations: tuple[str, ...]
    releases: tuple[str, ...]
    days: pd.DatetimeIndex
    counts: np.ndarray
    observed: np.ndarray
    cut: np.ndarray
    week: np.ndarray
    week_ends: tuple[str, ...]
    parent: np.ndarray
    period: int = 7

    @property
    def n_weeks(self) -> int:
        """Number of hub weeks."""
        return len(self.week_ends)

    def cut_week(self, r: int) -> int:
        """Return the hub week index holding release `r`'s cut-off.

        Parameters
        ----------
        r : int
            Release index.

        Returns
        -------
        int
            Index into the hub weeks.
        """
        return int(self.week[self.cut[r]])


def snapshot_arrays(
    daily: pd.DataFrame, vintage: str, locations: tuple[str, ...]
) -> SnapshotArrays:
    """Build the arrays of the releases on or before a vintage.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with columns `vintage`, `location`,
        `onset_date`, `observed` and `imputed`.
    vintage : str
        The forecast vintage. Later releases are not used.
    locations : tuple of str
        Locations to include. Each location's ancestors
        (`nowcastutils.hierarchy.parent`) are added, so the root
        is always the first node.

    Returns
    -------
    SnapshotArrays
        The arrays.

    Raises
    ------
    ValueError
        If no release is on or before the vintage.
    """
    rows = daily.loc[daily["vintage"].astype(str) <= vintage]
    if rows.empty:
        raise ValueError(f"No daily releases on or before {vintage}.")
    releases = tuple(sorted(str(v) for v in rows["vintage"].unique()))
    nodes: dict[str, None] = {}
    for loc in locations:
        chain: list[str] = []
        code: str | None = loc
        while code is not None and code not in nodes:
            chain.append(code)
            code = parent_of(code)
        for c in reversed(chain):
            nodes[c] = None
    nodes_t = tuple(sorted(nodes, key=lambda c: (depth(c), len(c), c)))
    first = as_date(str(rows["onset_date"].min()))
    cutoffs = [release_cutoff(rows, r) for r in releases]
    days = pd.date_range(first, epiweek_end(max(cutoffs)), freq="D")
    index = {d: i for i, d in enumerate(days.date)}
    cut = np.array([index[c] for c in cutoffs], dtype=np.int64)
    observed = np.arange(len(days))[None, :] <= cut[:, None]
    counts = np.zeros((len(nodes_t), len(releases), len(days)))
    take = rows.loc[rows["location"].isin(nodes_t)]
    li = take["location"].map({c: i for i, c in enumerate(nodes_t)})
    ri = take["vintage"].astype(str).map({r: i for i, r in enumerate(releases)})
    ti = pd.to_datetime(take["onset_date"]).dt.date.map(index)
    values = (take["observed"] + take["imputed"]).to_numpy(dtype=float)
    np.add.at(
        counts,
        (
            li.to_numpy(dtype=int),
            ri.to_numpy(dtype=int),
            ti.to_numpy(dtype=int),
        ),
        values,
    )
    counts = counts * observed[None]
    ends = [epiweek_end(d) for d in days.date]
    week_ends = tuple(sorted({e.isoformat() for e in ends}))
    where = {w: i for i, w in enumerate(week_ends)}
    week = np.array([where[e.isoformat()] for e in ends], dtype=np.int64)
    pos = {c: i for i, c in enumerate(nodes_t)}
    parent = np.array(
        [
            pos[p] if (p := parent_of(c)) is not None and p in pos else -1
            for c in nodes_t
        ],
        dtype=np.int64,
    )
    return SnapshotArrays(
        nodes_t, releases, days, counts, observed, cut, week,
        week_ends, parent,
    )  # fmt: skip


def blocks(snap: SnapshotArrays, period: int, end: int = 5) -> SnapshotArrays:
    """Regroup the days into blocks of `period` days ending on a weekday.

    The reference time a model nowcasts: 7-day blocks ending Saturday are
    the hub weeks, those ending another weekday the 7-day sums ending that
    day, and 1-day blocks the days. Empty days are added after the last
    day so the last block is whole; the releases and cut-offs are the
    same.

    Parameters
    ----------
    snap : SnapshotArrays
        The snapshots.
    period : int
        Days in a block, 1 or 7.
    end : int, default 5
        Weekday each 7-day block ends on, Monday 0 to Sunday 6.

    Returns
    -------
    SnapshotArrays
        The snapshots with `week`, `week_ends` and `period` of the blocks.

    Raises
    ------
    ValueError
        If `period` is not 1 or 7.
    """
    if period not in (1, 7):
        raise ValueError(f"A block is 1 or 7 days, not {period}.")
    first = snap.days[0]
    last = snap.days[-1]
    if period == 7:
        last = last + pd.Timedelta(days=(end - last.weekday()) % 7)
    days = pd.date_range(first, last, freq="D")
    extra = days.size - snap.days.size
    pad = ((0, 0), (0, 0), (0, extra))
    counts = np.pad(snap.counts, pad)
    observed = np.pad(snap.observed, pad[1:])
    ahead = ((end - days.weekday) % 7) if period == 7 else np.zeros(days.size)
    ends = days + pd.to_timedelta(ahead, unit="D")
    week_ends = tuple(sorted({e.date().isoformat() for e in ends}))
    where = {w: i for i, w in enumerate(week_ends)}
    week = np.array([where[e.date().isoformat()] for e in ends], dtype=np.int64)
    return replace(
        snap, days=days, counts=counts, observed=observed, week=week,
        week_ends=week_ends, period=period,
    )  # fmt: skip


def week_sum(snap: SnapshotArrays, x: np.ndarray) -> np.ndarray:
    """Sum day values to hub weeks.

    Parameters
    ----------
    snap : SnapshotArrays
        The snapshots, whose `week` maps each day to its hub week.
    x : numpy.ndarray
        Day values, shape `[L, T]`.

    Returns
    -------
    numpy.ndarray
        Week sums, shape `[L, W]`.
    """
    if snap.n_weeks == snap.week.size:
        # Blocks of one day are the days.
        return np.asarray(x)
    onehot = np.zeros((snap.week.size, snap.n_weeks))
    onehot[np.arange(snap.week.size), snap.week] = 1.0
    return np.asarray(x @ onehot)


def weekly_sums(
    snap: SnapshotArrays, r: int, upto: int | None = None
) -> np.ndarray:
    """Return release `r`'s counts summed to hub weeks.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index.
    upto : int, optional
        Last day index to sum. Defaults to the release's cut-off.

    Returns
    -------
    numpy.ndarray
        Weekly counts, `[L, W]`.
    """
    last = snap.cut[r] if upto is None else upto
    days = np.arange(snap.week.size) <= last
    return week_sum(snap, snap.counts[:, r, :] * days)


def weekly_counts(daily: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Sum one release's daily onsets to weeks ending on Saturday.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with columns `vintage`, `location`,
        `onset_date`, `observed` and `imputed`.
    vintage : str
        The release, ISO format.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` (the Saturday ending the
        onset week) and `total`, known plus imputed, for weeks with at
        least one onset.
    """
    rows = daily.loc[daily["vintage"].astype(str) == vintage]
    onset = pd.to_datetime(rows["onset_date"])
    end = onset + pd.to_timedelta((5 - onset.dt.weekday) % 7, unit="D")
    out = (
        rows.assign(
            target_end_date=end.dt.date.astype(str),
            total=rows["observed"] + rows["imputed"],
        )
        .groupby(["location", "target_end_date"], as_index=False)["total"]
        .sum()
    )
    kept: pd.DataFrame = out.loc[out["total"] > 0]
    return kept.reset_index(drop=True)
