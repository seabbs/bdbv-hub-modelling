# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""The baseline chain's delay at a release, for the report's delay check.

`baseline_share` gives the share of the final count each age has reported
under `randomwalknowcast`'s chain ladder (`nowcastbaselines.chain`), and
`plot_delay_check` plots the growth and spread each place uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nowcastbaselines.baselinenowcast.delays import Settings
from nowcastbaselines.baselinenowcast.nowcast import build_origin
from nowcasteval.plots import MODEL_COLOURS, THEMES
from nowcastutils.snapshots import snapshot_arrays

__all__ = ["baseline_share", "plot_delay_check"]


def baseline_share(
    daily: pd.DataFrame, vintage: str, locations: Sequence[str]
) -> pd.DataFrame:
    """Return randomwalknowcast's share reported by age at one release.

    The share is one over the growth to final of `plot_delay_check`.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, as for `make_tasks`.
    vintage : str
        The release.
    locations : sequence of str
        Places.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `age` (days since onset) and `share`.
    """
    snap = snapshot_arrays(daily, vintage, tuple(locations))
    origin = build_origin(snap, Settings(), errors=False)
    age = int(snap.cut[origin.o]) - np.arange(snap.week.size)
    seen = (age >= 0) & (age < origin.settings.max_delay)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "location": loc,
                    "age": age[seen],
                    "share": 1 / origin.growth[j, seen],
                }
            ).sort_values("age")
            for j, loc in enumerate(snap.locations)
        ],
        ignore_index=True,
    )


def plot_delay_check(
    daily: pd.DataFrame,
    vintage: str,
    locations: Sequence[str],
    mode: str = "light",
) -> Figure:
    r"""Plot the delay and spread each place uses at one release.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, as for `make_tasks`.
    vintage : str
        The release, the origin of the nowcast.
    locations : sequence of str
        Places to show.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Left, the growth $G$ from each age in days to final, chained over
        the release gap, on a log scale. Middle, the dispersion
        $1 / \phi_k$ of the new reports by weeks back, with the number of
        errors behind it.
        A place's label says at how many ages it uses its own delay
        factor and not its parent's.
    """
    snap = snapshot_arrays(daily, vintage, tuple(locations))
    origin = build_origin(snap, Settings(), errors=True)
    assert origin.size is not None
    assert origin.obs_n is not None
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    age = int(snap.cut[origin.o]) - np.arange(snap.week.size)
    seen = (age >= 0) & (age < origin.settings.max_delay)
    fig, (left, mid) = plt.subplots(
        1, 2, figsize=(6.4, 3.2), layout="constrained"
    )
    drawn: set[tuple[float, ...]] = set()
    for i, loc in enumerate(locations):
        j = snap.locations.index(loc)
        own = origin.fit.own[j]
        kind = f"own at {int(own.sum())} of {own.size} ages"
        colour = palette[i % len(palette)]
        order = np.argsort(age[seen])
        left.plot(
            age[seen][order],
            origin.growth[j, seen][order],
            color=colour,
            lw=1.2,
            label=f"{loc} ({kind})",
        )
        row = tuple(np.round(origin.size[j], 6))
        if row not in drawn:
            drawn.add(row)
            k = np.arange(origin.size.shape[1])
            style = {"color": colour, "ms": 3, "lw": 1}
            label = f"{loc} and places using it"
            mid.plot(k, 1 / origin.size[j], "o-", label=label, **style)
    left.set_yscale("log")
    left.set_xlabel("Age of the onset day (days)")
    left.set_ylabel("Growth to final")
    mid.set_xlabel("Weeks back")
    mid.set_ylabel("Dispersion of new reports, 1 / size")
    n_err = int(origin.obs_n.max())
    mid.set_title(f"At most {n_err} errors behind a fit", fontsize=8)
    mid.legend(fontsize=7, frameon=False)
    left.axhline(1, color=ink["muted"], lw=0.8, ls=":")
    left.legend(fontsize=7, frameon=False)
    return fig
