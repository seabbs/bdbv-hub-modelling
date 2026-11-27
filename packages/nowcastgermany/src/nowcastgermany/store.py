# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Our German forecasts, from the fine-tuning workflow's units.

The workflow's German units (`score_germany`, `score_run_germany`,
`nowcastgermany.checkpoint`) already nowcast every development date: the
reference unit (randomwalknowcast and zero-shot Chronos-2) and each
checkpoint's unit (the fine-tuned models). `nowcastgermany units` stacks
the reference and the paper's final checkpoint's unit as the German
development table (`units_forecasts`), which the paper's
`germany_development` rule scores. `release_paper_germany` publishes it
in the `paper-germany-<hash>` release (`UNITS`, with its note `NOTE`),
and Site results loads the newest one (`nowcastgermany stored-load`,
`load_units`), so the site and the paper show the same forecasts and no
German backtest runs for either.

Only development dates, nowcast dates before the split date, at the
hub's nowcast horizons are ever stacked or loaded (`development_rows`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

__all__ = [
    "NOTE",
    "UNITS",
    "UNITS_NOTE",
    "checkpoint_of",
    "development_rows",
    "load_units",
    "read_unit",
    "units_forecasts",
]

UNITS: Final = "germany-units.csv.gz"
"""Our development forecasts' file in the `paper-germany` release."""

NOTE: Final = "units.json"
"""The note naming the checkpoint (and, on the site, the release) shown."""

UNITS_NOTE: Final = "germany-units.json"
"""`NOTE` of the units, as the `paper-germany` release holds it."""

_DTYPE: Final = {"origin": str, "truth_vintage": str, "target_end_date": str}


def read_unit(root: Path | str) -> pd.DataFrame:
    """Stack a German unit's date files as the page scores them.

    The 7-day sums at the hub's nowcast horizons (0 days back or
    earlier), without the `target` column.

    Parameters
    ----------
    root : pathlib.Path or str
        The unit's folder, with `<date>.csv.gz` (`<SCORE_CACHE>/
        germany-days/reference` or `<run>/scores/germany-days/n<N>`).

    Returns
    -------
    pandas.DataFrame
        The rows.

    Raises
    ------
    ValueError
        If the unit has no files.
    """
    root = Path(root)
    files = sorted(root.glob("20*.csv.gz"))
    if not files:
        raise ValueError(f"No German unit files in {root}.")
    rows = pd.concat(
        [pd.read_csv(f, dtype=_DTYPE) for f in files], ignore_index=True
    )
    if "target" in rows:
        rows = rows.loc[rows["target"] == "7day"].drop(columns="target")
    return rows.loc[rows["horizon"] <= 0].reset_index(drop=True)


def development_rows(
    forecasts: pd.DataFrame, split_date: str, models: Sequence[str]
) -> pd.DataFrame:
    """Return the models' forecasts, refusing any held-out date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `germany_forecasts` output with truth (`with_truth`).
    split_date : str
        The first held-out nowcast date (`GermanyConfig.split_date`).
    models : sequence of str
        The models kept (the German config's `models`).

    Returns
    -------
    pandas.DataFrame
        The models' rows, sorted.

    Raises
    ------
    ValueError
        If a kept row has a nowcast date or a target end date on or after
        the split date, or there are none.
    """
    rows = forecasts.loc[forecasts["model"].isin(models)].copy()
    for col in ("origin", "target_end_date", "truth_vintage"):
        rows[col] = rows[col].astype(str)
    late = (rows["origin"] >= split_date) | (
        rows["target_end_date"] >= split_date
    )
    if late.any():
        first = rows.loc[late, "origin"].min()
        raise ValueError(
            f"{int(late.sum())} rows are held-out (on or after "
            f"{split_date}, from {first}); only development dates may be "
            "stored or shown."
        )
    if rows.empty:
        raise ValueError(f"No forecasts of {', '.join(models)}.")
    keys = [
        "model",
        "origin",
        "location",
        "target_end_date",
        "horizon",
        "quantile",
    ]
    return rows.sort_values(keys, kind="stable").reset_index(drop=True)


def units_forecasts(
    units: Iterable[Path | str], split_date: str, models: Sequence[str]
) -> pd.DataFrame:
    """Stack the units' development forecasts of the config's models.

    Parameters
    ----------
    units : iterable of pathlib.Path or str
        The units' folders (`read_unit`).
    split_date : str
        The first held-out nowcast date.
    models : sequence of str
        The models kept.

    Returns
    -------
    pandas.DataFrame
        `nowcastgermany backtest`'s forecast columns.
    """
    rows = pd.concat([read_unit(u) for u in units], ignore_index=True)
    return development_rows(rows, split_date, models)


def checkpoint_of(record: Path | str | None) -> str | None:
    """Return a checkpoint's release tag from its release record.

    Parameters
    ----------
    record : pathlib.Path or str or None
        The checkpoint's `published.json` (rule publish_checkpoint).

    Returns
    -------
    str or None
        Its tag, or None with no record.
    """
    if not record:
        return None
    return str(json.loads(Path(record).read_text())["tag"])


def load_units(
    store: Path | str, split_date: str, models: Sequence[str]
) -> pd.DataFrame:
    """Read the published development forecasts, refusing any held-out date.

    Parameters
    ----------
    store : pathlib.Path or str
        The unpacked `paper-germany` release (`paperpipe.stored fetch`).
    split_date : str
        The first held-out nowcast date.
    models : sequence of str
        The models kept.

    Returns
    -------
    pandas.DataFrame
        `nowcastgermany backtest`'s forecast columns.
    """
    frame = pd.read_csv(Path(store) / UNITS, dtype=_DTYPE)
    return development_rows(frame, split_date, models)
