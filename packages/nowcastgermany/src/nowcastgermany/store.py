# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Stored German forecasts of the fine-tuned Chronos-2 model.

The Pages build never runs `STORED_MODELS` on the German benchmark. The
workflow's German unit (`score_germany`, `nowcastgermany.checkpoint`)
already nowcasts every development date with each released checkpoint.
`publish_germany_unit` packs a released checkpoint's unit into a store
directory (`pack_unit`, `write_stored`) as the `ASSET` tarball and
attaches it to the checkpoint's own release (`nowcastgermany
unit-publish`). The Pages build downloads the asset of the checkpoint
config/finetune.toml pins (`pinned_tag`) and loads it (`nowcastgermany
ft-load`), so it is scored and shown like every other model; no other
checkpoint stands in.

Only development dates, nowcast dates before the split date, at the
hub's nowcast horizons are ever stored or loaded (`development_rows`).
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Final

import pandas as pd

from nowcastfoundation.forecaster import SLOTS_PREFIX
from nowcastfoundation.forecaster import release_tag as foundation_release_tag

__all__ = [
    "ASSET",
    "STORED_MODELS",
    "development_rows",
    "load_stored",
    "pack_unit",
    "pinned_tag",
    "read_unit",
    "release_tag",
    "stored_tags",
    "write_stored",
]

STORED_MODELS: Final = ("foundation",)
"""The models whose German forecasts come from the store: Germany is
daily only (`packages/nowcastgermany/config/germany.toml`)."""

ASSET: Final = "germany-forecasts.tar.gz"
"""The release asset with a checkpoint's stored German forecasts."""

UNIT: Final = "germany-days"
"""The German unit's folder in the score cache
(`nowcastgermany.checkpoint.GERMANY_UNIT`)."""

FORECASTS: Final = "forecasts.csv.gz"
MANIFEST: Final = "manifest.json"

_HASH = re.compile(r"[0-9a-f]{6,64}")


def release_tag(label: str) -> str:
    """Return the release tag of a checkpoint scorer label.

    Parameters
    ----------
    label : str
        A settings hash (a `chronos-ft-<hash>` release) or `slots-<hash>`
        (a `chronos-slots-<hash>` release).

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the label is neither.
    """
    if not _HASH.fullmatch(label.removeprefix(SLOTS_PREFIX)):
        raise ValueError(
            f"Not a checkpoint label: {label!r}. Give a settings hash or "
            "slots-<hash>."
        )
    return foundation_release_tag(label)


def pinned_tag(spec: str) -> str:
    """Return the release of the checkpoint config/finetune.toml pins.

    Parameters
    ----------
    spec : str
        The pin, `slots-<hash>` or a settings hash (`finetuned_spec`).

    Returns
    -------
    str
        Its release tag.
    """
    return release_tag(spec)


def read_unit(cache: Path | str, label: str) -> pd.DataFrame:
    """Stack a label's German unit files as the page scores them.

    The 7-day sums at the hub's nowcast horizons (0 days back or
    earlier), without the `target` column.

    Parameters
    ----------
    cache : pathlib.Path or str
        The score cache, with `<UNIT>/<label>/<date>.csv.gz`.
    label : str
        The checkpoint's label.

    Returns
    -------
    pandas.DataFrame
        The rows.

    Raises
    ------
    ValueError
        If the unit has no files.
    """
    root = Path(cache) / UNIT / label
    files = sorted(root.glob("20*.csv.gz"))
    if not files:
        raise ValueError(f"No German unit files in {root}.")
    dtype = {"origin": str, "truth_vintage": str, "target_end_date": str}
    rows = pd.concat(
        [pd.read_csv(f, dtype=dtype) for f in files], ignore_index=True
    )
    if "target" in rows:
        rows = rows.loc[rows["target"] == "7day"].drop(columns="target")
    return rows.loc[rows["horizon"] <= 0].reset_index(drop=True)


def pack_unit(
    cache: Path | str, label: str, out: Path | str, split_date: str
) -> Path:
    """Pack a label's development forecasts as the release asset.

    Parameters
    ----------
    cache : pathlib.Path or str
        The score cache.
    label : str
        The checkpoint's label (`slots-<hash>` or a settings hash).
    out : pathlib.Path or str
        The folder the asset is written to.
    split_date : str
        The first held-out nowcast date; any later row is refused.

    Returns
    -------
    pathlib.Path
        `<out>/ASSET`, holding `<release tag>/forecasts.csv.gz` and
        `manifest.json`.
    """
    tag = release_tag(label)
    rows = read_unit(cache, label)
    with tempfile.TemporaryDirectory() as tmp:
        write_stored(tmp, tag, rows, split_date, label, f"{UNIT}/{label}")
        path = Path(out) / ASSET
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(Path(tmp) / tag, arcname=tag)
    return path


def development_rows(
    forecasts: pd.DataFrame,
    split_date: str,
    models: tuple[str, ...] = STORED_MODELS,
) -> pd.DataFrame:
    """Return the models' forecasts, refusing any held-out date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `germany_forecasts` output with truth (`with_truth`), of any
        models.
    split_date : str
        The first held-out nowcast date (`GermanyConfig.split_date`).
    models : tuple of str
        The models kept.

    Returns
    -------
    pandas.DataFrame
        The models' rows, sorted.

    Raises
    ------
    ValueError
        If a row of the models has a nowcast date or a target end date on
        or after the split date, or there are none.
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
            f"{int(late.sum())} rows of {', '.join(models)} are held-out "
            f"(on or after {split_date}, from {first}); only development "
            "dates may be "
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


def _csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV with no timestamp.

    The same rows give the same bytes.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False).encode())
    path.write_bytes(buf.getvalue())


def write_stored(
    store: Path | str,
    tag: str,
    forecasts: pd.DataFrame,
    split_date: str,
    label: str,
    source: str = "",
) -> Path:
    """Write a checkpoint's development forecasts into a store directory.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root, one folder per release tag.
    tag : str
        The checkpoint's release tag, the directory written.
    forecasts : pandas.DataFrame
        Its forecasts (`development_rows` keeps `STORED_MODELS`).
    split_date : str
        The first held-out nowcast date.
    label : str
        The checkpoint scorer's label, for the manifest.
    source : str
        Where the forecasts came from, for the manifest.

    Returns
    -------
    pathlib.Path
        The directory written: `forecasts.csv.gz` and `manifest.json`.
    """
    rows = development_rows(forecasts, split_date)
    out = Path(store) / tag
    out.mkdir(parents=True, exist_ok=True)
    _csv_gz(rows, out / FORECASTS)
    manifest = {
        "tag": tag,
        "label": label,
        "models": sorted(set(rows["model"])),
        "split_date": split_date,
        "first_origin": str(rows["origin"].min()),
        "last_origin": str(rows["origin"].max()),
        "origins": int(rows["origin"].nunique()),
        "rows": len(rows),
        "source": source,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def stored_tags(store: Path | str) -> set[str]:
    """Return the release tags with stored forecasts.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root; missing means none.

    Returns
    -------
    set of str
    """
    root = Path(store)
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if (p / FORECASTS).exists()}


def load_stored(store: Path | str, tag: str, split_date: str) -> pd.DataFrame:
    """Read a checkpoint's stored forecasts, refusing any held-out date.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root.
    tag : str
        The release tag.
    split_date : str
        The first held-out nowcast date.

    Returns
    -------
    pandas.DataFrame
        `nowcastgermany backtest`'s forecast columns, `STORED_MODELS` only.
    """
    dtype = {"origin": str, "truth_vintage": str, "target_end_date": str}
    frame = pd.read_csv(Path(store) / tag / FORECASTS, dtype=dtype)
    return development_rows(frame, split_date)
