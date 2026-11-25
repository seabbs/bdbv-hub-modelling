"""Load the Ebola study's forecasts and data.

The BDBV Modeling Hub collects 100 sample trajectories of weekly confirmed
cases by week of symptom onset (`wk inc confirmed cases`) for each round,
named by its `reference_date` (a Saturday). Rounds close at 23:59
Kinshasa time (22:59 UTC) on the Tuesday four days before it.

Two sources give samples of the rounds.

- `hub_samples`: every team's files in a clone of the hub, our
  submissions and the hub's baseline (`HUB_BASELINE`) included. A file
  may hold several forecasts; the one with the latest `forecast_date` is
  kept.
- `release_samples`: our models a round did not submit, from that
  round's release (`round-<reference date>` on our repository), whose
  `hub.zip` holds every model's forecast in the hub's format. The
  submission ledger's table says which model went out under each hub id,
  and those models are left to the hub's copy.
  `write_release_samples` writes them for the workflow, and
  `round_samples` reads them back.

`load` gives every hub forecast's samples, one row per sample, in the
paper's schema with `truth` left for the workflow to attach.
`backtest_samples` reads our models' samples from the release backtest.
`release_observations` gives each release's counts from main's data, and
`truth_delay` D as the release backtest sets it, so the paper scores the
backtest against the count it was scored against.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from nowcastbdbv.data.tidy import load_target
from nowcasteval.backtest import load_config
from nowcasteval.nowcast import resolve_truth_delay

__all__ = [
    "COLUMNS",
    "HUB_BASELINE",
    "TARGET",
    "backtest_samples",
    "hub_samples",
    "load",
    "release_observations",
    "release_samples",
    "round_close",
    "round_of",
    "round_samples",
    "truth_delay",
    "write_release_samples",
]

TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's weekly round."""

COLUMNS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "sample_id",
    "value",
    "truth",
)
"""The columns of the long table every study shares."""

HUB_BASELINE: Final = "BDBVHub-baseline"
"""The hub's own baseline: an additive random walk from horizon -3."""

KEYS: Final = ["model", "location", "origin_date", "target_date"]
SAMPLE_COLUMNS: Final = [*KEYS, "sample_id", "value"]


def round_close(origin_date: pd.Timestamp) -> pd.Timestamp:
    """Return when a round closes: 22:59 UTC on the Tuesday before."""
    day = pd.Timestamp(origin_date).normalize() - pd.Timedelta(days=4)
    return (day + pd.Timedelta(hours=22, minutes=59)).tz_localize("UTC")


def round_of(release: pd.Timestamp) -> pd.Timestamp:
    """Return the round a release feeds: the first still open on its day.

    A round closes on the Tuesday four days before its Saturday
    (`round_close`), so a release on or before that Tuesday feeds it.
    """
    day = pd.Timestamp(release).normalize() + pd.Timedelta(days=4)
    return day + pd.Timedelta(days=(5 - day.weekday()) % 7)


def _samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a hub-layout frame to the sample columns."""
    out = frame.rename(
        columns={
            "reference_date": "origin_date",
            "target_end_date": "target_date",
            "output_type_id": "sample_id",
            "sample": "sample_id",
        }
    )
    for col in ("origin_date", "target_date"):
        out[col] = pd.to_datetime(out[col])
    out["value"] = out["value"].astype(float)
    out["sample_id"] = out["sample_id"].astype(int)
    return out[SAMPLE_COLUMNS].reset_index(drop=True)


def hub_samples(hub: str | Path) -> pd.DataFrame:
    """Read every team's weekly-round samples from a clone of the hub."""
    frames = []
    for path in sorted(Path(hub, "model-output").glob("*/*.csv")):
        frame = pd.read_csv(path)
        if "target" not in frame or "output_type" not in frame:
            continue
        frame = frame[
            (frame["target"] == TARGET) & (frame["output_type"] == "sample")
        ]
        if frame.empty:
            continue
        if "forecast_date" in frame:
            latest = frame["forecast_date"].max()
            frame = frame[frame["forecast_date"] == latest]
        frames.append(frame.assign(model=path.parent.name))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return _samples(pd.concat(frames, ignore_index=True))


def _submitted(ledger: str | Path) -> set[tuple[str, pd.Timestamp]]:
    """Return each model and round the ledger says went to the hub."""
    table = pd.read_csv(ledger, dtype=str, keep_default_na=False)
    table = table.loc[table["status"] == "submitted"]
    return set(
        zip(
            table["model"],
            pd.to_datetime(table["reference_date"]),
            strict=True,
        )
    )


def release_samples(releases: str | Path, ledger: str | Path) -> pd.DataFrame:
    """Read our unsubmitted models' forecasts from the round releases.

    `releases` holds a folder `round-<reference date>` per round with
    that round's `hub.zip`, whose `hub/**/<model>-samples.csv.gz` are
    every model's forecasts in the hub's format. Each round keeps its own
    rows. A model the ledger's table (`ledger`) says went to the hub that
    round is left out, as its forecast is the hub's copy
    (`hub_samples`). A round with no `hub.zip` gives nothing.
    """
    sent = _submitted(ledger)
    frames = []
    for folder in sorted(Path(releases).glob("round-*")):
        ref = pd.Timestamp(folder.name.removeprefix("round-"))
        archive = folder / "hub.zip"
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as z:
            for name in sorted(z.namelist()):
                if not name.endswith("-samples.csv.gz"):
                    continue
                model = Path(name).name.removesuffix("-samples.csv.gz")
                if (model, ref) in sent:
                    continue
                with z.open(name) as f:
                    frame = pd.read_csv(f, compression="gzip")
                frame = frame[
                    (frame["target"] == TARGET)
                    & (frame["output_type"] == "sample")
                    & (pd.to_datetime(frame["reference_date"]) == ref)
                ]
                if not frame.empty:
                    frames.append(_samples(frame.assign(model=model)))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_release_samples(
    releases: str | Path, ledger: str | Path, out: str | Path
) -> int:
    """Write `release_samples` to `out`, for `round_samples` to read.

    Returns the rows written.
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = release_samples(releases, ledger)
    samples.to_csv(path, index=False)
    return len(samples)


def round_samples(path: str | Path) -> pd.DataFrame:
    """Read a file `write_release_samples` wrote."""
    return _samples(pd.read_csv(path, dtype={"location": str}))


def load(
    hub: str | Path,
    releases: str | Path | None = None,
) -> pd.DataFrame:
    """Return every forecast's samples, with `truth` left empty.

    `hub` is a clone of the hub and `releases` our unsubmitted models'
    forecasts from the round releases (`round_samples`). A model and
    round in both is taken from the hub.
    """
    parts = [hub_samples(hub)]
    if releases is not None:
        parts.append(round_samples(releases))
    seen: set[tuple[str, pd.Timestamp]] = set()
    kept = []
    for part in parts:
        pairs = list(zip(part["model"], part["origin_date"], strict=True))
        kept.append(part[[p not in seen for p in pairs]])
        seen |= set(pairs)
    table = pd.concat([k for k in kept if not k.empty], ignore_index=True)
    table["truth"] = np.nan
    table["horizon"] = (
        (table["target_date"] - table["origin_date"]).dt.days // 7
    ).astype(int)
    table["study"] = "ebola"
    table["quantile_level"] = np.nan
    return table[list(COLUMNS)].reset_index(drop=True)


def release_observations(data: str | Path) -> pd.DataFrame:
    """Return each release's weekly counts, in the paper's schema.

    The tidy target table of main's data (`bdbvhub data`) has one
    row per release, place and week. A release's `as_of_date` is its
    cut-off, the latest onset day it covers, as the release backtest
    dates it.
    """
    target = load_target(data)
    target = target.loc[target["kind"] == "real"]
    return pd.DataFrame(
        {
            "study": "ebola",
            "location": target["location"],
            "target_date": pd.to_datetime(target["target_end_date"]),
            "as_of_date": pd.to_datetime(target["cutoff"]),
            "value": target["total"].astype(float),
        }
    ).reset_index(drop=True)


def truth_delay(data: str | Path, config: str | Path) -> int:
    """Return D as the release backtest sets it (`resolve_truth_delay`)."""
    return resolve_truth_delay(load_config(config), load_target(data))


def backtest_samples(
    path: str | Path, models: str | None = None
) -> pd.DataFrame:
    """Return release-backtest samples in the paper's schema.

    `bdbvhub backtest` forecasts from every past release, and
    `fit-store load` adds the stored fits' forecasts, to one table of
    samples (`results/forecasts.csv.gz`): `model`, `kind`, `vintage` (the
    release), `location`, `target_end_date`, `sample`, `value`, `as_of`
    and `horizon` (weeks from the round's reference date). Rows of real
    releases become sample forecasts from the release, with `truth` left
    for the workflow. `models` keeps the models matching a regular
    expression.
    """
    raw = pd.read_csv(path)
    raw = raw.loc[(raw["kind"] == "real") & raw["as_of"].isna()]
    if models:
        raw = raw.loc[raw["model"].str.match(models)]
    return pd.DataFrame(
        {
            "study": "ebola",
            "model": raw["model"],
            "location": raw["location"],
            "origin_date": pd.to_datetime(raw["vintage"]),
            "target_date": pd.to_datetime(raw["target_end_date"]),
            "horizon": raw["horizon"].astype(int),
            "quantile_level": np.nan,
            "sample_id": raw["sample"].astype(int),
            "value": raw["value"].astype(float),
            "truth": np.nan,
        }
    ).reset_index(drop=True)
