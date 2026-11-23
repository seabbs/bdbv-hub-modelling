"""Load the Ebola study's forecasts and data.

The BDBV Modeling Hub collects 100 sample trajectories of weekly confirmed
cases by week of symptom onset (`wk inc confirmed cases`) for each round,
named by its `reference_date` (a Saturday). Rounds close at 23:59
Kinshasa time (22:59 UTC) on the Tuesday four days before it.

Three sources give samples.

- `hub_samples`: every team's files in a clone of the hub, our
  submissions and the hub's baseline (`HUB_BASELINE`) included. A file
  may hold several forecasts; the one with the latest `forecast_date` is
  kept.
- `candidate_samples`: files of our models' forecasts in the layout of
  `data/candidate-forecasts.csv.gz`.
- `history_samples`: the fit store's history, a file of the first fit
  stored for each model and origin, kept only if it was made before its
  round closed. `write_store_history` writes it from the fit store
  branch (`store_samples`), for the workflow to publish as a release.

`load` gives every hub forecast's samples, one row per sample, in the
paper's schema with `truth` left for the workflow to attach.
`backtest_samples` reads our models' samples from the release backtest.
`release_observations` gives each release's counts from main's data, and
`truth_delay` D as the release backtest sets it, so the paper scores the
backtest against the count it was scored against.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from bdbvhub.data.tidy import load_target
from nowcasteval.backtest import load_config
from nowcasteval.nowcast import resolve_truth_delay

__all__ = [
    "COLUMNS",
    "HUB_BASELINE",
    "TARGET",
    "backtest_samples",
    "candidate_samples",
    "history_samples",
    "hub_samples",
    "load",
    "release_observations",
    "round_close",
    "round_of",
    "store_samples",
    "truth_delay",
    "write_store_history",
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


def candidate_samples(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read our models' samples from files of stored forecasts."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    if "kind" in frame:
        frame = frame[frame["kind"] == "real"]
    return _samples(frame)


def _git(repo: str | Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def _prospective(repo: str | Path, ref: str) -> Iterator[pd.DataFrame]:
    """Yield the samples of each prospective fit in the store's history.

    Every fit file ever committed to `ref` is read once for its meta, so
    only one fit is held at a time. For each model and origin the
    earliest good fit with real forecasts is chosen, and kept if it was
    made before the round it forecasts closed. The chosen fits are then
    read again, one by one.
    """
    log = _git(
        repo,
        "log",
        "--reverse",
        "--format=commit %H",
        "--name-only",
        "--diff-filter=A",
        ref,
    ).decode()
    added: dict[str, str] = {}
    commit = ""
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.removeprefix("commit ")
        elif line.endswith(".json.gz"):
            added.setdefault(line, commit)

    def read(path: str, commit: str) -> dict:
        return json.loads(
            gzip.decompress(_git(repo, "show", f"{commit}:{path}"))
        )

    def real(fit: dict) -> pd.DataFrame:
        table = fit["forecasts"]
        frame = pd.DataFrame(table["data"], columns=table["columns"])
        return frame[frame["kind"] == "real"]

    first: dict[tuple[str, str], tuple[str, str, str, pd.Timestamp]] = {}
    for path, commit in added.items():
        fit = read(path, commit)
        meta = fit["meta"]
        if meta["status"] != "ok":
            continue
        key = (meta["model"], meta["origin"])
        if key in first and first[key][0] <= meta["fitted_at"]:
            continue
        frame = real(fit)
        if frame.empty:
            continue
        origin = pd.Timestamp(frame["reference_date"].iloc[0])
        first[key] = (meta["fitted_at"], path, commit, origin)
    for fitted_at, path, commit, origin in first.values():
        if pd.Timestamp(fitted_at) <= round_close(origin):
            yield _samples(real(read(path, commit)))


def store_samples(
    repo: str | Path, ref: str = "origin/fit-store"
) -> pd.DataFrame:
    """Read the prospective fits from the fit store's history.

    For each model and origin the earliest good fit is kept, if it was
    made before the round it forecasts closed (`_prospective`).
    """
    frames = list(_prospective(repo, ref))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_store_history(repo: str | Path, ref: str, out: str | Path) -> int:
    """Write the fit store's prospective fits (`store_samples`) to `out`.

    The fits are written one at a time, so the history is never held
    whole. The file is what `history_samples` reads, so the paper reads
    the store's history without the branch. Returns the rows written.
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", newline="") as f:
        f.write(",".join(SAMPLE_COLUMNS) + "\n")
        for frame in _prospective(repo, ref):
            frame.to_csv(f, header=False, index=False)
            rows += len(frame)
    return rows


def history_samples(path: str | Path) -> pd.DataFrame:
    """Read a file `write_store_history` wrote, as `store_samples` gives."""
    return _samples(pd.read_csv(path, dtype={"location": str}))


def load(
    hub: str | Path,
    candidates: Iterable[str | Path] = (),
    history: str | Path | None = None,
) -> pd.DataFrame:
    """Return every forecast's samples, with `truth` left empty.

    `hub` is a clone of the hub. `candidates` are files of our models'
    stored forecasts and `history` the fit store's history
    (`history_samples`). A model and round in more than one source is
    taken from the first of hub, candidates and history.
    """
    parts = [hub_samples(hub), candidate_samples(candidates)]
    if history is not None:
        parts.append(history_samples(history))
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

    The tidy target table of main's data (`bdbvhubrenewal data`) has one
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

    `bdbvhubrenewal backtest` forecasts from every past release, and
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
