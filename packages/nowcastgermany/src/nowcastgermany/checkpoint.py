# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Score each published Chronos-2 checkpoint.

Each checkpoint (release `chronos-ft-<hash>` or `chronos-slots-<hash>`)
is a label, run beside `reference` (randomwalknowcast and zero-shot
Chronos-2, ragged and complete). A label's units forecast and cache;
`table` scores the caches into rows of `evidence/checkpoint-scores.csv`
with `rule` = `RULE`:

- `germany`: the development half of the German benchmark
  (packages/nowcastgermany/config/germany.toml), quantile WIS and log WIS on
  each model's own
  quantiles (`nowcastgermany.evaluation.germany_forecasts`), by level.
- `held_out`: the trainer's held-out loss from the release's
  settings.json (unit `release`).

BDBV is held out for the paper, so no BDBV rows are scored here.

Relative scores divide by randomwalknowcast's on the cells every model
and checkpoint of a level made. Every score comes from
`nowcasteval.scoring`, as the paper's and the dashboard's do. Rows are
keyed by checkpoint hash and run, with the checkpoint's outbreaks and
grid.

Usage:
  python -m nowcastgermany.checkpoint germany CACHE LABEL [SEGMENT]
  python -m nowcastgermany.checkpoint release CACHE HASH
  python -m nowcastgermany.checkpoint table CACHE EVIDENCE \\
    CHECKPOINTS HASH...
  python -m nowcastgermany.checkpoint run TRAIN SCORES REFERENCE RUN OUT
  python -m nowcastgermany.checkpoint daily TRAIN SCORES REFERENCE RUN OUT
  python -m nowcastgermany.checkpoint compare OUT TABLE...
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nowcasteval.labels import FINETUNED, FINETUNED_OF, base_model
from nowcastfoundation.sidecar import SIDECAR, with_sidecar
from nowcastutils import codekey

RULE: Final = "own-quantile-wis_crps-1000-common"
"""The scoring rule: WIS and CRPS on each model's own quantiles, from
1000 draws on the common random stream (`nowcasteval.marginals`)."""
BASELINE: Final = "randomwalknowcast"
REFERENCE: Final = "reference"
REFERENCE_PIN: Final = "e8a1f5ae338b"
"""The checkpoint the reference pins, which its models never load."""
GERMANY_CONFIG: Final = Path("packages/nowcastgermany/config/germany.toml")
"""The German evaluation's config, whose `nowcast.models` the units run."""
GERMANY_UNIT: Final = "germany-days"
"""The cache folder of the German units, which score every one of the
hub's 29 target days."""
GERMANY_CHUNK: Final = 5
"""German dates forecast together (`germany_unit`) unless `SCORE_DATES`
in the environment (local config) says otherwise. Each date has two
tasks (states, ages): small enough for `score.mem_mb`."""
FINAL_OUTBREAKS: Final = 8400
"""The reported model is the live run's final checkpoint."""
FINAL_GRID: Final = "r5"
CURRENT_RUNS: Final = frozenset({"one-model"})
"""Runs whose adapters this code loads (series bias, mixed layouts)."""
HELD_OUT_LEVELS: Final = ("leaf", "national", "intermediate")
GERMANY_CELL: Final = ["origin", "location", "target_end_date"]
COVERAGE: Final = (50, 90)
"""Central intervals scored for coverage: 50% and 90%."""
GERMANY_COVERAGE: Final = (50, 95)
"""The German rows' intervals: the hub's quantiles have no 0.05 or 0.95,
so there is no 90% interval and its rows keep the 95% one the hub
scored (`nowcastgermany.evaluation`)."""
WIS_PARTS: Final = ("dispersion", "overprediction", "underprediction")
"""WIS's components (Bracher et al. 2021), which sum to it."""
REPO: Final = "seabbs/bdbv-hub-modelling"
COLUMNS: Final = [
    "study",
    "level",
    "model",
    "checkpoint",
    "outbreaks",
    "n",
    "n_origins",
    "first_origin",
    "last_origin",
    "rel_wis",
    "rel_log_wis",
    "coverage_50",
    "coverage_90",
    "coverage_95",
    "grid",
    "rule",
    "rel_crps",
    "rel_log_crps",
    "loss",
    "loss_zero_shot",
    "run",
    "chosen",
    "target",
    "daily_rule",
    "layout",
    "window",
    "rel_to",
    "wis",
    "crps",
    "n_blocks",
    "horizon_weeks",
    "zone_method",
    "factor",
    "factor_level",
    "dispersion",
    "overprediction",
    "underprediction",
    "precision",
]
"""The evidence table's columns, those of rows of other studies too."""
SORT: Final = ["rule", "study", "level", "model", "outbreaks", "checkpoint"]


# Labels and checkpoints ------------------------------------------------


def german_models(config: Path = GERMANY_CONFIG) -> tuple[str, ...]:
    """Return the models the German evaluation lists (`nowcast.models`)."""
    return tuple(tomllib.loads(Path(config).read_text())["nowcast"]["models"])


def label_models(
    label: str, config: Path = GERMANY_CONFIG, layout: str = "ragged"
) -> tuple[str, ...]:
    """Return the models a label runs, of those the German config lists.

    The reference runs the baseline and the zero-shot Chronos-2 models
    (ragged and complete); a checkpoint runs the ones fine-tuned on
    simulations (`labels.FINETUNED`), `foundation` in the `layout` it was
    trained on (`segment_layout`).
    """
    models = german_models(config)
    if label == REFERENCE:
        zero = [m for m in models if m in FINETUNED_OF]
        return (BASELINE, *zero)
    return tuple(
        (
            f"foundation_{layout}"
            if m == "foundation" and layout != "ragged"
            else m
        )
        for m in models
        if m in FINETUNED
    )


def segment_layout(segment: Path | None) -> str:
    """Return the block layout a run's segment trained on.

    From its settings (`layout`, set only when not ragged); a release
    label, with no segment, is ragged.
    """
    if segment is None:
        return "ragged"
    settings = json.loads((Path(segment) / "settings.json").read_text())
    return str(settings.get("layout", "ragged"))


def run_of(settings: Mapping[str, Any]) -> str:
    """Return a checkpoint's run from its settings.

    `one-model`: the series bias with mixed-frequency layouts (a source
    such as `daily+weekly`); `attn-bias`: earlier series-bias trials;
    `slots-zeroinit` and `slots`: the slots runs; `no-slots`.
    """
    if settings.get("series_bias"):
        mixed = any("+" in str(s) for s in settings.get("sources", []))
        return "one-model" if mixed else "attn-bias"
    slots = settings.get("series_slots")
    if not slots:
        return "no-slots"
    return "slots-zeroinit" if slots.get("init") == "zero" else "slots"


def outbreaks_and_grid(settings: Mapping[str, Any]) -> tuple[int, str]:
    """Return a checkpoint's training outbreaks and grid, from settings.json.

    0 and `unknown` where the settings have none.
    """
    return (
        int(settings.get("n_outbreaks") or 0),
        str(settings.get("grid") or "unknown"),
    )


def chosen(rows: pd.DataFrame) -> pd.Series:
    """Mark the reported model's rows: the live run's final checkpoint."""
    return (
        rows["run"].astype(str).isin(CURRENT_RUNS)
        & (rows["grid"].astype(str) == FINAL_GRID)
        & (rows["outbreaks"] == FINAL_OUTBREAKS)
    )


def held_out_rows(checkpoint: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Return the trainer's held-out losses as rows."""
    by = settings.get("held_out_loss_by_level") or {}
    rows = [
        {
            "level": "all",
            "loss": settings.get("held_out_loss", np.nan),
            "loss_zero_shot": settings.get("held_out_loss_start", np.nan),
        }
    ]
    rows += [
        {"level": lv, "loss": by[lv]} for lv in HELD_OUT_LEVELS if lv in by
    ]
    out = pd.DataFrame(rows).assign(
        study="held_out",
        model="foundation_zeroshot",
        checkpoint=checkpoint,
        n=settings.get("n_validation", 0),
        n_origins=0,
        rule=RULE,
        target="7day",
    )
    out["loss"] = out["loss"].astype(float)
    return out.reindex(columns=COLUMNS)


# Scores ----------------------------------------------------------------


def tag(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """Name a label's rows by checkpoint and zero-shot model (`base_model`)."""
    return rows.assign(
        checkpoint="zero-shot" if label == REFERENCE else label,
        model=rows["model"].map(base_model),
    )


def matched(rows: pd.DataFrame, cell: list[str]) -> pd.DataFrame:
    """Keep, in each level, the cells every model and checkpoint made."""
    key = rows["model"] + "@" + rows["checkpoint"]
    out = []
    for _, part in rows.assign(_k=key).groupby("level"):
        n = part.groupby(cell)["_k"].nunique()
        full = n.index[n == part["_k"].nunique()]
        out.append(part.set_index(cell).loc[full].reset_index())
    return pd.concat(out, ignore_index=True).drop(columns="_k")


def relative(
    rows: pd.DataFrame,
    study: str,
    metrics: tuple[str, str],
    baseline: str = BASELINE,
    target: str = "7day",
) -> pd.DataFrame:
    """Return mean scores relative to `baseline`, by level and model.

    With the mean WIS and its components, and coverage, where the rows
    have them. Rows are tagged with `target`.
    """
    widths = (*COVERAGE, *GERMANY_COVERAGE)
    extra = [
        "wis",
        *WIS_PARTS,
        *(f"coverage_{p}" for p in dict.fromkeys(widths)),
    ]
    extra = [c for c in extra if c in rows and c not in metrics]
    keys = ["level", "model", "checkpoint"]
    mean = rows.groupby(keys)[[*metrics, *extra]].mean()
    mean["n"] = rows.groupby(keys).size()
    origins = rows.groupby(keys)["origin"]
    mean["n_origins"] = origins.nunique()
    mean["first_origin"] = origins.min()
    mean["last_origin"] = origins.max()
    base = mean.xs((baseline, "zero-shot"), level=("model", "checkpoint"))
    at = mean.index.get_level_values("level")
    for m in metrics:
        mean[f"rel_{m}"] = mean[m] / base[m].reindex(at).to_numpy()
    out = mean.reset_index().assign(study=study, rule=RULE, target=target)
    return out.reindex(columns=COLUMNS)


def merge_evidence(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add rule rows to the evidence, replacing the checkpoints' old ones.

    The rule's rows of checkpoints not in `new` are kept; rows of any
    other rule are dropped.
    """
    old = old.reindex(columns=COLUMNS)
    mine = set(new["checkpoint"].astype(str))
    old = old.loc[
        (old["rule"] == RULE) & ~old["checkpoint"].astype(str).isin(mine)
    ]
    out = pd.concat([old, new.reindex(columns=COLUMNS)], ignore_index=True)
    return out.sort_values([*SORT, "target"], kind="stable").reset_index(
        drop=True
    )


# Units -----------------------------------------------------------------


def score_dates() -> int:
    """Return the German dates a unit forecasts together.

    `SCORE_DATES` from the local config (`flowlib.machine.torch_env`),
    else `GERMANY_CHUNK`: more is faster and holds more in memory.
    """
    return int(os.environ.get("SCORE_DATES") or GERMANY_CHUNK)


PRECISION_FILE: Final = "precision.txt"


def record_precision(root: Path) -> None:
    """Write the precision a unit forecasts in to its folder."""
    from nowcastfoundation.backends.chronos2 import (
        inference_precision,
        torch_device,
    )

    Path(root).mkdir(parents=True, exist_ok=True)
    precision = inference_precision(torch_device())
    (Path(root) / PRECISION_FILE).write_text(precision + "\n")


def unit_precision(root: Path) -> str:
    """Return the precision a unit forecast in, fp32 if it records none."""
    path = Path(root) / PRECISION_FILE
    return path.read_text().strip() if path.exists() else "fp32"


def _pin(label: str, segment: Path | None = None) -> None:
    """Pin the checkpoint, so no process resolves `latest` mid-run.

    A run's own `segment` folder, when given, is the checkpoint
    (`forecaster.local_checkpoint`), else the release `label` names.
    """
    if segment is not None:
        os.environ["BDBV_CHRONOS_FT"] = str(segment)
        return
    os.environ["BDBV_CHRONOS_FT"] = (
        REFERENCE_PIN if label == REFERENCE else label
    )


def _free() -> None:
    """Free JAX's compiled functions, which grow with each origin."""
    import jax

    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV atomically, so a stopped run leaves no part.

    The temporary file is this process's own, as units sharing a cache
    write the same files (the shared truth).
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)


def german_tasks_dir(
    cache: Path,
    triangle: Path,
    n_samples: int,
    seed: int,
    max_delay: int | None = None,
) -> Path:
    """Return the store of the German tasks every checkpoint scores.

    Keyed on what the tasks are built from: the triangle file's bytes,
    the task-building code (`nowcastgermany.data` and the
    `nowcastutils.dates` it uses),
    the fields of the `ForecastTask` they are pickled as, the draws and
    seed, and the longest delay the models see (`max_delay`).
    """
    import hashlib
    import inspect

    from nowcastgermany import data as dg
    from nowcastutils import base, dates

    h = hashlib.sha256()
    with Path(triangle).open("rb") as f:
        for part in iter(lambda: f.read(1 << 24), b""):
            h.update(part)
    for module in (dg, dates):
        h.update(inspect.getsource(module).encode())
    fields = [f.name for f in dataclasses.fields(base.ForecastTask)]
    h.update(repr(fields).encode())
    h.update(f"{n_samples}-{seed}".encode())
    if max_delay is not None:
        h.update(f"max_delay-{max_delay}".encode())
    return Path(cache) / GERMANY_UNIT / f"tasks-{h.hexdigest()[:12]}"


TRUTH_FILE: Final = "truth.csv.gz"
"""The German truth the plots show (`evaluation.truth_context`)."""

WRITTEN_FILE: Final = "written.tsv"
"""A unit folder's record of each date and model written, with its key."""

EVERY: Final = "*"
"""The model of a record that covers every model of its date."""


def written(root: Path) -> dict[tuple[str, str], str]:
    """Return the key each date's rows of each model were written with.

    Lines are `date<TAB>model<TAB>key`, or `date<TAB>key` for every model
    of the date (`EVERY`); a later line replaces an earlier one.
    """
    path = Path(root) / WRITTEN_FILE
    if not path.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        p = line.split("\t")
        if len(p) == 2:
            out[(p[0], EVERY)] = p[1]
        elif len(p) == 3:
            out[(p[0], p[1])] = p[2]
    return out


def _record(root: Path, date: str, keys: Mapping[str, str]) -> None:
    with (Path(root) / WRITTEN_FILE).open("a") as f:
        f.writelines(f"{date}\t{m}\t{k}\n" for m, k in keys.items())


def model_key(model: str) -> str:
    """Return a digest of what a model's forecasts depend on, past its tasks.

    The repository's modules the model's own module and the German
    forecasting code (`nowcastgermany.evaluation`) import
    (`nowcastutils.codekey`), so a change to a module neither imports
    keeps the key, and the German config.
    """
    from nowcastutils.base import list_models

    spec = next(s for s in list_models() if s.name == model)
    factory = getattr(spec.factory, "func", spec.factory)
    files = codekey.module_files(
        [factory.__module__, "nowcastgermany.evaluation"]
    )
    return codekey.digest([*files, Path(GERMANY_CONFIG).resolve()])


def _reaches(path: Path, last: int, targets: Sequence[str]) -> bool:
    """Whether a unit's date file holds `targets` up to `last` days.

    Files written before the forecast days or a target were added are
    forecast again. A file with no `target` column holds 7-day sums.
    """
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty or int(rows["horizon"].max()) < last:
        return False
    held = set(rows["target"]) if "target" in rows else {"7day"}
    return set(targets) <= held


TRIANGLE_ENV = "GERMANY_TRIANGLE"
"""The environment variable with the German triangle's path, which the
workflow sets to the file its fetch rule wrote."""


def triangle_path(config: Any) -> Path:
    """Return the German triangle a unit reads.

    `TRIANGLE_ENV` when set (the workflow's fetched file), else
    `data/raw/germany/<the config's triangle file>`.
    """
    given = os.environ.get(TRIANGLE_ENV)
    if given:
        return Path(given)
    return Path("data/raw/germany") / Path(config.triangle).name


def germany_unit(cache: Path, label: str, segment: Path | None = None) -> None:
    """Forecast each German development date, own quantiles with truth.

    Each date is forecast at the German config's horizons (`horizons`):
    the hub's nowcast and the forecast days after it. The fine-tuned
    models load the release `label` names, or a run's own `segment`
    folder (`<run>/train/n<N>`) when given, cached as `label`.
    """
    from nowcastgermany import evaluation as de
    from nowcastgermany.data import hub_triangle

    _pin(label, segment)
    config = de.load_germany_config(
        "packages/nowcastgermany/config/germany.toml"
    )
    path = triangle_path(config)
    triangle = hub_triangle(path)
    # Built once for every checkpoint, then read back.
    tasks_dir = german_tasks_dir(
        cache, path, config.n_samples, config.seed, config.truth_delay
    )
    root = cache / GERMANY_UNIT / label
    root.mkdir(parents=True, exist_ok=True)
    record_precision(root)
    dates = de.nowcast_dates(config, held_out=False)
    # The plots' truth from `plot_start`, shared by every unit.
    _write(
        de.truth_context(triangle, config),
        cache / GERMANY_UNIT / TRUTH_FILE,
    )
    last = max(config.horizons)
    models = list(label_models(label, layout=segment_layout(segment)))
    # A date's rows are kept only if forecast from this task store: the
    # tasks' building code is in its name (`german_tasks_dir`). The
    # reference's do not depend on a checkpoint, so each model's are also
    # keyed on its own code and the German config (`model_key`), and a
    # change redoes that model's rows alone.
    tasks = tasks_dir.name
    want = {
        m: f"{tasks}-{model_key(m)}" if label == REFERENCE else tasks
        for m in models
    }
    have = written(root)
    groups: dict[tuple[str, ...], list[str]] = {}
    for d in dates:
        full = not _reaches(root / f"{d}.csv.gz", last, config.targets)
        need = tuple(
            m
            for m in models
            if full or have.get((d, m), have.get((d, EVERY))) != want[m]
        )
        if need:
            groups.setdefault(need, []).append(d)
    # Rows of a model the label no longer runs (a renamed or dropped
    # model) go from every date, also those nothing is redone on.
    gone = {d for (d, m) in have if m != EVERY and m not in models}
    for d in sorted(gone - {d for ds in groups.values() for d in ds}):
        out = root / f"{d}.csv.gz"
        if out.exists():
            old = pd.read_csv(out)
            _write(old.loc[old["model"].isin(models)], out)
    todo = [(need, d) for need, ds in groups.items() for d in ds]
    # Dates are forecast `score_dates()` at a time, Chronos-2 predicted for
    # all of them in a few batched passes, and written one file a date so
    # a stopped run resumes.
    step = score_dates()
    chunks = [
        (need, ds[i : i + step])
        for need, ds in groups.items()
        for i in range(0, len(ds), step)
    ]
    for i, (need, chunk) in enumerate(chunks):
        start = time.monotonic()
        timings: list[dict[str, Any]] = []
        fc = de.germany_forecasts(
            list(need),
            triangle,
            chunk,
            config.n_samples,
            config.seed,
            tasks_dir,
            max_delay=config.truth_delay,
            horizons=config.horizons,
            targets=config.targets,
            timings=timings,
        )
        # Seconds by model and date, for the `benchmarks` target.
        times = root / "timings.csv"
        pd.DataFrame(timings).assign(dates=len(chunk)).to_csv(
            times, mode="a", header=not times.exists(), index=False
        )
        fc = de.with_truth(fc, triangle, config.truth_delay)
        for date in chunk:
            mine = fc.loc[fc["origin"] == date].reset_index(drop=True)
            out = root / f"{date}.csv.gz"
            if len(need) < len(models) and out.exists():
                # The other models' rows of the date are kept.
                old = pd.read_csv(out)
                kept = old["model"].isin(models) & ~old["model"].isin(need)
                mine = pd.concat([old.loc[kept], mine], ignore_index=True)
            _write(mine, out)
            _record(root, date, {m: want[m] for m in need})
        _free()
        took = time.monotonic() - start
        done = len(dates) - len(todo) + sum(len(c) for _, c in chunks[: i + 1])
        print(
            f"germany {label} {len(chunk)} dates to {chunk[-1]} "
            f"({done}/{len(dates)}): {took:.0f} s",
            flush=True,
        )
    print(f"germany {label} done", flush=True)


def release_unit(cache: Path, label: str) -> None:
    """Fetch a release's settings.json, with its held-out sidecar if any."""
    from nowcastfoundation.forecaster import release_tag

    root = cache / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    tag_ = release_tag(label)
    out = str(root / f"{label}.json")
    subprocess.run(
        ["gh", "release", "download", tag_, "-R", REPO, "-p", "settings.json",
         "-O", out, "--clobber"],
        check=True,
    )  # fmt: skip
    view = subprocess.run(
        ["gh", "release", "view", tag_, "-R", REPO, "--json", "assets"],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip
    release = json.loads(view.stdout)
    # Offloaded checkpoints keep their held-out losses in a sidecar.
    if SIDECAR in {a["name"] for a in release.get("assets", [])}:
        sidecar = root / f"{label}-{SIDECAR}"
        subprocess.run(
            ["gh", "release", "download", tag_, "-R", REPO, "-p", SIDECAR,
             "-O", str(sidecar), "--clobber"],
            check=True,
        )  # fmt: skip
        settings = json.loads(Path(out).read_text())
        merged = with_sidecar(settings, json.loads(sidecar.read_text()))
        Path(out).write_text(json.dumps(merged, indent=1))


# Table -----------------------------------------------------------------


def _cached(root: Path, target: str = "7day") -> pd.DataFrame | None:
    files = sorted(root.glob("*.csv.gz"))
    if not files:
        return None
    dtype = {"origin": str, "target_end_date": str}
    rows = pd.concat([pd.read_csv(f, dtype=dtype) for f in files])
    if "target" not in rows:
        return rows if target == "7day" else None
    keep = rows["target"].fillna("7day") == target
    return rows.loc[keep].drop(columns="target")


def _with_precision(
    rows: pd.DataFrame, folders: Mapping[str, Path]
) -> pd.DataFrame:
    """Give each checkpoint's rows the precision its unit forecast in."""
    at = {
        ("zero-shot" if lb == REFERENCE else lb): unit_precision(folder)
        for lb, folder in folders.items()
    }
    return rows.assign(precision=rows["checkpoint"].map(at))


def germany_rows(
    folders: Mapping[str, Path],
    target: str = "7day",
    baseline: str = BASELINE,
) -> pd.DataFrame:
    """Return the German development half's rule rows.

    `folders` gives each label's cached German unit. Rows are of the view
    `target` (`7day` or `daily`), relative to zero-shot `baseline`.
    """
    from nowcastgermany import evaluation as de

    config = de.load_germany_config(
        "packages/nowcastgermany/config/germany.toml"
    )
    parts = []
    for label, folder in folders.items():
        fc = _cached(folder, target)
        if fc is None or fc.empty:
            continue
        # The table scores the hub's horizons only, as the paper scores
        # the teams; the forecast days are for the page.
        fc = fc.loc[(fc["origin"] < config.split_date) & (fc["horizon"] <= 0)]
        nat = de.score_germany(fc, config.truth_delay)
        logged = fc.assign(
            value=np.log1p(fc["value"].clip(lower=0)),
            truth=np.log1p(fc["truth"]),
        )
        log = de.score_germany(logged, config.truth_delay)
        log = log[["model", *GERMANY_CELL, "wis"]].rename(
            columns={"wis": "log_wis"}
        )
        sc = nat.merge(log, on=["model", *GERMANY_CELL])
        parts.append(tag(sc, label))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    sc = pd.concat(parts, ignore_index=True).rename(columns={"pool": "level"})
    out = relative(
        matched(sc, GERMANY_CELL),
        "germany",
        ("wis", "log_wis"),
        baseline,
        target,
    )
    return _with_precision(out, folders)


def _existing(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_csv(path)


def table(
    cache: Path, evidence: Path, checkpoints: Path, hashes: Sequence[str]
) -> pd.DataFrame:
    """Score the caches of `hashes` and the reference into the tables.

    Rows of other rules, and of this rule for other checkpoints, are kept.
    """
    meta = {}
    for h in hashes:
        settings = json.loads((cache / "heldout" / f"{h}.json").read_text())
        meta[h] = (settings, *outbreaks_and_grid(settings))
    labels = [REFERENCE, *hashes]
    # BDBV is held out for the paper: no BDBV rows are computed, and
    # those already in `evidence` are kept.
    new = pd.concat(
        [
            germany_rows({lb: cache / GERMANY_UNIT / lb for lb in labels}),
            *(held_out_rows(h, m[0]) for h, m in meta.items()),
        ],
        ignore_index=True,
    )
    new["outbreaks"] = new["checkpoint"].map({h: m[1] for h, m in meta.items()})
    new["outbreaks"] = new["outbreaks"].fillna(0).astype(int)
    new["grid"] = new["checkpoint"].map({h: m[2] for h, m in meta.items()})
    runs = {h: run_of(m[0]) for h, m in meta.items()}
    new["run"] = new["checkpoint"].map(runs).fillna("zero-shot")
    new["chosen"] = chosen(new)
    merged = merge_evidence(_existing(evidence, COLUMNS), new)
    merged.round(4).to_csv(evidence, index=False)
    ck = _existing(checkpoints, ["checkpoint", "outbreaks", "grid"])
    add = pd.DataFrame(
        [(h, m[1], m[2]) for h, m in meta.items()],
        columns=["checkpoint", "outbreaks", "grid"],
    )
    ck = pd.concat(
        [ck.loc[~ck["checkpoint"].astype(str).isin(add["checkpoint"])], add],
        ignore_index=True,
    )
    ck.to_csv(checkpoints, index=False)
    return new


def _segments(train: Path) -> list[Path]:
    """Return a run's saved segments `n<N>` with settings, smallest first."""
    return sorted(
        (
            d
            for d in Path(train).glob("n*")
            if d.name[1:].isdigit() and (d / "settings.json").is_file()
        ),
        key=lambda d: int(d.name[1:]),
    )


VIEWS: Final = {"7day": "7-day sums", "daily": "daily counts"}
"""The German views' names in the daily comparison's sets."""


def daily_table(
    train: Path, scores: Path, reference: Path, run: str
) -> pd.DataFrame:
    """Return a run's checkpoints on both German views, for the SI.

    Each saved segment's relative WIS on the 7-day sums (against the
    config's `baseline`) and on the daily counts (against its
    `daily_baseline`), by level, in the comparison layout (`set`, `run`,
    `outbreaks`, `score`) of `paperpipe.evidence.comparison_figure`.
    """
    from nowcastgermany import evaluation as de

    config = de.load_germany_config(
        "packages/nowcastgermany/config/germany.toml"
    )
    folders = {REFERENCE: Path(reference)} | {
        d.name: Path(scores) / d.name
        for d in _segments(train)
        if (Path(scores) / d.name).is_dir()
    }
    outbreaks = {
        d.name: outbreaks_and_grid(
            json.loads((d / "settings.json").read_text())
        )[0]
        for d in _segments(train)
    }
    bases = {"7day": config.baseline, "daily": config.daily_baseline}
    parts = []
    for target in config.targets:
        rows = germany_rows(folders, target, bases[target])
        rows = rows.loc[rows["checkpoint"].isin(outbreaks)]
        parts.append(
            pd.DataFrame(
                {
                    "set": f"Germany, {VIEWS[target]}, " + rows["level"],
                    "run": run,
                    "outbreaks": rows["checkpoint"].map(outbreaks),
                    "score": rows["rel_wis"],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_table(
    train: Path, scores: Path, reference: Path, run: str
) -> pd.DataFrame:
    """Score a run's own saved checkpoints into the tables' rows.

    Each saved segment `n<N>` of `train` is a checkpoint labelled `n<N>`:
    its German unit cached in `scores/n<N>` (rule score_run_germany),
    beside zero-shot's in `reference`, and its held-out losses from its
    settings and sidecar. Rows carry the run's name as `run`.
    """
    segments = _segments(train)
    settings = {}
    for d in segments:
        sidecar = d / SIDECAR
        settings[d.name] = with_sidecar(
            json.loads((d / "settings.json").read_text()),
            json.loads(sidecar.read_text()) if sidecar.is_file() else None,
        )
    folders = {REFERENCE: Path(reference)} | {
        d.name: Path(scores) / d.name
        for d in segments
        if (Path(scores) / d.name).is_dir()
    }
    new = pd.concat(
        [
            germany_rows(folders),
            *(held_out_rows(lb, s) for lb, s in settings.items()),
        ],
        ignore_index=True,
    )
    meta = {lb: outbreaks_and_grid(s) for lb, s in settings.items()}
    new["outbreaks"] = new["checkpoint"].map(
        {lb: m[0] for lb, m in meta.items()}
    )
    new["outbreaks"] = new["outbreaks"].fillna(0).astype(int)
    new["grid"] = new["checkpoint"].map({lb: m[1] for lb, m in meta.items()})
    new["run"] = np.where(new["checkpoint"] == "zero-shot", "zero-shot", run)
    new["chosen"] = False
    return new.reindex(columns=COLUMNS)


def comparison(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Return the runs' tables as one, with zero-shot's rows once."""
    rows = pd.concat(tables, ignore_index=True)
    zero = rows["run"] == "zero-shot"
    return pd.concat(
        [rows.loc[zero].drop_duplicates(), rows.loc[~zero]],
        ignore_index=True,
    )


def main(argv: Sequence[str]) -> None:
    """Run one unit, or rebuild the tables."""
    if len(argv) < 3:
        raise SystemExit(__doc__)
    study, cache = argv[0], Path(argv[1])
    if study == "germany" and len(argv) <= 4:
        segment = Path(argv[3]) if len(argv) == 4 else None
        germany_unit(cache, argv[2], segment)
    elif study == "release":
        release_unit(cache, argv[2])
    elif study == "run" and len(argv) == 6:
        out = run_table(cache, Path(argv[2]), Path(argv[3]), argv[4])
        out.round(4).to_csv(argv[5], index=False)
    elif study == "daily" and len(argv) == 6:
        out = daily_table(cache, Path(argv[2]), Path(argv[3]), argv[4])
        out.round(4).to_csv(argv[5], index=False)
    elif study == "compare" and len(argv) >= 3:
        tables = [pd.read_csv(p) for p in argv[2:]]
        comparison(tables).to_csv(cache, index=False)
    elif study == "table" and len(argv) >= 4:
        new = table(cache, Path(argv[2]), Path(argv[3]), argv[4:])
        print(new.round(3).to_string(index=False), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
