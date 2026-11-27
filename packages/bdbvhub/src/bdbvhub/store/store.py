# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""The fit store: one stored fit per stored model and real origin.

The fitted models are slow, so the docs never fit them. The random
walks (`SAMPLE_FREE`) are stored the same way, so the submission, the
backtest and the docs read one forecast of each at each origin.
The Fit store workflow (`.github/workflows/fit-store.yml`) fits each
(model, origin) pair once and stores the result in the `fit-store`
Actions artifact (`bdbvhub.store.artifact`), one file per fit:

```text
<model>/<origin>/<key>.json.gz
```

The key (`fit_key`) hashes only what reaches a fit: the task at the
origin, which holds only the releases up to the origin; the built model,
whose representation gives its variant and resolved sampler settings;
the BVDOutbreakSize release a BVD model uses; data a model reads beside
the task (`fit_inputs`, the zone fits' zone data up to the origin); the
code the model's fit can run (`fit_code`, without docstrings or
comments, and `run_forecasts`), less the zone backtest; the package data
files; and the runtime dependencies locked in `uv.lock`.
A new release adds one fit per model and leaves the older ones valid,
while a change to a model's code or settings refits every origin of the
models that run it, and no other.

`plan` lists the fits the store lacks, `fit` makes one, and `load`
writes the stored forecasts where the backtest and the docs read them.
A submission candidate's fit also holds its fit directory (`fit_files`),
which `unpack` writes out for the submission to forecast from, so the
submission uses the stored fit rather than fitting again. A walk's fit
directory holds its records, and `unpack` adds its stored forecasts
(`fitting.STORED_FORECASTS`).
A walk's stored fit also holds its run on every hub zone (`walk_zones`),
in a file of its own beside the fit (`zones_path`), as it holds about 20
times the fit's rows. `read_fit` leaves it alone; `stored_zones` reads
the zone runs of the fits `load` used, for the consumers that need them.
Where the current key has not been fitted yet (pending), `load` uses
the newest good stored fit of that pair and labels it pending
(`StoreStatus`). Where the current key's fit failed, the model has no
forecast at that origin: no older fit stands in for it.
`command` runs `bdbvhub fit-store`.
"""

from __future__ import annotations

import argparse
import ast
import base64
import dataclasses
import datetime as dt
import functools
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhub.models import get_model, list_models
from nowcastbdbv.backtest import vintage_task
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcastbdbv.data.zones import ZoneData, load_zone_data
from nowcastbdbv.zones import ZoneForecasts
from nowcasteval.backtest import (
    BacktestConfig,
    backtest_origins,
    load_config,
    merge_by_model,
    run_forecasts,
)
from nowcastutils.base import SAMPLE_COLUMNS, ForecastModel, ForecastTask
from nowcastutils.files import write_csv
from nowcastutils.hierarchy import is_zone

__all__ = [
    "FIT_FUNCTIONS",
    "FIT_LABELS",
    "KEY_PACKAGES",
    "KEY_SKIP",
    "SAMPLE_FREE",
    "STATUS_FILE",
    "STORE_DATA",
    "UNPACKED_FILE",
    "ZONES_SUFFIX",
    "ZONE_CODE",
    "ZONE_FITS_FILE",
    "StoreStatus",
    "StoredFit",
    "build_model",
    "code_digest",
    "command",
    "compact_forecasts",
    "current_keys",
    "failed",
    "fit",
    "fit_code",
    "fit_files",
    "fit_key",
    "label_origins",
    "load",
    "newest_fit",
    "origin_tasks",
    "plan",
    "read_fit",
    "read_zones",
    "samplers",
    "stack_forecasts",
    "store_path",
    "stored_delays",
    "stored_models",
    "stored_zone_delays",
    "stored_zone_forecasts",
    "stored_zones",
    "unpack",
    "walk_zones",
    "write_fit",
    "write_fit_dir",
    "zones_path",
]

FIT_FUNCTIONS: Final = {
    "packages/bdbvhub/src/bdbvhub/store/store.py": (
        "build_model",
        "fit",
        "fit_files",
        "walk_zones",
    ),
    "packages/nowcasteval/src/nowcasteval/backtest.py": (
        "_fit_rows",
        "run_forecasts",
    ),
}
"""The functions of the store and the backtest a stored fit runs.

Only these are in a fit's key, by their source. The rest of those
modules, and everything only they import, is loader and IO code: changing
it never refits.
"""

KEY_SKIP: Final = (
    "packages/nowcasteval/src/nowcasteval/backtest.py",
    "packages/bdbvhub/src/bdbvhub/store/digest.py",
    "packages/bdbvhub/src/bdbvhub/store/store.py",
    "packages/nowcasteval/src/nowcasteval/plots.py",
    "packages/bdbvhub/src/bdbvhub/reporting/site.py",
)
"""Loader, IO and report modules a model's code imports but a fit does not
run, left out of its key with what only they import (unless the model's
own module imports them). The store's and backtest's functions a fit runs
are keyed by source instead (`FIT_FUNCTIONS`)."""

RECORD_CODE: Final = "packages/bdbvhub/src/bdbvhub/submission/fitting.py"
"""The fit configurations and `write_records`, which write a candidate's
fit records (`fit_files`). The models with a fit directory
(`renewal.FIT_VARIANTS`) hold the whole file in their keys."""

ZONE_CODE: Final = "packages/nowcastbaselines/src/nowcastbaselines/zones.py"
"""The zone run of a walk (`walk_zones`), in the walks' keys."""

STORE_DATA: Final = "packages/nowcastbdbv/src/nowcastbdbv/data/*.csv"
"""The package's data files, which a fit may read (the province centres
of the renewal models' gravity kernel). Every stored fit's key holds
them."""

STATUS_FILE: Final = "fit-store.json"
"""What `load` used, written beside the results for the docs."""

ZONES_SUFFIX: Final = ".zones.csv.gz"
"""The suffix of a walk's zone run beside its fit (`zones_path`)."""

ZONE_FITS_FILE: Final = "zone-fits.csv.gz"
"""The zone samples of the stored zone fits, which `load` writes beside
the results: `model`, `vintage`, `location`, `target_end_date`, `sample`
and `value`."""

UNPACKED_FILE: Final = "store.json"
"""Which stored fit `unpack` wrote, in the fit directory."""

NEXT_ROUND_WEEKS: Final = 1
"""Weeks a stored fit forecasts past `[forecast] max_horizon`, so a round
with no new release is forecast from the last release's stored fit."""

ZONE_FIELDS: Final = ("zone_history", "zone_daily_history", "zone_codes")
"""Task fields in the fit key of a model only if it reads zones
(`reads_zones`) or is run on them (`SAMPLE_FREE`), so zone data never
refits the others."""

PLACE_FIELDS: Final = ("imputation_delay", "places")
"""Task fields in the fit key of a model only if it reads them
(`reads_places`)."""

SAMPLE_FREE: Final = ("randomwalk", "randomwalknowcast")
"""The models the store holds that fit nothing: the random walks. Each is
a submission candidate whose fit directory holds no samples of its own,
so `unpack` writes its stored forecasts there."""

COMMIT_FILE: Final = ".commit"
"""The fetched store's version, written by `fit-store-artifact fetch`."""


def stored_models() -> list[str]:
    """Return the registered models the store holds.

    Those marked fitted and the random walks (`SAMPLE_FREE`). Models only
    the paper's workflow runs (`ModelSpec.paper_only`) are left out.

    Returns
    -------
    list of str
        Model names.
    """
    return [
        s.name
        for s in list_models()
        if (s.fitted or s.name in SAMPLE_FREE) and not s.paper_only
    ]


def samplers(
    policy: Path | str = "packages/bdbvsubmission/policy.toml",
) -> dict[str, Any]:
    """Return each model's NUTS settings from the CI policy.

    A stored fit uses the sampler table of the submission candidate that
    runs the same model, so the docs show the fits the submission makes.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    dict
        Sampler table by model name.

    Raises
    ------
    ValueError
        If two candidates with a sampler table run the same model, as a
        stored fit has one.
    """
    raw = tomllib.loads(Path(policy).read_text())
    out: dict[str, Any] = {}
    for c in raw["candidates"]:
        if "model" not in c or "sampler" not in c:
            continue
        if c["model"] in out:
            msg = f"Two candidates in {policy} run {c['model']}."
            raise ValueError(msg)
        out[c["model"]] = dict(c["sampler"])
    return out


def build_model(name: str, sampler: Mapping[str, Any]) -> ForecastModel:
    """Build a registered model with its sampler table.

    Parameters
    ----------
    name : str
        The model name.
    sampler : mapping
        `samplers`. A model absent from it keeps its own settings.

    Returns
    -------
    ForecastModel
        The model. The environment is left as it was.
    """
    from nowcastrenewal.forecaster import SAMPLER_ENV

    old = os.environ.pop(SAMPLER_ENV, None)
    try:
        if name in sampler:
            os.environ[SAMPLER_ENV] = json.dumps(dict(sampler[name]))
        return get_model(name)
    finally:
        os.environ.pop(SAMPLER_ENV, None)
        if old is not None:
            os.environ[SAMPLER_ENV] = old


def _strip(source: str) -> str:
    """Dump Python source as its syntax tree, without docstrings."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        kept = [
            s
            for s in body
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
        ]
        node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.dump(tree)


KEY_PACKAGES: Final = (
    "nowcastutils",
    "nowcasteval",
    "nowcastbaselines",
    "nowcastfoundation",
    "nowcastrenewal",
    "nowcastgermany",
    "nowcastbdbv",
    "bdbvhub",
)
"""The workspace's packages, whose import lines leave a fit key: their
modules are in the key themselves (`fit_code`), so a module keeps its
key when it moves between them."""


def _own_import(node: ast.stmt) -> bool:
    """Whether a statement imports from `KEY_PACKAGES` or `__future__`."""

    def own(module: str) -> bool:
        top = module.split(".")[0]
        return top in KEY_PACKAGES or top == "__future__"

    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or own(node.module or "")
    if isinstance(node, ast.Import):
        return all(own(alias.name) for alias in node.names)
    return False


def _code_parts(source: str) -> list[str]:
    """Hash each top-level statement of a module, with what it imports.

    Docstrings, `__all__` and the package's own import lines are left
    out (`_own_import`), nested ones included. Each remaining statement is
    hashed as its syntax tree, together with the third-party module-level
    imports of the names it reads, so moving a function to another file
    keeps its hash while a change of library does not.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        kept = [
            s
            for s in body
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
            and not _own_import(s)
        ]
        node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    bound: dict[str, str] = {}
    parts = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                name = alias.asname or alias.name.split(".")[0]
                bound[name] = f"import {alias.name}"
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                name = alias.asname or alias.name
                bound[name] = f"from {stmt.module} import {alias.name}"
        elif isinstance(stmt, ast.Pass) or _is_all(stmt):
            continue
        else:
            reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
            uses = sorted(bound[n] for n in reads if n in bound)
            body = ast.dump(stmt) + "\0" + "\n".join(uses)
            parts.append(hashlib.sha256(body.encode()).hexdigest())
    return parts


def _is_all(stmt: ast.stmt) -> bool:
    """Whether a statement assigns `__all__`."""
    targets = (
        stmt.targets
        if isinstance(stmt, ast.Assign)
        else [stmt.target]
        if isinstance(stmt, (ast.AnnAssign, ast.AugAssign))
        else []
    )
    return any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)


def _fit_functions(root: Path) -> list[str]:
    """Dump each function in `FIT_FUNCTIONS` from `root`'s source."""
    out = []
    for rel, names in FIT_FUNCTIONS.items():
        tree = ast.parse((root / rel).read_text())
        defs = {
            n.name: n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = sorted(set(names) - set(defs))
        if missing:
            raise ValueError(f"{rel} has no {missing}")
        out += [_strip(ast.unparse(defs[name])) for name in names]
    return out


def fit_code(model: str, root: Path | str = ".") -> list[str]:
    """List the source files a stored fit of one model can run.

    The import closure of the model's module, with any other model's
    module it builds on (`bdbvhub.store.digest.model_code`), and the
    package's data files (`STORE_DATA`), without the loader, IO and report
    modules in `KEY_SKIP`. A model with a fit directory (a renewal
    variant or a walk) also holds the code that writes its fit records
    (`RECORD_CODE`), and a walk the code of its zone run (`ZONE_CODE`).
    The store's and the
    backtest's own code is keyed by function (`FIT_FUNCTIONS`), not by
    file.

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    list of str
        Paths relative to `root`, sorted.
    """
    from bdbvhub.store.digest import model_code
    from nowcastrenewal.forecaster import FIT_VARIANTS

    code = model_code(model, root, (), cli=False, skip=KEY_SKIP)
    if model in FIT_VARIANTS.values() or model in SAMPLE_FREE:
        code = sorted({*code, RECORD_CODE})
    if model in SAMPLE_FREE:
        code = sorted({*code, ZONE_CODE})
    data = [p.relative_to(root).as_posix() for p in Path(root).glob(STORE_DATA)]
    return sorted([*code, *data])


@functools.cache
def _export(root: str, lock: str) -> str:
    """Return `uv export` of the runtime dependencies; `lock` keys the cache."""
    del lock
    return subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--no-hashes",
            "--no-header",
            "--no-emit-project",
            "--no-emit-workspace",
            # No `# via <package>` lines, which name the workspace's own
            # packages and would change every key when modules move.
            "--no-annotate",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def code_digest(model: str, root: Path | str = ".") -> str:
    """Hash the code a stored fit of a model runs and its dependencies.

    Each top-level statement of the Python files in `fit_code` is hashed
    as its syntax tree, without docstrings, `__all__` or the package's own
    import lines, with the third-party imports it reads (`_code_parts`);
    other files by their name and bytes. The hashes are combined in sorted
    order, not by file, so comments, docstrings, renamed or moved modules
    and functions moved between files keep the key. The functions of the store
    and the backtest a fit runs (`FIT_FUNCTIONS`) are hashed the same
    way, read from `root`, with the Python minor version.
    The dependencies are `uv export --no-dev --all-extras --no-annotate`
    of `uv.lock`.

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    base = Path(root)
    names = fit_code(model, base)
    files: list[Path] = []
    for name in names:
        path = base / name
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    parts = []
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix == ".py":
            parts += _code_parts(path.read_text())
        else:
            body = path.name.encode() + b"\0" + path.read_bytes()
            parts.append(hashlib.sha256(body).hexdigest())
    digest = hashlib.sha256()
    for part in sorted(parts):
        digest.update(part.encode())
    for part in _fit_functions(base):
        digest.update(part.encode())
    digest.update(repr(sys.version_info[:2]).encode())
    lock = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        lock.update((base / name).read_bytes())
    digest.update(_export(str(base.resolve()), lock.hexdigest()).encode())
    return digest.hexdigest()


def origin_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    daily: pd.DataFrame,
    zones: ZoneData | None = None,
) -> dict[str, ForecastTask]:
    """Build the task of every real origin, as the store fits it.

    Each is the task the submission fits at that vintage
    (`vintage_task`), with `NEXT_ROUND_WEEKS` more weeks, so the
    submission forecasts its vintage's own round, or the next round when
    no new release came, from the stored fit. A stored task is asked for
    final counts only (no `report_by`), so it holds no release after its
    origin. The origins are `nowcasteval.backtest.backtest_origins`;
    earlier releases stay in the later tasks' histories.

    Parameters
    ----------
    target, delays, daily : pandas.DataFrame
        The tidy tables.
    config : BacktestConfig
        Horizons, samples and seed (`[forecast]`).
    zones : ZoneData, optional
        The health-zone tables, for the models that read zones.

    Returns
    -------
    dict
        Task by origin (the vintage date), in date order.
    """
    return {
        v: vintage_task(
            target,
            delays,
            config,
            v,
            daily,
            NEXT_ROUND_WEEKS,
            zones=zones,
        )[0]
        for v in backtest_origins(target, config)
    }


def _bvd_tag(model: ForecastModel, task: ForecastTask) -> str | None:
    """Return the BVDOutbreakSize release a BVD model uses, else None."""
    variant = getattr(model, "variant", None)
    if variant is None:
        return None
    from nowcastrenewal.forecaster import bvd_record, bvd_release, model_config

    if not model_config(variant).bvd:
        return None
    tag = bvd_record(bvd_release(task))["tag"]
    return None if tag is None else str(tag)


def _settings(model: ForecastModel) -> str:
    """Return a dataclass model's fields, else its class name."""
    return repr(model) if is_dataclass(model) else type(model).__qualname__


def fit_key(model: ForecastModel, task: ForecastTask, code: str) -> str:
    """Hash the inputs of one stored fit.

    The task's zone data (`ZONE_FIELDS`) are hashed only for a model that
    reads them, or that the store runs on the zones (`SAMPLE_FREE`).

    Parameters
    ----------
    model : ForecastModel
        The model, from `build_model`, before it fits anything.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    code : str
        `code_digest`.

    Returns
    -------
    str
        The first 16 hex digits of a SHA-256 hash.
    """
    digest = hashlib.sha256()
    zones = getattr(model, "reads_zones", False)
    reads_places = getattr(model, "reads_places", False)
    head = {
        "model": _settings(model),
        "code": code,
        "bvd": _bvd_tag(model, task),
    }
    # Data a model reads beside the task (the zone fits' zone data).
    inputs = getattr(model, "fit_inputs", None)
    if inputs is not None:
        head["inputs"] = inputs(task)
    digest.update(json.dumps(head, sort_keys=True).encode())
    on_zones = zones or model.name in SAMPLE_FREE
    for f in fields(task):
        if f.name in ZONE_FIELDS and not on_zones:
            continue
        if f.name in PLACE_FIELDS and not reads_places:
            continue
        value = getattr(task, f.name)
        if f.name == "places" and value is not None and not zones:
            value = value.loc[~value["location"].map(is_zone)]
        digest.update(f.name.encode())
        if isinstance(value, pd.DataFrame):
            digest.update(value.to_csv(index=False).encode())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:16]


Keys = dict[tuple[str, str], str]
"""Current key by (model, origin)."""


def current_keys(
    tasks: Mapping[str, ForecastTask],
    models: Sequence[str],
    sampler: Mapping[str, Any],
    code: str | Mapping[str, str],
) -> Keys:
    """Return the current key of every stored model at every origin.

    Parameters
    ----------
    tasks : mapping
        `origin_tasks`.
    models : sequence of str
        The stored models.
    sampler : mapping
        `samplers`.
    code : str or mapping
        `code_digest`, or each model's by name.

    Returns
    -------
    dict
        Key by (model, origin).
    """
    built = {m: build_model(m, sampler) for m in models}
    codes = {m: code if isinstance(code, str) else code[m] for m in models}
    return {
        (m, origin): fit_key(built[m], task, codes[m])
        for origin, task in tasks.items()
        for m in models
    }


def store_path(store: Path | str, model: str, origin: str, key: str) -> Path:
    """Return where a fit is stored.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root (as `fit-store-artifact fetch` unpacks it).
    model, origin, key : str
        The fit.

    Returns
    -------
    pathlib.Path
        `<store>/<model>/<origin>/<key>.json.gz`.
    """
    return Path(store) / model / origin / f"{key}.json.gz"


@dataclass(frozen=True)
class StoredFit:
    """One stored fit.

    Attributes
    ----------
    meta : dict
        `model`, `origin`, `key`, `commit` (the code fitted), `fitted_at`
        (ISO time), `settings` (the model's representation) and `status`
        (`"ok"`, or `"failed"` with `error` and no tables).
    forecasts : pandas.DataFrame
        `run_forecasts` output.
    fits : pandas.DataFrame
        The model's fit summaries (`fit_diagnostics`).
    delays : pandas.DataFrame
        The learnt delay (`delay_summary`), empty if the model has none.
    files : dict
        A submission candidate's fit directory (`fit_files`): file name
        to base64 contents. Empty for other models.
    zones : pandas.DataFrame
        A walk's zone run (`walk_zones`), written to its own file
        (`zones_path`) and left out by `read_fit` (`read_zones` reads
        it). Empty for other models.
    zone_delays : pandas.DataFrame
        The delay a walk learnt in its zone run, empty if it has none.
    """

    meta: dict[str, Any]
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    delays: pd.DataFrame
    files: dict[str, str] = field(default_factory=dict)
    zones: pd.DataFrame = field(default_factory=pd.DataFrame)
    zone_delays: pd.DataFrame = field(default_factory=pd.DataFrame)


_TABLES: Final = ("forecasts", "fits", "delays", "zone_delays")


def zones_path(store: Path | str, model: str, origin: str, key: str) -> Path:
    """Return where a walk's zone run is stored, beside its fit.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    model, origin, key : str
        The fit.

    Returns
    -------
    pathlib.Path
        `<store>/<model>/<origin>/<key>.zones.csv.gz`.
    """
    return Path(store) / model / origin / f"{key}{ZONES_SUFFIX}"


def write_fit(store: Path | str, stored: StoredFit) -> Path:
    """Write a fit to the store as gzipped JSON.

    A walk's zone run goes to its own file (`zones_path`).

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    stored : StoredFit
        The fit.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    m = stored.meta
    path = store_path(store, m["model"], m["origin"], m["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "meta": m,
        **{
            name: json.loads(
                getattr(stored, name).to_json(orient="split", index=False)
            )
            for name in _TABLES
        },
        "files": stored.files,
    }
    # No timestamp in the gzip header, so the bytes depend on the content.
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(json.dumps(body).encode())
    if not stored.zones.empty:
        write_csv(
            stored.zones, zones_path(store, m["model"], m["origin"], m["key"])
        )
    return path


def read_fit(path: Path | str) -> StoredFit:
    """Read a stored fit.

    Parameters
    ----------
    path : pathlib.Path or str
        A file written by `write_fit`.

    Returns
    -------
    StoredFit
        The fit.
    """
    body = json.loads(gzip.decompress(Path(path).read_bytes()))

    def table(name: str) -> pd.DataFrame:
        if name not in body:
            return pd.DataFrame()
        return pd.read_json(
            io.StringIO(json.dumps(body[name])),
            orient="split",
            dtype=False,
        )

    forecasts, fits, delays, zone_delays = (table(n) for n in _TABLES)
    return StoredFit(
        body["meta"],
        forecasts,
        fits,
        delays,
        body.get("files", {}),
        zone_delays=zone_delays,
    )


def read_zones(path: Path | str) -> pd.DataFrame:
    """Read a walk's stored zone run, text columns as categories.

    Parameters
    ----------
    path : pathlib.Path or str
        A file `write_fit` wrote (`zones_path`).

    Returns
    -------
    pandas.DataFrame
        The `walk_zones` rows, none if there is no such file.
    """
    if not Path(path).exists():
        return pd.DataFrame()
    text = dict.fromkeys(TEXT_COLUMNS, str)
    return compact_forecasts(pd.read_csv(path, dtype=text))


def _commit() -> str:
    """Return the commit being fitted: the CI's, else the checkout's."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def _meta(model: ForecastModel, origin: str, key: str) -> dict[str, Any]:
    return {
        "model": model.name,
        "origin": origin,
        "key": key,
        "commit": _commit(),
        "fitted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "settings": _settings(model),
        "status": "ok",
    }


def plan(
    keys: Keys,
    store: Path | str,
    retry_failed: bool = False,
    usable: bool = False,
) -> list[dict[str, str]]:
    """List the fits the store lacks.

    A fit that failed is stored as failed and is not planned again until
    its inputs change, unless `retry_failed`.
    With `usable`, a pair is not lacking when `unpack` can use a fit
    (`_choose`: the current key's, else, while it is pending, the newest
    good one), as the submission asks.

    Parameters
    ----------
    keys : dict
        `current_keys`.
    store : pathlib.Path or str
        The store root.
    retry_failed : bool
        Plan the fits stored as failed again.
    usable : bool
        Count an older stored fit of the pair as present.

    Returns
    -------
    list of dict
        `model`, `origin` and `key` of each fit to make, newest origin
        first, so the Nowcast page's fits come first.
    """
    to_fit = []
    for (model, origin), key in sorted(keys.items(), key=lambda x: x[0][1])[
        ::-1
    ]:
        path = store_path(store, model, origin, key)
        redo = retry_failed and path.exists()
        if redo:
            redo = read_fit(path).meta["status"] != "ok"
        if usable and _choose(Path(store), model, origin, key)[0]:
            continue
        if not path.exists() or redo:
            to_fit.append({"model": model, "origin": origin, "key": key})
    return to_fit


def fit(
    name: str, task: ForecastTask, sampler: Mapping[str, Any], code: str
) -> StoredFit:
    """Fit a model at an origin and return it for the store.

    A fit that raises is returned as failed (`failed`).

    Parameters
    ----------
    name : str
        The model name.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    sampler : mapping
        `samplers`.
    code : str
        `code_digest`.

    Returns
    -------
    StoredFit
        The forecasts, fit summaries and learnt delay.
    """
    model = build_model(name, sampler)
    meta = _meta(model, task.vintage, fit_key(model, task, code))
    fits: list[pd.DataFrame] = []
    learnt: list[pd.DataFrame] = []
    start = time.monotonic()
    zones, zone_delays = pd.DataFrame(), pd.DataFrame()
    try:
        forecasts = run_forecasts([model], [task], fits, learnt)
        files = fit_files(model, task, sampler, time.monotonic() - start)
        if name in SAMPLE_FREE:
            zones, zone_delays = walk_zones(build_model(name, sampler), task)
    except Exception as err:
        return failed(meta, err)
    return StoredFit(
        meta,
        forecasts,
        pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(),
        pd.concat(learnt, ignore_index=True) if learnt else pd.DataFrame(),
        files,
        zones,
        zone_delays,
    )


def walk_zones(
    model: ForecastModel, task: ForecastTask
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a walk on every hub zone of the task's provinces.

    The zone run of the submission, the hub comparison, the zone backtest
    and the docs (`nowcastbaselines.zones.zone_forecast`): each zone from
    its own release snapshots up to the origin (`zone_daily_history`),
    for every hub zone (`zone_codes`), whether or not it has cases.

    Parameters
    ----------
    model : ForecastModel
        A walk (`SAMPLE_FREE`), before it forecasts anything.
    task : ForecastTask
        The task at the origin (`origin_tasks`), with its zone data.

    Returns
    -------
    tuple of pandas.DataFrame
        The final-count zone samples as `run_forecasts` gives them
        (`model`, `kind`, `vintage`, `reference_date`, `horizon`,
        `location`, `target_end_date`, `sample` and `value`), none if
        the task has no zone data, and the delay the walk learnt on the
        zones (`delay_summary`), none if it has none.
    """
    from nowcastbaselines.zones import zone_forecast
    from nowcastutils.snapshots import weekly_counts

    daily = task.zone_daily_history
    if daily is None or not task.zone_codes:
        return pd.DataFrame(), pd.DataFrame()
    counts = weekly_counts(daily, task.vintage)
    samples = zone_forecast(model, task, counts, task.zone_codes, daily)
    ref = pd.Timestamp(task.reference_date)
    weeks = (pd.to_datetime(samples["target_end_date"]) - ref).dt.days // 7
    zones = samples.assign(
        model=model.name,
        kind=task.kind,
        vintage=task.vintage,
        reference_date=task.reference_date.isoformat(),
        horizon=weeks,
    )
    cols = ["model", "kind", "vintage", "reference_date", "horizon"]
    summary = getattr(model, "delay_summary", None)
    return (
        zones[[*cols, *SAMPLE_COLUMNS]],
        pd.DataFrame() if summary is None else summary,
    )


def fit_files(
    model: ForecastModel,
    task: ForecastTask,
    sampler: Mapping[str, Any],
    seconds: float = 0.0,
) -> dict[str, str]:
    """Return a submission candidate's fit directory, after its forecast.

    The directory `bdbvhub fit --config` would write: the model's
    `write_fit_dir` (samples and `ppc.json`), with `diagnostics.json`
    and `fit.json` (`write_records`). A walk (`SAMPLE_FREE`) writes its
    configuration's records and the copula it joined its draws by
    (`fitting.COPULA_FILE`); `unpack` adds its forecasts. A model that is
    no candidate's (`renewal.FIT_VARIANTS`) has none.

    Parameters
    ----------
    model : ForecastModel
        The model, after it forecast `task`.
    task : ForecastTask
        The task.
    sampler : mapping
        `samplers`.
    seconds : float
        How long the fit took.

    Returns
    -------
    dict
        File name to base64 contents, or empty.
    """
    from bdbvhub.submission.fitting import (
        COPULA_FILE,
        get_config,
        write_records,
    )
    from nowcastrenewal.forecaster import FIT_VARIANTS

    write = getattr(model, "write_fit_dir", None)
    names = [c for c, v in FIT_VARIANTS.items() if v == model.name]
    if model.name in SAMPLE_FREE:
        names = [model.name]
    elif write is None or not names:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        if write is None:
            config = get_config(names[0])
            diag = config.fit(task, sampler.get(model.name, {}), out)
            copula = getattr(model, "copula_fit", None)
            if copula is not None:
                (out / COPULA_FILE).write_text(
                    json.dumps(copula, default=str) + "\n"
                )
        else:
            diag = write(task, out, config_name=names[0])
        write_records(
            get_config(names[0]),
            task,
            sampler.get(model.name, {}),
            out,
            model=model.name,
            diagnostics=diag,
            seconds=seconds,
        )
        return {
            p.name: base64.b64encode(p.read_bytes()).decode()
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "progress.log"
        }


def failed(meta: Mapping[str, Any], error: BaseException) -> StoredFit:
    """Record a fit that failed, so the store does not retry it.

    Parameters
    ----------
    meta : mapping
        The fit's meta.
    error : BaseException
        Why it failed.

    Returns
    -------
    StoredFit
        Meta only, with `status` `"failed"` and `error`.
    """
    empty = pd.DataFrame()
    record = {**meta, "status": "failed", "error": repr(error)}
    return StoredFit(record, empty, empty, empty)


def unpack(
    store: Path | str,
    keys: Keys,
    model: str,
    origin: str,
    out: Path | str,
) -> tuple[str, StoredFit | None]:
    """Write a stored candidate fit's directory for the submission.

    The fit of the current key is used when stored (same code and data,
    same fit), else, while it is pending, the newest stored fit of the
    model and origin (`_choose`), chosen for each model on its own.
    Writes the fit directory (`write_fit_dir`) and `UNPACKED_FILE`,
    which says which fit it came from (its key and commit, and the
    current key) and at which store commit.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    keys : dict
        `current_keys`.
    model, origin : str
        The fit.
    out : pathlib.Path or str
        The fit directory to write.

    Returns
    -------
    tuple
        `"ok"`, `"missing"` (no fit, or one with no fit directory) or
        `"failed"`, and the stored fit if there is one.
    """
    current = keys[(model, origin)]
    stored, bad = _choose(Path(store), model, origin, current)
    if bad:
        return "failed", read_fit(store_path(store, model, origin, current))
    if stored is None:
        return "missing", None
    if not stored.files:
        return "missing", stored
    target = Path(out)
    write_fit_dir(store, stored, target)
    commit = Path(store) / COMMIT_FILE
    record = {
        "model": model,
        "origin": origin,
        "key": stored.meta["key"],
        "current_key": current,
        "commit": stored.meta["commit"],
        "fitted_at": stored.meta["fitted_at"],
        "store_commit": (
            commit.read_text().strip() if commit.exists() else None
        ),
    }
    (target / UNPACKED_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return "ok", stored


def write_fit_dir(store: Path | str, stored: StoredFit, out: Path) -> None:
    """Write a stored fit's directory, as the submission forecasts from it.

    The fit's files, and for a walk (`SAMPLE_FREE`) its stored forecasts
    (`fitting.STORED_FORECASTS`) and its zone run as the zone fits give
    theirs (`nowcastrenewal.zones.ZONES_FILE`), if it has one.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    stored : StoredFit
        The fit, read from `store`.
    out : pathlib.Path
        The fit directory.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, body in stored.files.items():
        (out / name).write_bytes(base64.b64decode(body))
    meta = stored.meta
    if meta["model"] not in SAMPLE_FREE:
        return
    from bdbvhub.submission.fitting import STORED_FORECASTS
    from nowcastrenewal.zones import ZONES_FILE

    write_csv(stored.forecasts, out / STORED_FORECASTS)
    zones = read_zones(
        zones_path(store, meta["model"], meta["origin"], meta["key"])
    )
    if not zones.empty:
        write_csv(zones[list(SAMPLE_COLUMNS)], out / ZONES_FILE)


@dataclass(frozen=True)
class StoreStatus:
    """What `load` used for one (model, origin) pair.

    Attributes
    ----------
    model, origin : str
        The pair.
    key : str
        The key of the current code and data.
    used : str or None
        The key of the fit loaded, None if none was: the store has none,
        the current key's fit failed, or only current fits were asked for.
    commit : str or None
        The commit the loaded fit was made at.
    current : bool
        The loaded fit is the current key's.
    failed : bool
        The current key's fit failed, so the model has no forecast here.
    pending : bool
        The current key has not been fitted yet; a fit in `used` is an
        older one standing in until it is.
    """

    model: str
    origin: str
    key: str
    used: str | None
    commit: str | None
    current: bool
    failed: bool
    pending: bool = False


def _choose(
    store: Path, model: str, origin: str, key: str
) -> tuple[StoredFit | None, bool]:
    """Return the fit to use for a pair, and whether the current key failed.

    The current key's fit when it is stored and good. When it failed,
    none: the model has no forecast here, and no fit of other code or
    data stands in. When it has not been fitted yet (pending), the newest
    good fit of the pair.

    Returns
    -------
    tuple
        The fit to load (or None) and whether the current key failed.
    """
    found = [read_fit(p) for p in (store / model / origin).glob("*.json.gz")]
    current = [f for f in found if f.meta["key"] == key]
    if current:
        failed = current[0].meta["status"] != "ok"
        return (None, True) if failed else (current[0], False)
    good = [f for f in found if f.meta["status"] == "ok"]
    if not good:
        return None, False
    return max(good, key=lambda f: f.meta["fitted_at"]), False


def newest_fit(store: Path | str, keys: Keys, model: str) -> StoredFit | None:
    """Return the fit `load` uses at the newest origin that has one.

    At each origin of the model in `keys`, newest first, the fit of the
    current key is used when stored and good, else, while it is pending,
    the newest good fit of older code (`_choose`). An origin whose
    current fit failed is passed over.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    keys : dict
        `current_keys`.
    model : str
        The stored model.

    Returns
    -------
    StoredFit or None
        The fit, with its origin and key in `meta`; None if no origin has
        a good fit.
    """
    origins = sorted((o for m, o in keys if m == model), reverse=True)
    for origin in origins:
        chosen, _ = _choose(Path(store), model, origin, keys[(model, origin)])
        if chosen is not None:
            return chosen
    return None


def load(
    store: Path | str,
    keys: Keys,
    results: Path | str = "results",
    min_horizon: int | None = None,
    max_horizon: int | None = None,
    current_only: bool = False,
) -> list[StoreStatus]:
    """Write the stored fits where the backtest and the docs read them.

    Each pair's fit is `_choose`'s: the current key's, none if it failed,
    and while it is pending the newest good older fit, unless
    `current_only`.

    Every origin goes to `forecasts.csv.gz`, from `min_horizon` to
    `max_horizon`: a stored fit holds the submission's weeks, which start
    earlier and end later.
    Fit summaries go to `backtest-fits.csv`, learnt delays to
    `delay-summary.csv` and the zone fits' final-count zone samples to
    `ZONE_FITS_FILE`. Each file keeps the rows of models not stored.
    What was used is written to `STATUS_FILE`.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    keys : dict
        `current_keys`.
    results : pathlib.Path or str
        The results directory.
    min_horizon, max_horizon : int, optional
        The backtest's weeks (`[forecast]`). None keeps every week.
    current_only : bool
        Load only the current keys' fits: a pending pair loads none.

    Returns
    -------
    list of StoreStatus
        One per (model, origin), newest origin first.
    """
    from nowcastrenewal.forecaster import DELAY_SUMMARY_FILE

    out = Path(results)
    out.mkdir(parents=True, exist_ok=True)
    status: list[StoreStatus] = []
    loaded: list[StoredFit] = []
    forecasts: list[pd.DataFrame] = []
    zones: list[pd.DataFrame] = []
    low = -(10**6) if min_horizon is None else min_horizon
    high = 10**6 if max_horizon is None else max_horizon
    for (model, origin), key in sorted(keys.items(), key=lambda x: x[0][1])[
        ::-1
    ]:
        chosen, failed = _choose(Path(store), model, origin, key)
        pending = not failed and (chosen is None or chosen.meta["key"] != key)
        if current_only and pending:
            chosen = None
        if chosen is not None:
            # Each fit's forecasts are cut to the backtest's weeks and
            # compacted as read (`compact_forecasts`), as are its zone
            # samples, so the store's samples never sit in memory as text
            # at once.
            forecasts.append(compact_forecasts(chosen.forecasts, low, high))
            zones.append(compact_forecasts(_zone_samples(chosen)))
            loaded.append(
                dataclasses.replace(chosen, forecasts=pd.DataFrame(), files={})
            )
        status.append(
            StoreStatus(
                model,
                origin,
                key,
                None if chosen is None else chosen.meta["key"],
                None if chosen is None else chosen.meta["commit"],
                chosen is not None and chosen.meta["key"] == key,
                failed,
                pending,
            )
        )
    for name, frames in (
        ("forecasts.csv.gz", [stack_forecasts(forecasts)]),
        ("backtest-fits.csv", [f.fits for f in loaded]),
        (DELAY_SUMMARY_FILE, [f.delays for f in loaded]),
        (ZONE_FITS_FILE, [stack_forecasts(zones)]),
    ):
        _merge(out / name, frames)
    (out / STATUS_FILE).write_text(
        json.dumps([vars(s) for s in status], indent=1) + "\n"
    )
    return status


def _used(
    results: Path | str,
    models: Sequence[str],
    origins: Sequence[str] | None,
) -> list[tuple[str, str, str]]:
    """Return the (model, origin, key) of each fit `load` used."""
    status = Path(results) / STATUS_FILE
    if not status.exists():
        return []
    return [
        (s["model"], s["origin"], s["used"])
        for s in json.loads(status.read_text())
        if s.get("used")
        and s["model"] in models
        and (origins is None or s["origin"] in origins)
    ]


FIT_LABELS: Final = {
    "failed": "the fit for this code failed",
    "pending": "not yet fitted for this code",
}
"""Why a stored model has no forecast at an origin (`label_origins`)."""


def label_origins(
    origins: pd.DataFrame, status: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Say which stored fit each scored model at each origin came from.

    Adds `fit` (`"current"`, `"pending"`: an older fit standing in for
    one not made yet, or `"failed"`), `fit_key` and `fit_commit` from the
    `load` record. A stored model with no fit loaded at an origin of the
    table, and no row there, is listed with no forecast and why.

    Parameters
    ----------
    origins : pandas.DataFrame
        The release backtest's origins: `origin`, `model`, `status` and
        `reason`, with any count columns.
    status : sequence of mapping
        The `load` record (`STATUS_FILE`).

    Returns
    -------
    pandas.DataFrame
        The origins with the fit columns; rows not from the store keep
        them empty.
    """
    out = origins.assign(
        origin=origins["origin"].astype(str),
        fit=None,
        fit_key=None,
        fit_commit=None,
    )
    scored = set(out["origin"])
    added = []
    for s in status:
        origin = str(s["origin"])
        if origin not in scored:
            continue
        label = (
            "current"
            if s.get("current")
            else "failed"
            if s.get("failed")
            else "pending"
        )
        here = (out["model"] == s["model"]) & (out["origin"] == origin)
        if not s.get("used"):
            here &= out["status"] == "no forecast"
        if here.any():
            out.loc[here, ["fit", "fit_key", "fit_commit"]] = [
                label,
                s.get("used"),
                s.get("commit"),
            ]
        elif (
            not s.get("used")
            and not (
                (out["model"] == s["model"]) & (out["origin"] == origin)
            ).any()
        ):
            added.append(
                {
                    "origin": origin,
                    "model": s["model"],
                    "status": "no forecast",
                    "reason": FIT_LABELS[label],
                    "fit": label,
                }
            )
    if added:
        out = pd.concat([out, pd.DataFrame(added)], ignore_index=True)
    counts = [c for c in ("n_reported", "n_complete") if c in out]
    out[counts] = out[counts].fillna(0).astype(int)
    return out.sort_values("origin", kind="stable").reset_index(drop=True)


def stored_zones(
    store: Path | str,
    results: Path | str,
    models: Sequence[str] = SAMPLE_FREE,
    origins: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the walks' stored zone runs of the fits `load` used.

    `load` chooses each pair's fit (`_choose`: the current key's, else,
    while it is pending, the newest good one) and records it in
    `STATUS_FILE`. The zone runs
    of those fits are read only here, for the consumers that need them.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root `load` read.
    results : pathlib.Path or str
        Where `load` wrote `STATUS_FILE`.
    models : sequence of str
        The walks to read.
    origins : sequence of str, optional
        Only these origins. All if None.

    Returns
    -------
    pandas.DataFrame
        `walk_zones` rows, text columns as categories; none if `load` has
        not run or used no fit with a zone run.
    """
    return stack_forecasts(
        [
            read_zones(zones_path(store, m, o, k))
            for m, o, k in _used(results, models, origins)
        ]
    )


def stored_zone_delays(
    store: Path | str,
    results: Path | str,
    models: Sequence[str] = SAMPLE_FREE,
    origins: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the delays the walks learnt in the zone runs `load` used.

    Parameters
    ----------
    store, results, models, origins
        As for `stored_zones`.

    Returns
    -------
    pandas.DataFrame
        The `zone_delays` of each fit, with `origin`.
    """
    frames = [
        read_fit(store_path(store, m, o, k)).zone_delays.assign(origin=o)
        for m, o, k in _used(results, models, origins)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stored_zone_forecasts(
    store: Path | str,
    results: Path | str,
    daily_zones: pd.DataFrame,
    models: Sequence[str] = SAMPLE_FREE,
    origins: Sequence[str] | None = None,
    min_horizon: int | None = None,
    max_horizon: int | None = None,
) -> ZoneForecasts:
    """Return the stored zone runs as the zone backtest scores them.

    Each origin keeps the zones with cases in its release and the
    backtest's weeks, as `load` keeps them for the provinces, for the
    zone backtest (`nowcastbdbv.zones.score_zones`).

    Parameters
    ----------
    store, results, models, origins
        As for `stored_zones`.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    min_horizon, max_horizon : int, optional
        The backtest's weeks (`[forecast]`). None keeps every week.

    Returns
    -------
    ZoneForecasts
        The runs, samples under each model's name and delays by origin.
    """
    from nowcastutils.snapshots import weekly_counts

    low = -(10**6) if min_horizon is None else min_horizon
    high = 10**6 if max_horizon is None else max_horizon
    runs, samples = [], []
    for model, origin, key in _used(results, models, origins):
        zones = read_zones(zones_path(store, model, origin, key))
        if zones.empty:
            continue
        active = set(weekly_counts(daily_zones, origin)["location"])
        mine = zones.loc[
            zones["location"].astype(str).isin(active)
            & zones["horizon"].between(low, high)
        ]
        runs.append({"origin": origin, "model": model})
        samples.append(
            mine[list(SAMPLE_COLUMNS)]
            .astype({"location": str, "target_end_date": str})
            .assign(key=model, origin=origin)
        )
    delays = stored_zone_delays(store, results, models, origins)
    return ZoneForecasts(
        pd.DataFrame(runs, columns=["origin", "model"]),
        (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()),
        delays,
    )


def stored_delays(results: Path | str) -> pd.DataFrame:
    """Return the fitted models' stored delay summaries.

    `fit-store load` writes them to `results`.

    Parameters
    ----------
    results : pathlib.Path or str
        The results directory.

    Returns
    -------
    pandas.DataFrame
        The delay summaries of the fitted models, with `model` and
        `vintage`, and no rows if no fits were loaded.
    """
    from nowcastrenewal.forecaster import DELAY_SUMMARY_FILE

    path = Path(results) / DELAY_SUMMARY_FILE
    if not path.exists():
        return pd.DataFrame(columns=["model", "vintage"])
    learnt = pd.read_csv(path)
    fitted = [s.name for s in list_models() if s.fitted]
    return learnt.loc[learnt["model"].isin(fitted)]


def _zone_samples(fit: StoredFit) -> pd.DataFrame:
    """Return a stored zone fit's final-count zone samples, else none."""
    from nowcastrenewal.zones import ZONES_FILE

    body = fit.files.get(ZONES_FILE)
    if body is None:
        return pd.DataFrame()
    zones = pd.read_csv(
        io.BytesIO(base64.b64decode(body)),
        compression="gzip",
        dtype={"target_end_date": str},
    )
    if "as_of" in zones:
        zones = zones.loc[zones["as_of"].isna()].drop(columns="as_of")
    return zones.assign(
        model=fit.meta["model"], vintage=fit.meta["origin"]
    ).reset_index(drop=True)


TEXT_COLUMNS: Final = (
    "model",
    "kind",
    "vintage",
    "reference_date",
    "location",
    "target_end_date",
)
"""Stored forecasts' text columns, held as categories when loading."""


def compact_forecasts(
    frame: pd.DataFrame, low: int = -(10**6), high: int = 10**6
) -> pd.DataFrame:
    """Return a fit's forecasts at weeks `low` to `high`, text as categories.

    A stored fit with the zones holds about 600,000 sample rows. As text
    they take about 250 MB, and the store's fits together more than a CI
    runner's memory.
    """
    if "horizon" in frame:
        frame = frame.loc[frame["horizon"].between(low, high)]
    cols = [c for c in TEXT_COLUMNS if c in frame]
    return frame.astype(dict.fromkeys(cols, "category")).reset_index(drop=True)


def stack_forecasts(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate forecasts, text columns as categories shared by all.

    `pd.concat` turns categories that differ between frames back into
    text, so each text column is first given every frame's categories.
    The values equal `pd.concat` of the frames.
    """
    frames = [compact_forecasts(f) for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    shared = {}
    for col in TEXT_COLUMNS:
        if all(col in f for f in frames):
            seen = {c for f in frames for c in f[col].cat.categories}
            cats = sorted(seen, key=str)
            shared[col] = pd.CategoricalDtype(cats)
    frames = [f.astype(shared) for f in frames]
    return pd.concat(frames, ignore_index=True)


def _merge(path: Path, frames: Iterable[pd.DataFrame]) -> None:
    """Merge stored rows into a results file by model."""
    rows = [f for f in frames if not f.empty]
    if rows:
        table = pd.concat(rows, ignore_index=True)
        write_csv(merge_by_model(path, table), path)


def command(args: argparse.Namespace) -> int:
    """Run `bdbvhub fit-store plan|fit|load|unpack`.

    `plan` prints the fits to make and writes them as a GitHub Actions
    matrix (`--out`), with every current key (`--keys`) for pruning;
    `--origins` keeps only those origins.
    `fit` fits one model at one origin and writes it to `--store`.
    `load` writes the stored fits into `--results`; with
    `--current-only`, only the current keys' fits.
    `unpack` writes the stored fit of `--model` at `--origin` into the
    fit directory `--out`, for the submission.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed `fit-store` arguments.

    Returns
    -------
    int
        0 on success, 1 if the fit failed or the arguments are wrong.
        `unpack` returns 3 if the store has no fit at the current key
        and 4 if that fit failed.
    """
    sampler = samplers(args.policy)
    if args.action == "fit":
        from nowcastutils.devices import host_devices

        # Before anything starts JAX, so the chains run in parallel.
        chains = sampler.get(args.model, {}).get("num_chains", 1)
        host_devices(int(chains))
    config = load_config(args.config)
    tasks = origin_tasks(
        load_target(args.data),
        load_delays(args.data),
        config,
        load_daily(args.data),
        load_zone_data(args.data),
    )
    models = args.models or stored_models()
    one = args.action in ("fit", "unpack")
    names = [args.model] if one else models
    unknown = sorted(set(names) - set(stored_models()))
    if unknown:
        print(
            f"ERROR: {unknown} not in the stored models {stored_models()}.",
            file=sys.stderr,
        )
        return 1
    code = {m: code_digest(m) for m in names}
    if args.action == "fit":
        if args.origin not in tasks:
            print(
                f"ERROR: fit needs --origin in {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        stored = fit(args.model, tasks[args.origin], sampler, code[args.model])
        path = write_fit(args.store, stored)
        print(f"Wrote {path} ({stored.meta['status']}).")
        if stored.meta["status"] != "ok":
            print(f"ERROR: {stored.meta['error']}", file=sys.stderr)
            return 1
        return 0
    if args.action == "unpack":
        if args.origin not in tasks or not args.out:
            print(
                f"ERROR: unpack needs --out and --origin in {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        keys = current_keys(
            {args.origin: tasks[args.origin]}, names, sampler, code
        )
        state, found = unpack(
            args.store, keys, args.model, args.origin, args.out
        )
        key = keys[(args.model, args.origin)]
        used = "" if found is None else found.meta["key"]
        older = f", older fit {used} used" if used not in ("", key) else ""
        print(f"{args.model} at {args.origin} ({key}): {state}{older}.")
        if state == "failed" and found is not None:
            print(f"ERROR: {found.meta.get('error')}", file=sys.stderr)
        return {"ok": 0, "missing": 3, "failed": 4}[state]
    if args.origins:
        unknown = sorted(set(args.origins) - set(tasks))
        if unknown:
            print(
                f"ERROR: {', '.join(unknown)} not among the origins the "
                f"store fits, {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        tasks = {o: t for o, t in tasks.items() if o in set(args.origins)}
    keys = current_keys(tasks, models, sampler, code)
    if args.action == "plan":
        to_fit = plan(keys, args.store, args.retry_failed, args.usable)
        for t in to_fit:
            print(f"To fit: {t['model']} at {t['origin']} ({t['key']}).")
        print(f"{len(to_fit)} of {len(keys)} fits to make.")
        if args.out:
            Path(args.out).write_text(json.dumps({"include": to_fit}))
        if args.keys:
            Path(args.keys).write_text(
                json.dumps([f"{m}/{o}/{k}" for (m, o), k in keys.items()])
            )
        return 0
    for s in load(
        args.store,
        keys,
        args.results,
        config.min_horizon,
        config.max_horizon,
        current_only=args.current_only,
    ):
        if s.current:
            state = "current"
        elif s.failed:
            state = "the fit for this code failed, no forecast"
        else:
            state = "pending, not fitted for this code"
            if s.used:
                state += f", older fit {s.used} used"
        print(f"{s.model} at {s.origin}: {state}.")
    return 0
