# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Blocks and batches for fine-tuning Chronos-2.

Cuts outbreaks from the simulator (nowcastfoundation.simulator) into
training blocks
(`outbreak_blocks`) and packs them into batches (`plan_batches`),
without torch.
`nowcastfoundation.finetune.stream` trains on them and
publishes each checkpoint as the release `chronos-ft-<hash>`, which the
`_ft` models load the newest one
(`nowcastfoundation.forecaster.resolve_checkpoint`).
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nowcastfoundation import forecaster as foundation
from nowcastfoundation.finetune import (
    BINNED,
    MIXED,
    ROLLING,
    draw_named,
    release_days,
    release_weekday,
    training_block,
    weekly_cutoffs,
)
from nowcastutils.base import NATIONAL

CONTEXT = 364
"""Steps of context in a block, left-padded with missing values."""


LAG_RULE: Final = (
    "one lag count a block: the largest of its places' truncation-aware "
    "99% delays from the steps complete at the cut-off, at most the "
    "history before it (nowcastfoundation.forecaster.lag_count, as "
    "forecasts use)"
)
"""How a block's release lags are set (`training_block`)."""


DAYS_PER_STEP = {
    "weekly": 7,
    "daily": 1,
    "3day": 3,
    "daily+weekly": 1,
    "daily+rolling7": 1,
    "daily+3day": 1,
    "daily+rolling3": 1,
}
"""Days in a step of each source."""


NATIVE_CADENCE = {1: "daily", 3: "3day", 7: "weekly"}
"""The regular release cadence of each step width, in days."""


BLOCK_DESIGN: dict[str, Any] = {
    "source_shares": {
        "daily": 0.25,
        "weekly": 0.25,
        "3day": 0.15,
        "daily+weekly": 0.125,
        "daily+rolling7": 0.125,
        "daily+3day": 0.05,
        "daily+rolling3": 0.05,
    },
    "cadences": {
        "daily": {
            "daily": 0.6,
            "weekly": 0.2,
            "daily_gaps": 0.1,
            "weekdays": 0.1,
        },
        "3day": {"3day": 0.8, "3day_gaps": 0.2},
        "weekly": {"weekly": 0.8, "weekly_gaps": 0.2},
    },
    "whole_system": True,
}
"""The block design: each layout's share of blocks (`source_shares`), the
release cadences of each step width with their shares (`cadences`,
`finetune.release_days`), and whole-system blocks (`whole_system`).

Releases follow the layout's steps, daily, every 3 days or weekly.
Daily layouts also get weekly releases (the BDBV hub's daily data), daily
releases on weekdays only, and missed releases (`_gaps`); 3-day and weekly
layouts get missed releases. Daily blocks with weekly or 3-day totals have
them either on period-end days only (`finetune.MIXED`) or as rolling sums
on every day (`finetune.ROLLING`). Overall, 36% of blocks have daily
releases, 32% weekly (12% daily blocks, 20% weekly blocks), 12% 3-day,
6% daily with gaps, 6% weekdays only, 5% weekly with gaps and 3% 3-day
with gaps. Each block is a whole simulated system (`whole_system`), or
its country alone.
"""

ROOT = Path(__file__).resolve().parents[5]
"""The repository root, which the hashed paths are relative to: this
module is `packages/nowcastfoundation/src/nowcastfoundation/finetune/`
under it in the workspace's editable install."""

GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"
"""The grid file the simulations are drawn from (`use_grid`)."""

SOURCE_SHARES = dict(BLOCK_DESIGN["source_shares"])
"""Share of blocks of each layout (`BLOCK_DESIGN`)."""

CADENCES = {k: dict(v) for k, v in BLOCK_DESIGN["cadences"].items()}
"""Release cadences of each step width and their shares (`BLOCK_DESIGN`)."""

WHOLE_SYSTEM = bool(BLOCK_DESIGN["whole_system"])
"""Whether each block is a whole simulated system (`whole_system`), rather
than a drawn share of its places (`choose_places`)."""

SOURCES = tuple(SOURCE_SHARES)
"""Sources blocks are cut from, drawn at `SOURCE_SHARES`."""


def use_grid(grid_file: Path | str = GRID_FILE) -> None:
    """Set the parents from the grid file (`grid_parents`).

    Raises
    ------
    FileNotFoundError
        If the grid file is missing, so a wrong path is not silently
        trained with the even parent mix.
    """
    global PARENTS, SIMULATOR, CODE
    path = Path(grid_file)
    if not path.is_file():
        raise FileNotFoundError(f"no grid file at {path}")
    PARENTS = grid_parents(path)
    # Named relative to the repository, so the hash does not depend on
    # where the checkout is.
    name = os.path.relpath(path.resolve(), ROOT.resolve())
    SIMULATOR = simulator_files(Path(name).as_posix())
    CODE = code_files(SIMULATOR)


def native_cadence(source: str) -> str:
    """Return the regular cadence of a source's steps (`NATIVE_CADENCE`)."""
    return NATIVE_CADENCE[DAYS_PER_STEP[source]]


def draw_cadence(source: str, rng: np.random.Generator) -> str:
    """Draw a block's release cadence at its source's shares.

    The shares are the block design's (`CADENCES`), or the release
    layer's where it sets them (`RELEASE_LAYER`).
    """
    width = native_cadence(source)
    mix = (RELEASE_LAYER or {}).get("cadences", {}).get(width, CADENCES[width])
    return str(rng.choice(list(mix), p=list(mix.values())))


RELEASE_LAYER: dict[str, Any] | None = None
"""The release layer (`finetune.release_layer`, `check_release_layer`),
drawn per block at fine-tune time from the simulations already made: a
late start to a share of blocks' releases (`release_start`), a share of
blocks with no lag series (`complete_block`) and the release cadences of
each step width (`draw_cadence`). None leaves each
block's releases as the block design draws them."""

NAMED_PARAMETERS: Final = {
    "uniform": ("lower", "upper"),
    "beta": ("a", "b"),
    "lognormal": ("median", "sdlog"),
}
"""The parameters each named distribution needs (`finetune.draw_named`)."""


def read_release_layer(path: Path | str) -> dict[str, Any]:
    """Return the release layer in a YAML file, checked."""
    import yaml

    layer = yaml.safe_load(Path(path).read_text()) or {}
    check_release_layer(layer)
    return layer


def check_release_layer(layer: dict[str, Any]) -> None:
    """Raise unless every part of the release layer can be drawn.

    `start` needs a `share` from 0 to 1 and a named `delay_days`
    distribution; `no_lags` a `share` from 0 to 1. Each step width of
    `cadences` (`NATIVE_CADENCE`) needs
    shares of that width's cadences (`CADENCES`) summing to one.

    Raises
    ------
    ValueError
        Naming the part that cannot be drawn.
    """
    extra = set(layer) - {"start", "cadences", "no_lags"}
    if extra:
        raise ValueError(f"release layer: no part {sorted(extra)}")
    if "start" in layer:
        start = layer["start"]
        share = start.get("share")
        if not isinstance(share, (int, float)) or not 0 <= share <= 1:
            raise ValueError(f"release layer: start share {share!r}")
        spec = start.get("delay_days", {})
        needs = NAMED_PARAMETERS.get(spec.get("dist"))
        if needs is None or any(k not in spec for k in needs):
            raise ValueError(f"release layer: start delay_days {spec!r}")
    if "no_lags" in layer:
        share = layer["no_lags"].get("share")
        if not isinstance(share, (int, float)) or not 0 <= share <= 1:
            raise ValueError(f"release layer: no_lags share {share!r}")
    for width, mix in layer.get("cadences", {}).items():
        if width not in CADENCES:
            raise ValueError(f"release layer: no step width {width!r}")
        unknown = set(mix) - set(CADENCES[width])
        if unknown:
            raise ValueError(
                f"release layer: {width} has no cadence {sorted(unknown)}"
            )
        shares = list(mix.values())
        if min(shares) < 0 or abs(sum(shares) - 1) > 1e-9:
            raise ValueError(f"release layer: {width} shares {mix}")


def release_start(
    days: list[str],
    first: str,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[str], int]:
    """Return a block's releases from a drawn start day, and its delay.

    With chance `spec["share"]`, a delay in whole days is drawn from
    `spec["delay_days"]` (`finetune.draw_named`), and every release
    before `first` plus the delay is dropped; the cut's is always kept.
    The first kept release then holds every onset reported before it, as
    the BDBV hub's first release held onsets from months before. A block
    whose delay reaches its cut has the cut's release alone. Otherwise
    the releases are as given and the delay is 0.
    """
    if rng.random() >= spec["share"]:
        return list(days), 0
    delay = max(0, int(draw_named(spec["delay_days"], rng)))
    start = dt.date.fromisoformat(first[:10]) + dt.timedelta(days=delay)
    kept = [d for d in days[:-1] if dt.date.fromisoformat(d) >= start]
    return kept + list(days[-1:]), delay


def check_sources(sources: tuple[str, ...]) -> None:
    """Raise unless every source has its own triangle or is binned.

    A source neither in `foundation.SOURCES` nor in `BINNED` would be cut
    as daily without notice.
    """
    bad = [
        s
        for s in sources
        if (
            s not in foundation.SOURCES
            and s not in BINNED
            and s not in MIXED
            and s not in ROLLING
        )
        or s not in DAYS_PER_STEP
    ]
    if bad:
        raise ValueError(f"sources {bad} are neither native nor binned")


def check_package(root: Path) -> None:
    """Raise unless `nowcastfoundation` is imported from `root`'s packages.

    A venv linked from another worktree imports that worktree's package,
    whose blocks may differ from this one's.
    """
    import nowcastfoundation

    where = Path(nowcastfoundation.__file__).resolve()
    if not where.is_relative_to((Path(root) / "packages").resolve()):
        raise RuntimeError(
            f"nowcastfoundation is imported from {where}, not this worktree "
            f"({Path(root).resolve()})"
        )


MAX_SERIES = 2000
"""Most series in a block of sampled places, whose series count is drawn
log-uniformly up to it, and in a batch of several blocks."""

CUTS = 32
"""Snapshots (cut-off slots) of each observation draw, spread over its
course: 32 blocks from one draw."""

SIMULATOR_PY = (
    "packages/nowcastfoundation/src/nowcastfoundation/simulator",
    "packages/nowcastrenewal/src/nowcastrenewal/latent",
    "packages/nowcastrenewal/src/nowcastrenewal/components",
    "packages/nowcastrenewal/src/nowcastrenewal/reporting",
    "packages/nowcastrenewal/src/nowcastrenewal/simulation",
)
"""The simulator's Python: its entry point and draws, the latent step and
the shared components it composes."""


def simulator_files(grid_file: str) -> tuple[str, ...]:
    """Return the simulator's files: the grid and its Python.

    The Python is every file of `SIMULATOR_PY`, found by glob so a new file
    is hashed without listing it.
    """
    python = sorted(
        str(p.relative_to(ROOT))
        for folder in SIMULATOR_PY
        for p in (ROOT / folder).glob("*.py")
    )
    return (grid_file, *python)


def workflow_finetune(key: str) -> Any:
    """Return the workflow's fine-tuning setting `key`.

    `finetune.<key>` in workflow/config.yaml: the trainer's default.
    """
    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    return config["finetune"][key]


DEFAULT_GRID_FILE = "config/finetune/grid.yaml"
"""The grid file when there is no workflow config: the package alone."""


def workflow_grid_file() -> str:
    """Return the workflow's grid file (`simulate.grid_file`).

    `DEFAULT_GRID_FILE` outside a checkout, where the package is
    installed alone and has no workflow config to read.
    """
    import yaml

    path = ROOT / "workflow" / "config.yaml"
    if not path.is_file():
        return DEFAULT_GRID_FILE
    config = yaml.safe_load(path.read_text())
    return str(config["simulate"]["grid_file"])


def code_files(simulator: tuple[str, ...]) -> dict[str, str]:
    """Return `CODE` for a simulator's files, each named by its path."""
    files = (
        *simulator,
        "packages/nowcastfoundation/src/nowcastfoundation/finetune/chronos.py",
        "packages/nowcastfoundation/src/nowcastfoundation/finetune/__init__.py",
    )
    return {path: path for path in files}


SIMULATOR = simulator_files(workflow_grid_file())
"""The simulator's files, with the workflow's grid until a run sets its
own (`use_grid`)."""

CODE = code_files(SIMULATOR)
"""Files the checkpoint depends on, hashed into its settings: each
file's name in `code_sha256`, then its path. The paths also go into the
settings outside the hash (`code_paths`), for `finetune.verify`."""

TRAINER = "packages/nowcastfoundation/src/nowcastfoundation/finetune/stream.py"
"""The trainer's name in `code_sha256` (its path is its own file)."""

FOUNDATION_PATH = (
    "packages/nowcastfoundation/src/nowcastfoundation/forecaster.py"
)
"""Where the module of `FOUNDATION_CODE` is now (`code_paths`
`foundation`)."""

FOUNDATION_CODE = (foundation.hub_triangle, foundation.layout_block)
"""The functions of `foundation` the blocks are laid out with, hashed into
the settings."""

PREFIXES = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
"""Outbreaks seen at each saved checkpoint. The last is every training
outbreak."""

EVAL_BLOCKS = 100
"""Held-out blocks whose loss is logged during training."""

EVAL_EVERY = 100
"""Steps between logged held-out losses."""

OUTPUT_PATCH = 16
"""Chronos-2's output patch length."""

LOSS_HORIZON_DAYS: int | None = None
"""Days after the cut-off that a training block's horizon, and so its
loss, runs to; None for the whole horizon, to `AHEAD_WEEKS` past the
cut-off's week."""

SIM_DELAY_WEEKS = 8
"""The simulations' longest delay in weeks, after which a count is
complete, unless an outbreak's settings give `max_delay` in days."""


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256(path: str) -> str:
    """Return the SHA-256 of a hashed code file, its path from `ROOT`."""
    return _sha256(ROOT / path)


QUIET_WINDOW_DAYS = 28
"""Days up to a cut-off over which a block's reported cases are counted
for the quiet rule (`quiet`)."""

QUIET_MIN_CASES: int | None = None
"""Cases a block needs reported across its places in the
`QUIET_WINDOW_DAYS` up to its cut-off to be trained on (`quiet`); None
keeps every block."""


def reported_cases(
    block: dict[str, Any], days_per_step: int, days: int
) -> float:
    """Return a block's cases reported in `days` up to its cut-off.

    Only what is known at the cut-off counts, never the counts after it.
    Each step counts once, at its latest report: before `start`, the
    targets' counts; from `start` to the cut-off, each lag series'
    value at the step's lag from the cut-off (lag series hold a step's
    count as reported at that lag, so summing them would count a case
    once per lag). The places are summed within each level (national,
    intermediate, leaf) and the busiest level counts, so a total does
    not count its places' cases again, and a block of some places with
    their country counts the country's cases. `block` is
    `finetune.training_block`'s, before padding.
    """
    x, places = block["context"], block["places"]
    t, n = block["n_targets"], len(places)
    lags, extra = divmod(block["n_covariates"], n)
    if extra:
        raise ValueError(
            f"{block['n_covariates']} lag series over {n} places: not the "
            "same number a place"
        )
    if block["nowcast"] > lags:
        raise ValueError(
            f"nowcast of {block['nowcast']} steps past {lags} lags: it would "
            "read the next place's rows"
        )
    start = block["start"]
    cut = start + block["nowcast"]
    lo = max(cut - -(-days // days_per_step), 0)
    count = np.nansum(x[:n, lo:start], axis=1)
    rows = np.arange(n)
    for step in range(max(lo, start), cut):
        count += np.nan_to_num(x[t + rows * lags + (cut - 1 - step), step])
    levels = np.array(target_levels(places))
    return float(max(count[levels == lv].sum() for lv in set(levels)))


def quiet(block: dict[str, Any], days_per_step: int, min_cases: int) -> bool:
    """Return whether a block has fewer than `min_cases` reported cases.

    The cases are those reported across its places in the
    `QUIET_WINDOW_DAYS` up to its cut-off (`reported_cases`), so the
    rule depends only on what is known at the cut-off.
    """
    return reported_cases(block, days_per_step, QUIET_WINDOW_DAYS) < min_cases


LAYOUT = "ragged"
"""The layout of the training blocks (`nowcastfoundation.forecaster.LAYOUTS`):
`ragged`, the counts to the last complete step and each place's lag
series; or `complete`, the same blocks without their lag series
(`complete_block`), so the context ends at the last complete step."""


def complete_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return a block without its lag series: the complete layout.

    The targets keep their counts to the last complete step (`start`) and
    the truth after it; the nowcast and forecast steps are as before.
    `block` is `finetune.training_block`'s, before padding.
    """
    targets = block["context"][: block["n_targets"]]
    return block | {
        "context": targets,
        "future_covariates": targets[:, block["start"] :].copy(),
        "n_covariates": 0,
        "n_future_covariates": 0,
    }


def _pad(
    block: dict[str, Any],
    context: int,
    loss_days: int | None = None,
    days_per_step: int = 1,
) -> dict[str, Any]:
    """Pad a block to `context` steps and its horizon in whole patches.

    Chronos-2 predicts in patches of `OUTPUT_PATCH` steps, so a block's
    prediction length is its horizon, its nowcast and forecast steps,
    rounded up to whole patches; the steps past the horizon are missing.
    With `loss_days`, the horizon stops at the nowcast steps and the
    steps covering `loss_days` days after them, of `days_per_step` days
    each. Long context is cropped. The arrays are numpy, so blocks are
    cut without torch; the trainer makes them tensors
    (`finetune.stream.tensors`).
    """
    start, h = block["start"], block["horizon"]
    if loss_days is not None:
        h = min(h, block["nowcast"] + -(-loss_days // days_per_step))
    length = -(-h // OUTPUT_PATCH) * OUTPUT_PATCH
    x = block["context"]
    keep = min(start, context)
    out = np.full((x.shape[0], context + length), np.nan, dtype=np.float32)
    out[:, context - keep : context + h] = x[:, start - keep : start + h]
    return {
        "context": out,
        "future_covariates": out[:, context:],
        "n_targets": block["n_targets"],
        "n_covariates": block["n_covariates"],
        "n_future_covariates": block["n_future_covariates"],
        "prediction_length": length,
        "nowcast": block["nowcast"],
    }


def strata(settings: pd.Series) -> list[str]:
    """Return an outbreak's strata, with or without cases."""
    places = [f"P{i:02d}" for i in range(1, int(settings["n_places"]) + 1)]
    n_ages = settings.get("n_ages", 1)
    n_ages = 1 if pd.isna(n_ages) else int(n_ages)
    ages = [f"A{i}" for i in range(1, n_ages + 1)]
    if len(places) > 1 and len(ages) > 1:
        return [f"{p}-{a}" for p in places for a in ages]
    return ages if len(ages) > 1 else places


LEVEL_MIX = ("all_levels", "leaves_national", "leaves", "national", "single")
"""A block's levels: the country, intermediate totals and leaves (as the
deployed block: national, provinces, zones), leaves and the country,
leaves only, the country only, or one leaf only. The last two are
univariate series from the same outbreaks. All levels need nested strata
(regions of places, or places by ages)."""

PARENT_MIX = ("exact", "unallocated", "late_parent")
"""How the country of a block with places relates to them, drawn evenly:
their sum, more than their sum (cases with no place), or reported through
a later stream (`finetune.country_counts`). A block of some of the
places is also not their sum. Parent means belongs to, not sums to."""

PARENTS: dict[str, Any] | None = None
"""The grid's parents, from its file's `aggregation` (`grid_parents`), or
None before `use_grid` or for a grid with no `aggregation`, whose parents
are `PARENT_MIX` drawn evenly with `finetune.PARENT_DRAWS`."""


def grid_parents(path: Path) -> dict[str, Any] | None:
    """Return the grid file's parents, or None if it sets none.

    The share of blocks with places whose country is their exact sum,
    has unallocated cases, or is reported later (`aggregation.parents`,
    weights), and the named distributions of the unallocated share and
    the lag (`finetune.country_counts`).
    """
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    aggregation = config.get("aggregation")
    if aggregation is None:
        return None
    weights = aggregation["parents"]
    total = sum(weights.values())
    dists = config["distributions"]
    return {
        "shares": {k: w / total for k, w in weights.items()},
        "unallocated_share": dists[aggregation["unallocated_share"]],
        "parent_lag_days": dists[aggregation["parent_lag_days"]],
    }


UNIVARIATE_SHARE = 0.2
"""Share of blocks that are one series, the country or one leaf; the rest
are multivariate, shared evenly among their mixes."""

REGIONS = 1.0
"""Chance an outbreak of four or more flat places is grouped into 2 to 6
regions for its blocks, so blocks can nest places in regions in the
country."""


def regions(places: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Return each place's id in a seeded grouping into 2 to 6 regions."""
    k = int(rng.integers(2, min(6, len(places) // 2) + 1))
    order = rng.permutation(len(places))
    group = {places[i]: 1 + j % k for j, i in enumerate(order)}
    return {p: f"R{group[p]}-{p}" for p in places}


def target_levels(places: tuple[str, ...]) -> tuple[str, ...]:
    """Return the level of each place of a block.

    The levels are national, intermediate (a total of strata) or leaf.
    """
    nested = any("-" in p for p in places)
    return tuple(
        "national"
        if p == NATIONAL
        else "leaf"
        if "-" in p or not nested
        else "intermediate"
        for p in places
    )


def whole_system(
    provinces: list[str], rng: np.random.Generator
) -> tuple[tuple[str, ...], str]:
    """Return a block of a whole simulated system and its level mix.

    Every stratum with every total it is part of: the country, each
    group's total (a place's over its ages, or a region's over its
    places), each age group's total across places, and the leaves; flat
    places are the country and its places. At `UNIVARIATE_SHARE`, and for
    a system of one stratum, the country alone. Strata are never cut out
    of the system they were simulated with.
    """
    if len(provinces) < 2 or rng.random() < UNIVARIATE_SHARE:
        return (NATIONAL,), "national"
    if not any("-" in p for p in provinces):
        return (NATIONAL, *sorted(provinces)), "leaves_national"
    groups = sorted({p.split("-")[0] for p in provinces})
    ages = sorted(
        {
            parts[1]
            for parts in (p.split("-") for p in provinces)
            if len(parts) > 1 and re.fullmatch(r"A\d+", parts[1])
        }
    )
    return (NATIONAL, *groups, *ages, *sorted(provinces)), "all_levels"


BLOCK_LIMIT: int | None = None
"""Most series in a block (`finetune.block_limit`), the training machine's
block capacity, or None for no limit. A whole system over it is cut into
views (`system_views`)."""

VIEW_RULE: Final = (
    "a whole system over the block limit becomes views that each fit: its "
    "top levels aggregated up (strata into their places or regions, then "
    "those into the country) and each region or place with its leaves and "
    "its own total, as many leaves a block as fit"
)
"""The rule `system_views` follows, recorded in the settings."""


def system_views(
    places: tuple[str, ...], cap: int
) -> list[tuple[str, tuple[str, ...], list[str]]]:
    """Return the views of a whole system that hold at most `cap` places.

    Each view is its kind, its places and the levels collapsed in it. A
    system within `cap` is one view, `whole`. Over it, `top` is the
    country with its places or regions, the strata summed into them, or
    the country alone if those do not fit; and each `subtree` is a place
    or region with its leaves and its own total (the country, for flat
    places), as many leaves as fit with it, so every leaf is in one.
    Totals are sums of what the system simulated, never estimates.
    """
    places = tuple(places)
    if len(places) <= cap:
        return [("whole", places, [])]
    nested = any("-" in p for p in places)
    if nested:
        leaves = [p for p in places if "-" in p]
        groups = sorted({p.split("-")[0] for p in leaves})
        top: tuple[str, ...] = (NATIONAL, *groups)
        collapsed = ["strata"]
        if len(top) > cap:
            top, collapsed = (NATIONAL,), ["strata", "places"]
        trees = [
            (g, [p for p in leaves if p.split("-")[0] == g]) for g in groups
        ]
    else:
        top, collapsed = (NATIONAL,), ["places"]
        trees = [(NATIONAL, [p for p in places if p != NATIONAL])]
    out = [("top", top, collapsed)]
    for total, members in trees:
        if cap < 2:
            out += [("subtree", (m,), []) for m in members]
            continue
        step = cap - 1
        out += [
            ("subtree", (total, *members[i : i + step]), [])
            for i in range(0, len(members), step)
        ]
    return out


def choose_places(
    provinces: list[str], budget: int, rng: np.random.Generator
) -> tuple[tuple[str, ...], str]:
    """Return a block's places and its level mix (`LEVEL_MIX`).

    At most `budget` places are below the country. All levels are the
    country, the totals of the chosen leaves' groups (and, for places by
    ages, the age groups' totals) and the leaves.
    """
    nested = any("-" in p for p in provinces)
    if rng.random() < UNIVARIATE_SHARE:
        mix = str(rng.choice(LEVEL_MIX[3:]))
    else:
        mix = str(rng.choice(LEVEL_MIX[:3] if nested else LEVEL_MIX[1:3]))
    # A block's size gives way to its level mix: leaves need at least one
    # place below the country, all levels a group's total and a leaf.
    budget = max(budget, 2 if mix == "all_levels" else 1)
    if mix == "national":
        return (NATIONAL,), mix
    if mix == "single":
        return (str(rng.choice(provinces)),), mix
    if mix in ("leaves_national", "leaves"):
        k = min(budget, len(provinces))
        picked = tuple(str(p) for p in sorted(rng.choice(provinces, k, False)))
        return ((NATIONAL,) if mix == "leaves_national" else ()) + picked, mix
    groups: dict[str, list[str]] = {}
    for p in provinces:
        groups.setdefault(p.split("-")[0], []).append(p)
    ages = sorted(
        {
            parts[1]
            for parts in (p.split("-") for p in provinces)
            if len(parts) > 1 and re.fullmatch(r"A\d+", parts[1])
        }
    )
    chosen: list[str] = []
    leaves: list[str] = []
    for parent in rng.permutation(sorted(groups)):
        room = budget - len(chosen) - len(leaves) - 1
        if room < 1:
            break
        chosen.append(str(parent))
        leaves.extend(sorted(groups[parent])[:room])
        if room < len(groups[parent]):
            break
    if not chosen:
        # Too small a budget for a group: one group's total and a leaf.
        parent = sorted(groups)[0]
        chosen, leaves = [parent], [sorted(groups[parent])[0]]
    totals = sorted(chosen)
    if ages and rng.random() < 0.5:
        room = budget - len(chosen) - len(leaves)
        totals += ages[: max(room, 0)]
    return (NATIONAL, *totals, *sorted(leaves)), mix


AGGREGATE_END = "a random weekday, or 3-day offset, per block"
"""The day a block's period totals end on (`finetune.period_totals`):
any, as the German target is a 7-day sum ending on every day."""


def block_options(source: str, rng: np.random.Generator) -> dict[str, Any]:
    """Return a block's period end, drawn from `rng`.

    For a source with period totals, a random end day. Reports come in
    weekly batches only from the grid's weekday reporting.
    """
    end = int(rng.integers(7 if MIXED.get(source) == "weekly" else 3))
    return {"period_end": end if source in MIXED else None}


def outbreak_blocks(
    counts: pd.DataFrame,
    provinces: list[str],
    rng: np.random.Generator,
    delay_weeks: int = SIM_DELAY_WEEKS,
    cuts: int = CUTS,
    picks: set[tuple[str, int]] | None = None,
    empty: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the training blocks of one outbreak.

    Each source is cut at `cuts` forecast origins spread over the
    outbreak's course, one in each of `cuts` equal stretches of its
    cut-offs, so they fall in its rising, peak and falling phases. With
    `picks`, only those (source, slot) blocks are cut. A block's series
    count is drawn log-uniformly from 1 to `MAX_SERIES`,
    and its places (`choose_places`) are about that many series at its
    lag, with the country first. Its release lags follow `LAG_RULE`, the
    rule forecasts use. A block whose places have no release
    rows by its cut (strata with no cases) is skipped and, with `empty`,
    described there. Each block's releases run up to its cut at a cadence
    drawn for its source (`draw_cadence`), recorded as its `cadence`.
    """
    weekday = release_weekday(rng)
    first = str(counts["report_date"].min())
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=delay_weeks + 5)
    cutoffs = weekly_cutoffs(first, last.isoformat(), weekday)
    # Flat places may nest in regions, each a total of its places.
    # Only places nest in regions; age groups total across places.
    flat = len(provinces) >= 4 and all(
        "-" not in p and p.startswith("P") for p in provinces
    )
    if flat and rng.random() < REGIONS:
        nest = regions(provinces, rng)
        counts = counts.assign(location=counts["location"].map(nest))
        provinces = sorted(nest.values())
    out = []
    for source in SOURCES:
        for j in range(cuts):
            if picks is not None and (source, j) not in picks:
                continue
            width = DAYS_PER_STEP[source]
            # At least two steps of context before one lag.
            least = -(-3 * width // 7)
            if len(cutoffs) <= least:
                continue
            share = (j + rng.random()) / cuts
            cut = least + int(share * (len(cutoffs) - least))
            # The most lags the history before the cut-off allows, with
            # two steps of context: the block's lags are at most this.
            history = (
                dt.date.fromisoformat(cutoffs[cut])
                - dt.date.fromisoformat(first[:10])
            ).days // width
            if history - 2 < 1:
                continue
            places: tuple[str, ...] = (NATIONAL,)
            mix = "national"
            # A place's rows at most: its counts, its lags and, with period
            # totals, one more.
            rows = history - 2 + 1 + (source in MIXED or source in ROLLING)
            if WHOLE_SYSTEM:
                # The whole system, however many series: a block is never
                # skipped for its size.
                places, mix = whole_system(provinces, rng)
            elif len(provinces) > 1:
                size = np.exp(rng.uniform(0, np.log(MAX_SERIES)))
                fits = MAX_SERIES // rows - 1
                budget = min(round(size / rows) - 1, fits)
                places, mix = choose_places(provinces, budget, rng)
            # How the country relates to its places, drawn from a child
            # generator, so the other draws are as without it.
            parent_rng, option_rng = rng.spawn(2)
            parent = "exact"
            parents = PARENTS
            if len(places) > 1 and places[0] == NATIONAL:
                if parents is None:
                    parent = str(parent_rng.choice(PARENT_MIX))
                else:
                    kinds = list(parents["shares"])
                    shares = list(parents["shares"].values())
                    parent = str(parent_rng.choice(kinds, p=shares))
            options = block_options(source, option_rng)
            # The releases follow the source's cadence (`CADENCES`), up to
            # the slot's cut.
            cadence = draw_cadence(source, option_rng)
            days = release_days(first, cutoffs[cut], cadence, option_rng)
            # A late start to the releases (`RELEASE_LAYER`), from a child
            # generator, so the other draws are as without it.
            start_days, no_lags = 0, False
            if RELEASE_LAYER is not None:
                layer_rng = option_rng.spawn(1)[0]
                if "start" in RELEASE_LAYER:
                    days, start_days = release_start(
                        days, first, RELEASE_LAYER["start"], layer_rng
                    )
                # Counts alone, as when only the latest data are at hand.
                if "no_lags" in RELEASE_LAYER:
                    share = RELEASE_LAYER["no_lags"]["share"]
                    no_lags = bool(layer_rng.random() < share)
            totals = any("-" in p for p in provinces)
            # A view with the country draws its cases as the whole did.
            country_rng = copy.deepcopy(parent_rng)
            block = training_block(
                counts,
                source,
                places,
                days,
                totals=totals,
                parent=parent,
                rng=parent_rng,
                parent_draws=parents,
                **options,
            )
            if block is None:
                if empty is not None:
                    empty.append(f"{source} cut {j}: {', '.join(places)}")
                continue
            if block["start"] < 2:
                continue
            if QUIET_MIN_CASES is not None and quiet(
                block, DAYS_PER_STEP[source], QUIET_MIN_CASES
            ):
                continue
            whole: dict[str, Any] = {"kind": "whole"}
            views: list[
                tuple[dict[str, Any], tuple[str, ...], str, str, dict[str, Any]]
            ] = [(block, places, mix, parent, whole)]
            series = int(block["context"].shape[0])
            if BLOCK_LIMIT is not None and series > BLOCK_LIMIT:
                views = []
                cap = max(1, int(BLOCK_LIMIT * len(places) // series))
                while True:
                    views = []
                    for kind, view, collapsed in system_views(places, cap):
                        has_country = len(view) > 1 and view[0] == NATIONAL
                        how = parent if has_country else "exact"
                        part = training_block(
                            counts,
                            source,
                            view,
                            days,
                            totals=totals,
                            parent=how,
                            rng=copy.deepcopy(country_rng),
                            parent_draws=parents,
                            **options,
                        )
                        if part is None or part["start"] < 2:
                            if empty is not None:
                                empty.append(
                                    f"{source} cut {j} {kind}: "
                                    f"{', '.join(view)}"
                                )
                            continue
                        n = int(part["context"].shape[0])
                        record: dict[str, Any] = {
                            "kind": kind,
                            "collapsed": collapsed,
                            "places_before": len(places),
                            "places_after": len(view),
                            "series_before": series,
                            "series_after": n,
                        }
                        view_mix = (
                            "national"
                            if view == (NATIONAL,)
                            else "leaves_national"
                            if kind == "top"
                            else "subtree"
                        )
                        views.append((part, view, view_mix, how, record))
                    # A view's lags can differ from the whole's: one still
                    # over the limit takes fewer places.
                    fits = all(
                        v[0]["context"].shape[0] <= BLOCK_LIMIT for v in views
                    )
                    if fits or cap == 1:
                        break
                    cap -= 1
            for part, view, view_mix, how, record in views:
                if LAYOUT == "complete" or no_lags:
                    part = complete_block(part)
                padded = _pad(
                    part, CONTEXT, LOSS_HORIZON_DAYS, DAYS_PER_STEP[source]
                )
                levels = target_levels(view)
                padded["levels"] = tuple(
                    levels[i] for i in part["target_places"]
                )
                padded["places"] = tuple(view)
                padded["target_places"] = list(part["target_places"])
                padded["parent"] = how
                padded |= options
                padded["mix"] = view_mix
                padded["source"] = source
                padded["cadence"] = cadence
                padded["release_start_days"] = start_days
                padded["no_lags"] = no_lags
                padded["first_release"] = days[0]
                padded["releases"] = len(days)
                padded["cut_date"] = days[-1]
                padded["cut"] = j
                padded["view"] = record
                out.append(padded)
    return out


def size_class(n: int) -> int:
    """Return a block's size class, by powers of two of its series."""
    return int(np.floor(np.log2(n)))


def plan_batches(
    sizes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    most: int | None = None,
) -> list[list[int]]:
    """Return batches of block indices that use every block once.

    Blocks are shuffled and packed in that order into batches of one size
    class until a batch holds `batch_size` series, so batches of small
    blocks pack tightly and a large block is a batch on its own. No batch
    of several blocks holds more than `most` series (the trainer passes
    `MAX_SERIES`). The batches are then shuffled.
    """
    filling: dict[int, list[int]] = {}
    held: dict[int, int] = {}
    batches = []
    for i in rng.permutation(len(sizes)):
        c = size_class(int(sizes[i]))
        if (
            most is not None
            and filling.get(c)
            and (held[c] + int(sizes[i]) > most)
        ):
            batches.append(filling.pop(c))
            held[c] = 0
        filling.setdefault(c, []).append(int(i))
        held[c] = held.get(c, 0) + int(sizes[i])
        if held[c] >= batch_size:
            batches.append(filling.pop(c))
            held[c] = 0
    batches.extend(b for b in filling.values() if b)
    return [batches[j] for j in rng.permutation(len(batches))]


def plan_by_length(
    lengths: np.ndarray,
    sizes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    most: int | None = None,
) -> list[tuple[int, list[int]]]:
    """Return batches of blocks that share a prediction length.

    Blocks of each prediction length are batched by `plan_batches`, and
    the batches of every length are then shuffled together. Each batch is
    `(length, indices)`.
    """
    plan = []
    for length in sorted(set(lengths.tolist())):
        idx = np.flatnonzero(lengths == length)
        for batch in plan_batches(sizes[idx], batch_size, rng, most):
            plan.append((int(length), [int(idx[i]) for i in batch]))
    return [plan[j] for j in rng.permutation(len(plan))]


def _order(outbreak: str) -> tuple[int, int]:
    """Sort key of an outbreak id, `<seed>-<index>`."""
    seed, index = outbreak.split("-")
    return int(seed), int(index)
