"""Nowcast panels by checkpoint, for any configured dataset.

A dataset's forecasts are one folder per label (`reference`,
`reference-complete`, or a checkpoint such as `slots-<hash>`), each with
one file per nowcast date. Rows carry `quantile`, `value`, `truth` and
optionally `target` and `score`. A row with no target is the 7-day
target.

Every row goes to the configured reference it matches, else, in a
reference folder, to its model (`model:<name>`) if `nowcasteval.labels`
names it (`known`; others, such as hub teams or a model's name before a
rename, are left out), else to the checkpoint of a folder a run claims;
excluded and
unclaimed checkpoints are left out. The page draws `levels`; coverage,
bias and the WIS use the seven cached levels.

Bands and coverage are 50% and 90%, as in the paper: the 90% interval
from the 5% and 95% quantiles where a file has them (`OUTER`), else the
95% interval, as the German hub's quantiles stop at 2.5% and 97.5%.
"""

from __future__ import annotations

import csv
import functools
import gzip
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcasteval import labels
from nowcasteval.scoring import wis_quantile

from . import paperstyle
from .paperstyle import WIS_PARTS
from .rows import num, sources
from .runs import INTERIM, Classifier, Reference, Run, split_label

SEVEN = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
"""The cached quantile levels."""

LEVELS = tuple(float(q) for q in SEVEN)

OUTER = ("0.05", "0.95")
"""The 90% interval's quantiles, used for bands and coverage if present."""

PAGE_LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)
"""Default levels of `build`; the page draws all seven (`LEVELS`)."""

REFERENCE = "reference"
"""Reference folders start with this (`reference`, `reference-complete`)."""

TRUTH_FILE = "truth.csv.gz"
"""A dataset's truth context file (`nowcastgermany.checkpoint.TRUTH_FILE`)."""

TARGET = "7day"

LOG = logging.getLogger(__name__)

RUN_MODEL = "foundation"
"""The model a run's checkpoint is in its score files."""


_UNKNOWN: set[str] = set()
"""Model names already logged as unknown."""


def known(model: str) -> bool:
    """Return whether `nowcasteval.labels` names a model; log it if not.

    A model with no name of its own (an old name in files written before a
    rename, a hub team) is not shown; it is logged once.
    """
    if model in labels.MODEL_LABELS:
        return True
    if model not in _UNKNOWN:
        _UNKNOWN.add(model)
        LOG.info("not shown, not a model nowcasteval.labels names: %s", model)
    return False


def variant_id(model: str, run_hash: str) -> str:
    """Return the series id of a checkpoint's variant model.

    A fine-tuned model other than `RUN_MODEL` (`labels.FINETUNED`, such
    as `foundation_baselinenowcast`) in a checkpoint's files is a series
    of its own, `<model>:<hash>`.
    """
    return f"{model}:{run_hash}"


def variant_of(series: str) -> tuple[str, str] | None:
    """Return a variant series' (model, checkpoint hash), else None."""
    model, sep, run_hash = series.partition(":")
    if sep and model in labels.FINETUNED and model != RUN_MODEL:
        return model, run_hash
    return None


"""The target of rows that name none (files scored before the column)."""


def wis(q: Sequence[float], y: float) -> float:
    """Return the WIS of the seven cached quantiles at truth `y`."""
    return float(wis_quantile([y], [list(q)], list(LEVELS))["wis"][0])


def wis_each(scored: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    """Return `wis` of each (seven quantiles, truth), scored in one pass."""
    if not scored:
        return []
    q = np.array([qs for qs, _ in scored], dtype=float)
    y = np.array([y for _, y in scored], dtype=float)
    return [float(v) for v in wis_quantile(y, q, list(LEVELS))["wis"]]


def wis_parts(scored: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    """Return mean WIS parts, natural scale then log scale (`log1p`).

    `WIS_PARTS` of the forecasts' seven quantiles, then the log-scale WIS
    and its `WIS_PARTS`, as the shared scorer takes log WIS: of log(x + 1)
    of the truth and the quantiles.
    """
    q = np.array([qs for qs, _ in scored], dtype=float)
    y = np.array([y for _, y in scored], dtype=float)
    nat = wis_quantile(y, q, list(LEVELS))
    log = wis_quantile(np.log1p(y), np.log1p(np.clip(q, 0, None)), list(LEVELS))
    out = [nat[p].mean() for p in WIS_PARTS] + [log["wis"].mean()]
    return [
        round(float(v), 4) for v in [*out, *(log[p].mean() for p in WIS_PARTS)]
    ]


def cdf_at(q: Sequence[float], y: float) -> float:
    """Return the forecast CDF at `y`, linear between the seven quantiles.

    Below the lowest quantile it is half its level, above the highest
    half the remaining tail.
    """
    if y <= q[0]:
        return LEVELS[0] / 2
    if y >= q[-1]:
        return 1 - (1 - LEVELS[-1]) / 2
    for i in range(len(q) - 1):
        if q[i] <= y <= q[i + 1]:
            if q[i + 1] == q[i]:
                return (LEVELS[i] + LEVELS[i + 1]) / 2
            return LEVELS[i] + (LEVELS[i + 1] - LEVELS[i]) * (y - q[i]) / (
                q[i + 1] - q[i]
            )
    return 0.5


def stats(
    pairs: Sequence[tuple[Sequence[float], float]],
    outer: Sequence[Sequence[float] | None] | None = None,
) -> list[float]:
    """Return 50% and outer coverage, the mean bias, 1 - 2 F(y), and band.

    The outer interval is 90% where every forecast has its `OUTER`
    quantiles (`outer`, one pair or None per forecast), else 95%; the
    last value is its level, 90 or 95.
    """
    n = len(pairs)
    ninety = outer is not None and all(o is not None for o in outer)
    bounds = (
        [(o[0], o[1]) for o in outer if o is not None]
        if ninety and outer is not None
        else [(q[0], q[6]) for q, _ in pairs]
    )
    return [
        round(sum(q[2] <= y <= q[4] for q, y in pairs) / n, 3),
        round(
            sum(
                lo <= y <= hi
                for (lo, hi), (_, y) in zip(bounds, pairs, strict=True)
            )
            / n,
            3,
        ),
        round(sum(1 - 2 * cdf_at(q, y) for q, y in pairs) / n, 3),
        90 if ninety else 95,
    ]


def is_reference(folder: str) -> bool:
    """Return whether a cache folder holds reference forecasts."""
    return folder.startswith(REFERENCE)


def read_file(
    path: Path, folder: str, which: Classifier
) -> dict[tuple[str, str, str, str], dict[str, dict[str, Any]]]:
    """Return a quantile file's rows by (location, series, source, target)."""
    ref_folder = is_reference(folder)
    checkpoint = "zero-shot" if ref_folder else folder
    run_hash = None if ref_folder else split_label(folder)[0]
    refs = {r.key for r in which.references}
    by_model: dict[str, str | None] = {}
    out: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            model = r["model"]
            if model not in by_model:
                series = which.series(
                    {"model": model, "checkpoint": checkpoint}
                )
                if series is None and ref_folder:
                    # Only models nowcasteval.labels names are shown.
                    series = f"model:{model}" if known(model) else None
                elif series is not None and series not in refs:
                    # A checkpoint's own model, or one of its variants as
                    # a series of its own; any other model is not shown.
                    series = (
                        run_hash
                        if model == RUN_MODEL
                        else variant_id(model, run_hash)
                        if model in labels.FINETUNED
                        else None
                    )
                by_model[model] = series
            series = by_model[model]
            value = num(r.get("value"))
            if (
                series is None
                or (
                    r.get("quantile") not in SEVEN
                    and r.get("quantile") not in OUTER
                )
                or value is None
            ):
                continue
            target = (r.get("target") or "").strip() or TARGET
            for src in sources(model):
                entry = out.setdefault(
                    (r["location"], series, src, target), {}
                ).setdefault(
                    r["target_end_date"],
                    {"q": {}, "y": None, "v": "", "s": None, "l": None},
                )
                entry["q"][r["quantile"]] = value
                y = num(r.get("truth"))
                if y is not None:
                    entry["y"], entry["v"] = y, r.get("truth_vintage") or ""
                entry["s"] = num(r.get("score"))
                entry["l"] = num(r.get("latest"))
    return out


def wis_split(
    scored: Sequence[tuple[str, Sequence[float], float]],
    origin: str,
    each: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return mean WIS of the nowcast part and of each forecast week ahead.

    `scored` holds (target date, seven quantiles, truth). The nowcast part
    is the target dates up to the nowcast date `origin`; the forecast part
    the dates after it, by week ahead (days 1 to 7 after are week 1), as
    the held-out loss and the paper split them (`labels.HORIZON_PARTS`).
    Each part is [mean WIS, dates], the mean None with no dates. `each`
    is each date's WIS, if already scored (`wis_each`).
    """
    if each is None:
        each = wis_each([(q, y) for _, q, y in scored])
    now: list[float] = []
    ahead: dict[str, list[float]] = {}
    for (day, _, _), w in zip(scored, each, strict=True):
        days = _offset(day, origin)
        if days <= 0:
            now.append(w)
        else:
            ahead.setdefault(str(-(-days // 7)), []).append(w)

    def mean(v: list[float]) -> list[Any]:
        return [round(sum(v) / len(v), 4) if v else None, len(v)]

    return {
        "nowcast": mean(now),
        "forecast": {k: mean(ahead[k]) for k in sorted(ahead, key=int)},
    }


Range = tuple[int | None, int | None]
"""Days from the nowcast date, first and last; None is open."""


ROOT = Path(__file__).parents[3]


def horizons(
    dataset: Mapping[str, Any],
) -> tuple[Range | None, dict[str, Range], str | None]:
    """Return a dataset's horizon days, control choices and default.

    From its `horizons`: `days`, a German config (TOML, its `horizons`:
    the nowcast to `forecast_days` ahead) or [first, last]; `ranges`,
    name: [first, last], None for the end of `days`; and `default`. With
    none, every day is shown and there is no control.
    """
    spec = dataset.get("horizons") or {}
    if not spec:
        return None, {}, None
    raw = spec.get("days")
    days: Range | None
    if isinstance(raw, str):
        from nowcastgermany.evaluation import load_germany_config

        hs = load_germany_config(ROOT / raw).horizons
        days = (min(hs), max(hs))
    elif raw:
        days = (int(raw[0]), int(raw[1]))
    else:
        days = None
    lo, hi = days or (None, None)
    ranges = {
        str(name): (
            lo if span[0] is None else int(span[0]),
            hi if span[1] is None else int(span[1]),
        )
        for name, span in (spec.get("ranges") or {}).items()
    }
    return days, ranges, spec.get("default")


def held_out_from(german_config: Path | str) -> str:
    """Return a German config's split date, its held-out half's first date."""
    from nowcastgermany.evaluation import load_germany_config

    return str(load_germany_config(ROOT / german_config).split_date)


@functools.lru_cache(maxsize=1 << 16)
def _offset(day: str, origin: str) -> int:
    """Return the days from `origin` to `day`."""
    return (pd.Timestamp(day) - pd.Timestamp(origin)).days


def _within(day: str, origin: str | None, span: Range | None) -> bool:
    """Return whether target date `day` is in `span` days of `origin`."""
    if span is None or origin is None:
        return True
    d = _offset(day, origin)
    lo, hi = span
    return (lo is None or d >= lo) and (hi is None or d <= hi)


def _scored(
    scored: Sequence[tuple[Sequence[float], float, float | None]],
    outers: Sequence[Sequence[float] | None],
    each: Sequence[float],
) -> dict[str, Any]:
    """Return the stats, mean score and, for WIS, its parts of forecasts.

    `each` is each forecast's WIS (`wis_each`), used where the file has
    no score of its own.
    """
    out: dict[str, Any] = {
        "stats": stats([(q, y) for q, y, _ in scored], outers),
        "score": round(
            sum(
                s if s is not None else w
                for (_, _, s), w in zip(scored, each, strict=True)
            )
            / len(scored),
            2,
        ),
    }
    # A file's own score may not be WIS, so it has no parts.
    if all(s is None for _, _, s in scored):
        out["parts"] = wis_parts([(q, y) for q, y, _ in scored])
    return out


def digest(
    rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    picks: Sequence[int],
    origin: str | None = None,
    days: Range | None = None,
    ranges: Mapping[str, Range] | None = None,
) -> list[tuple]:
    """Return one file's series, stats, score, truth and latest by key.

    Kept with the file (`build`'s `files`), so a rebuild after a new
    file only assembles these. Where every date has the `OUTER`
    quantiles, the outermost band drawn is 90%, from them, in place of
    the 2.5% and 97.5% quantiles (`series["band"]`), else 95%.

    Target dates outside `days` of the nowcast date `origin` are left
    out, so every model is drawn and scored over the same days. Each of
    `ranges` (name: days) gets its own stats, score and parts
    (`entry["ranges"]`), and each day's WIS is kept by its days from
    `origin` (`entry["byday"]`), for the page's horizon control.
    """
    out = []
    for (loc, sid, src, tgt), all_dates in rows.items():
        dates = {d: e for d, e in all_dates.items() if _within(d, origin, days)}
        times, drawn, scored, outers, days_ = [], [], [], [], []
        ninety = all(all(k in dates[d]["q"] for k in OUTER) for d in dates)
        seen: dict[str, tuple[str, float]] = {}
        last: dict[str, float] = {}
        for day in sorted(dates):
            e = dates[day]
            if not all(k in e["q"] for k in SEVEN):
                continue
            seven = [e["q"][k] for k in SEVEN]
            outer = [e["q"][k] for k in OUTER] if ninety else None
            band = list(seven)
            if outer is not None:
                band[0], band[-1] = outer
            times.append(day)
            drawn.append([round(band[i], 1) for i in picks])
            if e["y"] is not None:
                scored.append((seven, e["y"], e["s"]))
                days_.append(day)
                outers.append(outer)
                seen[day] = (e["v"], e["y"])
            if e["l"] is not None:
                last[day] = e["l"]
        entry: dict[str, Any] | None = None
        if times:
            entry = {
                "series": {"t": times, "q": drawn, "band": 90 if ninety else 95}
            }
            if scored:
                each = wis_each([(q, y) for q, y, _ in scored])
                entry |= _scored(scored, outers, each)
                by_range = {}
                for name, span in (ranges or {}).items():
                    keep = [
                        i
                        for i, d in enumerate(days_)
                        if _within(d, origin, span)
                    ]
                    if keep:
                        by_range[name] = _scored(
                            [scored[i] for i in keep],
                            [outers[i] for i in keep],
                            [each[i] for i in keep],
                        )
                if by_range:
                    entry["ranges"] = by_range
                if all(s is None for _, _, s in scored) and origin is not None:
                    entry["byday"] = {
                        str(_offset(d, origin)): round(w, 4)
                        for d, w in zip(days_, each, strict=True)
                    }
                if "parts" in entry and origin is not None:
                    entry["split"] = wis_split(
                        [
                            (d, q, y)
                            for d, (q, y, _) in zip(days_, scored, strict=True)
                        ],
                        origin,
                        each,
                    )
        out.append((loc, sid, src, tgt, entry, seen, last))
    return out


def _put(d: dict, keys: Sequence[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def folders(
    cache: Path, run_list: Sequence[Run], unit: str | None = None
) -> list[tuple[Path, str]]:
    """Return each forecast folder and the checkpoint name it stands for.

    The shared cache's folders by their own names, and, with `unit`, each
    run's own scored checkpoints (`<run>/scores/<unit>/n<N>`, and interim
    ones in `interim/s<step>`) as `<run>-<hash>`, the hash of its
    `train/n<N>/settings.json` (or `train/interim/s<step>/`).
    """
    out = [(p, p.name) for p in sorted(cache.iterdir()) if p.is_dir()]
    for r in run_list if unit else ():
        if r.folder is None:
            continue
        scored = Path(r.folder, "scores", unit)
        for p in [
            *sorted(scored.glob("n*")),
            *sorted(scored.glob(f"{INTERIM}/s*")),
        ]:
            rel = p.relative_to(scored)
            settings = Path(r.folder, "train", rel, "settings.json")
            try:
                key = json.loads(settings.read_text()).get("hash")
            except (OSError, ValueError):
                continue
            if p.is_dir() and key:
                out.append((p, f"{r.key}-{key}"))
    return out


def build(
    cache: Path | str,
    run_list: Sequence[Run],
    references: Sequence[Reference],
    levels: Sequence[float] = PAGE_LEVELS,
    excluded: Iterable[str] = (),
    files: dict | None = None,
    order: Sequence[tuple[str, str]] = (),
    sources: Iterable[str] | None = None,
    unit: str | None = None,
    days: Range | None = None,
    ranges: Mapping[str, Range] | None = None,
    until: str | None = None,
) -> dict[str, Any] | None:
    """Return every location's nowcasts by series, target, source, origin.

    `until` is the first nowcast date of a held-out half (`held_out_from`):
    no file of that date or later is read.

    `days` and `ranges` are the horizon's days every model is shown over
    and the horizon control's choices (`digest`).

    `sources` are the model inputs shown (a dataset's `sources`, such as
    Germany's daily only); None shows every input.

    `files` keeps each file's computed rows (`digest`) by its time and
    size, so a rebuild reads and computes only new or changed files.
    `order` is the location groups, (name, regex), locations are sorted
    by. Returns None if the folder does not exist.
    """
    import re

    cache = Path(cache)
    if not cache.is_dir():
        return None
    files = {} if files is None else files
    which = Classifier(run_list, references, excluded)
    keep = None if sources is None else set(sources)
    picks = [SEVEN.index(f"{lv:g}") for lv in levels]
    series: dict = {}
    score: dict = {}
    parts: dict = {}
    split: dict = {}
    by_range: dict = {}
    byday: dict = {}
    stat: dict = {}
    truth: dict = {}
    latest: dict = {}
    origins: set[str] = set()
    targets: set[str] = set()
    found = folders(cache, run_list, unit)
    for folder, name in found:
        ref = is_reference(name)
        if not ref and which.run_of(name) is None:
            continue
        for path in sorted(folder.glob("*.csv.gz")):
            origin = path.name.removesuffix(".csv.gz")
            if until is not None and origin >= until:
                continue
            if ref:
                origins.add(origin)
            try:
                st = path.stat()
            except OSError:
                continue
            key = (
                st.st_mtime_ns,
                st.st_size,
                tuple(picks),
                days,
                tuple(sorted((ranges or {}).items())),
            )
            if str(path) not in files or files[str(path)][0] != key:
                rows = read_file(path, name, which)
                files[str(path)] = (
                    key,
                    digest(rows, picks, origin, days, ranges),
                )
            for loc, sid, src, tgt, entry, seen, last in files[str(path)][1]:
                if keep is not None and src not in keep:
                    continue
                targets.add(tgt)
                for day, (vintage, y) in seen.items():
                    known = truth.setdefault(loc, {}).setdefault(tgt, {})
                    if day not in known or vintage >= known[day][0]:
                        known[day] = (vintage, y)
                for day, v in last.items():
                    _put(latest, [loc, tgt, day], v)
                if entry is None:
                    continue
                keys = [loc, sid, tgt, src, origin]
                _put(series, keys, entry["series"])
                if "stats" in entry:
                    _put(stat, keys, entry["stats"])
                    _put(score, keys, entry["score"])
                    if "parts" in entry:
                        _put(parts, keys, entry["parts"])
                    if "split" in entry:
                        _put(split, keys, entry["split"])
                    if "byday" in entry:
                        _put(byday, keys, entry["byday"])
                    for span, part in entry.get("ranges", {}).items():
                        for k, v in part.items():
                            _put(by_range, [loc, span, k, *keys[1:]], v)

    # A dataset's truth context (the German plots' truth before the first
    # nowcast date) adds days no forecast file has; a file's own truth
    # wins.
    context = cache / TRUTH_FILE
    if context.is_file():
        for r in pd.read_csv(context, dtype=str).itertuples(index=False):
            known = truth.setdefault(r.location, {}).setdefault(r.target, {})
            if r.target_end_date not in known:
                known[r.target_end_date] = (r.truth_vintage, float(r.truth))

    def rank(loc: str) -> tuple[int, str]:
        hit = (i for i, (_, rx) in enumerate(order) if re.match(rx, loc))
        return next(hit, len(order)), loc

    if not origins:
        origins = {
            o
            for loc in series.values()
            for s in loc.values()
            for t in s.values()
            for src in t.values()
            for o in src
        }
    return {
        "series": series,
        "score": score,
        "parts": parts,
        "split": split,
        "ranges": by_range,
        "byday": byday,
        "stats": stat,
        "truth": truth,
        "latest": latest,
        "sets": sorted({s for loc in series.values() for s in loc}),
        "locs": sorted(series, key=rank),
        "targets": sorted(targets),
        "origins": sorted(origins),
        "folders": {
            split_label(name)[0]: name
            for _, name in found
            if not is_reference(name)
        },
    }


def _signature(
    cache: Path, run_list: Sequence[Run], unit: str | None = None
) -> tuple:
    out = []
    files = [
        f for d, _ in folders(cache, run_list, unit) for f in d.glob("*.csv.gz")
    ]
    for f in sorted(files):
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def cached_build(
    cache: Path | str,
    run_list: Sequence[Run],
    references: Sequence[Reference],
    store: dict,
    levels: Sequence[float] = PAGE_LEVELS,
    excluded: Iterable[str] = (),
    order: Sequence[tuple[str, str]] = (),
    sources: Iterable[str] | None = None,
    unit: str | None = None,
    days: Range | None = None,
    ranges: Mapping[str, Range] | None = None,
    until: str | None = None,
) -> dict[str, Any] | None:
    """Return `build`'s result, reused while no file of the folders changes."""
    cache = Path(cache)
    shown = None if sources is None else tuple(sorted(sources))
    sig = (
        _signature(cache, run_list, unit) if cache.is_dir() else None,
        tuple(levels),
        shown,
        days,
        tuple(sorted((ranges or {}).items())),
        until,
    )
    if store.get("sig") != sig or "built" not in store:
        store["built"] = build(
            cache,
            run_list,
            references,
            levels,
            excluded,
            store.setdefault("files", {}),
            order,
            shown,
            unit,
            days,
            ranges,
            until,
        )
        store["sig"] = sig
    return store["built"]


def location(built: Mapping[str, Any] | None, loc: str) -> dict[str, Any]:
    """Return one location's page data: series, scores, stats, truth."""
    if not built or loc not in built["series"]:
        return {}

    def line(by_target: Mapping[str, Mapping[str, Any]], pick=lambda v: v):
        return {
            tgt: {"t": sorted(d), "y": [pick(d[k]) for k in sorted(d)]}
            for tgt, d in by_target.items()
        }

    return {
        "series": built["series"][loc],
        "score": built["score"].get(loc, {}),
        "parts": built.get("parts", {}).get(loc, {}),
        "split": built.get("split", {}).get(loc, {}),
        "ranges": built.get("ranges", {}).get(loc, {}),
        "byday": built.get("byday", {}).get(loc, {}),
        "stats": built["stats"].get(loc, {}),
        "truth": line(built["truth"].get(loc, {}), lambda v: v[1]),
        "latest": line(built["latest"].get(loc, {})),
    }


def meta(
    built: Mapping[str, Any],
    run_list: Sequence[Run],
    references: Sequence[Reference],
    sizes: Mapping[str, int],
    excluded: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return each series' label, run, size and colour, in page order.

    References first, in their configured order, then other reference
    models by name and each checkpoint's variants by size, then each
    run's checkpoints by size, in the run's colour.
    """
    which = Classifier(run_list, references, excluded)
    refs = {r.key: r for r in references}
    by_key = {r.key: r for r in run_list}
    out = []
    for s in built["sets"]:
        if s in refs:
            out.append(
                {
                    "id": s,
                    "label": refs[s].label,
                    "run": "ref",
                    "n": 0,
                    "color": refs[s].colour,
                    "dash": refs[s].dash,
                }
            )
        elif s.startswith("model:"):
            model = s.removeprefix("model:")
            out.append(
                {
                    "id": s,
                    "label": labels.model_label(model),
                    "run": "ref",
                    "n": 1,
                    "color": paperstyle.model_colour(model),
                    "dash": paperstyle.model_dash(model),
                }
            )
        elif variant_of(s):
            model, key = variant_of(s)
            run = which.run_of(built["folders"].get(key, key))
            if run is None:
                continue
            n = sizes.get(key)
            # The run is named only when there are several.
            name = f", {by_key[run].label}" if len(run_list) > 1 else ""
            out.append(
                {
                    "id": s,
                    "label": f"{labels.model_label(model)}{name}"
                    + (f" n{n}" if n else ""),
                    "run": "ref",
                    "n": 2 + (n or 0),
                    "color": paperstyle.model_colour(model),
                    "dash": paperstyle.model_dash(model),
                }
            )
        else:
            run = which.run_of(built["folders"].get(s, s))
            if run is None:
                continue
            n = sizes.get(s) or 0
            # An interim checkpoint, exported mid-segment, says so.
            what = f"n{n} " if n else ""
            if INTERIM in built["folders"].get(s, ""):
                what += f"{INTERIM} "
            label = f"{by_key[run].label} {what}{s}"
            out.append({"id": s, "label": label, "run": run, "n": n})
    ref_rank = {k: i for i, k in enumerate(refs)}
    run_rank = {r.key: i for i, r in enumerate(run_list)}
    out.sort(
        key=lambda m: (
            (0, ref_rank.get(m["id"], len(ref_rank)), m["n"], m["id"])
            if m["run"] == "ref"
            else (1, run_rank[m["run"]], m["n"], m["id"])
        )
    )
    for m in out:
        if m["run"] != "ref":
            m["color"] = by_key[m["run"]].colour
    return out
