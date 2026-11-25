"""The simulated data dashboard: what the simulations under ROOT look like.

Its sources are each round's simulations (`<ROOT>/rounds/r<N>/sims/`)
and each run's (`<ROOT>/runs/<run>/sims/`), as the simulator writes them:
`counts-<seed>.csv.gz` (counts by stratum, reference and report date),
`settings-<seed>.csv` (each simulation's drawn settings) and, for rounds,
`latent-<seed>.csv.gz` (infections, onsets and R).

`Index` summarises a source, one entry per simulation from its first
observation draw: its grid factors and drawn values, each with the grid
layer it belongs to (`layers`), its size, strata, delays and weekly
counts by reference date. A batch is summarised once and cached on disk
until its files change. `simulation` gives one simulation's detail for
the deep dive. `page` is the browser page, in the fine-tuning
dashboard's style; workflow/scripts/dashboard.py serves it under /sims/.
"""

from __future__ import annotations

import functools
import json
import math
import re
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LAYER_ORDER = ("infections", "strata", "observation", "delays", "reporting")
"""The grid's layers, in the order the grid file describes them."""


_ID = re.compile(r"^(\d+)-(\d+)$")
"""A simulation's id: its batch's seed and its number in the batch."""


def layers(grid: Path | str) -> dict[str, str]:
    """Return each grid setting's layer: its factors' and distributions'."""
    spec = yaml.safe_load(Path(grid).read_text())
    out = dict(spec.get("factor_layers") or {})
    for name, dist in (spec.get("distributions") or {}).items():
        if isinstance(dist, Mapping) and dist.get("layer"):
            out[name] = dist["layer"]
    return out


def notes(grid: Path | str) -> dict[str, str]:
    """Return each grid setting's note, for the page's tooltips.

    A drawn value's `why`; a factor's comment lines just above it in the
    grid's `factors`.
    """
    text = Path(grid).read_text()
    spec = yaml.safe_load(text)
    out = {
        name: str(d["why"])
        for name, d in (spec.get("distributions") or {}).items()
        if isinstance(d, Mapping) and d.get("why")
    }
    factors = set(spec.get("factors") or {})
    block, comment = False, []
    for line in text.splitlines():
        if not line.startswith(" "):
            block = line.startswith("factors:")
            comment = []
            continue
        if not block:
            continue
        if line.startswith("  #"):
            comment.append(line.strip().lstrip("#").strip())
            continue
        key = re.match(r"^  (\w+):", line)
        if key and key.group(1) in factors and comment:
            out.setdefault(key.group(1), " ".join(comment))
        comment = []
    return out


def sources(root: Path | str, hidden: Iterable[int] = ()) -> dict[str, Path]:
    """Return the folders of simulations under `root`, by name.

    Rounds (`r<N>`, in round order), then runs; only folders with a
    batch's settings. Rounds in `hidden` (the grid's test round, held out
    for the paper) are never listed.
    """
    root = Path(root)
    skip = {f"r{int(r)}" for r in hidden}

    def has(folder: Path) -> bool:
        return any(folder.glob("settings-*.csv"))

    out = {}
    rounds = sorted(
        (root / "rounds").glob("r*/sims"),
        key=lambda p: (len(p.parent.name), p.parent.name),
    )
    for folder in rounds:
        if has(folder) and folder.parent.name not in skip:
            out[folder.parent.name] = folder
    for folder in sorted((root / "runs").glob("*/sims")):
        if has(folder) and folder.parent.name not in out:
            out[folder.parent.name] = folder
    return out


def _seed(path: Path) -> int:
    return int(path.name.split("-")[1].split(".")[0])


def _number(text: str) -> float | None:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(v: float, digits: int = 4) -> float:
    if v == 0 or not math.isfinite(v):
        return 0.0
    return round(v, max(0, digits - 1 - math.floor(math.log10(abs(v)))))


def _first_draw(counts: pd.DataFrame) -> pd.DataFrame:
    """Return each simulation's first observation draw, keyed by its id."""
    parts = counts["outbreak"].str.rsplit("-", n=1, expand=True)
    counts = counts.assign(id=parts[0], draw=pd.to_numeric(parts[1]))
    first = counts.groupby("id")["draw"].transform("min")
    return counts[counts["draw"] == first]


def _read_counts(path: Path) -> pd.DataFrame:
    counts = pd.read_csv(
        path,
        usecols=[
            "outbreak",
            "location",
            "reference_date",
            "report_date",
            "count",
        ],
        dtype={"outbreak": str, "location": str},
    )
    counts = _first_draw(counts)
    ref = pd.to_datetime(counts["reference_date"])
    rep = pd.to_datetime(counts["report_date"])
    return counts.assign(ref=ref, delay=(rep - ref).dt.days, rep=rep)


def _read_settings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


DELAY_POINTS = (0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 42)
"""The delays (days) a simulation's reporting delay distribution is kept
at: the share of its cases reported by each."""

CACHE_VERSION = 2
"""The batch summaries' version: a cached summary of another is redone."""


def _derived(rows: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    """Return what a simulation's drawn settings produced, from its counts.

    Statistics: the share reported within 14 and 28 days, releases a
    week (report dates over their span), the median gap between report
    dates and the days from the first reference date to the first report
    (cadence), the largest weekly log growth, and the observation noise
    (the mean absolute log ratio of each day's final count to its
    centred 7-day mean). Curves: the delay distribution's cumulative
    share at `DELAY_POINTS`, and the reference- and report-date weekday
    profiles (Monday first, each weekday's counts over their mean).
    """
    total = float(rows["count"].sum()) or 1.0
    by_delay = rows.groupby("delay")["count"].sum()
    cdf = [
        _round(float(by_delay[by_delay.index <= d].sum()) / total, 4)
        for d in DELAY_POINTS
    ]
    reports = np.sort(rows["rep"].unique())
    days = (reports[-1] - reports[0]) / np.timedelta64(1, "D") + 1
    gaps = np.diff(reports) / np.timedelta64(1, "D")
    daily = rows.groupby("ref")["count"].sum()
    daily = daily.reindex(
        pd.date_range(daily.index.min(), daily.index.max()), fill_value=0
    )
    smooth = daily.rolling(7, center=True, min_periods=1).mean()
    noise = np.abs(
        np.log1p(daily.clip(lower=0)) - np.log1p(smooth.clip(lower=0))
    ).mean()
    first = rows["ref"].min()
    weekly = rows.groupby((rows["ref"] - first).dt.days // 7)["count"].sum()
    growth = np.diff(np.log1p(weekly.clip(lower=0).to_numpy()))

    def profile(dates: pd.Series) -> list[float]:
        by = rows.groupby(dates.dt.day_of_week)["count"].sum()
        by = by.reindex(range(7), fill_value=0).to_numpy(dtype=float)
        mean = by.mean()
        return [_round(v / mean, 3) if mean else 0.0 for v in by]

    stats = {
        "within14": _round(
            float(by_delay[by_delay.index < 14].sum()) / total, 3
        ),
        "within28": _round(
            float(by_delay[by_delay.index < 28].sum()) / total, 3
        ),
        "releases": _round(7 * len(reports) / days, 3),
        "gap": _round(float(np.median(gaps)), 3) if len(gaps) else 0.0,
        "first_report": int((reports[0] - first) / np.timedelta64(1, "D")),
        "growth": _round(float(max(growth.max(), 0.0)), 3)
        if len(growth)
        else 0.0,
        "noise": _round(float(noise), 3),
    }
    curves = {
        "cdf": cdf,
        "refdow": profile(rows["ref"]),
        "repdow": profile(rows["rep"]),
    }
    return stats, curves


def _summary(rows: pd.DataFrame) -> tuple[dict[str, float], list[int], str]:
    """Return a simulation's statistics, weekly counts and first date."""
    first = rows["ref"].min()
    total = float(rows["count"].sum())
    week = (rows["ref"] - first).dt.days // 7
    weekly = rows.groupby(week)["count"].sum()
    series = weekly.reindex(range(int(week.max()) + 1), fill_value=0)
    peak = int(np.argmax(series.to_numpy()))
    weighted = total if total else 1.0
    stats = {
        "cases": int(total),
        "strata": int(rows["location"].nunique()),
        "days": int(rows["ref"].nunique()),
        "weeks": len(series),
        "peak": int(series.max()),
        "peak_at": _round(peak / max(len(series) - 1, 1), 3),
        "delay": _round(
            float((rows["count"] * rows["delay"]).sum()) / weighted
        ),
        "within7": _round(
            float(rows.loc[rows["delay"] < 7, "count"].sum()) / weighted, 3
        ),
        "negative": _round(float((rows["count"] < 0).mean()), 3),
    }
    return stats, [int(v) for v in series], first.strftime("%Y-%m-%d")


def summarise_batch(
    counts: Path, settings: Path, layer_of: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return one entry per simulation of a batch.

    Each has its id, its grid factors (`f`) and single drawn values
    (`v`), its statistics (`s`, from the first draw's counts with every
    stratum summed, `_derived`'s among them), its curves (`x`: the delay
    distribution and weekday profiles), its weekly counts by reference
    date (`w`) and its first reference date.
    """
    table = _read_settings(settings)
    by_id = dict(list(_read_counts(counts).groupby("id")))
    out = []
    for _, row in table.iterrows():
        oid = row["outbreak"]
        if oid not in by_id:
            continue
        factors, values = {}, {}
        for name in layer_of:
            if name not in row or row[name] == "":
                continue
            v = _number(row[name])
            if v is not None:
                values[name] = _round(v)
            elif ";" not in row[name]:
                factors[name] = row[name]
        stats, series, first = _summary(by_id[oid])
        more, curves = _derived(by_id[oid])
        out.append(
            {
                "id": oid,
                "f": factors,
                "v": values,
                "s": stats | more,
                "w": series,
                "x": curves,
                "first": first,
            }
        )
    return out


def _stamp(*paths: Path) -> list[int]:
    return [n for p in paths for n in (p.stat().st_mtime_ns, p.stat().st_size)]


class Index:
    """Each source's simulations, summarised a batch at a time.

    A batch's summary is cached in `cache/<source>/<seed>.json` with its
    files' times and sizes, and summarised again when they change.
    `refresh` brings every source up to date; `data` returns a source's
    last one (built now if it has none yet).
    """

    def __init__(
        self,
        root: Path | str,
        cache: Path | str,
        layer_of: Mapping[str, str],
        hidden: Iterable[int] = (),
        notes: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.notes = dict(notes or {})
        self.hidden = frozenset(int(r) for r in hidden)
        self.cache = Path(cache)
        self.layer_of = dict(layer_of)
        self.built: dict[str, dict[str, Any]] = {}
        self.encoded: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def sources(self) -> dict[str, Path]:
        """Return the sources under ROOT now."""
        return sources(self.root, self.hidden)

    def _batch(self, name: str, settings: Path) -> list[dict[str, Any]]:
        seed = _seed(settings)
        counts = settings.with_name(f"counts-{seed}.csv.gz")
        if not counts.exists():
            return []
        stamp = [CACHE_VERSION, *_stamp(counts, settings)]
        cached = self.cache / name / f"{seed}.json"
        try:
            got = json.loads(cached.read_text())
            if got["stamp"] == stamp:
                return got["rows"]
        except (OSError, ValueError, KeyError):
            pass
        try:
            rows = summarise_batch(counts, settings, self.layer_of)
        except Exception:
            # A batch still being written, or a broken one: left out now,
            # read again on the next refresh.
            return []
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"stamp": stamp, "rows": rows}))
        return rows

    def build(self, name: str, folder: Path) -> dict[str, Any]:
        """Return a source's index, its columns one list each."""
        batches = sorted(folder.glob("settings-*.csv"), key=_seed)
        rows = [r for b in batches for r in self._batch(name, b)]
        factor_names = sorted({k for r in rows for k in r["f"]})
        value_names = sorted({k for r in rows for k in r["v"]})
        factors = {}
        for k in factor_names:
            levels = sorted({r["f"].get(k, "") for r in rows})
            code = {lv: i for i, lv in enumerate(levels)}
            factors[k] = {
                "levels": levels,
                "codes": [code[r["f"].get(k, "")] for r in rows],
            }
        stat_names = list(rows[0]["s"]) if rows else []
        return {
            "source": name,
            "batches": sum(
                1
                for b in batches
                if b.with_name(f"counts-{_seed(b)}.csv.gz").exists()
            ),
            "ids": [r["id"] for r in rows],
            "first": [r["first"] for r in rows],
            "layers": {k: self.layer_of[k] for k in factor_names + value_names},
            # The grid's own order of its settings, and their notes.
            "order": list(self.layer_of),
            "notes": self.notes,
            "factors": factors,
            "values": {k: [r["v"].get(k) for r in rows] for k in value_names},
            "stats": {k: [r["s"][k] for r in rows] for k in stat_names},
            "series": [r["w"] for r in rows],
            "derived": {
                k: [r["x"][k] for r in rows]
                for k in ("cdf", "refdow", "repdow")
            },
            "delay_points": list(DELAY_POINTS),
        }

    def refresh(self) -> None:
        """Bring every source's index up to date."""
        found = self.sources()
        for name, folder in found.items():
            built = self.build(name, folder)
            with self.lock:
                self.built[name] = built
                self.encoded[name] = _encode(built)
        with self.lock:
            for gone in set(self.built) - set(found):
                self.built.pop(gone)
                self.encoded.pop(gone)

    def _ensure(self, name: str) -> bool:
        if name in self.built:
            return True
        folder = self.sources().get(name)
        if folder is None:
            return False
        built = self.build(name, folder)
        with self.lock:
            self.built[name] = built
            self.encoded[name] = _encode(built)
        return True

    def data(self, name: str) -> dict[str, Any] | None:
        """Return a source's index, None for a source with none."""
        return self.built[name] if self._ensure(name) else None

    def encoded_data(self, name: str) -> bytes | None:
        """Return a source's index as JSON, None for a source with none."""
        return self.encoded[name] if self._ensure(name) else None

    def listing(self) -> list[dict[str, Any]]:
        """Return the sources, each with its simulations if indexed.

        Each says whether it is a round's or a run's; the default is the
        run whose simulations changed last, else the last round.
        """
        found = self.sources()
        runs = [k for k, f in found.items() if f.parent.parent.name == "runs"]
        newest = max(runs, key=lambda k: _changed(found[k]), default=None)
        default = newest or (list(found)[-1] if found else None)
        return [
            {
                "key": k,
                "simulations": len(self.built[k]["ids"])
                if k in self.built
                else None,
                "kind": "run" if k in runs else "round",
                "default": k == default,
            }
            for k in found
        ]


def _changed(folder: Path) -> int:
    """Return when a source's simulations last changed (ns)."""
    return max(
        (p.stat().st_mtime_ns for p in folder.glob("settings-*.csv")),
        default=0,
    )


def _encode(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _clean(values: pd.Series) -> list[float | None]:
    return [None if not math.isfinite(v) else _round(float(v)) for v in values]


@functools.lru_cache(maxsize=16)
def _batch_rows(path: str, stamp: tuple) -> dict[str, pd.DataFrame]:
    """Return a batch's counts by simulation, decoded once (`stamp` keys).

    The deep dive reads a simulation from here, so each batch is read and
    decompressed once while it is among the last 16 used.
    """
    del stamp
    return dict(list(_read_counts(Path(path)).groupby("id")))


@functools.lru_cache(maxsize=16)
def _latent_rows(path: str, stamp: tuple) -> dict[str, pd.DataFrame]:
    """Return a batch's latent values (all strata summed) by simulation."""
    del stamp
    lat = pd.read_csv(path, dtype={"outbreak": str, "location": str})
    lat = lat[lat["location"] == "all"]
    return dict(list(lat.groupby("outbreak")))


@functools.lru_cache(maxsize=32)
def _detail(
    folder: str, oid: str, layer_items: tuple, stamp: tuple
) -> dict[str, Any] | None:
    # `stamp` (the files' times and sizes) keys the cache only, so a
    # batch that changes is read again.
    del stamp
    layer_of = dict(layer_items)
    seed = _ID.match(oid).group(1)
    path = Path(folder)
    counts_path = path / f"counts-{seed}.csv.gz"
    rows = _batch_rows(str(counts_path), tuple(_stamp(counts_path))).get(oid)
    if rows is None or rows.empty:
        return None
    first, last = rows["ref"].min(), rows["ref"].max()
    dates = pd.date_range(first, last, freq="D")
    final = (
        rows.pivot_table(
            index="ref", columns="location", values="count", aggfunc="sum"
        )
        .reindex(dates, fill_value=0)
        .fillna(0)
    )
    total = rows["count"].sum() or 1
    by_delay = rows.groupby("delay")["count"].sum()
    by_delay = by_delay.reindex(
        range(int(by_delay.index.max()) + 1), fill_value=0
    )
    known = []
    for share in (0.25, 0.5, 0.75):
        at = first + pd.Timedelta(days=round(share * (last - first).days))
        seen = rows[(rows["rep"] <= at) & (rows["ref"] <= at)]
        upto = pd.date_range(first, at, freq="D")
        by_ref = seen.groupby("ref")["count"].sum().reindex(upto, fill_value=0)
        known.append(
            {"at": at.strftime("%Y-%m-%d"), "counts": [int(v) for v in by_ref]}
        )
    ref_week = (rows["ref"] - first).dt.days // 7
    delay_week = rows["delay"].clip(lower=0) // 7
    triangle = (
        rows.groupby([ref_week, delay_week])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    triangle = triangle.reindex(
        index=range(int(ref_week.max()) + 1),
        columns=range(int(delay_week.max()) + 1),
        fill_value=0,
    )
    settings = _read_settings(path / f"settings-{seed}.csv")
    row = settings[settings["outbreak"] == oid]
    listed = []
    if not row.empty:
        for name, value in row.iloc[0].items():
            if name == "outbreak" or value == "":
                continue
            listed.append(
                {
                    "layer": layer_of.get(name, "other"),
                    "name": name,
                    "value": value,
                }
            )
    order = {k: i for i, k in enumerate((*LAYER_ORDER, "other"))}
    listed.sort(key=lambda s: order.get(s["layer"], len(order)))
    latent = None
    latent_path = path / f"latent-{seed}.csv.gz"
    if latent_path.exists():
        lat = _latent_rows(str(latent_path), tuple(_stamp(latent_path))).get(
            oid, pd.DataFrame()
        )
        if not lat.empty:
            lat = lat.sort_values("day")
            latent = {"dates": list(lat["date"].astype(str))} | {
                k: _clean(lat[k].astype(float))
                for k in ("infections", "onsets", "r", "reff")
            }
    return {
        "id": oid,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "strata": {str(k): [int(v) for v in final[k]] for k in final.columns},
        "known": known,
        "delays": [_round(float(v) / total, 4) for v in by_delay],
        "triangle": {
            "counts": [[int(v) for v in r] for r in triangle.to_numpy()]
        },
        "settings": listed,
        "latent": latent,
    }


def simulation(
    folder: Path | str, oid: str, layer_of: Mapping[str, str]
) -> dict[str, Any] | None:
    """Return one simulation's detail, None if there is no such one.

    Its first draw's final counts by stratum and reference date, the
    counts as known at a quarter, half and three quarters of the way
    through, the share reported at each delay, its reporting triangle by
    week of reference and week of delay, its settings with their layers,
    and its infections, onsets and R where the batch kept them.
    """
    m = _ID.match(oid)
    if not m:
        return None
    path = Path(folder)
    files = [
        path / f"counts-{m.group(1)}.csv.gz",
        path / f"settings-{m.group(1)}.csv",
    ]
    if not all(f.exists() for f in files):
        return None
    latent = path / f"latent-{m.group(1)}.csv.gz"
    stamp = tuple(_stamp(*files, *([latent] if latent.exists() else [])))
    return _detail(str(path), oid, tuple(sorted(layer_of.items())), stamp)


def page() -> str:
    """Return the simulations page, in the fine-tuning dashboard's style."""
    from nowcasteval import labels

    from . import page as ft

    palette = json.dumps(list(labels.CATEGORICAL))
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Simulations</title>"
        f"<style>{ft.ui_colours()}{ft._asset('style.css')}"
        f"{ft._asset('sims.css')}</style></head><body>"
        f"{ft._asset('sims.html')}"
        f"<script>window.simsPalette={palette};window.simsLayers="
        f"{json.dumps(list(LAYER_ORDER))};\n"
        f"{ft._asset('sims.js')}</script></body></html>"
    )
