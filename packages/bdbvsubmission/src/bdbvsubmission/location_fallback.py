"""Fall back at one location rather than for the whole file.

A candidate whose only failing checks name locations (`Check.locations`)
is kept: each failing location's province subtree (the province, its
health zones, and their wk R_t rows) comes from the next candidate in the
chain that converged and passes there. National is re-summed per sample
from the merged provinces plus the kept candidate's unassigned component
(`national.json`), so the file stays coherent. The merged file then goes
through the format checks and the hub's validation again; if it fails
either, the candidate is ruled out.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .check_forecast import (
    Check,
    _unassigned,
    check_coverage,
    check_national_sum,
    check_rt,
    check_samples,
    check_schema,
    check_values,
    check_zones,
)
from .common import blocks, read_json
from .hubdata import is_province, read_samples

LOCATION_CHECKS = frozenset({"values", "samples"})
"""Checks whose failures can be tied to single locations."""


def province_of(location: str) -> str | None:
    """The province (`CD-XX`) a province or health zone belongs to."""
    parts = location.split("-")
    if len(parts) < 2 or parts[0] != "CD":
        return None
    return "-".join(parts[:2])


def failing_provinces(checks: Sequence[Mapping[str, Any]]) -> set[str] | None:
    """Provinces behind every failing check, or None if any failure is
    about the whole file (or national) rather than a province."""
    out: set[str] = set()
    for ch in checks:
        if not blocks(str(ch["status"])):
            continue
        locs = ch.get("locations") or []
        if ch["name"] not in LOCATION_CHECKS or not locs:
            return None
        for loc in locs:
            p = province_of(str(loc))
            if p is None:
                return None
            out.add(p)
    return out


def _in(location: str, provinces: set[str]) -> bool:
    return province_of(location) in provinces


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def _sample_ids(rows: Sequence[Mapping[str, str]]) -> list[str]:
    ids = {r["output_type_id"] for r in rows if r["output_type"] == "sample"}

    def key(x: str) -> tuple[int, Any]:
        try:
            return (0, int(float(x)))
        except ValueError:
            return (1, x)

    return sorted(ids, key=key)


def merge(
    base: Path,
    donors: Mapping[str, Path],
    sidecar: Mapping[str, Any],
    out: Path,
) -> None:
    """Write `base` with each province in `donors` taken from that file.

    Donor sample ids are mapped in order onto the base's, so national can
    be re-summed sample by sample.
    """
    header, rows = _read(base)
    ids = _sample_ids(rows)
    swapped = set(donors)
    merged = [r for r in rows if not _in(r["location"], swapped)]
    for prov, path in donors.items():
        _, drows = _read(path)
        dids = _sample_ids(drows)
        if len(dids) != len(ids):
            raise ValueError(f"{path}: {len(dids)} samples, not {len(ids)}")
        to_base = dict(zip(dids, ids, strict=True))
        for r in drows:
            if province_of(r["location"]) != prov:
                continue
            r = dict(r)
            if r["output_type"] == "sample":
                r["output_type_id"] = to_base[r["output_type_id"]]
            merged.append({k: r.get(k, "") for k in header})
    national = str(sidecar["national"])
    if sidecar.get("national_source", "sum") != "own":
        extra = _unassigned(sidecar)
        parts: dict[tuple[str, str, str], list[float]] = {}
        for r in merged:
            if r["output_type"] == "sample" and is_province(r["location"]):
                k = (r["target"], r["output_type_id"], r["target_end_date"])
                parts.setdefault(k, []).append(float(r["value"]))
        for r in merged:
            if r["location"] != national or r["output_type"] != "sample":
                continue
            k = (r["target"], r["output_type_id"], r["target_end_date"])
            if k in parts:
                sid = str(int(float(k[1])))
                add = extra.get((sid, k[2]), 0.0)
                r["value"] = repr(math.fsum([*parts[k], add]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(merged)


def recheck(
    path: Path,
    expected: Sequence[str],
    settings: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    tasks: Mapping[str, Any] | None,
) -> list[Check]:
    """The format checks that can rule a file out, on a merged file."""
    s = read_samples(path)
    ref = path.name[:10]
    return [
        check_schema(s, ref, path.name, tasks),
        check_values(s),
        check_rt(s, ref, tasks),
        check_samples(s, int(settings["samples"])),
        check_coverage(
            s,
            dt.date.fromisoformat(ref),
            expected,
            int(settings["horizon_weeks"]),
            int(settings["nowcast_weeks"]),
        ),
        check_zones(s, False, float(settings["sum_tolerance"])),
        check_national_sum(s, float(settings["sum_tolerance"]), sidecar),
    ]


def _submission(cdir: Path) -> Path | None:
    files = sorted((cdir / "submission").glob("**/*.csv"))
    return files[0] if files else None


def _converged(cdir: Path) -> bool:
    fit = cdir / "fit-summary.json"
    return not fit.exists() or not blocks(str(read_json(fit)["status"]))


def try_merge(
    cdir: Path,
    later: Sequence[tuple[str, Path]],
    out: Path,
    settings: Mapping[str, Any],
    tasks: Mapping[str, Any] | None,
    validate: Callable[[Path, Path], str] | None,
) -> tuple[Path, dict[str, str], list[str]] | None:
    """Merge a candidate whose failures are all at provinces.

    Returns the merged candidate directory, the donor of each province
    and why, or None when the candidate cannot be kept.
    """
    checks_path = cdir / "checks.json"
    base = _submission(cdir)
    side = cdir / "submission" / "national.json"
    if base is None or not checks_path.exists() or not side.exists():
        return None
    need = failing_provinces(read_json(checks_path)["checks"])
    if not need:
        return None
    donors: dict[str, Path] = {}
    by: dict[str, str] = {}
    for prov in sorted(need):
        for cid, ddir in later:
            dsub = _submission(ddir)
            dchecks = ddir / "checks.json"
            if dsub is None or not dchecks.exists() or not _converged(ddir):
                continue
            bad = failing_provinces(read_json(dchecks)["checks"])
            if bad is None or prov in bad:
                continue
            donors[prov], by[prov] = dsub, cid
            break
        else:
            return None
    sidecar = read_json(side)
    dest = out / "merged" / cdir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cdir, dest)
    merged = _submission(dest)
    assert merged is not None
    try:
        merge(base, donors, sidecar, merged)
    except (ValueError, KeyError):
        return None
    expected = sorted(read_samples(base).locations)
    if any(
        blocks(c.status)
        for c in recheck(merged, expected, settings, sidecar, tasks)
    ):
        return None
    if (
        validate is not None
        and validate(merged, dest / "hub-validation.log") != "pass"
    ):
        return None
    notes = [
        f"location fallback: {p} (province and its zones) from {by[p]}"
        for p in sorted(by)
    ]
    return dest, by, notes
