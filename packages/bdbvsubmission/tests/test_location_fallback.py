"""A check that fails at one location falls back there only (Sam,
2026-11-24): the failing province's subtree comes from the next
candidate in the chain, and national is re-summed."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from conftest import POLICY, REF, make_samples, write_json, write_samples

from bdbvsubmission.check_forecast import check_values
from bdbvsubmission.common import read_json
from bdbvsubmission.hubdata import read_samples
from bdbvsubmission.select_candidate import run_select

SIDECAR = {
    "national": "CD",
    "national_source": "sum",
    "modelled_provinces": ["CD-HU", "CD-IT", "CD-NK"],
    "unassigned": [],
}


def _nan_at(loc: str) -> Any:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == loc:
                r[6] = float("nan")
                return

    return mutate


def candidate(
    root: Path,
    cid: str,
    *,
    seed: int,
    mutate: Any = None,
    fit: str = "pass",
) -> Path:
    d = root / cid
    rows = make_samples(seed=seed, mutate=mutate)
    sub = d / "submission" / "model-output" / "m" / f"{REF}-seabbs-m.csv"
    write_samples(sub, rows)
    write_json(d / "submission" / "national.json", SIDECAR)
    write_json(d / "fit-summary.json", {"status": fit, "failures": []})
    write_json(d / "ppc-summary.json", {"status": "pass"})
    values = check_values(read_samples(sub))
    write_json(
        d / "checks.json",
        {
            "status": values.status,
            "checks": [
                {
                    "name": "values",
                    "status": values.status,
                    "summary": values.summary,
                    "locations": values.locations,
                }
            ],
        },
    )
    write_json(d / "run.json", {"model": cid})
    return sub


def test_values_name_the_failing_location(tmp_path: Path) -> None:
    sub = candidate(tmp_path, "bvd", seed=1, mutate=_nan_at("CD-IT"))
    assert check_values(read_samples(sub)).locations == ["CD-IT"]


def test_a_failing_province_comes_from_the_next_candidate(
    tmp_path: Path,
) -> None:
    root, out = tmp_path / "c", tmp_path / "gate"
    candidate(root, "bvd", seed=1, mutate=_nan_at("CD-IT"))
    donor = candidate(root, "renewalsp", seed=2)
    prov = run_select(POLICY, root, out, None)
    assert prov["selected"] == "bvd"
    assert prov["locations_substituted"] == {"CD-IT": "renewalsp"}
    assert any("CD-IT" in w for w in prov["warnings"])
    merged = next((out / "selected" / "submission").glob("**/*.csv"))
    s = read_samples(merged)
    d = read_samples(donor)
    base = read_samples(next((root / "bvd" / "submission").glob("**/*.csv")))
    assert s.values["CD-IT"] == d.values["CD-IT"]
    assert s.values["CD-NK"] == base.values["CD-NK"]
    for sid, traj in s.values["CD"].items():
        for w, v in traj.items():
            parts = [s.values[p][sid][w] for p in ("CD-HU", "CD-IT", "CD-NK")]
            assert math.isclose(v, math.fsum(parts))
    assert read_json(out / "provenance.json")["selected"] == "bvd"


def test_a_file_level_failure_still_rules_the_candidate_out(
    tmp_path: Path,
) -> None:
    root, out = tmp_path / "c", tmp_path / "gate"
    candidate(root, "bvd", seed=1, mutate=_nan_at("CD"))
    candidate(root, "renewalsp", seed=2)
    prov = run_select(POLICY, root, out, None)
    assert prov["selected"] == "renewalsp"
    assert not prov.get("locations_substituted")


def test_no_donor_for_the_province_rules_the_candidate_out(
    tmp_path: Path,
) -> None:
    root, out = tmp_path / "c", tmp_path / "gate"
    candidate(root, "bvd", seed=1, mutate=_nan_at("CD-IT"))
    candidate(root, "renewalsp", seed=2, mutate=_nan_at("CD-IT"))
    candidate(root, "main", seed=3, fit="fail")
    prov = run_select(POLICY, root, out, None)
    assert prov["selected"] not in ("bvd", "renewalsp")


def test_a_merged_file_failing_hub_validation_falls_through(
    tmp_path: Path,
) -> None:
    root, out = tmp_path / "c", tmp_path / "gate"
    candidate(root, "bvd", seed=1, mutate=_nan_at("CD-IT"))
    candidate(root, "renewalsp", seed=2)
    seen: list[Path] = []

    def validate(sub: Path, log: Path) -> str:
        seen.append(sub)
        log.write_text("")
        return "fail" if "merged" in sub.parts else "pass"

    prov = run_select(POLICY, root, out, validate)
    assert prov["selected"] == "renewalsp"
    assert not prov.get("locations_substituted")
    assert any("merged" in p.parts for p in seen)
