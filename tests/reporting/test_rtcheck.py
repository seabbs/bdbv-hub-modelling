from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhub.reporting import rtcheck
from nowcastbdbv.data.tidy import load_health_zones, load_target
from nowcastutils.hierarchy import is_zone, parent

pytest.importorskip("nowcastrenewal")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "fixtures" / "data"
LEVELS = (
    0.01,
    0.025,
    *(round(0.05 * k, 2) for k in range(1, 20)),
    0.975,
    0.99,
)


def _rt_rows(locations, weeks, rng):
    rows = []
    for loc in locations:
        for week in weeks:
            values = np.sort(rng.lognormal(0.0, 0.2, len(LEVELS)))
            rows += [
                (loc, week, q, v) for q, v in zip(LEVELS, values, strict=True)
            ]
    return pd.DataFrame(
        rows, columns=["location", "target_end_date", "output_type_id", "value"]
    )


def test_rt_bands_take_the_median_and_50_and_90_intervals() -> None:
    rows = _rt_rows(["CD"], ["2026-10-10"], np.random.default_rng(1))
    bands = rtcheck.rt_bands(rows)
    assert list(bands.columns) == [
        "location",
        "target_end_date",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    ]
    mid = rows.loc[rows["output_type_id"] == 0.5, "value"].iloc[0]
    assert bands["q50"].iloc[0] == pytest.approx(mid)
    with pytest.raises(ValueError, match="lack the levels"):
        rtcheck.rt_bands(rows.loc[rows["output_type_id"] != 0.25])


def test_submitted_weeks_are_the_reference_week_and_the_one_before() -> None:
    assert rtcheck.submitted_weeks("2026-11-21") == ("2026-11-14", "2026-11-21")


def test_bvd_rt_maps_areas_and_zones_to_hub_codes() -> None:
    ts = pd.DataFrame(
        {
            "area": ["National", "Haut-Uele", "haut_uele", "Ituri"],
            "date": ["2026-11-01"] * 4,
            "median": [1.0, 1.2, 9.0, 0.9],
            "lower_90": [0.8, 1.0, 9.0, 0.7],
            "upper_90": [1.2, 1.4, 9.0, 1.1],
            "series": ["rt", "rt", "cases", "rt"],
        }
    )
    zones = pd.DataFrame(
        {
            "members": ["CD-IT-03"],
            "as_of": ["2026-11-07"],
            "log_mean": [0.0],
            "log_sd": [0.1],
        }
    )
    out = rtcheck.bvd_rt(
        ts, zones, {"haut_uele": "CD-HU", "ituri": "CD-IT"}, 1.6448536
    )
    assert list(out["location"]) == ["CD", "CD-HU", "CD-IT", "CD-IT-03"]
    zone = out.iloc[-1]
    assert zone["median"] == pytest.approx(1.0)
    assert zone["upper_90"] == pytest.approx(np.exp(0.16448536))


def _fit_dir(tmp_path: Path, with_rt: bool = True) -> tuple[Path, list[str]]:
    target = load_target(DATA)
    vintage = sorted(target.loc[target["kind"] == "real", "vintage"])[-1]
    zones = load_health_zones(DATA)
    zones = zones.loc[zones["vintage"] == vintage]
    rows = target.loc[
        (target["vintage"] == vintage) & (target["kind"] == "real")
    ]
    weeks = sorted(rows["target_end_date"].unique())[-6:]
    busy = zones.groupby("location")["total"].sum().sort_values()
    fitted = list(busy.index[-3:])
    provinces = sorted({parent(z) for z in fitted})
    rng = np.random.default_rng(2)
    samples = pd.DataFrame(
        [
            (loc, w, k, float(rng.poisson(5)))
            for loc in [*provinces, *fitted]
            for w in weeks
            for k in range(1, 21)
        ],
        columns=["location", "target_end_date", "sample", "value"],
    )
    out = tmp_path / "fit"
    out.mkdir()
    samples.to_csv(out / "samples.csv.gz", index=False)
    if with_rt:
        _rt_rows(["CD", *provinces, *fitted], weeks, rng).to_csv(
            out / "rt.csv.gz", index=False
        )
    (out / "fit.json").write_text(
        json.dumps(
            {
                "model": "renewalbvd",
                "vintage": vintage,
                "reference_date": weeks[-1],
            }
        )
    )
    (out / "diagnostics.json").write_text(json.dumps({"bvd": None}))
    return out, [*provinces, *fitted]


def test_the_page_shows_each_location_with_rt_and_lists_silent_zones(
    tmp_path: Path,
) -> None:
    fit, shown = _fit_dir(tmp_path)
    page = rtcheck.rt_check_from_fit(fit, DATA)
    assert page.startswith("<!doctype html>")
    # One picker entry and one panel (light and dark) per location.
    for code in ["CD", *shown]:
        assert f"({code})</option>" in page
    assert page.count('class="panel"') == len(shown) + 1
    assert page.count("<svg") == 2 * (len(shown) + 1)
    # Zones with cases in the fitted provinces but no Rt are listed.
    zones = load_health_zones(DATA)
    vintage = json.loads((fit / "fit.json").read_text())["vintage"]
    zones = zones.loc[zones["vintage"] == vintage]
    counts = zones.groupby("location")["total"].sum()
    provinces = {c for c in shown if not is_zone(c)}
    silent = [
        z
        for z in counts.index[counts > 0]
        if parent(z) in provinces and z not in shown
    ]
    assert silent
    assert f"Zones with cases but no Rt ({len(silent)})" in page
    for z in silent:
        assert f"({z})" in page
    # The submitted weeks and the time indexing are stated.
    ref = json.loads((fit / "fit.json").read_text())["reference_date"]
    assert " and ".join(rtcheck.submitted_weeks(ref)) in page
    assert "week of infection" in page
    assert "week of onset" in page


def test_a_fit_without_rt_is_refused(tmp_path: Path) -> None:
    fit, _ = _fit_dir(tmp_path, with_rt=False)
    with pytest.raises(FileNotFoundError, match=r"rt\.csv\.gz"):
        rtcheck.rt_check_from_fit(fit, DATA)


def test_cli_writes_the_page(tmp_path: Path) -> None:
    from bdbvhub.cli import main

    fit, _ = _fit_dir(tmp_path)
    out = tmp_path / "review" / "rt-check.html"
    assert main(["rt-check", "--fit", str(fit), "--data", str(DATA),
                 "--out", str(out)]) == 0  # fmt: skip
    assert out.read_text().startswith("<!doctype html>")
