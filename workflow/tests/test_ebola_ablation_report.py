"""The Ebola ablation page (workflow/scripts/ebola_ablation_report.py)."""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import ebola_ablation_report as er  # noqa: E402

MODELS = {
    "randomwalknowcast": 1.0,
    "foundation_zeroshot": 1.5,
    "foundation": 3.7,
    "foundation_zeroshot_pseudoweeks": 1.4,
    "foundation_pseudoweeks": 0.9,
}


def _summary(scores: dict[str, float]) -> dict:
    weeks = (-1, 0, 1)
    return {
        "overall": [
            {
                "model": m,
                "relative_crps": v,
                "relative_log_crps": v / 2,
                "n": 10,
            }
            for m, v in scores.items()
        ],
        "by_weeks_back": [
            {
                "model": m,
                "weeks_back": k,
                "relative_crps": v,
                "relative_log_crps": v / 2,
            }
            for m, v in scores.items()
            for k in weeks
        ],
        "calibration": [
            {
                "model": m,
                "level": level,
                "weeks_back": k,
                "median_over_truth": v,
                "median_over_reported": 1.0,
                "coverage_90": 0.9,
            }
            for m, v in scores.items()
            for level in ("national", "province")
            for k in weeks
        ],
    }


def _variant(root: Path, side: str, name: str, cutoffs: list[str]) -> Path:
    """A variant folder: its summary, daily releases and scored origins."""
    d = root / side / name
    (d / "data").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_summary(MODELS)))
    pd.DataFrame(
        {
            "vintage": cutoffs,
            "location": "CD",
            "onset_date": cutoffs,
            "observed": 1,
            "imputed": 0,
            "cutoff": cutoffs,
        }
    ).to_csv(d / "data" / "daily.csv", index=False)
    runs = d / "forecasts" / "randomwalknowcast"
    runs.mkdir(parents=True)
    with gzip.open(runs / "release-runs.csv.gz", "wt") as f:
        pd.DataFrame(
            {
                "origin": cutoffs[-3:],
                "model": "randomwalknowcast",
                "status": "forecast",
                "reason": "",
            }
        ).to_csv(f, index=False)
    return d


CUTS = ["2026-08-18", "2026-10-07", "2026-10-14", "2026-10-21"]


def test_the_release_history_counts_releases_at_each_scored_origin(
    tmp_path,
) -> None:
    d = _variant(tmp_path, "bdbv", "as_is", CUTS)
    counts, gap = er.release_history(d, "randomwalknowcast")
    assert counts == [2, 3, 4]
    assert gap == 50


def _releases() -> pd.DataFrame:
    rows = [
        ("1-100", "daily", "daily", 0, 2),
        ("1-100", "daily+rolling7", "weekly", 30, 5),
        ("1-100", "daily", "daily", 0, 40),
        ("1-100", "weekly", "weekly", 0, 3),
        ("held-out", "daily", "daily", 0, 1),
    ]
    return pd.DataFrame(
        rows,
        columns=["part", "source", "cadence", "release_start_days", "releases"],
    )


def test_training_counts_are_the_daily_step_training_blocks() -> None:
    t = er.training_releases(_releases(), [2, 7])
    got = dict(zip(t["blocks"].index, t["blocks"], strict=True))
    assert got == {
        "All": 3,
        "2 releases or fewer": 1,
        "7 releases or fewer": 2,
        "Late start (first release backfills)": 1,
    }
    assert t.loc["All", "share"] == 1.0


def test_the_page_is_computed_from_the_runs(tmp_path) -> None:
    for name in ("daily", "few", "fresh"):
        _variant(tmp_path, "sim", name, CUTS)
    _variant(tmp_path, "bdbv", "as_is", CUTS)
    releases = tmp_path / "releases.csv"
    _releases().to_csv(releases, index=False)
    out = tmp_path / "page.html"
    er.report(
        [*sorted((tmp_path / "sim").iterdir()), tmp_path / "bdbv" / "as_is"],
        out,
        checkpoint="slots-c44bbe122e6c",
        reference="randomwalknowcast",
        releases=releases,
        root=ROOT,
    )
    html = out.read_text()
    assert "slots-c44bbe122e6c" in html
    assert "BDBV releases" in html
    assert "BDBV hub data" in html
    assert "Pseudo weeks" in html
    assert "Release counts" in html
    # The training blocks come from releases.csv.
    # Training blocks at or below the most releases BDBV had (4).
    assert "4 releases or fewer" in html
    # Scores come from the summaries.
    assert "3.70" in html
    assert "0.90" in html
    # The grid suggestions are in the issue, not the page.
    assert "Suggestions" not in html


def test_the_page_needs_no_training_releases(tmp_path) -> None:
    _variant(tmp_path, "sim", "few", CUTS)
    out = tmp_path / "page.html"
    er.report(
        [tmp_path / "sim" / "few"],
        out,
        checkpoint="slots-c44bbe122e6c",
        reference="randomwalknowcast",
        releases=None,
        root=ROOT,
    )
    assert "BDBV releases" in out.read_text()
