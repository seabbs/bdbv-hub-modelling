"""The Ebola ablation (workflow/scripts/ebola_ablation.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import ebola_ablation as ea  # noqa: E402


def _reports() -> pd.DataFrame:
    """Two provinces' reports: 1,000 cases of one onset day each."""
    ref = pd.Timestamp("2026-09-01")
    return pd.DataFrame(
        {
            "province": ["CD-IT", "CD-IT", "CD-NK"],
            "ref": [ref, ref, ref],
            "rep": [ref + pd.Timedelta(days=d) for d in (2, 9, 30)],
            "count": [600, 400, 1000],
        }
    )


def test_the_delay_line_is_in_days_and_reported_by_the_vintage() -> None:
    rng = np.random.default_rng(1)
    line = ea.delay_line(_reports(), "2026-09-15", cap=500, rng=rng)
    assert set(line.columns) >= {"vintage", "location", "onset_week", "delay"}
    assert (line["vintage"] == "2026-09-15").all()
    # CD-NK's cases are reported after the vintage.
    assert set(line["location"]) == {"CD-IT"}
    assert set(line["delay"]) <= {2, 9}
    assert len(line) == 500
    assert (line["onset_week"] == "2026-09-05").all()
    share = (line["delay"] == 2).mean()
    assert 0.5 < share < 0.7


def test_the_delay_line_keeps_every_case_under_the_cap() -> None:
    rng = np.random.default_rng(1)
    line = ea.delay_line(_reports(), "2026-10-15", cap=5000, rng=rng)
    assert line.groupby("location").size().to_dict() == {
        "CD-IT": 1000,
        "CD-NK": 1000,
    }
    assert (line.loc[line["location"] == "CD-NK", "delay"] == 30).all()


def _summary(models: dict[str, float]) -> dict:
    return {
        "overall": [
            {
                "model": m,
                "relative_crps": v,
                "relative_log_crps": v,
                "n": 10,
            }
            for m, v in models.items()
        ],
        "by_weeks_back": [
            {
                "model": m,
                "weeks_back": 0,
                "relative_crps": v,
                "relative_log_crps": v,
            }
            for m, v in models.items()
        ],
        "calibration": [
            {
                "model": m,
                "level": "national",
                "weeks_back": 0,
                "median_over_truth": 1.0,
                "median_over_reported": 1.0,
                "coverage_90": 0.9,
            }
            for m in models
        ],
    }


def test_the_report_shows_each_variant_with_and_without_pseudo_weeks(
    tmp_path,
) -> None:
    scores = {
        "foundation_zeroshot": 1.5,
        "foundation": 3.7,
        "foundation_zeroshot_pseudoweeks": 1.4,
        "foundation_pseudoweeks": 1.1,
    }
    for side, variant in (("sim", "few"), ("bdbv", "as_is")):
        d = tmp_path / side / variant
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(_summary(scores)))
    out = tmp_path / "report.html"
    ea.report(
        [tmp_path / "sim" / "few", tmp_path / "bdbv" / "as_is"],
        out,
        checkpoint="slots-c44bbe122e6c",
    )
    html = out.read_text()
    assert "BDBV releases" in html
    assert "BDBV hub data" in html
    assert "Pseudo weeks" in html
    for v in ("3.70", "1.10", "1.40"):
        assert v in html
    assert "slots-c44bbe122e6c" in html
