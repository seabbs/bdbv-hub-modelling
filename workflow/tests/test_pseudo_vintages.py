"""Pseudo vintages from reference/report pairs (pseudo_vintages.py)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import pseudo_vintages as pv  # noqa: E402

LOCATIONS = pd.DataFrame(
    {
        "location": ["CD", "CD-IT", "CD-TO", "CD-TO-15"],
        "location_name": ["DRC", "Ituri", "Tshopo", "Tshopo"],
        "location_type": ["country", "province", "province", "health zone"],
    }
)


def _line() -> pd.DataFrame:
    """Four Ituri cases with onsets in the week ending 5 September."""
    return pd.DataFrame(
        {
            "week_ending": "2026-09-05",
            "location_name": ["Ituri", "Ituri", "Ituri", "Tshopo"],
            "days_sym_to_report": [0, 3, 20, 1],
            "as_of": "2026-11-22",
        }
    )


def test_cases_reported_by_a_cutoff_are_bounded_by_their_onset_week() -> None:
    got = pv.observed_vintage(_line(), "2026-09-05", LOCATIONS)
    it = got.loc[("CD-IT", "2026-09-05")]
    # Delay 0: reported by 5 Sep whatever its onset day. Delay 3: only
    # with an onset by 2 Sep. Delay 20: not before 21 Sep.
    assert (it["lower"], it["upper"]) == (1, 2)
    assert it["even"] == 1 + 4 / 7
    # Tshopo is the province, not its namesake zone.
    assert ("CD-TO", "2026-09-05") in got.index


def test_a_cutoff_after_every_report_counts_every_case() -> None:
    got = pv.observed_vintage(_line(), "2026-10-01", LOCATIONS)
    assert got["lower"].sum() == got["upper"].sum() == 4


def test_missing_onsets_count_exactly_by_report_date() -> None:
    missing = pd.DataFrame(
        {
            "as_of": "2026-11-22",
            "report_date": ["2026-09-01", "2026-09-04", "2026-09-10"],
            "location": "CD-IT",
            "location_type": "province",
            "n_miss_sym_onset": [2, 1, 5],
        }
    )
    got = pv.missing_vintage(missing, "2026-09-05")
    assert got.to_dict() == {"CD-IT": 3}


def test_pseudo_cutoffs_are_weekly_and_the_real_releases() -> None:
    cuts = pv.cutoffs(["2026-08-18", "2026-10-09"], "2026-08-01")
    assert cuts[:3] == ["2026-08-01", "2026-08-08", "2026-08-15"]
    assert "2026-08-18" in cuts
    assert "2026-10-09" in cuts
    assert cuts == sorted(cuts)
