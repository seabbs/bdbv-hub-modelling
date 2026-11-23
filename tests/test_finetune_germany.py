"""The German real-data training blocks (the real-data ablation)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.finetune import germany as fg
from nowcastfoundation.finetune.stream import BLOCK_KEYS


def _wide(days: int = 160) -> pd.DataFrame:
    """A hub triangle: two states and Germany's two age groups."""
    rows = []
    places = [
        ("DE-BW", "00+"),
        ("DE-BY", "00+"),
        ("DE", "00-04"),
        ("DE", "05-14"),
    ]
    for loc, age in places:
        for date in pd.date_range("2021-03-01", periods=days):
            row = {"date": date.date().isoformat(), "location": loc}
            row["age_group"] = age
            row |= {
                f"value_{d}d": 1.0 if d in (0, 3, 9) else 0.0 for d in range(81)
            }
            row["value_>80d"] = 5.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_window_ends_before_the_scored_half():
    # Trained only on reports before the first scored German date, with
    # every horizon's truth as complete as the benchmark's truth.
    as_of, truth_until = fg.window("2021-11-22", 40)
    assert as_of == "2021-11-21"
    assert truth_until == "2021-10-12"
    cuts = fg.cut_dates("2021-04-20", truth_until)
    last = dt.date.fromisoformat(cuts[-1]) + dt.timedelta(weeks=4)
    assert last.isoformat() <= truth_until < "2021-11-22"


def test_counts_are_the_changes_as_reported_by_a_date():
    counts = fg.german_counts(_wide(), "2021-03-05")
    assert set(counts["location"]) == {"CD-BW", "CD-BY", "CD-A0", "CD-A1"}
    assert counts["report_date"].max() <= "2021-03-05"
    # A change is reported on its date plus its delay; after 80 days left out.
    first = counts.loc[
        (counts["reference_date"] == "2021-03-01") & (counts["count"] > 0)
    ]
    assert set(first["report_date"]) <= {"2021-03-01", "2021-03-04"}


@pytest.mark.slow
def test_german_blocks_are_training_blocks_the_trainer_reads():
    wide = _wide()
    counts = fg.german_counts(wide, "2021-07-31")
    blocks = fg.german_blocks(
        counts, ["2021-06-01", "2021-06-02"], "2021-07-31"
    )
    assert blocks
    for b in blocks:
        assert set(BLOCK_KEYS) <= set(b)
        assert b["source"] == fg.SOURCE
        assert b["mix"] == "germany"
        assert b["parent"] == "exact"
        assert b["places"][0] == "CD"
        assert len(b["target_places"]) == b["n_targets"]
        assert np.isfinite(b["context"]).any()
    with pytest.raises(ValueError, match="truth"):
        fg.german_blocks(counts, ["2021-07-20"], "2021-07-31")
