"""The German real-data training blocks (the real-data ablation)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.finetune.stream import BLOCK_KEYS
from nowcastgermany import finetune_blocks as fg


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


def test_the_cutter_imports_without_the_evaluation_module():
    # The package-alone job imports it with only nowcastgermany's own
    # dependencies; the evaluation module reaches bdbvhub.
    import subprocess
    import sys

    code = (
        "import sys, nowcastgermany.finetune_blocks; "
        "print('nowcastgermany.evaluation' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_german_blocks_keep_the_real_dates(monkeypatch):
    # The German tasks score on the real dates, so
    # a block is cut on its own date, not moved to the Saturday after.
    counts = fg.german_counts(_wide(), "2021-07-31")
    seen = []

    def block(own, _source, _view, cutoffs, max_lag):  # noqa: ARG001
        seen.append((own, cutoffs))

    monkeypatch.setattr(fg, "training_block", block)
    fg.german_blocks(counts, ["2021-06-01"], "2021-07-31")
    assert seen
    for own, cutoffs in seen:
        assert cutoffs[-1] == "2021-06-01"  # a Tuesday
        assert own["reference_date"].min() == "2021-03-01"


def _germany_config() -> dict:
    import tomllib
    from pathlib import Path

    root = Path(__file__).parents[3]
    path = root / "packages" / "nowcastgermany" / "config" / "germany.toml"
    return tomllib.loads(path.read_text())


def test_the_realdata_window_is_the_development_half():
    # The real-data checkpoint sees the German data as available on the
    # last day of the development half, with the benchmark's truth delay.
    german = _germany_config()
    window = fg.realdata_window()
    split = dt.date.fromisoformat(german["nowcast"]["split_date"])
    assert window.as_of == (split - dt.timedelta(days=1)).isoformat()
    assert window.as_of == "2022-02-09"
    assert window.truth_delay == german["truth"]["delay"]
    assert window.truth_until == "2021-12-31"
    cuts = window.cuts()
    assert (cuts[0], cuts[-1]) == ("2021-04-20", "2021-12-03")
    assert window.sources == ("daily+rolling7", "weekly")
    assert window.held_out_dates == 14


@pytest.mark.slow
def test_realdata_blocks_never_read_past_the_development_half():
    # Reports after the window's date are dropped before any block is
    # cut, and no block's truth runs past its truth date.
    window = fg.realdata_window()
    counts = fg.german_counts(_wide(400), window.as_of)
    assert counts["report_date"].max() <= window.as_of
    cuts = ["2021-11-30", "2021-12-03"]
    blocks = fg.german_blocks(
        counts, cuts, window.truth_until, sources=window.sources
    )
    assert {b["source"] for b in blocks} == {"daily+rolling7", "weekly"}
    for b in blocks:
        assert b["data"] == "germany"
        assert b["truth_end"] <= window.truth_until < window.as_of
        lags = fg.LAGS if b["source"] == fg.SOURCE else fg.WEEKLY_LAGS
        assert b["lag"] in lags


def test_weekly_blocks_take_weekly_releases_on_the_cut_weekday(monkeypatch):
    counts = fg.german_counts(_wide(), "2021-07-31")
    seen = []

    def block(own, source, _view, cutoffs, max_lag):  # noqa: ARG001
        seen.append((source, cutoffs, max_lag))

    monkeypatch.setattr(fg, "training_block", block)
    fg.german_blocks(counts, ["2021-06-01"], "2021-07-31", sources=("weekly",))
    assert seen
    for source, cutoffs, lag in seen:
        assert source == "weekly"
        assert cutoffs[-1] == "2021-06-01"
        days = {dt.date.fromisoformat(c).weekday() for c in cutoffs}
        assert days == {dt.date(2021, 6, 1).weekday()}
        assert lag in fg.WEEKLY_LAGS
