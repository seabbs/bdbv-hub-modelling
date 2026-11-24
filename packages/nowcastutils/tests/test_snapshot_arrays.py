"""Release snapshots on the onset clock."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastutils.releases import release_cutoff
from nowcastutils.snapshots import blocks, snapshot_arrays, weekly_sums

VINTAGES = ("2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24")


DATA = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "data"


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(DATA / "daily.csv")


@pytest.fixture(scope="module")
def snap(daily: pd.DataFrame):
    return snapshot_arrays(daily, "2026-10-24", ("CD-IT", "CD-NK"))


def test_cutoff_is_the_release_s_latest_onset_day(daily: pd.DataFrame) -> None:
    cuts = [release_cutoff(daily, v).isoformat() for v in VINTAGES]
    assert cuts == ["2026-08-18", "2026-10-07", "2026-10-16", "2026-10-24"]


def test_ages_count_from_the_cutoff_not_the_release_date(snap) -> None:
    days = snap.days
    second = snap.releases.index("2026-10-09")
    assert days[snap.cut[second]].date() == dt.date(2026, 10, 7)
    # Age 0 at the second release is 10-07, not 10-09.
    assert snap.cut[second] - np.flatnonzero(days == "2026-10-07")[0] == 0


def test_cells_after_the_cutoff_are_unobserved(snap) -> None:
    for r in range(len(snap.releases)):
        assert snap.observed[r].sum() == snap.cut[r] + 1
        assert (snap.counts[:, r][:, ~snap.observed[r]] == 0).all()


def test_one_cutoff_for_every_location(daily: pd.DataFrame) -> None:
    quiet = daily.loc[
        ~((daily["location"] == "CD-HU") & (daily["vintage"] == "2026-10-24"))
    ]
    a = snapshot_arrays(quiet, "2026-10-24", ("CD-HU",))
    b = snapshot_arrays(daily, "2026-10-24", ("CD-HU",))
    assert (a.cut == b.cut).all()
    assert a.observed.shape == (4, len(a.days))


def test_national_is_the_sum_of_provinces(daily: pd.DataFrame) -> None:
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-10-24", provinces)
    kids = snap.parent == 0
    assert np.allclose(snap.counts[0], snap.counts[kids].sum(axis=0))


def test_hierarchy_and_node_order(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-24", ("CD-IT", "CD-IT-01"))
    assert snap.locations == ("CD", "CD-IT", "CD-IT-01")
    assert snap.parent.tolist() == [-1, 0, 1]


def test_releases_after_the_vintage_are_not_used(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT",))
    assert snap.releases == VINTAGES[:3]
    with pytest.raises(ValueError, match="No daily releases"):
        snapshot_arrays(daily, "2026-01-01", ("CD-IT",))


def test_weeks_run_sunday_to_saturday(snap) -> None:
    ends = pd.to_datetime(list(snap.week_ends))
    assert (ends.day_of_week == 5).all()
    assert snap.week_ends[-1] == "2026-10-24"
    last = snap.releases.index("2026-10-24")
    weekly = weekly_sums(snap, last)
    assert weekly.shape == (len(snap.locations), snap.n_weeks)
    release = pd.read_csv(DATA / "target.csv")
    rows = release.loc[
        (release["vintage"] == "2026-10-24")
        & (release["location"] == "CD")
        & (release["target_end_date"] == "2026-10-17")
    ]
    assert (
        weekly[0, snap.week_ends.index("2026-10-17")] == rows["total"].iloc[0]
    )


def test_seven_day_blocks_ending_saturday_are_the_hub_weeks(snap) -> None:
    again = blocks(snap, 7, 5)
    assert again.week_ends == snap.week_ends
    assert (again.week == snap.week).all()
    assert (again.counts == snap.counts).all()
    assert again.period == 7


def test_blocks_end_on_their_weekday_and_cover_the_cutoff(snap) -> None:
    for end in range(7):
        b = blocks(snap, 7, end)
        ends = pd.to_datetime(list(b.week_ends))
        assert (ends.weekday == end).all()
        assert (np.diff(ends).astype("timedelta64[D]").astype(int) == 7).all()
        # The days are those of the snapshots, then empty to the last end.
        n = snap.days.size
        assert (b.days[:n] == snap.days).all()
        assert b.days[-1] == ends[-1]
        assert (b.counts[:, :, :n] == snap.counts).all()
        assert (b.counts[:, :, n:] == 0).all()
        assert (b.cut == snap.cut).all()
        # Every day is in the block ending on or after it, within 7 days.
        lag = (ends[b.week] - b.days).days
        assert ((lag >= 0) & (lag < 7)).all()


def test_one_day_blocks_are_the_days(snap) -> None:
    b = blocks(snap, 1)
    assert b.n_weeks == b.days.size
    assert (b.week == np.arange(b.days.size)).all()
    assert b.period == 1
