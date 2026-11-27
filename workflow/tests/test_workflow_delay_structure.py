"""The structure of reporting delays (flowlib.delay_structure)."""

import numpy as np
import pandas as pd
import pytest
from flowlib import delay_structure as ds

START = pd.Timestamp("2026-01-05")  # a Monday


def _rows(spec, days=56, stratum="A"):
    """Counts by reference and report date: `spec(day)` gives the
    (delay, count) pairs of each reference day."""
    out = []
    for i in range(days):
        ref = START + pd.Timedelta(days=i)
        for delay, count in spec(i):
            out.append(
                {
                    "stratum": stratum,
                    "ref": ref,
                    "rep": ref + pd.Timedelta(days=delay),
                    "count": count,
                }
            )
    return pd.DataFrame(out)


def test_a_fixed_delay_is_the_same_by_reference_and_by_release():
    rows = _rows(lambda _: [(2, 10)])
    got = ds.structure(rows)
    assert got["horizon"] == 2
    ref = got["by_ref"]
    assert ref["mean"] == pytest.approx([2.0] * len(ref["mean"]))
    assert ref["q50"] == [2] * len(ref["q50"])
    rel = got["by_rel"]
    assert rel["mean"] == pytest.approx([2.0] * len(rel["mean"]))
    # Weeks start on the first reference date, as dates.
    assert ref["week"][0] == "2026-01-05"
    assert rel["week"][0] == "2026-01-05"


def test_a_delay_that_lengthens_shows_by_reference_week():
    # The delay is 1 day in the first four weeks and 5 after.
    rows = _rows(lambda i: [(1 if i < 28 else 5, 10)])
    got = ds.structure(rows)
    means = got["by_ref"]["mean"]
    assert means[0] == pytest.approx(1.0)
    assert means[-1] == pytest.approx(5.0)
    sc = ds.scalars(got)
    assert sc["delay_trend"] > 3
    assert sc["delay_drift"] > 0.3
    # The time curve is each tenth's mean over the overall mean.
    curve = ds.curves(got)["delay_time"]
    assert len(curve) == ds.TIME_BINS
    assert curve[0] < 1 < curve[-1]


def test_reference_dates_too_late_to_be_complete_are_left_out():
    # Every reference date reported 3 days later; the last report date
    # is the last reference date's, so the last 3 days are incomplete.
    rows = _rows(lambda _: [(0, 5), (3, 5)])
    last = rows["ref"].max()
    rows = rows[rows["rep"] <= last]
    got = ds.structure(rows)
    assert got["horizon"] == 3
    # Complete reference dates only: half on the day, half 3 days later.
    assert got["by_ref"]["mean"][0] == pytest.approx(1.5)
    assert got["by_ref"]["mean"][-1] == pytest.approx(1.5)


def test_weekday_effects_in_delays_and_in_reporting():
    # Reported every Monday only: the delay is the days to the next
    # Monday, so the weekday of reference sets it.
    def spec(i):
        return [((7 - i % 7) % 7, 7)]

    got = ds.structure(_rows(spec))
    dow = got["dow"]
    # Monday's cases wait 0 days, Tuesday's 6: 3 days either side of
    # the mean.
    assert dow["ref_delay"][0] == pytest.approx(-3.0)
    assert dow["ref_delay"][1] == pytest.approx(3.0)
    assert dow["release_days"] == [1.0, 0, 0, 0, 0, 0, 0]
    assert dow["rep_share"][0] == pytest.approx(7.0)
    sc = ds.scalars(got)
    assert sc["dow_delay"] == pytest.approx(6.0)
    assert sc["release_weekdays"] == 1


def test_release_cadence_and_the_wait_for_a_release():
    def spec(i):
        return [((7 - i % 7) % 7, 7)]

    got = ds.structure(_rows(spec))
    rel = got["releases"]
    # A release every 7 days.
    assert rel["gaps"][6] == 1.0
    assert sum(rel["gaps"]) == pytest.approx(1.0)
    assert rel["per_week"] == pytest.approx(1.0, abs=0.2)
    sc = ds.scalars(got)
    # All of the delay is the wait for the next release.
    assert sc["wait"] == pytest.approx(3.0)
    assert sc["wait_share"] == pytest.approx(1.0)


def test_daily_releases_wait_nothing():
    got = ds.structure(_rows(lambda _: [(0, 5), (4, 5)]))
    assert got["releases"]["gaps"][0] == 1.0
    sc = ds.scalars(got)
    assert sc["wait"] == 0
    assert sc["wait_share"] == 0
    assert sc["release_weekdays"] == 7


def test_delays_by_stratum_with_the_parent_and_its_children():
    fast = _rows(lambda _: [(1, 10)], stratum="P01-A1")
    slow = _rows(lambda _: [(5, 10)], stratum="P01-A2")
    other = _rows(lambda _: [(3, 20)], stratum="P02-A1")
    got = ds.structure(pd.concat([fast, slow, other]), parent=ds.sim_parent)
    by = {s["name"]: s for s in got["strata"]}
    assert by["all"]["level"] == "total"
    assert by["all"]["mean"] == pytest.approx(3.0)
    assert by["P01"]["level"] == "parent"
    assert by["P01"]["mean"] == pytest.approx(3.0)
    assert by["P01-A1"]["level"] == "child"
    assert by["P01-A1"]["mean"] == pytest.approx(1.0)
    assert by["P01-A2"]["q50"] == 5
    # Each parent is followed by its children.
    names = [s["name"] for s in got["strata"]]
    assert names.index("P01") < names.index("P01-A1")
    assert names.index("P02") < names.index("P02-A1")
    # The children's means weighted by their counts: 1, 5, 3, 3.
    assert ds.scalars(got)["strata_spread"] == pytest.approx(
        np.sqrt(2), abs=1e-3
    )


def test_one_stratum_has_no_spread():
    got = ds.structure(_rows(lambda _: [(1, 10)]))
    assert [s["level"] for s in got["strata"]] == ["total"]
    assert ds.scalars(got)["strata_spread"] is None


def test_the_sim_parent_of_a_stratum():
    assert ds.sim_parent("P01-A1") == "P01"
    assert ds.sim_parent("P01") is None
    assert ds.sim_parent("A1") is None


def test_negative_updates_are_netted_before_the_summaries():
    # 10 reported on the day, 2 of them withdrawn 2 days later.
    rows = _rows(lambda _: [(0, 10), (2, -2), (1, 2)])
    got = ds.structure(rows)
    # Net by delay: 10 at 0, 2 at 1; the withdrawn 2 at 2 are dropped.
    assert got["by_ref"]["mean"][0] == pytest.approx(2 / 12, abs=1e-3)


def test_a_structure_is_plain_json():
    import json

    got = ds.structure(_rows(lambda i: [(i % 3, 4)]), parent=ds.sim_parent)
    json.dumps(got)
    json.dumps(ds.scalars(got))
    json.dumps(ds.curves(got))


def test_no_counts_have_no_structure():
    empty = pd.DataFrame(columns=["stratum", "ref", "rep", "count"])
    assert ds.structure(empty) is None


def test_the_curves_have_fixed_lengths_for_any_series():
    got = ds.curves(ds.structure(_rows(lambda _: [(0, 5), (7, 5)])))
    assert dict(zip(ds.DELAY_POINTS, got["cdf"], strict=True))[0] == 0.5
    assert dict(zip(ds.DELAY_POINTS, got["cdf"], strict=True))[7] == 1.0
    lengths = {
        "cdf": len(ds.DELAY_POINTS),
        "delay_time": ds.TIME_BINS,
        "ref_delay": 7,
        "refdow": 7,
        "repdow": 7,
        "release_days": 7,
        "gaps": ds.GAP_BINS,
    }
    assert {k: len(v) for k, v in got.items()} == lengths
    assert got["refdow"] == [1.0] * 7
