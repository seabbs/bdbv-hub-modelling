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
    many = _weekly_delays(lambda s, i: (i + len(s)) % 5, ["P01-A1", "P02"])
    json.dumps(ds.structure(many, parent=ds.sim_parent), allow_nan=False)
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


def _weekly_delays(delay_of, strata, days=84, count=lambda _s, _i: 20):
    """Rows where stratum `s`'s reference day `i` is all reported after
    `delay_of(s, i)` days, `count(s, i)` cases."""
    return pd.concat(
        [
            _rows(
                lambda i, s=s: [(delay_of(s, i), count(s, i))],
                days=days,
                stratum=s,
            )
            for s in strata
        ],
        ignore_index=True,
    )


def _wobble(i):
    """A delay that moves week to week: 1, 4, 2, 5, 3, 6, ... days."""
    return (1, 4, 2, 5, 3, 6)[(i // 7) % 6]


def test_shared_delay_changes_move_strata_together():
    rows = _weekly_delays(lambda _s, i: _wobble(i), ["A", "B", "C"])
    got = ds.structure(rows)
    by = got["by_stratum"]
    assert [x["name"] for x in by["strata"]] == ["A", "B", "C"]
    assert by["corr"] == pytest.approx(1.0)
    assert by["corr_min"] == pytest.approx(1.0)
    # One line per stratum, week by week, as the total.
    assert len(by["strata"][0]["mean"]) == len(by["week"])
    assert by["strata"][0]["mean"][:3] == pytest.approx([1.0, 4.0, 2.0])
    assert by["total"]["mean"][:3] == pytest.approx([1.0, 4.0, 2.0])


def test_a_stratum_whose_delay_lengthens_diverges_from_the_others():
    def delay(s, i):
        return 1 + i // 7 if s == "C" else _wobble(i)

    got = ds.structure(_weekly_delays(delay, ["A", "B", "C"]))
    by = {x["name"]: x for x in got["by_stratum"]["strata"]}
    # C's line rises week on week; A and B wobble together.
    assert by["C"]["mean"][-1] > by["C"]["mean"][0] + 5
    assert by["C"]["corr"] < by["A"]["corr"]
    sc = ds.scalars(got)
    assert sc["strata_corr"] < 0.9
    assert sc["strata_corr_min"] == pytest.approx(by["C"]["corr"])


def _counts(_s, i):
    """Weekly counts rising and falling: 10 to 60 a day."""
    return 10 * (1 + (1, 3, 5, 2, 4, 6)[(i // 7) % 6])


def test_delays_that_lengthen_with_the_counts_give_a_positive_slope():
    def delay(s, i):
        return _counts(s, i) // 10 + (2 if s == "B" else 0)

    got = ds.structure(_weekly_delays(delay, ["A", "B"], count=_counts))
    assert got["coupling"]["slope"] > 0
    assert got["coupling"]["corr"] == pytest.approx(1.0, abs=0.1)
    line = got["by_stratum"]["strata"][0]
    assert line["counts_slope"] > 0
    # The log count a day in each week.
    assert line["log_count"][0] == pytest.approx(np.log(20), abs=1e-3)
    sc = ds.scalars(got)
    assert sc["counts_slope"] > 0


def test_delays_that_shorten_with_the_counts_give_a_negative_slope():
    def delay(s, i):
        return 8 - _counts(s, i) // 10

    got = ds.structure(_weekly_delays(delay, ["A", "B"], count=_counts))
    assert got["coupling"]["slope"] < 0
    assert got["coupling"]["corr"] < -0.9
    # Reported by day 1 only in the busiest weeks (7 days' delay or less
    # elsewhere is 2 days or more): the share by day 1 rises with them.
    assert ds.scalars(got)["share1_counts_corr"] > 0


def test_an_over_reporting_burst_is_withdrawn_within_days():
    def spec(i):
        # Days 21 to 27: 50 first reported, 40 withdrawn 3 days later.
        if 21 <= i < 28:
            return [(0, 50), (3, -40)]
        return [(0, 10)]

    burst = _rows(spec, days=56, stratum="A")
    calm = _rows(lambda _: [(0, 10)], days=56, stratum="B")
    got = ds.structure(pd.concat([burst, calm], ignore_index=True))
    by = {x["name"]: x for x in got["by_stratum"]["strata"]}
    # The burst's week first reports 5 times its final count.
    assert by["A"]["first_ratio"][3] == pytest.approx(5.0)
    assert by["A"]["first_ratio"][0] == pytest.approx(1.0)
    assert by["B"]["first_ratio"] == pytest.approx([1.0] * 8)
    # 7 * 40 of A's 56 * 10 final cases were withdrawn.
    assert by["A"]["withdrawn"] == pytest.approx(0.5)
    assert by["B"]["withdrawn"] == 0
    sc = ds.scalars(got)
    assert sc["burst_ratio"] == pytest.approx(5.0)
    assert sc["withdrawn_share"] == pytest.approx(0.25)
    assert sc["withdrawn_max"] == pytest.approx(0.5)


def test_the_first_report_ratio_counts_revisions_as_reported():
    # Without a burst the first report is the whole count: the ratio is
    # 1 even though the netted delays drop the withdrawn 2.
    rows = _rows(lambda _: [(0, 12), (2, -2)])
    got = ds.structure(rows)
    assert got["by_stratum"]["total"]["first_ratio"][0] == pytest.approx(1.2)
    assert got["by_stratum"]["total"]["share1"][0] == pytest.approx(1.2)


def test_a_pure_time_pattern_is_the_time_part_of_the_noise():
    got = ds.structure(_weekly_delays(lambda _s, i: _wobble(i), "ABCD"))
    noise = got["noise"]
    assert noise["time_share"] == pytest.approx(1.0)
    assert noise["strata_share"] == pytest.approx(0.0, abs=1e-6)
    assert noise["residual_share"] == pytest.approx(0.0, abs=1e-6)
    assert ds.scalars(got)["noise_time"] == pytest.approx(1.0)


def test_fixed_stratum_delays_are_the_strata_part_of_the_noise():
    offset = {"A": 1, "B": 3, "C": 6, "D": 10}
    got = ds.structure(_weekly_delays(lambda s, _i: offset[s], "ABCD"))
    noise = got["noise"]
    assert noise["strata_share"] == pytest.approx(1.0)
    assert noise["time_share"] == pytest.approx(0.0, abs=1e-6)


def test_strata_moving_on_their_own_are_the_residual_part():
    # Each stratum's own week-to-week pattern, none shared, none fixed.
    pats = {"A": (1, 5, 3, 1, 5, 3), "B": (5, 3, 1, 5, 3, 1),
            "C": (3, 1, 5, 3, 1, 5)}  # fmt: skip

    def delay(s, i):
        return pats[s][(i // 7) % 6]

    noise = ds.structure(_weekly_delays(delay, "ABC"))["noise"]
    assert noise["residual_share"] > 0.9
    assert noise["residual_sd"] > 0


def test_one_stratum_has_only_a_time_part():
    noise = ds.structure(_weekly_delays(lambda _s, i: _wobble(i), "A"))["noise"]
    assert noise["time_sd"] > 0
    assert noise["strata_share"] is None


def test_the_drawn_noise_sds_by_layer():
    got = ds.drawn_noise(
        {
            "delay_jitter": "yes",
            "delay_jitter_sd": "0.12",
            "delay_strata_noise": "strata",
            "delay_strata_sd": "0.3",
            "delay_strata_jitter_sd": "0.05",
        }
    )
    assert got == {"time": 0.12, "strata": 0.3, "strata_time": 0.0}
    both = ds.drawn_noise(
        {
            "delay_jitter": "no",
            "delay_jitter_sd": "0.12",
            "delay_strata_noise": "both",
            "delay_strata_sd": "0.3",
            "delay_strata_jitter_sd": "0.05",
        }
    )
    assert both == {"time": 0.0, "strata": 0.3, "strata_time": 0.05}
    assert ds.drawn_noise({}) == {
        "time": None,
        "strata": None,
        "strata_time": None,
    }


def test_each_stratum_s_weekly_counts_over_its_mean():
    def count(s, i):
        return 10 * (1 + (i // 7) % 3) * (2 if s == "B" else 1)

    got = ds.structure(_weekly_delays(lambda _s, _i: 1, "AB", count=count))
    by = {x["name"]: x for x in got["by_stratum"]["strata"]}
    # The same shape at twice the size: the same line over its mean.
    assert by["A"]["rel_count"][:3] == pytest.approx([0.5, 1.0, 1.5])
    assert by["B"]["rel_count"] == pytest.approx(by["A"]["rel_count"])
    assert np.mean(by["A"]["rel_count"]) == pytest.approx(1.0, abs=1e-3)
    assert got["by_stratum"]["total"]["rel_count"][:3] == pytest.approx(
        [0.5, 1.0, 1.5]
    )
    # Strata that move together keep their shares.
    assert got["levels"]["corr"] == pytest.approx(1.0)
    assert got["levels"]["share_sd"] == pytest.approx(0.0, abs=1e-6)


def test_a_stratum_whose_counts_move_on_their_own_moves_its_share():
    def count(s, i):
        # C's ascertainment doubles half way; A and B stay put.
        return 20 * (2 if s == "C" and i >= 42 else 1) + 5 * ((i // 7) % 2)

    got = ds.structure(_weekly_delays(lambda _s, _i: 1, "ABC", count=count))
    by = {x["name"]: x for x in got["by_stratum"]["strata"]}
    # Shares are of the strata's total, so A's and B's move a little too.
    assert by["C"]["count_share_sd"] > by["A"]["count_share_sd"]
    assert by["C"]["rel_count"][-1] > by["C"]["rel_count"][0]
    sc = ds.scalars(got)
    assert sc["count_share_sd"] > 0.1
    assert sc["strata_count_corr"] < 0.9


def test_the_drawn_ascertainment_structure():
    got = ds.drawn_ascertainment(
        {
            "ascertainment_change": "drift",
            "ascertainment_change_strata": "related",
            "ascertainment_counts": "strata",
            "asc_counts_sign": "-1;-1;1",
            "asc_counts_slope": "0.2",
            "asc_noise_sd": "",
            "delay_change": "step",
        }
    )
    assert got == {
        "ascertainment_change": "drift",
        "ascertainment_change_strata": "related",
        "ascertainment_counts": "strata",
        "asc_counts_sign": "-1;-1;1",
        "asc_counts_slope": "0.2",
    }
    assert ds.drawn_ascertainment({}) == {}
