"""Weekly pseudo releases from the latest release and the delay line."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastutils.releases import pseudo_cutoffs, pseudo_releases


def _day(s: str, k: int = 0) -> str:
    return (dt.date.fromisoformat(s) + dt.timedelta(days=k)).isoformat()


def _release(
    cutoff: str, place: str = "CD-AA", days: int = 60, n: float = 10
) -> pd.DataFrame:
    """One release: `n` cases on each of `days` onset days to `cutoff`."""
    return pd.DataFrame(
        {
            "vintage": cutoff,
            "location": place,
            "onset_date": [_day(cutoff, -k) for k in range(days)],
            "observed": n,
            "imputed": 0.0,
            "cutoff": cutoff,
        }
    )


def _delays(
    delays: list[int], weeks: list[str], place: str = "CD-AA"
) -> pd.DataFrame:
    """A delay line: each of `delays` once in each onset week."""
    return pd.DataFrame(
        [
            {"location": place, "onset_week": w, "delay": d}
            for w in weeks
            for d in delays
        ]
    )


WEEKS = [_day("2026-01-03", 7 * k) for k in range(20)]


def test_cutoffs_fill_gaps_weekly_and_go_back_to_the_first_onset() -> None:
    real = ["2026-02-01", "2026-03-22", "2026-03-29", "2026-04-07"]
    got = pseudo_cutoffs(real, "2026-01-10")
    # Back from the first release to the first onset; every 7 days back
    # from a release across a gap while a week remains after the release
    # before; none between releases 7 or 9 days apart.
    assert got == [
        "2026-01-11",
        "2026-01-18",
        "2026-01-25",
        "2026-02-08",
        "2026-02-15",
        "2026-02-22",
        "2026-03-01",
        "2026-03-08",
        "2026-03-15",
    ]


def test_no_delays_leave_the_releases_as_they_are() -> None:
    rows = _release("2026-03-01")
    for delays in (None, _delays([], WEEKS)):
        out = pseudo_releases(rows, delays)
        pd.testing.assert_frame_equal(out, rows)


def test_reported_on_onset_gives_each_pseudo_release_every_case() -> None:
    rows = _release("2026-03-01")
    out = pseudo_releases(rows, _delays([0], WEEKS))
    real = out.loc[out["vintage"] == "2026-03-01"]
    pd.testing.assert_frame_equal(real.reset_index(drop=True), rows)
    pseudo = out.loc[out["vintage"] != "2026-03-01"]
    assert sorted(pseudo["vintage"].unique())[-1] == "2026-02-22"
    assert (pseudo["cutoff"] == pseudo["vintage"]).all()
    assert (pseudo["onset_date"] <= pseudo["cutoff"]).all()
    assert np.allclose(pseudo["observed"], 10)
    # Each pseudo release has every onset day to its cut-off.
    at = pseudo.loc[pseudo["vintage"] == "2026-02-22"]
    assert at["onset_date"].max() == "2026-02-22"
    assert len(at) == len(rows.loc[rows["onset_date"] <= "2026-02-22"])


def test_counts_are_the_share_reported_by_each_pseudo_cutoff() -> None:
    rows = _release("2026-03-01")
    # Half the cases reported on onset, half ten days after.
    out = pseudo_releases(rows, _delays([0, 10], WEEKS))
    at = out.loc[out["vintage"] == "2026-02-22"].set_index("onset_date")
    age = (
        pd.Timestamp("2026-02-22") - pd.to_datetime(at.index)
    ).days.to_numpy()
    want = np.where(age >= 10, 10.0, 5.0)
    assert np.allclose(at["observed"].to_numpy(), want)


def test_imputed_onsets_are_scaled_as_observed_ones() -> None:
    rows = _release("2026-03-01").assign(imputed=4.0)
    out = pseudo_releases(rows, _delays([0, 10], WEEKS))
    at = out.loc[out["vintage"] == "2026-02-22"]
    assert np.allclose(at["imputed"] / at["observed"], 0.4)


def test_negative_delays_count_as_reported_on_onset() -> None:
    rows = _release("2026-03-01")
    out = pseudo_releases(rows, _delays([-3], WEEKS))
    pseudo = out.loc[out["vintage"] != "2026-03-01"]
    assert np.allclose(pseudo["observed"], 10)


def test_only_the_latest_release_is_scaled() -> None:
    early = _release("2026-01-15", n=1)
    late = _release("2026-03-01", n=10)
    rows = pd.concat([early, late], ignore_index=True)
    out = pseudo_releases(rows, _delays([0], WEEKS))
    pd.testing.assert_frame_equal(
        out.loc[out["vintage"] == "2026-01-15"].reset_index(drop=True), early
    )
    gap = out.loc[out["vintage"].between("2026-01-16", "2026-02-28")]
    assert sorted(gap["vintage"].unique()) == [
        "2026-01-25",
        "2026-02-01",
        "2026-02-08",
        "2026-02-15",
        "2026-02-22",
    ]
    assert np.allclose(gap["observed"], 10)


def test_places_without_delays_take_their_parent_or_all() -> None:
    rows = pd.concat(
        [
            _release("2026-03-01", "CD-AA"),
            _release("2026-03-01", "CD-BB"),
            _release("2026-03-01", "CD-AA-01"),
            _release("2026-03-01", "CD"),
        ],
        ignore_index=True,
    )
    delays = pd.concat(
        [_delays([0], WEEKS, "CD-AA"), _delays([100], WEEKS, "CD-BB")]
    )
    out = pseudo_releases(rows, delays)
    at = out.loc[out["vintage"] == "2026-02-22"].groupby("location")[
        "observed"
    ]
    young = at.min()
    # The zone takes its province's delays; the country every place's.
    assert young["CD-AA-01"] == pytest.approx(10)
    # No case of CD-BB is reported within 100 days, so it has no rows.
    assert "CD-BB" not in young.index
    assert young["CD"] == pytest.approx(5)


def test_an_onset_week_without_delays_takes_the_places_others() -> None:
    rows = _release("2026-03-01")
    # Delays only for the first weeks, all reported on onset.
    out = pseudo_releases(rows, _delays([0], WEEKS[:3]))
    pseudo = out.loc[out["vintage"] != "2026-03-01"]
    assert np.allclose(pseudo["observed"], 10)


def test_zero_counts_are_left_out() -> None:
    rows = _release("2026-03-01")
    out = pseudo_releases(rows, _delays([30], WEEKS))
    assert (out["observed"] + out["imputed"] > 0).all()
