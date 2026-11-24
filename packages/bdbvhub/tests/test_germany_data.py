"""The German hospitalisation nowcasting hub data as forecast tasks."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcasteval.backtest import run_forecasts
from nowcastgermany.data import (
    HORIZONS,
    hub_changes,
    hub_forecasts,
    hub_target,
    hub_triangle,
    level,
    nowcast_tasks,
    place_code,
    read_triangle,
)
from nowcastgermany.pairing import hub_weeks
from nowcastutils.base import NATIONAL

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)

NOWCAST = "2021-10-20"
"""A Wednesday, the last report date of the fixture."""


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def tasks(triangle: pd.DataFrame):
    return nowcast_tasks(triangle, NOWCAST, n_samples=20, seed=1)


def test_hub_triangle_accumulates_changes_by_report_date(
    tmp_path: Path,
) -> None:
    nan = float("nan")
    # The second Meldedatum has two days reported when the file was made.
    values = {f"value_{d}d": [0.0, nan] for d in range(81)}
    values["value_0d"] = [5.0, 3.0]
    values["value_1d"] = [0.0, 0.0]
    values["value_2d"] = [-1.0, nan]
    values["value_>80d"] = [4.0, nan]
    wide = pd.DataFrame(
        {
            "date": ["2021-11-01", "2021-11-02"],
            "location": ["DE", "DE-BY"],
            "age_group": ["00+", "00+"],
            **values,
        }
    )
    path = tmp_path / "triangle.csv"
    wide.to_csv(path, index=False)
    long = hub_triangle(path).set_index(
        ["location", "reference_date", "report_date"]
    )["confirm"]
    first = long.loc[("DE", "2021-11-01")]
    assert first.loc["2021-11-01"] == 5
    assert first.loc["2021-11-03"] == 4
    # Changes after 80 days are put at 81 days, as epinowcast does.
    assert first.loc["2022-01-21"] == 8
    assert len(first) == 82
    # Missing changes are report dates after the file was made.
    assert list(long.loc[("DE-BY", "2021-11-02")]) == [3, 3]


def test_states_and_age_groups_are_separate_tasks_summing_to_germany(
    tasks,
) -> None:
    states, ages = tasks
    assert states.locations == (place_code("DE-BE"), place_code("DE-BY"))
    assert ages.locations == tuple(
        place_code("DE", a)
        for a in ("00-04", "05-14", "15-34", "35-59", "60-79", "80+")
    )
    for task in tasks:
        places = set(task.daily_history["location"])
        assert places == {NATIONAL, *task.locations}


def test_age_groups_are_germany_s_only(triangle: pd.DataFrame) -> None:
    # The hub's triangle also has age groups for each state, which must
    # not add to Germany's.
    state_ages = triangle.loc[
        (triangle["location"] == "DE") & (triangle["age_group"] != "00+")
    ].assign(location="DE-BY")
    both = pd.concat([triangle, state_ages], ignore_index=True)
    want = nowcast_tasks(triangle, NOWCAST, n_samples=20, seed=1)[1]
    got = nowcast_tasks(both, NOWCAST, n_samples=20, seed=1)[1]
    pd.testing.assert_frame_equal(got.daily_history, want.daily_history)


def test_each_release_carries_every_meldedatum_reported_by_then(
    triangle: pd.DataFrame,
) -> None:
    # The hub's triangle stops a Meldedatum's rows 81 days on, so a
    # release must carry the last count of older dates forward rather than
    # drop them.
    delay = pd.to_datetime(triangle["report_date"]) - pd.to_datetime(
        triangle["reference_date"]
    )
    short = triangle.loc[delay.dt.days <= 10]
    states = nowcast_tasks(short, NOWCAST, n_samples=20, seed=1)[0]
    daily = states.daily_history
    latest = daily.loc[daily["vintage"] == daily["vintage"].max()]
    first = short.loc[
        (short["location"] == "DE") & (short["age_group"] == "00+")
    ].sort_values("report_date")
    want = first.groupby("reference_date")["confirm"].last()
    got = latest.loc[latest["location"] == NATIONAL].set_index("onset_date")
    shift = states.reference_date - dt.date.fromisoformat(NOWCAST)
    moved = {
        (dt.date.fromisoformat(d) + shift).isoformat(): v
        for d, v in want.items()
        if v != 0
    }
    assert got["observed"].to_dict() == moved


def test_a_count_corrected_to_zero_is_not_carried(
    triangle: pd.DataFrame,
) -> None:
    rows = triangle.loc[
        (triangle["location"] == "DE")
        & (triangle["age_group"] == "00+")
        & (triangle["reference_date"] == "2021-09-01")
    ].sort_values("report_date")
    zero = rows.iloc[[-1]].assign(report_date="2021-10-01", confirm=0)
    keep = ~(
        (triangle["location"] == "DE")
        & (triangle["age_group"] == "00+")
        & (triangle["reference_date"] == "2021-09-01")
        & (triangle["report_date"] >= "2021-10-01")
    )
    both = pd.concat([triangle.loc[keep], zero], ignore_index=True)
    daily = nowcast_tasks(both, NOWCAST, n_samples=20, seed=1)[0].daily_history
    late = daily.loc[
        (daily["location"] == NATIONAL)
        & (daily["onset_date"] == "2021-09-01")
        & (daily["vintage"] >= "2021-10-01")
    ]
    assert late.empty


def test_tasks_keep_real_dates(tasks) -> None:
    for task in tasks:
        # 2021-10-20 is a Wednesday; nothing moves to a Saturday.
        assert task.reference_date == dt.date(2021, 10, 20)
        assert task.cutoff == NOWCAST
        # Every horizon's day is a target date of the one task.
        assert task.target_end_dates == tuple(
            (task.reference_date + dt.timedelta(days=h)).isoformat()
            for h in sorted(HORIZONS)
        )
        assert task.daily_history["vintage"].max() == NOWCAST


def test_weekly_models_forecast_the_sums_that_end_hub_weeks(tasks) -> None:
    for task in tasks:
        weekly = hub_weeks(task)
        # The Saturdays of the hub's days back from Wednesday 2021-10-20.
        assert weekly.target_end_dates == tuple(
            (task.reference_date + dt.timedelta(days=h)).isoformat()
            for h in (-25, -18, -11, -4)
        )
        assert weekly.daily_history is task.daily_history


def test_a_nowcast_date_has_one_states_and_one_age_groups_task(
    tasks,
) -> None:
    # One forecast with one context end gives every 7-day sum.
    assert len(tasks) == 2
    assert tasks[0].daily_history["vintage"].max() == tasks[0].cutoff
    assert tasks[0].cutoff == tasks[1].cutoff


def test_releases_are_the_counts_reported_by_each_report_date(
    triangle: pd.DataFrame, tasks
) -> None:
    states = tasks[0]
    daily = states.daily_history.set_index(
        ["vintage", "location", "onset_date"]
    )
    # Meldedatum 2021-09-01 as reported by 2021-09-10.
    want = triangle.loc[
        (triangle["location"] == "DE-BY")
        & (triangle["age_group"] == "00+")
        & (triangle["reference_date"] == "2021-09-01")
        & (triangle["report_date"] == "2021-09-10"),
        "confirm",
    ].item()
    got = daily.loc[("2021-09-10", place_code("DE-BY"), "2021-09-01")]
    assert got["observed"] == want
    assert got["imputed"] == 0
    assert got["cutoff"] == "2021-09-10"


def test_weekly_target_sums_each_release_to_hub_weeks(tasks) -> None:
    states = tasks[0]
    daily = states.daily_history
    week = daily.loc[
        (daily["vintage"] == states.vintage)
        & (daily["location"] == NATIONAL)
        & (daily["onset_date"] > "2021-10-16")
    ]
    target = states.target.set_index(["location", "target_end_date"])
    assert target.loc[(NATIONAL, "2021-10-23"), "total"] == (
        week["observed"].sum()
    )
    assert set(states.target_history["vintage"]) == set(daily["vintage"])


@pytest.mark.slow
def test_models_forecast_the_tasks_and_map_back(tasks) -> None:
    # A model of weekly steps forecasts the hub weeks of each task.
    model = get_model("randomwalk")
    out = pd.concat(
        hub_forecasts(run_forecasts([model], [w]), w, NOWCAST)
        for w in map(hub_weeks, tasks)
    )
    assert set(out["horizon"]) == {-4, -11, -18, -25}
    assert set(out["forecast_date"]) == {NOWCAST}
    assert (
        pd.to_datetime(out["target_end_date"])
        - pd.to_datetime(out["forecast_date"])
    ).dt.days.equals(out["horizon"])
    assert set(out["location"]) == {
        "DE",
        "DE-BE",
        "DE-BY",
        "00-04",
        "05-14",
        "15-34",
        "35-59",
        "60-79",
        "80+",
    }
    # Germany comes from the states task, the sum of its states.
    assert out.groupby(["location", "horizon"]).size().eq(20).all()


def test_hub_target_is_each_7_day_sum_as_each_release_reports_it(
    triangle: pd.DataFrame,
) -> None:
    out = hub_target(triangle, ["2021-10-01"]).set_index(
        ["location", "vintage"]
    )
    rows = triangle.loc[
        (triangle["location"] == "DE-BY")
        & (triangle["reference_date"].between("2021-09-25", "2021-10-01"))
        & (triangle["report_date"] <= "2021-10-05")
    ]
    want = rows.groupby("reference_date")["confirm"].last().sum()
    got = out.loc[("DE-BY", "2021-10-05")]
    assert got["total"] == want
    assert got["cutoff"] == "2021-10-05"
    assert got["target_end_date"] == "2021-10-01"
    assert out.index.get_level_values("vintage").min() == "2021-10-01"
    assert set(out.index.get_level_values("location")) == {
        "DE", "DE-BE", "DE-BY", "00-04", "05-14", "15-34", "35-59",
        "60-79", "80+",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("place", "want"),
    [("DE", "national"), ("DE-BY", "states"), ("80+", "age")],
)
def test_levels_are_the_paper_s(place: str, want: str) -> None:
    assert level(place) == want


def test_tasks_take_the_horizons_asked_for(triangle: pd.DataFrame) -> None:
    # Days after the nowcast date are forecasts, in the same task.
    horizons = tuple(range(28, -29, -1))
    states = nowcast_tasks(
        triangle, NOWCAST, n_samples=20, seed=1, horizons=horizons
    )[0]
    ref = states.reference_date
    got = [
        (dt.date.fromisoformat(w) - ref).days for w in states.target_end_dates
    ]
    assert got == sorted(horizons)


def test_within_delay_keeps_the_counts_reported_by_that_delay() -> None:
    from nowcastgermany.data import within_delay

    # One day's count, reported 5 by day 0, 8 by day 40 and 10 by day 41.
    tri = pd.DataFrame(
        {
            "reference_date": ["2021-01-01"] * 3,
            "location": ["DE"] * 3,
            "age_group": ["00+"] * 3,
            "confirm": [5, 8, 10],
            "report_date": ["2021-01-01", "2021-02-10", "2021-02-11"],
        }
    )
    cut = within_delay(tri, 40)
    assert cut["confirm"].tolist() == [5, 8]
    assert within_delay(tri, 41).equals(tri)


def test_hub_changes_put_the_tail_at_81_days_and_drop_unreported() -> None:
    wide = pd.DataFrame(
        {
            "date": ["2021-05-01"],
            "location": ["DE"],
            "age_group": ["00+"],
            **{f"value_{d}d": [1.0] for d in range(81)},
            "value_>80d": [2.0],
        }
    )
    wide.loc[0, "value_80d"] = None
    long = hub_changes(wide).set_index("delay")["change"]
    assert long.loc[81] == 2.0
    assert 80 not in long.index
    assert len(long) == 81


def test_daily_truth_is_each_day_s_count_reported_by_the_delay(
    triangle: pd.DataFrame,
) -> None:
    # The daily view's truth: a day's count as reported `delay` days after
    # it, straight from the triangle's rows; none where that report is not
    # in the data yet.
    from nowcastgermany.evaluation import daily_truth

    days = ["2021-09-01", "2021-09-20", "2021-10-15"]
    got = daily_truth(triangle, days, 10)
    for place, age in [("DE", "00+"), ("DE-BY", "00+"), ("DE", "80+")]:
        name = age if age != "00+" else place
        for day in days[:2]:
            by = (
                dt.date.fromisoformat(day) + dt.timedelta(days=10)
            ).isoformat()
            raw = triangle.loc[
                (triangle["location"] == place)
                & (triangle["age_group"] == age)
                & (triangle["reference_date"] == day)
                & (triangle["report_date"] <= by)
            ].sort_values("report_date")
            row = got.loc[
                (got["location"] == name) & (got["target_end_date"] == day)
            ]
            assert row["truth"].tolist() == [float(raw["confirm"].iloc[-1])]
            assert row["truth_vintage"].tolist() == [by]
    # 2021-10-15 + 10 days is after the last report date.
    assert got.loc[got["target_end_date"] == days[2], "truth"].isna().all()


def test_german_scoring_has_no_weekly_step_special_case() -> None:
    # Every German model forecasts every day of the one task; only the
    # pairing evidence, whose walks have weekly steps, keeps hub weeks.
    from nowcastgermany import data, evaluation, pairing

    assert not hasattr(data, "weekly_task")
    assert not hasattr(evaluation, "_own")
    assert callable(pairing.hub_weeks)
