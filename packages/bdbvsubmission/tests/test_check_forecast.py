from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    AS_OF,
    HUB,
    POLICY,
    POLICY_PATH,
    REF,
    make_samples,
    weeks,
    write_samples,
)

from bdbvsubmission.check_forecast import check_plausible, main, run_checks
from bdbvsubmission.common import read_json
from bdbvsubmission.hubdata import Observed, Samples, read_observed

OBS = read_observed(HUB / "time-series.csv", AS_OF)
TASKS = read_json(HUB / "tasks.json")
LOCS = ["CD", "CD-IT", "CD-NK", "CD-HU"]


def statuses(path: Path, **kwargs: Any) -> dict[str, str]:
    checks = run_checks(
        path,
        reference_date=REF,
        settings=POLICY.checks,
        observed=OBS,
        expected_locations=LOCS,
        **{"national_sidecar": _sidecar(), "tasks": TASKS, **kwargs},
    )
    return {c.name: c.status for c in checks}


def sub(tmp_path: Path, **kwargs: Any) -> Path:
    return write_samples(
        tmp_path / f"{REF}-seabbs-renewal.csv",
        make_samples(**kwargs),
    )


def test_plausible_file_passes(tmp_path: Path) -> None:
    s = statuses(sub(tmp_path))
    assert s.pop("baseline") == "skip" and s.pop("previous") == "skip"
    # Zones are required only for baseline, and the file has no wk R_t.
    assert s.pop("zones") == "skip"
    assert s.pop("rt") == "skip"
    assert set(s.values()) == {"pass"}, s
    assert "plausible" in s


def test_negative_and_nan_values_fail(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][6] = -3.0
        rows[1][6] = float("nan")

    assert statuses(sub(tmp_path, mutate=mutate))["values"] == "fail"


def test_unparsable_value_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][6] = "abc"

    assert statuses(sub(tmp_path, mutate=mutate))["values"] == "fail"


def test_wrong_sample_count_fails(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path, n=99))["samples"] == "fail"


def test_ragged_trajectory_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        del rows[3]

    s = statuses(sub(tmp_path, mutate=mutate))
    assert s["samples"] == "fail"


def test_missing_location_and_horizon_warn(tmp_path: Path) -> None:
    last = weeks()[-1]

    def mutate(rows: list[list[Any]]) -> None:
        rows[:] = [r for r in rows if r[2] != last]

    s = statuses(
        sub(tmp_path, levels={"CD-IT": 200.0, "CD-NK": 80.0}, mutate=mutate)
    )
    assert s["coverage"] == "warn"


def test_national_not_sum_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == "CD":
                r[6] += 1
                break

    assert statuses(sub(tmp_path, mutate=mutate))["national_sum"] == "fail"


def _checks(path: Path, **kwargs: Any) -> dict[str, Any]:
    checks = run_checks(
        path,
        reference_date=REF,
        settings=POLICY.checks,
        observed=OBS,
        expected_locations=LOCS,
        **{"national_sidecar": _sidecar(), "tasks": TASKS, **kwargs},
    )
    return {c.name: c for c in checks}


def _sidecar(**kw: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "national": "CD",
        "modelled_provinces": ["CD-HU", "CD-IT", "CD-NK"],
        "unassigned": [],
        **kw,
    }


def _bump_national(week: str, sample: int, by: float) -> Any:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == "CD" and r[2] == week and r[5] == sample:
                r[6] += by

    return mutate


def test_national_exact_with_sidecar(tmp_path: Path) -> None:
    c = _checks(sub(tmp_path), national_sidecar=_sidecar())["national_sum"]
    assert c.status == "pass"
    assert "3 modelled provinces" in c.summary


def test_national_allows_only_float_noise(tmp_path: Path) -> None:
    w = weeks()[-1]
    noisy = sub(tmp_path, mutate=_bump_national(w, 1, 1e-12))
    assert statuses(noisy)["national_sum"] == "pass"
    off = sub(tmp_path, mutate=_bump_national(w, 1, 1e-6))
    assert statuses(off)["national_sum"] == "fail"


def test_national_off_by_one_fails(tmp_path: Path) -> None:
    path = sub(tmp_path, mutate=_bump_national(weeks()[0], 7, 1.0))
    c = _checks(path, national_sidecar=_sidecar())["national_sum"]
    assert c.status == "fail"
    assert "sample 7" in c.summary


def test_national_missing_modelled_province_warns(tmp_path: Path) -> None:
    side = _sidecar(modelled_provinces=["CD-HU", "CD-IT", "CD-NK", "CD-TO"])
    c = _checks(sub(tmp_path), national_sidecar=side)["national_sum"]
    assert c.status == "warn"
    assert "CD-TO" in c.summary


def test_national_unassigned_component_adds_up(tmp_path: Path) -> None:
    w = weeks()[-1]
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    assert statuses(path)["national_sum"] == "fail"
    side = _sidecar(
        unassigned=[{"target_end_date": w, "sample": 3, "value": 4.0}]
    )
    c = _checks(path, national_sidecar=side)["national_sum"]
    assert c.status == "pass"
    assert "unassigned" in c.summary
    short = _sidecar(
        unassigned=[{"target_end_date": w, "sample": 3, "value": 3.0}]
    )
    assert (
        _checks(path, national_sidecar=short)["national_sum"].status == "fail"
    )


def test_a_models_own_national_skips_the_sum(tmp_path: Path) -> None:
    # A model that forecasts the country itself (Chronos-2) writes its own
    # national rows; the hub does not require them to sum (hub-spec.md).
    path = sub(tmp_path, mutate=_bump_national(weeks()[0], 7, 5.0))
    side = _sidecar(national_source="own")
    c = _checks(path, national_sidecar=side)["national_sum"]
    assert c.status == "skip"
    assert "own" in c.summary
    side = _sidecar(national_source="sum")
    assert _checks(path, national_sidecar=side)["national_sum"].status == (
        "fail"
    )


def test_cli_reads_national_sidecar(tmp_path: Path) -> None:
    w = weeks()[-1]
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    side = tmp_path / "national.json"
    side.write_text(
        json.dumps(
            _sidecar(
                unassigned=[{"target_end_date": w, "sample": 3, "value": 4.0}]
            )
        )
    )
    args = [
        "--id",
        "main",
        "--submission",
        str(path),
        "--reference-date",
        REF,
        "--observed",
        str(HUB / "time-series.csv"),
        "--policy",
        str(POLICY_PATH),
        "--out",
        str(tmp_path / "o"),
        "--national-sidecar",
    ]
    main([*args, str(side), "--no-fail"])
    got = read_json(tmp_path / "o" / "checks.json")["checks"]
    assert {c["name"]: c["status"] for c in got}["national_sum"] == "pass"


def test_schema_failures(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][4] = "quantile"
        rows[1][2] = "2026-11-06"  # a Friday

    assert statuses(sub(tmp_path, mutate=mutate))["schema"] == "fail"
    extra = write_samples(
        tmp_path / "x" / f"{REF}-m.csv",
        [[*r, "x"] for r in make_samples()],
        header=[
            "reference_date",
            "target",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
            "forecast_date",
            "extra",
        ],
    )
    assert statuses(extra)["schema"] == "fail"


def test_location_not_in_tasks_fails(tmp_path: Path) -> None:
    s = statuses(sub(tmp_path, levels={"CD-XX": 10.0}))
    assert s["schema"] == "fail"


@pytest.mark.parametrize(
    ("growth", "status"), [(1.5, "pass"), (2.5, "warn"), (5.0, "warn")]
)
def test_growth_levels(tmp_path: Path, growth: float, status: str) -> None:
    assert statuses(sub(tmp_path, growth=growth))["growth"] == status


def test_bounds_warn_on_explosion(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path, growth=8.0))["bounds"] == "warn"


@pytest.mark.parametrize(
    ("scale", "status"), [(1.0, "pass"), (3.0, "warn"), (8.0, "warn")]
)
def test_observed_levels(tmp_path: Path, scale: float, status: str) -> None:
    levels = {
        k: v * 1.2 * scale
        for k, v in {"CD-IT": 200.0, "CD-NK": 80.0, "CD-HU": 15.0}.items()
    }
    assert statuses(sub(tmp_path, levels=levels))["observed"] == status


def test_baseline_and_previous_only_warn(tmp_path: Path) -> None:
    base = write_samples(tmp_path / "b" / "base.csv", make_samples(seed=2))
    far = write_samples(
        tmp_path / "p" / "prev.csv",
        make_samples(levels={"CD-IT": 2000.0, "CD-NK": 800.0, "CD-HU": 150.0}),
    )
    s = statuses(sub(tmp_path), baseline=base, previous=far)
    assert s["baseline"] == "pass" and s["previous"] == "warn"
    s = statuses(sub(tmp_path), baseline=far)
    assert s["baseline"] == "warn"


def test_randomwalk_skips_baseline(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path), kind="randomwalk")["baseline"] == "skip"


def test_cli_writes_report(tmp_path: Path) -> None:
    path = sub(tmp_path, mutate=lambda rows: rows[0].__setitem__(6, -1.0))
    side = tmp_path / "national.json"
    side.write_text(json.dumps(_sidecar()))
    args = [
        "--national-sidecar",
        str(side),
        "--id",
        "main",
        "--submission",
        str(path),
        "--reference-date",
        REF,
        "--observed",
        str(HUB / "time-series.csv"),
        "--policy",
        str(POLICY_PATH),
        "--tasks",
        str(HUB / "tasks.json"),
        "--out",
        str(tmp_path / "o"),
        "--locations",
        ",".join(LOCS),
    ]
    assert main(args) == 1
    assert main([*args, "--no-fail"]) == 0
    data = read_json(tmp_path / "o" / "checks.json")
    assert data["status"] == "fail"
    assert (
        "Forecast checks for `main`: FAIL"
        in (tmp_path / "o" / "checks.md").read_text()
    )


def test_cli_missing_file_is_error(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--id",
                "m",
                "--submission",
                str(tmp_path / "none.csv"),
                "--national-sidecar",
                str(tmp_path / "national.json"),
                "--reference-date",
                REF,
                "--observed",
                str(HUB / "time-series.csv"),
                "--policy",
                str(POLICY_PATH),
                "--out",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert read_json(tmp_path / "checks.json")["status"] == "error"


def with_zones(rows: list[list[Any]], skip: str = "", off: float = 0.0) -> None:
    """Split each province's samples evenly over two zones."""
    for r in list(rows):
        loc = r[3]
        if loc == "CD" or loc == skip:
            continue
        for k in (1, 2):
            rows.append(
                [*r[:3], f"{loc}-0{k}", r[4], r[5], r[6] / 2 + off, *r[7:]]
            )


def test_zones_required_and_present(tmp_path: Path) -> None:
    f = sub(tmp_path, mutate=with_zones)
    checks = {c.name: c for c in run_checks(
        f, reference_date=REF,
        settings=POLICY.checks,
        national_sidecar=_sidecar(),
        observed=read_observed(HUB / "time-series.csv"),
        require_zones=True,
    )}  # fmt: skip
    assert checks["zones"].status == "pass"
    # Zones do not count as provinces in the national sum.
    assert checks["national_sum"].status == "pass"


def test_missing_zones_fail_when_required(tmp_path: Path) -> None:
    obs = read_observed(HUB / "time-series.csv")
    none = sub(tmp_path)
    check = {c.name: c for c in run_checks(
        none, reference_date=REF,
        settings=POLICY.checks,
        national_sidecar=_sidecar(), observed=obs, require_zones=True
    )}["zones"]  # fmt: skip
    assert check.status == "fail"
    part = sub(tmp_path / "p", mutate=lambda r: with_zones(r, skip="CD-NK"))
    check = {c.name: c for c in run_checks(
        part, reference_date=REF,
        settings=POLICY.checks,
        national_sidecar=_sidecar(), observed=obs, require_zones=True
    )}["zones"]  # fmt: skip
    assert check.status == "fail"
    assert "CD-NK" in check.summary
    # Not required: no zones is fine.
    check = {c.name: c for c in run_checks(
        none, reference_date=REF,
        settings=POLICY.checks,
        national_sidecar=_sidecar(), observed=obs
    )}["zones"]  # fmt: skip
    assert check.status == "skip"


def test_zones_that_do_not_add_up_warn(tmp_path: Path) -> None:
    f = sub(tmp_path, mutate=lambda r: with_zones(r, off=1.0))
    check = {c.name: c for c in run_checks(
        f, reference_date=REF,
        settings=POLICY.checks,
        national_sidecar=_sidecar(),
        observed=read_observed(HUB / "time-series.csv"),
        require_zones=True,
    )}["zones"]  # fmt: skip
    assert check.status == "warn"


def test_plausible_forecast_passes(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path))["plausible"] == "pass"


def test_wildly_high_median_warns(tmp_path: Path) -> None:
    # Far above the last 4 observed weeks' maximum: a warning.
    assert statuses(sub(tmp_path, growth=8.0))["plausible"] == "warn"


def test_heavy_upper_tail_is_ruled_out(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        last = max(r[2] for r in rows)
        for r in rows:
            if r[3] == "CD-IT" and r[5] <= 5 and r[2] == last:
                r[6] = r[6] * 100

    checks = {c.name: c for c in run_checks(
        sub(tmp_path, mutate=mutate, national=False), reference_date=REF,
        settings=POLICY.checks, national_sidecar=_sidecar(),
        observed=read_observed(HUB / "time-series.csv"),
    )}  # fmt: skip
    assert checks["plausible"].status == "warn"
    assert "97.5%" in " ".join(checks["plausible"].details)


def test_collapse_to_near_zero_is_ruled_out(tmp_path: Path) -> None:
    low = {"CD-IT": 1.0, "CD-NK": 80.0, "CD-HU": 15.0}
    checks = {c.name: c for c in run_checks(
        sub(tmp_path, levels=low), reference_date=REF,
        settings=POLICY.checks, national_sidecar=_sidecar(),
        observed=read_observed(HUB / "time-series.csv"),
    )}  # fmt: skip
    assert checks["plausible"].status == "warn"
    assert "CD-IT" in checks["plausible"].summary


# Mandima in the 2026-11-15 release: 34, 54, 39, then 2 and 0.
MANDIMA = {
    "2026-10-17": 34.0,
    "2026-10-24": 54.0,
    "2026-10-31": 39.0,
    "2026-11-07": 2.0,
    "2026-11-14": 0.0,
}


def _mandima_check(medians: dict[str, float]) -> Any:
    s = Samples(columns=[], rows=0)
    for i in range(1, 11):
        for w, m in medians.items():
            s.values["CD-IT-26"][str(i)][w] = m
    obs = Observed(as_of="2026-11-15", counts={"CD-IT-26": MANDIMA})
    return check_plausible(s, obs, dt.date(2026, 11, 21), 3, POLICY.checks)


def _mandima_medians(nov7: float) -> dict[str, float]:
    return {
        "2026-10-31": 39.0,
        "2026-11-07": nov7,
        "2026-11-14": 3.0,
        "2026-11-21": 3.0,
        "2026-11-28": 3.0,
        "2026-12-05": 3.0,
        "2026-12-12": 3.0,
    }


def test_low_median_at_the_reported_count_passes() -> None:
    # The median for 11-07 (2) is what is already reported for that week,
    # under 1/20 of the recent maximum 54: the data, not a collapse.
    assert _mandima_check(_mandima_medians(2.0)).status == "pass"


def test_low_median_below_the_reported_count_warns() -> None:
    # 10-31 was reported as 39; a median of 1 is under 1/20 of 54.
    medians = {**_mandima_medians(2.0), "2026-10-31": 1.0}
    check = _mandima_check(medians)
    assert check.status == "warn"
    assert "2026-10-31" in " ".join(check.details)


def test_plausible_thresholds_are_settings(tmp_path: Path) -> None:
    loose = {
        **POLICY.checks,
        "plausible_median_ratio": 1e6,
        "plausible_q975_ratio": 1e9,
    }
    checks = {c.name: c for c in run_checks(
        sub(tmp_path, growth=8.0), reference_date=REF,
        settings=loose, national_sidecar=_sidecar(),
        observed=read_observed(HUB / "time-series.csv"),
    )}  # fmt: skip
    assert checks["plausible"].status == "pass"


def _with_forecast_date(
    rows: list[list[object]], value: str
) -> list[list[object]]:
    return [[*r[:7], value] for r in rows]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("20/11/2026", "not YYYY-MM-DD"),
        ("", "not YYYY-MM-DD"),
        ("2026-11-08", "after"),
    ],
)
def test_forecast_date_must_be_valid_and_not_after_reference(
    tmp_path: Path, value: str, message: str
) -> None:
    path = write_samples(
        tmp_path / "x" / f"{REF}-m.csv",
        _with_forecast_date(make_samples(), value),
    )
    assert statuses(path)["schema"] == "fail"
    schema = next(
        c
        for c in run_checks(
            path,
            reference_date=REF,
            settings=POLICY.checks,
            national_sidecar=_sidecar(),
            observed=OBS,
            expected_locations=LOCS,
            tasks=TASKS,
        )
        if c.name == "schema"
    )
    assert message in str(schema)


def test_forecast_date_column_is_required(tmp_path: Path) -> None:
    path = write_samples(
        tmp_path / "x" / f"{REF}-m.csv",
        [r[:7] for r in make_samples()],
        header=[
            "reference_date",
            "target",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ],
    )
    assert statuses(path)["schema"] == "fail"


def test_a_new_forecast_has_one_forecast_date(tmp_path: Path) -> None:
    rows = make_samples()
    rows[0][7] = "2026-11-01"
    path = write_samples(tmp_path / "x" / f"{REF}-m.csv", rows)
    assert statuses(path)["schema"] == "fail"


def test_direct_model_is_checked_against_the_baseline(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    write_samples(base, make_samples())
    assert (
        statuses(sub(tmp_path), kind="direct", baseline=base)["baseline"]
        == "pass"
    )


RT_LEVELS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
             0.975, 0.99)  # fmt: skip
RT_WEEKS = ("2026-10-31", REF)


def _rt_rows(
    value: Any = lambda q: 0.6 + q, levels: tuple[float, ...] = RT_LEVELS
) -> list[list[Any]]:
    """`wk R_t` quantile rows for the reference week and the week before."""
    return [
        [REF, "wk R_t", w, loc, "quantile", q, value(q), "2026-11-02"]
        for loc in ("CD", "CD-IT")
        for w in RT_WEEKS
        for q in levels
    ]


def _rt_tasks() -> dict[str, Any]:
    """The fixture's tasks with the hub's optional wk R_t task added."""
    tasks: dict[str, Any] = json.loads(json.dumps(TASKS))
    rnd = tasks["rounds"][-1]
    rt = json.loads(json.dumps(rnd["model_tasks"][0]))
    rt["task_ids"]["target"] = {"required": None, "optional": ["wk R_t"]}
    rt["task_ids"]["target_end_date"] = {
        "required": None, "optional": list(RT_WEEKS)}  # fmt: skip
    rt["output_type"] = {
        "quantile": {
            "output_type_id": {"required": list(RT_LEVELS)},
            "is_required": False,
            "value": {"type": "double", "minimum": 0},
        }
    }
    rnd["model_tasks"].append(rt)
    return tasks


def _rt_sub(tmp_path: Path, rows: list[list[Any]]) -> Path:
    return write_samples(
        tmp_path / f"{REF}-seabbs-renewal.csv", make_samples() + rows
    )


def test_rt_quantiles_beside_the_counts_pass(tmp_path: Path) -> None:
    """The counts are checked as before, and the wk R_t rows on their own."""
    s = statuses(_rt_sub(tmp_path, _rt_rows()), tasks=_rt_tasks())
    assert s.pop("rt") == "pass"
    for name in ("schema", "values", "samples", "plausible", "national_sum"):
        assert s[name] == "pass", (name, s)


def test_no_rt_rows_skip_the_rt_check(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path))["rt"] == "skip"


@pytest.mark.parametrize(
    "rows",
    [
        _rt_rows(value=lambda q: 1.0 - q),  # decreasing in the level
        _rt_rows(value=lambda q: q - 1),  # negative
        _rt_rows(value=lambda q: q * math.nan),  # not finite
        _rt_rows(levels=RT_LEVELS[:-1]),  # a level missing
    ],
)
def test_bad_rt_quantiles_fail(tmp_path: Path, rows: list[list[Any]]) -> None:
    assert statuses(_rt_sub(tmp_path, rows), tasks=_rt_tasks())["rt"] == "fail"


def test_rt_week_the_round_does_not_take_fails(tmp_path: Path) -> None:
    rows = [[*r[:2], "2026-10-17", *r[3:]] for r in _rt_rows()]
    assert statuses(_rt_sub(tmp_path, rows), tasks=_rt_tasks())["rt"] == "fail"


def _place_check(counts: dict[str, float], median: float, q975: float) -> Any:
    weeks = ("2026-11-14", "2026-11-21", "2026-11-28")
    s = Samples(columns=[], rows=0)
    # 40 draws: 39 at the median and one high draw to set the 97.5%.
    for i in range(1, 41):
        for w in weeks:
            s.values["CD-HU"][str(i)][w] = q975 if i == 40 else median
    obs = Observed(as_of="2026-11-15", counts={"CD-HU": counts})
    return check_plausible(s, obs, dt.date(2026, 11, 21), 1, POLICY.checks)


def test_poisson_limits_of_the_recent_maximum_replace_the_floor() -> None:
    from bdbvsubmission.check_forecast import poisson_limits

    # The exact 95% limits of a Poisson mean given a count of 0, 1 and 54.
    assert poisson_limits(0, 0.975) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert poisson_limits(1, 0.975) == pytest.approx((0.0253, 5.572), abs=1e-3)
    lo54, hi54 = poisson_limits(54, 0.975)
    assert lo54 == pytest.approx(40.56, abs=0.05)
    assert hi54 == pytest.approx(70.46, abs=0.05)
    assert "plausible_floor" not in POLICY.checks
    none = {"2026-10-31": 0.0, "2026-11-07": 0.0, "2026-11-14": 0.0}
    # A place with no recent case may forecast an introduction ...
    assert _place_check(none, 0.0, 5.0).status == "pass"
    # ... but not a median over 5 times its upper limit, 3.69.
    assert _place_check(none, 30.0, 40.0).status == "warn"
    one = {"2026-10-31": 1.0, "2026-11-07": 0.0, "2026-11-14": 0.0}
    # A median of 0 is plausible after a recent maximum of 1.
    assert _place_check(one, 0.0, 18.5).status == "pass"
    big = {"2026-10-31": 54.0, "2026-11-07": 40.0, "2026-11-14": 30.0}
    # Under 1/20 of the lower limit, 40.6, once that is a case or more.
    assert _place_check(big, 1.0, 10.0).status == "warn"
    assert _place_check(big, 3.0, 10.0).status == "pass"


def test_plausibility_only_warns_and_never_rules_a_candidate_out(
    tmp_path: Path,
) -> None:
    """Plausibility is for a human to read in the PR body (Sam,
    2026-11-24): a wildly high forecast warns, and every other check is
    unchanged."""
    s = statuses(sub(tmp_path, growth=8.0))
    assert s["plausible"] == "warn"
    assert "fail" not in {s[n] for n in ("schema", "values", "samples")}
