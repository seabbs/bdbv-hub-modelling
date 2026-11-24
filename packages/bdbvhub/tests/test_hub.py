from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbvhub.submission.hub import (
    HUB_COLUMNS,
    model_id,
    render_model_metadata,
    run_date,
    to_hub_table,
    write_model_metadata,
    write_submission,
)
from bdbvhub.submission.validate import (
    Hub,
    validate_metadata,
    validate_submission,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "hub_validation" / "fixtures"
HUB = Hub.load(FIXTURES / "hub-snapshot")
"""The pinned hub config, in which the round of 2026-10-31 is open."""
METADATA = ROOT / "hub" / "model-metadata"
MID = "seabbs-renewal"

FORECAST_DATE = "2026-10-20"
"""On or before every reference date these tests use."""


def _failures(table: pd.DataFrame, root: Path) -> list[str]:
    """Write a table to the hub layout and list the checks it fails."""
    path = write_submission(table, root, MID)
    rep = validate_submission(
        path, HUB, metadata_dirs=[METADATA], skip_window=True
    )
    return [c.check for c in rep.failed]


def _table(samples: pd.DataFrame, ref: str, **kw: object) -> pd.DataFrame:
    kw.setdefault("forecast_date", FORECAST_DATE)
    return to_hub_table(samples, ref, **kw)  # type: ignore[arg-type]


def _samples(n: int = 100) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": 10.4}
        for loc in ("CD-IT", "CD")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


def test_to_hub_table_is_valid(tmp_path: Path) -> None:
    table = _table(_samples(), "2026-10-31")
    assert tuple(table.columns) == HUB_COLUMNS
    assert (table["forecast_date"] == FORECAST_DATE).all()
    assert (table["value"] == 10).all()
    assert _failures(table, tmp_path) == []


def test_submission_path(tmp_path: Path) -> None:
    table = _table(_samples(), "2026-10-31")
    path = write_submission(table, tmp_path, "seabbs-randomwalk")
    assert path.name == "2026-10-31-seabbs-randomwalk.csv"
    assert path.parent.name == "seabbs-randomwalk"


def test_model_ids() -> None:
    assert model_id("randomwalk") == "seabbs-randomwalk"
    assert model_id("a" * 14) == f"seabbs-{'a' * 14}"
    with pytest.raises(ValueError, match="letters"):
        model_id("bad-name")
    with pytest.raises(ValueError, match="14"):
        model_id("a" * 15)


def test_metadata_is_valid(tmp_path: Path) -> None:
    record = render_model_metadata(
        "randomwalk",
        model_name="RandomWalk baseline",
        methods="Short.",
        methods_long="Long.",
        data_inputs="Hub data.",
    )
    path = write_model_metadata(record, tmp_path)
    assert path.name == "seabbs-randomwalk.yml"
    assert validate_metadata(path, HUB).valid
    back = yaml.safe_load(path.read_text())
    assert "language model" not in back["model_details"]["methods_long"]
    # The team's HPRU funding, in the schema's `team_funding` field.
    assert "NIHR207404" in back["team_funding"]


# Gaps found by the hub validation bundle -----------------------------------


def _provinces(n: int = 100, value: float = 10.4) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": value}
        for loc in ("CD-IT", "CD-NK")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


def test_validation_needs_same_ids_in_every_location(
    tmp_path: Path,
) -> None:
    table = _table(_provinces(), "2026-10-31")
    mask = (table["location"] == "CD-IT") & (table["output_type_id"] == 1)
    table.loc[mask, "output_type_id"] = 101
    assert _failures(table, tmp_path) == ["spl_compound_taskid_set"]


def test_writer_makes_national_the_sum_of_provinces() -> None:
    samples = _provinces(value=10.4)
    samples.loc[samples["location"] == "CD-NK", "value"] = 2.6
    national = samples.loc[samples["location"] == "CD-IT"].assign(
        location="CD", value=999.0
    )
    table = _table(pd.concat([samples, national]), "2026-10-31")
    wide = table.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    assert (wide["CD"] == wide["CD-IT"] + wide["CD-NK"]).all()
    assert (wide["CD"] == 13).all()
    assert table["output_type_id"].dtype == "int64"
    assert sorted(table["output_type_id"].unique()) == list(range(1, 101))


def test_writer_adds_an_unassigned_component() -> None:
    unassigned = pd.DataFrame(
        {
            "target_end_date": "2026-10-31",
            "sample": range(1, 101),
            "value": 2.0,
        }
    )
    table = _table(_provinces(), "2026-10-31", unassigned=unassigned)
    cd = table.loc[table["location"] == "CD"].set_index(
        ["output_type_id", "target_end_date"]
    )["value"]
    assert (cd.xs("2026-10-31", level=1) == 22).all()
    assert (cd.xs("2026-10-24", level=1) == 20).all()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.loc[~((s["location"] == "CD-NK") & (s["sample"] == 5))],
        lambda s: s.loc[
            ~(
                (s["location"] == "CD-NK")
                & (s["target_end_date"] == "2026-11-07")
            )
        ],
        lambda s: s.assign(sample=s["sample"] + 1),
    ],
    ids=["missing-sample", "missing-week", "ids-not-1-to-100"],
)
def test_writer_refuses_inconsistent_samples(change) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"sample|week"):
        _table(change(_provinces()), "2026-10-31")


# forecast_date ---------------------------------------------------------------


def test_forecast_date_defaults_to_the_run_date_in_utc() -> None:
    table = to_hub_table(_samples(), "2026-10-31")
    assert (table["forecast_date"] == run_date().isoformat()).all()
    late = dt.datetime(
        2026, 10, 26, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
    )
    assert run_date(late) == dt.date(2026, 10, 27)


def _with_national(value: float = 999.4) -> pd.DataFrame:
    samples = _provinces(value=10.4)
    national = samples.loc[samples["location"] == "CD-IT"].assign(
        location="CD", value=value
    )
    return pd.concat([samples, national], ignore_index=True)


def test_writer_keeps_a_models_own_national_rows(tmp_path: Path) -> None:
    # A model that forecasts the country itself, as the Chronos-2 joint
    # block does, keeps its own national draws. The hub does not require
    # the national value to be the sum of the provinces.
    table = _table(_with_national(), "2026-10-31", national="own")
    cd = table.loc[table["location"] == "CD", "value"]
    assert (cd == 999).all()
    assert len(cd) == 100 * 3
    assert _failures(table, tmp_path) == []


def test_own_national_rows_must_be_complete() -> None:
    samples = _with_national()
    samples = samples.loc[
        ~((samples["location"] == "CD") & (samples["sample"] == 1))
    ]
    with pytest.raises(ValueError, match="CD"):
        _table(samples, "2026-10-31", national="own")
    with pytest.raises(ValueError, match="national"):
        _table(_provinces(), "2026-10-31", national="own")


def test_only_the_chronos_models_give_their_own_national() -> None:
    from bdbvhub.submission.hub import own_national

    assert own_national("foundation")
    assert own_national("foundation_zeroshot")
    for name in ("renewal", "renewalbvd", "randomwalknowcast", "randomwalk"):
        assert not own_national(name)
    assert not own_national(None)


def test_committed_metadata_passes_the_hub_schema() -> None:
    # The files the next submission uploads: each passes the hub's schema
    # (methods at most 200 characters, model_abbr at most 16).
    for path in sorted(METADATA.glob("*.yml")):
        report = validate_metadata(path, HUB)
        assert report.valid, (path.name, report)


RT_HUB = Hub.load(FIXTURES / "hub-snapshot-rt")
"""The hub config with the optional `wk R_t` target (hub tasks.json at
b7e6842f5, 2026-11-23): quantiles of the reference week and the week
before."""

RT_LEVELS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
             0.975, 0.99)  # fmt: skip


def _rt(weeks: tuple[str, ...]) -> pd.DataFrame:
    """A fit's `rt.csv.gz`: increasing quantiles per location and week."""
    return pd.DataFrame(
        [
            {"location": loc, "target_end_date": w, "output_type_id": q,
             "value": 0.5 + q}
            for loc in ("CD", "CD-IT")
            for w in weeks
            for q in RT_LEVELS
        ]
    )  # fmt: skip


def test_rt_end_dates_are_the_rounds_wk_rt_weeks() -> None:
    from bdbvhub.submission.hub import rt_end_dates

    tasks = FIXTURES / "hub-snapshot-rt" / "hub-config" / "tasks.json"
    assert rt_end_dates(tasks, "2026-11-28") == ("2026-11-21", "2026-11-28")
    # A round the target is not open for has none.
    assert rt_end_dates(tasks, "2026-10-31") == ()


def test_rt_rows_keep_the_rounds_weeks_as_quantiles() -> None:
    from bdbvhub.submission.hub import RT_TARGET, rt_table

    rt = _rt(("2026-11-14", "2026-11-21", "2026-11-28"))
    out = rt_table(
        rt, "2026-11-28", ("2026-11-21", "2026-11-28"), FORECAST_DATE
    )
    assert tuple(out.columns) == HUB_COLUMNS
    assert set(out["target"]) == {RT_TARGET}
    assert set(out["output_type"]) == {"quantile"}
    assert sorted(set(out["target_end_date"])) == ["2026-11-21", "2026-11-28"]
    assert len(out) == 2 * 2 * 23


def test_counts_and_rt_together_pass_the_hub_checks(tmp_path: Path) -> None:
    from bdbvhub.submission.hub import rt_table

    weeks = ["2026-11-21", "2026-11-28", "2026-12-05"]
    samples = pd.DataFrame(
        [
            {"location": loc, "target_end_date": w, "sample": s,
             "value": 10.0}
            for loc in ("CD-IT", "CD")
            for s in range(1, 101)
            for w in weeks
        ]
    )  # fmt: skip
    counts = _table(samples, "2026-11-28")
    rt = rt_table(
        _rt(("2026-11-21", "2026-11-28")),
        "2026-11-28",
        ("2026-11-21", "2026-11-28"),
        FORECAST_DATE,
    )
    path = write_submission(pd.concat([counts, rt]), tmp_path, MID)
    rep = validate_submission(
        path, RT_HUB, metadata_dirs=[METADATA], skip_window=True
    )
    assert [c.check for c in rep.failed] == []


def test_nowcast_adds_a_fits_rt_for_the_rounds_weeks(tmp_path: Path) -> None:
    """`nowcast --fit` adds the fit's `RT_FILE` as `wk R_t` rows for the
    weeks the round takes; a fit without one adds nothing."""
    from bdbvhub.cli import _with_rt
    from nowcastrenewal.forecaster import RT_FILE

    tasks = FIXTURES / "hub-snapshot-rt" / "hub-config" / "tasks.json"
    counts = _table(_samples(), "2026-11-28")
    fit = tmp_path / "fit"
    fit.mkdir()
    assert _with_rt(counts, fit, tasks, "2026-11-28", FORECAST_DATE) is counts
    _rt(("2026-11-14", "2026-11-21", "2026-11-28")).to_csv(
        fit / RT_FILE, index=False
    )
    out = _with_rt(counts, fit, tasks, "2026-11-28", FORECAST_DATE)
    rt = out.loc[out["target"] == "wk R_t"]
    assert sorted(set(rt["target_end_date"])) == ["2026-11-21", "2026-11-28"]
    assert len(out) == len(counts) + len(rt)
    # A round with no wk R_t task adds nothing.
    assert len(_with_rt(counts, fit, tasks, "2026-10-31", FORECAST_DATE)) == (
        len(counts)
    )


def test_the_submission_plot_reads_the_count_samples_only(
    tmp_path: Path,
) -> None:
    """The PR's plot draws the count samples; `wk R_t` quantile rows in the
    same file are left out."""
    from bdbvhub.reporting.data_plots import read_submission
    from bdbvhub.submission.hub import rt_table

    counts = _table(_samples(), "2026-11-28")
    rt = rt_table(_rt(("2026-11-28",)), "2026-11-28", ("2026-11-28",),
                  FORECAST_DATE)  # fmt: skip
    path = write_submission(pd.concat([counts, rt]), tmp_path, MID)
    fc = read_submission(path, "2026-11-22")
    assert len(fc) == len(counts)
    assert fc["sample"].dtype == "int64"
