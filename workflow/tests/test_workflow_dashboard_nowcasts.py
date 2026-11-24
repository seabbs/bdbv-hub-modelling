"""Nowcast panels for any dataset, from quantile forecasts."""

import csv
import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest
from dashboard_helpers import saved_run
from flowlib.dashboard import nowcasts, runs

SEVEN = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
PAGE = (0.025, 0.25, 0.5, 0.75, 0.975)


def _quantiles(
    path,
    model,
    truth=10.0,
    base=8.0,
    target=None,
    score=None,
    location="DE",
    week="2022-02-01",
    outer=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "location",
        "target_end_date",
        "quantile",
        "value",
        "truth",
        "truth_vintage",
    ]
    cols += ["target"] if target else []
    cols += ["score"] if score is not None else []
    values = [(q, base + i) for i, q in enumerate(SEVEN)]
    values += list(zip(("0.05", "0.95"), outer or (), strict=False))
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for q, value in values:
            row = {
                "model": model,
                "location": location,
                "target_end_date": week,
                "quantile": q,
                "value": value,
                "truth": truth,
                "truth_vintage": "2022-03-01",
            }
            if target:
                row["target"] = target
            if score is not None:
                row["score"] = score
            w.writerow(row)


def _setup():
    rs = [
        saved_run(
            Path(tempfile.mkdtemp()),
            "s",
            ("aaaaaa", "bbbbbb"),
            colour="#2a78d6",
        )
    ]
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#0a0",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)(?!KIT)",
                "checkpoint": "zero-shot",
                "role": "start",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
                "role": "baseline",
            },
        ],
        [],
    )
    return rs, refs


def test_quantile_series_by_target_with_old_files_as_7day(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    _quantiles(
        tmp_path / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
        target="daily",
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert built["targets"] == ["7day", "daily"]
    one = built["series"]["DE"]["aaaaaa"]["daily"]["daily"]["2022-02-09"]
    assert one["q"] == [[8.0, 10.0, 11.0, 12.0, 14.0]]
    assert "7day" in built["series"]["DE"]["zero"]


def test_stats_and_score_from_the_seven_levels(tmp_path):
    # Truth 10.5: inside 50% (10..12)? no; inside 95% (8..14) yes.
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=9.5,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    cov50, cov95, bias, band = built["stats"]["DE"]["zero"]["7day"]["daily"][
        "2022-02-09"
    ]
    # No 5% and 95% quantiles (the German hub's): the outer band is 95%.
    assert (cov50, cov95, band) == (0.0, 1.0, 95)
    assert (
        built["series"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]["band"]
        == 95
    )
    # CDF at 9.5 between 0.1 (9) and 0.25 (10): 0.175; bias 1 - 0.35.
    assert bias == pytest.approx(0.65)
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == (
        pytest.approx(nowcasts.wis([8, 9, 10, 11, 12, 13, 14], 9.5), abs=0.01)
    )


def test_a_file_score_wins_over_wis(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        score=3.25,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == 3.25


def test_other_reference_models_are_shown_under_their_name(tmp_path):
    _quantiles(tmp_path / "reference" / "2022-02-09.csv.gz", "KIT-nowcast")
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert "model:KIT-nowcast" in built["sets"]
    meta = {m["id"]: m for m in nowcasts.meta(built, rs, refs, {})}
    assert meta["model:KIT-nowcast"]["label"] == "KIT-nowcast"
    assert meta["model:KIT-nowcast"]["run"] == "ref"


def test_excluded_and_unclaimed_checkpoints_are_left_out(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    _quantiles(
        tmp_path / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
    )
    _quantiles(
        tmp_path / "ffffff" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged_ft"
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE, excluded={"aaaaaa"})
    assert set(built["sets"]) == {"zero"}


def test_checkpoints_of_a_run_take_its_colour_by_size(tmp_path):
    for h in ("aaaaaa", "bbbbbb"):
        _quantiles(
            tmp_path / f"slots-{h}" / "2022-02-09.csv.gz",
            "fm_chronos_daily_ragged_ft",
        )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    meta = nowcasts.meta(built, rs, refs, {"aaaaaa": 250, "bbbbbb": 500})
    got = [(m["n"], m["color"]) for m in meta if m["run"] == "s"]
    assert got == [(250, "#2a78d6"), (500, "#2a78d6")]


def test_one_location_has_truth_and_latest_by_target(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    got = nowcasts.location(built, "DE")
    assert set(got) == {
        "series",
        "score",
        "parts",
        "split",
        "ranges",
        "byday",
        "stats",
        "truth",
        "latest",
    }
    assert got["truth"]["7day"] == {"t": ["2022-02-01"], "y": [10.0]}


def test_wis_parts_on_both_scales_sum_to_each_scale_s_wis(tmp_path):
    from nowcasteval.scoring import wis_quantile

    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=15.5,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    parts = built["parts"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    q = [8.0, 9, 10, 11, 12, 13, 14]
    nat = wis_quantile([15.5], [q], nowcasts.LEVELS)
    log = wis_quantile(np.log1p([15.5]), [np.log1p(q)], nowcasts.LEVELS)
    want = (
        [nat[p][0] for p in nowcasts.WIS_PARTS]
        + [log["wis"][0]]
        + [log[p][0] for p in nowcasts.WIS_PARTS]
    )
    assert parts == pytest.approx(want, abs=1e-3)
    # The truth is above every quantile: underprediction only.
    assert parts[1] == 0
    assert parts[0] + parts[2] == pytest.approx(
        built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"], abs=0.01
    )
    assert nowcasts.location(built, "DE")["parts"] == built["parts"]["DE"]


def test_a_file_score_has_no_parts(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        score=3.25,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert "zero" not in built["parts"].get("DE", {})


def test_with_5_and_95_quantiles_bands_and_coverage_are_90(tmp_path):
    # 90% interval 9.6 to 12.4: truth 9.5 is outside it, inside 95%.
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=9.5,
        outer=(9.6, 12.4),
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    got = built["stats"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    assert got[1] == 0.0
    assert got[3] == 90
    series = built["series"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    assert series["band"] == 90
    # The outermost band drawn is the 90% interval.
    assert series["q"][0][0] == 9.6
    assert series["q"][0][-1] == 12.4
    # The WIS stays on the seven levels.
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == (
        pytest.approx(nowcasts.wis([8, 9, 10, 11, 12, 13, 14], 9.5), abs=0.01)
    )


def test_wis_splits_into_nowcast_and_forecast_weeks_ahead():
    # Target dates up to the nowcast date are the nowcast part; later ones
    # the forecast part, by week ahead (days 1 to 7 are week 1).
    q = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    scored = [
        ("2022-02-09", q, 10.0),
        ("2022-02-02", q, 9.0),
        ("2022-02-10", q, 12.0),
        ("2022-02-16", q, 13.0),
        ("2022-02-17", q, 11.0),
    ]
    got = nowcasts.wis_split(scored, "2022-02-09")
    w = {d: nowcasts.wis(qs, y) for d, qs, y in scored}
    assert got["nowcast"] == pytest.approx(
        [(w["2022-02-09"] + w["2022-02-02"]) / 2, 2], abs=1e-4
    )
    assert got["forecast"] == {
        "1": pytest.approx(
            [(w["2022-02-10"] + w["2022-02-16"]) / 2, 2], abs=1e-4
        ),
        "2": pytest.approx([w["2022-02-17"], 1], abs=1e-4),
    }


def test_the_split_reaches_each_location_s_page_data(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=9.5,
        week="2022-02-16",
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    split = nowcasts.location(built, "DE")["split"]
    got = split["zero"]["7day"]["daily"]["2022-02-09"]
    assert got["nowcast"] == [None, 0]
    assert list(got["forecast"]) == ["1"]


def test_a_dataset_shows_only_the_model_inputs_its_config_lists(tmp_path):
    # Germany is daily only: weekly-input rows in its files are left out,
    # and a model with no input of its own (randomwalknowcast) is shown
    # under the daily input alone, not under a weekly one.
    day = tmp_path / "reference" / "2022-02-09.csv.gz"
    _quantiles(day, "fm_chronos_weekly_ragged")
    _quantiles(
        tmp_path / "reference" / "2022-02-10.csv.gz", "randomwalknowcast"
    )
    _quantiles(
        tmp_path / "reference" / "2022-02-11.csv.gz", "fm_chronos_daily_ragged"
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE, sources={"daily"})
    for by_target in built["series"]["DE"].values():
        assert set(by_target["7day"]) == {"daily"}


def test_a_run_s_own_score_folders_are_its_checkpoints(tmp_path):
    # A run scores its saved checkpoints into <run>/scores/<unit>/n<N>;
    # each is the run's checkpoint of that segment's hash.
    cache = tmp_path / "cache" / "germany-days"
    _quantiles(cache / "reference" / "2022-02-09.csv.gz", "randomwalknowcast")
    run = tmp_path / "run"
    (run / "train" / "n100").mkdir(parents=True)
    (run / "train" / "n100" / "settings.json").write_text(
        '{"hash": "abc123def456", "n_outbreaks": 100}'
    )
    _quantiles(
        run / "scores" / "germany-days" / "n100" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
    )
    _, refs = _setup()
    rs = [runs.make_run("runa", str(run), live=True)]
    built = nowcasts.build(cache, rs, refs, levels=PAGE, unit="germany-days")
    assert "abc123def456" in built["sets"]
    assert "rw" in built["sets"]
    assert built["folders"]["abc123def456"] == "runa-abc123def456"


def test_plotted_truth_is_the_raw_count_in_both_german_views(tmp_path):
    # The page's truth for the 7-day sums and for the daily counts, from a
    # unit file, equals what the hub triangle's rows give directly: each
    # day's count as reported `delay` days after the view's end day.
    import datetime as dt
    from pathlib import Path

    import pandas as pd

    from nowcastgermany.data import read_triangle
    from nowcastgermany.evaluation import with_truth

    fixture = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "germany"
        / "germany_covid19_hosp.csv.gz"
    )
    triangle = read_triangle(fixture)
    delay, ends = 10, ["2021-09-20", "2021-09-27"]
    fc = pd.DataFrame(
        [
            {
                "model": "fm_chronos_daily_ragged",
                "target": target,
                "origin": "2021-10-01",
                "location": "DE",
                "target_end_date": end,
                "horizon": 0,
                "quantile": float(q),
                "value": 1.0,
            }
            for target in ("7day", "daily")
            for end in ends
            for q in SEVEN
        ]
    )
    unit = tmp_path / "reference" / "2021-10-01.csv.gz"
    unit.parent.mkdir(parents=True)
    with_truth(fc, triangle, delay).to_csv(unit, index=False)

    def raw(day: str, by: str) -> float:
        rows = triangle.loc[
            (triangle["location"] == "DE")
            & (triangle["age_group"] == "00+")
            & (triangle["reference_date"] == day)
            & (triangle["report_date"] <= by)
        ]
        return float(rows.sort_values("report_date")["confirm"].iloc[-1])

    def back(end: str, k: int) -> str:
        return (dt.date.fromisoformat(end) - dt.timedelta(days=k)).isoformat()

    def by(end: str) -> str:
        return (
            dt.date.fromisoformat(end) + dt.timedelta(days=delay)
        ).isoformat()

    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    truth = nowcasts.location(built, "DE")["truth"]
    assert truth["daily"] == {"t": ends, "y": [raw(e, by(e)) for e in ends]}
    assert truth["7day"] == {
        "t": ends,
        "y": [sum(raw(back(e, k), by(e)) for k in range(7)) for e in ends],
    }


def _days(path, model, origin, offsets, target=None):
    # One quantile set per target date, `offsets` days from `origin`, each
    # with its own truth so each day's WIS differs.
    path.parent.mkdir(parents=True, exist_ok=True)
    o = np.datetime64(origin)
    cols = ["model", "location", "target_end_date", "quantile", "value"]
    cols += ["truth", "truth_vintage", "target"]
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for k, d in enumerate(offsets):
            for i, q in enumerate(SEVEN):
                w.writerow(
                    {
                        "model": model,
                        "location": "DE",
                        "target_end_date": str(o + np.timedelta64(d, "D")),
                        "quantile": q,
                        "value": 8.0 + i,
                        "truth": 9.0 + 2 * k,
                        "truth_vintage": "2022-03-01",
                        "target": target or "7day",
                    }
                )


Q7 = [8.0 + i for i in range(7)]


def test_rows_outside_the_configured_days_are_left_out(tmp_path):
    _days(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        "2022-02-09",
        [-30, -5, 3, 40],
    )
    rs, refs = _setup()
    built = nowcasts.build(
        tmp_path, rs, refs, levels=nowcasts.LEVELS, days=(-28, 28)
    )
    one = built["series"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    assert one["t"] == ["2022-02-04", "2022-02-12"]
    got = built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    want = (nowcasts.wis(Q7, 11.0) + nowcasts.wis(Q7, 13.0)) / 2
    assert got == pytest.approx(want, abs=0.01)


def test_each_horizon_range_has_its_own_scores(tmp_path):
    _days(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        "2022-02-09",
        [-5, 0, 3],
    )
    rs, refs = _setup()
    ranges = {"all": (None, None), "nowcast": (None, 0), "forecast": (1, None)}
    built = nowcasts.build(
        tmp_path,
        rs,
        refs,
        levels=nowcasts.LEVELS,
        days=(-28, 28),
        ranges=ranges,
    )
    page = nowcasts.location(built, "DE")
    # By range, then score, stats or parts, keyed as the page's own.
    by = {
        k: {p: v[p]["zero"]["7day"]["daily"]["2022-02-09"] for p in v}
        for k, v in page["ranges"].items()
    }
    w = [nowcasts.wis(Q7, y) for y in (9.0, 11.0, 13.0)]
    assert by["nowcast"]["score"] == pytest.approx((w[0] + w[1]) / 2, abs=0.01)
    assert by["forecast"]["score"] == pytest.approx(w[2], abs=0.01)
    assert by["all"]["score"] == pytest.approx(sum(w) / 3, abs=0.01)
    for part in by.values():
        assert len(part["stats"]) == 4
        assert len(part["parts"]) == 7


def test_wis_by_day_of_the_horizon_reaches_the_page(tmp_path):
    _days(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        "2022-02-09",
        [-5, 3],
        target="daily",
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    page = nowcasts.location(built, "DE")
    got = page["byday"]["zero"]["daily"]["daily"]["2022-02-09"]
    assert got == {
        "-5": pytest.approx(nowcasts.wis(Q7, 9.0), abs=1e-4),
        "3": pytest.approx(nowcasts.wis(Q7, 11.0), abs=1e-4),
    }


def test_the_german_panels_take_their_days_from_the_german_config():
    days, ranges, default = nowcasts.horizons(
        {
            "horizons": {
                "days": "config/germany.toml",
                "ranges": {"all": [None, None], "ahead": [1, None]},
                "default": "all",
            }
        }
    )
    assert days == (-28, 28)
    assert ranges == {"all": (-28, 28), "ahead": (1, 28)}
    assert default == "all"
    assert nowcasts.horizons({}) == (None, {}, None)


def test_ranges_keep_every_folder_s_files(tmp_path):
    for origin in ("2022-02-02", "2022-02-09"):
        _days(
            tmp_path / "reference" / f"{origin}.csv.gz",
            "fm_chronos_daily_ragged",
            origin,
            [-5, 3],
        )
    _days(
        tmp_path / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
        "2022-02-09",
        [-5, 3],
    )
    rs, refs = _setup()
    built = nowcasts.build(
        tmp_path,
        rs,
        refs,
        levels=nowcasts.LEVELS,
        days=(-28, 28),
        ranges={"nowcast": (-28, 0)},
    )
    score = built["score"]["DE"]
    zero = sorted(score["zero"]["7day"]["daily"])
    assert zero == ["2022-02-02", "2022-02-09"]
    assert list(score["aaaaaa"]["7day"]["daily"]) == ["2022-02-09"]


def test_one_file_s_wis_is_scored_in_one_pass_whatever_its_dates(
    tmp_path, monkeypatch
):
    # Scoring each date apart took minutes for a German run's files.
    calls = []
    real = nowcasts.wis_quantile

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(nowcasts, "wis_quantile", counted)
    rs, refs = _setup()
    ranges = {"all": (None, None), "nowcast": (None, 0), "forecast": (1, None)}
    passes = []
    for n in (4, 40):
        _days(
            tmp_path / str(n) / "reference" / "2022-02-09.csv.gz",
            "fm_chronos_daily_ragged",
            "2022-02-09",
            range(-n // 2, n // 2),
        )
        calls.clear()
        nowcasts.build(
            tmp_path / str(n),
            rs,
            refs,
            levels=nowcasts.LEVELS,
            ranges=ranges,
        )
        passes.append(len(calls))
    assert passes[0] == passes[1]


def test_a_run_s_interim_checkpoint_is_shown_labelled_interim(tmp_path):
    # An interim checkpoint exported mid-segment, <run>/train/interim/s<step>,
    # scored into <run>/scores/<unit>/interim/s<step>, is one of the run's
    # checkpoints, labelled interim with its outbreaks.
    cache = tmp_path / "cache" / "germany-days"
    _quantiles(cache / "reference" / "2022-02-09.csv.gz", "randomwalknowcast")
    run = tmp_path / "run"
    (run / "train" / "interim" / "s4810").mkdir(parents=True)
    (run / "train" / "interim" / "s4810" / "settings.json").write_text(
        '{"hash": "interim-s4810-abc123def456", "n_outbreaks": 2000,'
        ' "interim": true}'
    )
    _quantiles(
        run
        / "scores"
        / "germany-days"
        / "interim"
        / "s4810"
        / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
    )
    _, refs = _setup()
    rs = [runs.make_run("runa", str(run), live=True)]
    built = nowcasts.build(cache, rs, refs, levels=PAGE, unit="germany-days")
    assert "abc123def456" in built["sets"]
    sizes = {"abc123def456": 2000}
    shown = nowcasts.meta(built, rs, refs, sizes)
    one = next(m for m in shown if m["id"] == "abc123def456")
    assert one["run"] == "runa"
    assert one["n"] == 2000
    assert "interim" in one["label"]


def test_a_unit_with_forecast_days_has_its_split_on_every_date(tmp_path):
    # The horizon split panel reads `split` by date: a unit with forecast
    # days has its nowcast part and weeks ahead on each date it scored.
    for origin in ("2022-02-08", "2022-02-09"):
        _days(
            tmp_path / "reference" / f"{origin}.csv.gz",
            "fm_chronos_daily_ragged",
            origin,
            [-3, 0, 2, 9],
        )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    split = nowcasts.location(built, "DE")["split"]["zero"]["7day"]["daily"]
    assert set(split) == {"2022-02-08", "2022-02-09"}
    for part in split.values():
        assert part["nowcast"][1] == 2
        assert set(part["forecast"]) == {"1", "2"}


def test_the_truth_context_reaches_the_page_before_the_first_forecast(
    tmp_path,
):
    # A dataset's truth.csv.gz (the German plot context) adds truth days
    # before any forecast file's, never replacing a file's own truth.
    _days(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        "2022-02-09",
        [-5, 3],
    )
    import pandas as pd

    pd.DataFrame(
        {
            "location": "DE",
            "target": "7day",
            "target_end_date": ["2021-06-01", "2022-02-04"],
            "truth": [5.0, 999.0],
            "truth_vintage": "2022-03-01",
        }
    ).to_csv(tmp_path / "truth.csv.gz", index=False)
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    page = nowcasts.location(built, "DE")
    line = page["truth"]["7day"]
    got = dict(zip(line["t"], line["y"], strict=True))
    assert got["2021-06-01"] == 5.0
    assert got["2022-02-04"] == 9.0
