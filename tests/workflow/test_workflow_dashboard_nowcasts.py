"""Nowcast panels for any dataset, from quantile forecasts."""

import csv
import gzip

import numpy as np
import pytest
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
    rs = runs.load_runs([], "s=slots-,hue=217", [])
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


def test_checkpoints_of_a_run_are_shaded_by_size(tmp_path):
    for h in ("aaaaaa", "bbbbbb"):
        _quantiles(
            tmp_path / f"slots-{h}" / "2022-02-09.csv.gz",
            "fm_chronos_daily_ragged_ft",
        )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    meta = nowcasts.meta(built, rs, refs, {"aaaaaa": 250, "bbbbbb": 500})
    cols = [m["color"] for m in meta if m["run"] == "s"]
    assert cols == ["hsl(217,80%,66%)", "hsl(217,80%,38%)"]


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
