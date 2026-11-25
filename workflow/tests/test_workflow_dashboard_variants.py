"""Each series takes its own model's rows, never another variant's.

A checkpoint's file holds `foundation` and its variants (on
baselinenowcast's point nowcast, which has no spread at horizons up to
0, and on pseudo weeks); the reference folder holds the zero-shot forms
and `randomwalknowcast`.
"""

import csv
import gzip
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml
from flowlib.dashboard import context, nowcasts, runs

CONFIG = Path(__file__).parents[1] / "config.yaml"
SEVEN = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
DAY = "2022-02-01"


def _write(path, models):
    """Write one file with each model's quantiles (model: (base, step))."""
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
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for model, (base, step) in models.items():
            for i, q in enumerate(SEVEN):
                w.writerow(
                    {
                        "model": model,
                        "location": "DE",
                        "target_end_date": DAY,
                        "quantile": q,
                        "value": base + i * step,
                        "truth": 10,
                        "truth_vintage": "2022-03-01",
                    }
                )


def _refs():
    config = yaml.safe_load(CONFIG.read_text())
    return context.references(config["dashboard"]["references"])


def _build(tmp_path):
    cache = tmp_path / "cache" / "germany-days"
    _write(
        cache / "reference" / "2022-02-09.csv.gz",
        {
            "foundation_zeroshot": (1, 1),
            "foundation_zeroshot_complete": (2, 1),
            "foundation_zeroshot_baselinenowcast": (50, 0),
            "randomwalknowcast": (3, 1),
        },
    )
    run = tmp_path / "run"
    (run / "train" / "n100").mkdir(parents=True)
    (run / "train" / "n100" / "settings.json").write_text(
        '{"hash": "abc123def456", "n_outbreaks": 100}'
    )
    _write(
        run / "scores" / "germany-days" / "n100" / "2022-02-09.csv.gz",
        {
            "foundation": (4, 1),
            "foundation_baselinenowcast": (60, 0),
            "foundation_pseudoweeks": (70, 1),
        },
    )
    rs = [runs.make_run("runa", str(run), live=True)]
    refs = _refs()
    built = nowcasts.build(
        cache,
        rs,
        refs,
        levels=(0.025, 0.25, 0.5, 0.75, 0.975),
        unit="germany-days",
    )
    return built, rs, refs


def _median(built, sid):
    return built["series"]["DE"][sid]["7day"]["daily"]["2022-02-09"]["q"][0][2]


def test_zero_shot_and_each_checkpoint_take_their_own_model(tmp_path):
    built, _, _ = _build(tmp_path)
    assert _median(built, "zero") == 4  # foundation_zeroshot: 1 + 3
    assert _median(built, "complete") == 5
    assert _median(built, "rw") == 6
    assert _median(built, "abc123def456") == 7  # foundation: 4 + 3


def test_the_variants_are_their_own_series(tmp_path):
    built, rs, refs = _build(tmp_path)
    assert _median(built, "zero_baselinenowcast") == 50
    assert _median(built, "foundation_baselinenowcast:abc123def456") == 60
    assert _median(built, "foundation_pseudoweeks:abc123def456") == 73
    got = {
        m["id"]: m
        for m in nowcasts.meta(built, rs, refs, {"abc123def456": 100})
    }
    assert got["zero_baselinenowcast"]["label"] == "Zero-shot (baselinenowcast)"
    two = got["foundation_baselinenowcast:abc123def456"]
    assert two["run"] == "ref"
    assert two["label"] == "Fine-tuned (baselinenowcast) n100"


def test_the_variants_are_off_by_default(tmp_path):
    _, _, refs = _build(tmp_path)
    roles = {r.key: r.role for r in refs}
    assert roles["zero"] == "start"
    assert roles["zero_baselinenowcast"] not in ("start", "baseline")


def test_every_series_carries_its_interval_quantiles(tmp_path):
    built, _, _ = _build(tmp_path)
    for sid, by_target in built["series"]["DE"].items():
        for by_src in by_target.values():
            for by_origin in by_src.values():
                for entry in by_origin.values():
                    assert len(entry["q"][0]) == 5, sid
                    if "baselinenowcast" not in sid:
                        q = entry["q"][0]
                        assert q[0] < q[1] < q[2] < q[3] < q[4], sid


def _days(path, models, origin, offsets):
    """Write each model's quantiles on days around `origin`.

    A model is (base, step): its median moves with the day, its spread is
    `step` a level.
    """
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
    day0 = date.fromisoformat(origin)
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for model, (base, step) in models.items():
            for k in offsets:
                day = (day0 + timedelta(days=k)).isoformat()
                for i, q in enumerate(SEVEN):
                    w.writerow(
                        {
                            "model": model,
                            "location": "DE",
                            "target_end_date": day,
                            "quantile": q,
                            "value": base + 2 * k + (i - 3) * step,
                            "truth": 20,
                            "truth_vintage": "2022-03-01",
                        }
                    )


def test_wis_by_horizon_is_the_checkpoint_s_own(tmp_path):
    # The dashboard's WIS at each day from the nowcast date matches WIS
    # computed straight from the checkpoint's own rows, with a variant's
    # rows (no spread) in the same file.
    origin, offsets = "2022-02-09", range(-3, 4)
    cache = tmp_path / "cache" / "germany-days"
    _days(
        cache / "reference" / f"{origin}.csv.gz",
        {"randomwalknowcast": (18, 1)},
        origin,
        offsets,
    )
    run = tmp_path / "run"
    (run / "train" / "n100").mkdir(parents=True)
    (run / "train" / "n100" / "settings.json").write_text(
        '{"hash": "abc123def456", "n_outbreaks": 100}'
    )
    _days(
        run / "scores" / "germany-days" / "n100" / f"{origin}.csv.gz",
        {"foundation": (15, 2), "foundation_baselinenowcast": (40, 0)},
        origin,
        offsets,
    )
    rs = [runs.make_run("runa", str(run), live=True)]
    built = nowcasts.build(cache, rs, _refs(), unit="germany-days")
    got = built["byday"]["DE"]["abc123def456"]["7day"]["daily"][origin]
    for k in offsets:
        q = [15 + 2 * k + (i - 3) * 2 for i in range(7)]
        assert got[str(k)] == pytest.approx(nowcasts.wis(q, 20), abs=1e-3)


def test_the_dashboard_scores_at_the_german_scorer_s_levels():
    # WIS is the hub's seven levels (score_germany); the 5% and 95%
    # quantiles the models also give are for the 90% bands only.
    from nowcastgermany.evaluation import QUANTILE_LEVELS

    assert tuple(float(q) for q in nowcasts.SEVEN) == QUANTILE_LEVELS


def test_wis_by_horizon_is_the_scorer_s(tmp_path):
    # A file with the nine levels the models forecast: the dashboard's
    # WIS at each horizon is score_quantiles' at the scorer's levels.
    import pandas as pd

    from nowcasteval.scoring import score_quantiles
    from nowcastgermany.evaluation import QUANTILE_LEVELS

    origin = "2022-02-09"
    path = tmp_path / "reference" / f"{origin}.csv.gz"
    path.parent.mkdir(parents=True)
    nine = (
        "0.025",
        "0.05",
        "0.1",
        "0.25",
        "0.5",
        "0.75",
        "0.9",
        "0.95",
        "0.975",
    )
    day0 = date.fromisoformat(origin)
    rows = []
    for k in range(-3, 4):
        day = (day0 + timedelta(days=k)).isoformat()
        for i, q in enumerate(nine):
            rows.append(
                {
                    "model": "randomwalknowcast",
                    "location": "DE",
                    "target_end_date": day,
                    "quantile": q,
                    "value": 10 + 3 * k + (i - 4) * (2 + k % 3),
                    "truth": 14,
                    "truth_vintage": "2022-03-01",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    built = nowcasts.build(tmp_path, [], _refs())
    got = built["byday"]["DE"]["rw"]["7day"]["daily"][origin]
    forecasts = frame.assign(
        quantile=frame["quantile"].astype(float),
        horizon=[
            (date.fromisoformat(d) - day0).days
            for d in frame["target_end_date"]
        ],
    )[["location", "target_end_date", "horizon", "quantile", "value"]]
    truth = frame[["location", "target_end_date", "truth"]].drop_duplicates()
    want = score_quantiles(
        forecasts[forecasts["quantile"].isin(QUANTILE_LEVELS)],
        truth,
        keys=["location", "target_end_date"],
        quantile_levels=QUANTILE_LEVELS,
    )
    for h, w in zip(want["horizon"], want["wis"], strict=True):
        assert got[str(h)] == pytest.approx(w, abs=1e-3)
    # With the 5% and 95% quantiles too, WIS would differ.
    levels = tuple(float(q) for q in nine)
    all_nine = score_quantiles(
        forecasts,
        truth,
        keys=["location", "target_end_date"],
        quantile_levels=levels,
    )
    assert any(
        abs(got[str(h)] - w) > 1e-3
        for h, w in zip(all_nine["horizon"], all_nine["wis"], strict=True)
    )
