"""Each series takes its own model's rows, never the two-step models'.

A checkpoint's file holds `foundation` and `foundation_twostep`; the
reference folder holds `foundation_zeroshot`, its complete-data and
two-step forms and `randomwalknowcast`. The two-step models copy the
reported count, with no spread, at horizons up to 0.
"""

import csv
import gzip
from pathlib import Path

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
            "foundation_zeroshot_twostep": (50, 0),
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
        {"foundation": (4, 1), "foundation_twostep": (60, 0)},
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


def test_the_two_step_models_are_their_own_series(tmp_path):
    built, rs, refs = _build(tmp_path)
    assert _median(built, "zero_twostep") == 50
    assert _median(built, "twostep:abc123def456") == 60
    got = {
        m["id"]: m
        for m in nowcasts.meta(built, rs, refs, {"abc123def456": 100})
    }
    assert got["zero_twostep"]["label"] == "Two-step zero-shot (point nowcast)"
    two = got["twostep:abc123def456"]
    assert two["run"] == "ref"
    assert two["label"].startswith("Two-step fine-tuned (point nowcast)")
    assert "n100" in two["label"]


def test_the_two_step_models_are_off_by_default(tmp_path):
    _, _, refs = _build(tmp_path)
    roles = {r.key: r.role for r in refs}
    assert roles["zero"] == "start"
    assert roles["zero_twostep"] not in ("start", "baseline")


def test_every_series_carries_its_interval_quantiles(tmp_path):
    built, _, _ = _build(tmp_path)
    for sid, by_target in built["series"]["DE"].items():
        for by_src in by_target.values():
            for by_origin in by_src.values():
                for entry in by_origin.values():
                    assert len(entry["q"][0]) == 5, sid
                    if not sid.startswith(("twostep:", "zero_twostep")):
                        q = entry["q"][0]
                        assert q[0] < q[1] < q[2] < q[3] < q[4], sid
