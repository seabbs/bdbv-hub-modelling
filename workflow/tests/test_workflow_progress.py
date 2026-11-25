"""A run's fine-tuning progress tables, for its SI figure."""

import csv
import json

from flowlib import progress


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_the_loss_rows_hold_training_validation_and_checkpoints(tmp_path):
    train = tmp_path / "train"
    write(
        train / "history.json",
        [
            {"step": 0},
            {"step": 10, "loss": 0.5},
            {"step": 20, "loss": 0.4},
            {"step": 20, "n_outbreaks": 100, "hash": "h1"},
        ],
    )
    write(
        train / "evals" / "step-0000000.json",
        {"step": 0, "by_source": {"daily": 0.6}, "by_level": {"leaf": 0.7}},
    )
    write(
        train / "evals" / "step-0000020.json",
        {"step": 20, "by_source": {"daily": 0.3, "weekly": 0.35}},
    )
    write(train / "n100" / "settings.json", {"n_outbreaks": 100, "steps": 20})
    write(
        train / "n100" / "held-out.json",
        {"held_out_loss_by_level": {"leaf": 0.2}},
    )
    got = progress.loss_rows(train, [100, 200])
    training = [r for r in got if r["kind"] == "training"]
    assert [(r["step"], r["loss"]) for r in training] == [(10, 0.5), (20, 0.4)]
    by_source = {
        (r["series"], r["step"]): r["loss"]
        for r in got
        if r["kind"] == "validation"
    }
    assert by_source == {
        ("daily", 0): 0.6,
        ("daily", 20): 0.3,
        ("weekly", 20): 0.35,
    }
    by_level = {
        (r["series"], r["step"], r["outbreaks"]): r["loss"]
        for r in got
        if r["kind"] == "validation by level"
    }
    assert by_level == {("leaf", 0, 0): 0.7, ("leaf", 20, 100): 0.2}
    marks = [r for r in got if r["kind"] == "checkpoint"]
    assert [(r["step"], r["outbreaks"]) for r in marks] == [(20, 100)]


def test_a_run_with_no_history_has_no_loss_rows(tmp_path):
    assert progress.loss_rows(tmp_path / "train", [100]) == []


def test_the_seven_day_rows_take_each_checked_checkpoint(tmp_path):
    train = tmp_path / "train"
    check = {"ratio": 1.02, "inside90": 0.9, "rough_ratio": 1.2, "n": 50}
    write(train / "n100" / "settings.json", {"n_outbreaks": 100})
    write(train / "n100" / "seven-day.json", {"seven_day": check})
    write(train / "n200" / "settings.json", {"n_outbreaks": 200})
    got = progress.seven_day_rows(train)
    assert got == [{"outbreaks": 100, **check}]


def test_the_bdbv_rows_take_each_released_checkpoint_s_check(tmp_path):
    train, ftc = tmp_path / "train", tmp_path / "ft-check"
    write(train / "n100" / "settings.json", {"n_outbreaks": 100})
    write(train / "n100" / "published.json", {"hash": "abc"})
    write(train / "n200" / "settings.json", {"n_outbreaks": 200})
    write(train / "n200" / "published.json", {"hash": "def"})
    data = ftc / "slots-abc" / "bdbv-ft-check-slots-abc-data"
    data.mkdir(parents=True)
    (data / "overall.csv").write_text(
        "model,relative_crps,n\nrandomwalknowcast,1.0,9\nfoundation,0.8,9\n"
    )
    got = progress.bdbv_rows(train, ftc)
    assert got == [
        {"outbreaks": 100, "model": "randomwalknowcast", "relative_crps": 1.0},
        {"outbreaks": 100, "model": "foundation", "relative_crps": 0.8},
    ]


def test_the_progress_tables_are_written_with_headers_when_empty(tmp_path):
    out = tmp_path / "progress"
    progress.write_progress(
        tmp_path / "train",
        [100],
        tmp_path / "ft-check",
        out / "loss.csv",
        out / "seven-day.csv",
        out / "bdbv.csv",
    )
    assert read(out / "loss.csv") == []
    header = (out / "seven-day.csv").read_text().splitlines()[0]
    assert header == ",".join(progress.SEVEN_DAY_COLUMNS)
