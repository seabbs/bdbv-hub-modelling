"""Offloaded evaluation results and how readers merge them (WP10)."""

import json

import pytest
from flowlib import evaluations


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_checkpoint_settings_merge_the_held_out_sidecar(tmp_path):
    where = tmp_path / "train" / "n500"
    write(where / "settings.json", {"hash": "h", "steps": 10})
    assert evaluations.checkpoint_settings(where) == {"hash": "h", "steps": 10}
    write(where / "held-out.json", {"held_out_loss": 0.2, "steps": 99})
    got = evaluations.checkpoint_settings(where)
    # The sidecar adds losses; it never overrides the trainer's settings.
    assert got == {"hash": "h", "steps": 10, "held_out_loss": 0.2}


def test_history_takes_each_step_s_offloaded_losses(tmp_path):
    train = tmp_path / "train"
    history = [
        {"step": 0, "lr": {"lora": 1e-5}},
        {"step": 200, "loss": 0.5, "lr": {"lora": 9e-6}},
    ]
    write(
        train / "evals" / "step-0000000.json",
        {
            "step": 0,
            "held_out_loss": 0.4,
            "by_source": {"daily": 0.3},
            "kinds": ["start"],
            "device": "cuda",
            "seconds": 1.0,
        },
    )
    write(
        train / "evals" / "step-0000200.json",
        {
            "step": 200,
            "eval_loss": 0.35,
            "kinds": ["monitor"],
        },
    )
    write(
        train / "evals" / "step-0000400.json",
        {
            "step": 400,
            "eval_loss": 0.3,
            "kinds": ["monitor"],
        },
    )
    got = evaluations.merged_history(history, train)
    assert got[0]["held_out_loss"] == 0.4
    assert got[0]["by_source"] == {"daily": 0.3}
    assert got[0]["lr"] == {"lora": 1e-5}
    assert "device" not in got[0]
    assert "kinds" not in got[0]
    assert got[1]["eval_loss"] == 0.35
    assert got[1]["loss"] == 0.5
    assert got[2] == {"step": 400, "eval_loss": 0.3}
    assert [h["step"] for h in got] == [0, 200, 400]


def test_pending_snapshots_are_those_without_results(tmp_path):
    evals = tmp_path / "train" / "evals"
    for k in (0, 200, 400):
        (evals / f"step-{k:07d}" / "adapter").mkdir(parents=True)
        write(evals / f"step-{k:07d}" / "request.json", {"step": k})
    (evals / "step-0000600.tmp" / "adapter").mkdir(parents=True)
    write(evals / "step-0000200.json", {"step": 200})
    got = evaluations.pending(tmp_path / "train")
    assert [p.name for p in got] == ["step-0000000", "step-0000400"]
    # One segment's steps: after the previous checkpoint's, up to its own.
    got = evaluations.pending(tmp_path / "train", after=0, upto=400)
    assert [p.name for p in got] == ["step-0000400"]
    got = evaluations.pending(tmp_path / "train", after=-1, upto=200)
    assert [p.name for p in got] == ["step-0000000"]


def test_start_loss_comes_from_step_0_else_an_earlier_checkpoint(tmp_path):
    train = tmp_path / "train"
    assert evaluations.start_loss(train) is None
    write(train / "n250" / "settings.json", {"held_out_loss_start": 0.41})
    assert evaluations.start_loss(train) == pytest.approx(0.41)
    write(train / "evals" / "step-0000000.json", {"held_out_loss": 0.40})
    assert evaluations.start_loss(train) == pytest.approx(0.40)


def test_start_loss_falls_back_to_an_earlier_sidecar(tmp_path):
    # An offloaded trainer's own settings never hold it; the first
    # segment's sidecar does.
    train = tmp_path / "train"
    write(train / "n250" / "settings.json", {"hash": "h"})
    write(train / "n250" / "held-out.json", {"held_out_loss_start": 0.39})
    assert evaluations.start_loss(train) == pytest.approx(0.39)


def test_the_blocks_file_is_the_trainer_s():

    pytest.importorskip("torch")
    from flowlib import segments

    from nowcastfoundation import sidecar
    from nowcastfoundation.finetune import stream as fs

    assert sidecar.HELD_OUT_BLOCKS == fs.HELD_OUT_BLOCKS
    assert segments.EVAL_MODES == fs.EVAL_MODES


def test_the_check_alerts_only_when_zero_shot_is_not_beaten(tmp_path):
    where = tmp_path / "n800"
    write(where / "settings.json", {})
    losses = {"held_out_loss": 0.3, "held_out_loss_start": 0.4}
    write(where / "held-out.json", losses)
    text, line = evaluations.check_report(where, "runa", 800, [100, 800, 1600])
    assert text == "runa n800 beats zero-shot: held-out 0.3000 vs 0.4000"
    assert line == text
    losses = {"held_out_loss": 0.5, "held_out_loss_start": 0.4}
    write(where / "held-out.json", losses)
    text, line = evaluations.check_report(where, "runa", 800, [1600, 100])
    assert "does NOT beat" in text
    assert line == f"ALERT {text}; training goes on, check before n1600"


def test_the_validation_curve_is_the_loss_by_level_at_each_checkpoint(tmp_path):
    train = tmp_path / "train"
    write(train / "evals" / "step-0000000.json", {"by_level": {"leaf": 0.4}})
    write(train / "n250" / "settings.json", {"n_outbreaks": 250})
    write(
        train / "n250" / "held-out.json",
        {"held_out_loss_by_level": {"leaf": 0.3}},
    )
    write(
        train / "n500" / "settings.json",
        {"n_outbreaks": 500, "held_out_loss_by_level": {"leaf": 0.2}},
    )
    got = evaluations.validation_curve(train, "runa", [250, 500])
    assert got == [
        {
            "run": "runa",
            "blocks": "leaf",
            "outbreaks_seen": 0,
            "validation_loss": 0.4,
        },
        {
            "run": "runa",
            "blocks": "leaf",
            "outbreaks_seen": 250,
            "validation_loss": 0.3,
        },
        {
            "run": "runa",
            "blocks": "leaf",
            "outbreaks_seen": 500,
            "validation_loss": 0.2,
        },
    ]


def test_a_run_with_no_start_loss_has_no_validation_curve(tmp_path):
    train = tmp_path / "train"
    write(
        train / "n250" / "settings.json",
        {"n_outbreaks": 250, "held_out_loss_by_level": {"leaf": 0.3}},
    )
    assert evaluations.validation_curve(train, "runa", [250]) == []
