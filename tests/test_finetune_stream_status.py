"""The trainer's status file (`stream-status.json`) names its stage."""

import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("chronos")

from nowcastfoundation.finetune import stream as fs


class CutError(Exception):
    """Stops the trainer once it starts cutting."""


def test_the_held_out_cut_is_its_own_stage(tmp_path, monkeypatch):
    # The cut takes about 20 minutes: the status must not still say the
    # trainer waits for held-out simulations.
    out = tmp_path / "train"
    seen = {}

    def cut(*_args, **_kwargs):
        seen.update(json.loads((out / "stream-status.json").read_text()))
        raise CutError

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["a", "b"])
    monkeypatch.setattr(fs, "code_commit", lambda: "c")
    monkeypatch.setattr(fs, "checkpoint_large_batches", lambda *_: None)
    monkeypatch.setattr(fs, "ProcessPoolExecutor", lambda *_, **__: None)
    monkeypatch.setattr(fs, "cut_in_workers", cut)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_stream",
            str(tmp_path / "sims"),
            str(out),
            "--train-seeds",
            "1",
            "2",
            "--held-out-seeds",
            "3",
            "4",
            "--holdout",
            "2",
        ],
    )
    with pytest.raises(CutError):
        fs.main()
    assert seen["stage"] == "cutting held-out blocks of 2 outbreaks"
