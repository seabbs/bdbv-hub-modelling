"""Backfilling the held-out evaluation of checkpoints that lack it.

After the final checkpoint the trainer scores, on its GPU, each saved
checkpoint without held-out losses, newest first.
"""

import json

import pytest

from nowcastfoundation import sidecar
from nowcastfoundation.finetune import evaluate as fe
from nowcastfoundation.finetune import stream as fs


def _saved(train, n, **settings):
    (train / f"n{n}").mkdir(parents=True)
    (train / f"n{n}" / "settings.json").write_text(json.dumps(settings))


def test_pending_checkpoints_are_the_unscored_saved_ones_newest_first(tmp_path):
    _saved(tmp_path, 100)
    _saved(tmp_path, 200)
    (tmp_path / "n200" / sidecar.SIDECAR).write_text("{}")
    _saved(tmp_path, 400)
    _saved(tmp_path, 800, held_out_loss=0.3)
    # n1600 is not saved.
    got = fs.pending_checkpoints(tmp_path, [100, 200, 400, 800, 1600])
    assert got == [tmp_path / "n400", tmp_path / "n100"]


def test_the_start_loss_is_step_zero_s_or_an_earlier_checkpoint_s(tmp_path):
    assert sidecar.start_loss(tmp_path) is None
    _saved(tmp_path, 100, held_out_loss_start=0.9)
    assert sidecar.start_loss(tmp_path) == 0.9
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "step-0000000.json").write_text(
        json.dumps({"held_out_loss": 0.8})
    )
    assert sidecar.start_loss(tmp_path) == 0.8


def test_a_checkpoint_s_sidecar_holds_its_losses_and_the_start(
    tmp_path, monkeypatch
):
    _saved(tmp_path, 400)
    monkeypatch.setattr(
        fe, "score", lambda *_a, **_k: ({"held_out_loss": 0.5}, 12.0)
    )
    with pytest.raises(ValueError, match="base Chronos-2"):
        fe.write_checkpoint_sidecar(tmp_path / "n400", None, [], "cuda")
    _saved(tmp_path, 100, held_out_loss_start=0.9)
    got = fe.write_checkpoint_sidecar(tmp_path / "n400", None, [], "cuda")
    written = json.loads((tmp_path / "n400" / sidecar.SIDECAR).read_text())
    assert got == written
    assert written == {
        "held_out_loss_start": 0.9,
        "held_out_loss": 0.5,
        "device": "cuda",
        "seconds": 12.0,
    }
