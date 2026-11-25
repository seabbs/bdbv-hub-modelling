"""One synthetic validation: the full held-out set, step 0 and checkpoints.

The subset monitor and the step snapshots are no longer scored, so the
loss section reads none of them: its validation panel is the full set
by checkpoint, with that evaluation's nowcast and forecast parts and its
losses by layout and level.
"""

import json
from pathlib import Path

from flowlib.dashboard import logs, runs, training

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard" / "page"


def _run(tmp_path):
    folder = tmp_path / "run"
    train = folder / "train"
    (train / "evals").mkdir(parents=True)
    (train / "history.json").write_text(
        json.dumps([{"step": 0}, {"step": 200, "loss": 0.45}])
    )
    (train / "evals" / "step-0000000.json").write_text(
        json.dumps(
            {
                "step": 0,
                "held_out_loss": 0.5,
                "held_out_loss_nowcast": 0.2,
                "held_out_loss_forecast": 0.3,
                "by_source": {"weekly": 0.3},
                "by_level": {"national": 0.4},
            }
        )
    )
    # An old step snapshot's subset losses: not read.
    (train / "evals" / "step-0000200.json").write_text(
        json.dumps(
            {
                "step": 200,
                "eval_loss": 0.3,
                "eval_loss_nowcast": 0.1,
                "eval_loss_forecast": 0.2,
                "by_source": {"weekly": 0.25},
            }
        )
    )
    return runs.make_run("r", str(folder), live=True)


CKS = [
    (
        "n250",
        {
            "steps": 1000,
            "n_outbreaks": 250,
            "held_out_loss": 0.35,
            "held_out_loss_nowcast": 0.1,
            "held_out_loss_forecast": 0.25,
            "held_out_loss_by_source": {"weekly": 0.2},
            "held_out_loss_by_level": {"national": 0.3},
        },
        None,
    )
]


def test_the_loss_plot_reads_the_full_set_only(tmp_path):
    rows = [{"step": 100, "loss": 0.5}]
    got = training.loss_run(
        _run(tmp_path), rows, CKS, lambda s: s / 4, None, "#111"
    )
    for subset in ("held", "held_parts", "bysrc"):
        assert subset not in got
    assert got["full0"] == 0.5
    assert got["ck"] == [[250, 0.35]]
    assert got["parts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["ck_parts"] == {
        "nowcast": [[250, 0.1]],
        "forecast": [[250, 0.25]],
    }
    assert got["src0"] == {"weekly": 0.3}
    assert got["ck_src"] == {"weekly": [[250, 0.2]]}
    assert got["lvl0"] == {"national": 0.4}
    assert got["ck_lvl"] == {"national": [[250, 0.3]]}


def test_the_log_keeps_no_held_out_loss(tmp_path):
    path = tmp_path / "n100.log"
    path.write_text(
        "2026-11-17 10:00:00 [train] step 200: loss 0.5, held-out (first 100)"
        " 0.4 (nowcast 0.1, forecast 0.3), by source weekly 0.12\n"
    )
    row = logs.Log(path).rows()[0]
    assert row["step"] == 200
    assert row["loss"] == 0.5
    for subset in ("held", "parts", "src"):
        assert subset not in row


def test_the_validation_panel_is_the_full_set_by_checkpoint():
    js = (PAGE / "loss.js").read_text()
    assert "subset" not in js
    assert "Validation loss on the full held-out set" in js
