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

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["3-0001", "3-0002"])
    monkeypatch.setattr(fs, "sims_sha256", lambda *_: {})
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


def test_a_resumed_run_s_status_carries_its_step_and_segment(
    tmp_path, monkeypatch
):
    # During the cuts before the first step, the status shows the step and
    # outbreaks the run resumes from, the segment and its target; not 0.
    out = tmp_path / "train"
    (out / "state").mkdir(parents=True)
    (out / "state" / "state.json").write_text(
        json.dumps({"step": 37374, "done": 1600})
    )
    seen = {}

    def cut(*_args, **_kwargs):
        seen.update(json.loads((out / "stream-status.json").read_text()))
        raise CutError

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["3-0001", "3-0002"])
    monkeypatch.setattr(fs, "sims_sha256", lambda *_: {})
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
            "--outbreaks",
            "3200",
            "--until",
            "3200",
        ],
    )
    with pytest.raises(CutError):
        fs.main()
    assert (seen["step"], seen["outbreaks"]) == (37374, 1600)
    assert (seen["segment"], seen["target"]) == ("n3200", 3200)


def test_a_fresh_run_s_status_starts_at_step_zero(tmp_path):
    assert fs.resume_progress(tmp_path / "state", 400, 3200) == {
        "step": 0,
        "outbreaks": 0,
        "segment": "n400",
        "target": 400,
    }


def test_cut_only_saves_the_held_out_and_first_chunk_without_the_gpu(
    tmp_path, monkeypatch
):
    # Lead 13:00: Run B's GPU must never wait on a cut. Its held-out
    # blocks and first chunk are cut on the CPU, under the keys the
    # trainer will look up, before it takes the GPU.
    out = tmp_path / "train"
    view = {"kind": "whole"}
    held = ([("3-0001-1", {"source": "daily", "cut": 0, "view": view})], [])
    first = ([("1-0001-1", {"source": "daily"})], [])

    def no_model(*_args, **_kwargs):
        raise AssertionError("cut-only loaded the model")

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["3-0001", "3-0002"])
    monkeypatch.setattr(fs, "code_commit", lambda: "c")
    monkeypatch.setattr(fs, "checkpoint_large_batches", lambda *_: None)
    monkeypatch.setattr(fs, "ProcessPoolExecutor", lambda *_, **__: None)
    monkeypatch.setattr(fs, "cut_in_workers", lambda *_a, **_k: held)
    monkeypatch.setattr(fs, "sims_sha256", lambda _d, ids: {"n": len(ids)})
    monkeypatch.setattr(fs, "next_chunk", lambda *_: ["1-0001", "1-0002"])
    monkeypatch.setattr(fs, "submit_cut", lambda *_a, **_k: "futures")
    monkeypatch.setattr(fs, "_join", lambda _f: first)
    import chronos

    monkeypatch.setattr(chronos.Chronos2Pipeline, "from_pretrained", no_model)
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
            "--cut-only",
        ],
    )
    fs.main()
    assert (out / fs.HELD_OUT_KEY).exists()
    key = json.loads((out / fs.HELD_OUT_KEY).read_text())
    saved = fs.load_chunk(
        out, key | {"outbreaks": ["1-0001", "1-0002"], "sims_sha256": {"n": 2}}
    )
    assert saved is not None
    assert saved[0][0][0] == "1-0001-1"


def test_cut_only_on_a_resumed_run_cuts_the_chunk_after_the_one_training(
    tmp_path, monkeypatch
):
    # Lead 15:08: Run B takes the lag jitter from its saved state; its
    # held-out blocks and the chunk after the one it trains are cut on
    # the CPU first, so the restart at that chunk's end loads them.
    out = tmp_path / "train"
    (out / "state").mkdir(parents=True)
    state = {
        "step": 24100,
        "done": 2,
        "trained": ["1-0001", "1-0002", "1-0009"],
        "chunk": ["1-0003", "1-0004"],
    }
    (out / "state" / "state.json").write_text(json.dumps(state))
    view = {"kind": "whole"}
    held = ([("3-0001-1", {"source": "daily", "cut": 0, "view": view})], [])
    later = ([("1-0005-1", {"source": "daily"})], [])
    asked = []

    def chunk_after(_sims, _seeds, after, size):
        asked.append((list(after), size))
        return ["1-0005", "1-0006"]

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["3-0001", "3-0002"])
    monkeypatch.setattr(fs, "code_commit", lambda: "c")
    monkeypatch.setattr(fs, "checkpoint_large_batches", lambda *_: None)
    monkeypatch.setattr(fs, "ProcessPoolExecutor", lambda *_, **__: None)
    monkeypatch.setattr(fs, "cut_in_workers", lambda *_a, **_k: held)
    monkeypatch.setattr(fs, "sims_sha256", lambda _d, ids: {"n": len(ids)})
    monkeypatch.setattr(fs, "next_chunk", chunk_after)
    monkeypatch.setattr(fs, "submit_cut", lambda *_a, **_k: "futures")
    monkeypatch.setattr(fs, "_join", lambda _f: later)
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
            "--outbreaks",
            "10",
            "--cut-only",
        ],
    )
    fs.main()
    assert asked == [(["1-0001", "1-0002", "1-0003", "1-0004"], 6)]
    key = json.loads((out / fs.HELD_OUT_KEY).read_text())
    saved = fs.load_chunk(
        out, key | {"outbreaks": ["1-0005", "1-0006"], "sims_sha256": {"n": 2}}
    )
    assert saved is not None


def test_the_status_names_the_model_load_after_saved_held_out_blocks(
    tmp_path, monkeypatch
):
    # Lead 03:20: Run B loaded its saved held-out blocks and went on to
    # the model, but its status still said it waited for held-out
    # simulations.
    import chronos

    out = tmp_path / "train"
    seen = {}

    def load(*_args, **_kwargs):
        seen.update(json.loads((out / "stream-status.json").read_text()))
        raise CutError

    monkeypatch.setattr(fs, "ready_outbreaks", lambda *_: ["3-0001", "3-0002"])
    monkeypatch.setattr(fs, "sims_sha256", lambda *_: {})
    monkeypatch.setattr(fs, "code_commit", lambda: "c")
    monkeypatch.setattr(fs, "checkpoint_large_batches", lambda *_: None)
    monkeypatch.setattr(fs, "ProcessPoolExecutor", lambda *_, **__: None)
    monkeypatch.setattr(fs, "held_out_blocks", lambda *_: [])
    monkeypatch.setattr(chronos.Chronos2Pipeline, "from_pretrained", load)
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
    assert seen["stage"] == "loading the model"


def test_the_status_names_a_checkpoint_s_evaluation():
    # Overseer 04:31: Run B's status said 'training' through the n100
    # evaluation (~20 min on the GPU), which the dashboard reads.
    import inspect

    assert fs.checkpoint_stage(100) == "evaluating checkpoint n100"
    source = inspect.getsource(fs.main)
    named = source.index("report(stage=checkpoint_stage(done))")
    # Just before the checkpoint's held-out evaluation.
    gap = source.index("parts = held_out_parts(model, val", named) - named
    assert gap < 400
