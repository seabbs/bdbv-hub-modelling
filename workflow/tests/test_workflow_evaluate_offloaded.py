"""The runner of the evaluation rules (evaluate_offloaded.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("chronos")

ROOT = Path(__file__).parents[2]


def runner():
    spec = importlib.util.spec_from_file_location(
        "evaluate_offloaded", ROOT / "workflow/scripts/evaluate_offloaded.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(train, step, kinds):
    where = train / "evals" / f"step-{step:07d}"
    (where / "adapter").mkdir(parents=True)
    (where / "request.json").write_text(
        json.dumps({"step": step, "kinds": kinds, "n_validation": 2})
    )


@pytest.fixture
def fake(monkeypatch):
    mod = runner()
    calls = {"loads": 0, "evals": []}

    class Base:
        def to(self, _device):
            return self

    def load_base(_device):
        calls["loads"] += 1
        return Base()

    def evaluate(adapter, _base, _blocks, kinds, _batch, real=()):
        calls["evals"].append((Path(adapter), list(kinds), len(real)))
        if "start" in kinds:
            return {"held_out_loss": 0.4}
        if "checkpoint" in kinds:
            return {"held_out_loss": 0.25, "held_out_loss_by_level": {}}
        return {"eval_loss": 0.3}

    monkeypatch.setattr(mod.fe, "load_base", load_base)
    monkeypatch.setattr(mod.fe, "evaluate", evaluate)
    monkeypatch.setattr(mod.fe, "load_blocks", lambda *_a, **_k: [{}, {}])
    monkeypatch.setattr(mod.fe.series_bias, "attach_for", lambda *_a: None)
    return mod, calls


def checkpoint_at(train, n, steps):
    where = train / f"n{n}"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "settings.json").write_text(
        json.dumps({"n_validation": 2, "n_outbreaks": n, "steps": steps})
    )
    return where


def test_checkpoint_writes_the_sidecar_with_the_start_loss(tmp_path, fake):
    mod, calls = fake
    train = tmp_path / "train"
    (train / "n500" / "finetuned-ckpt").mkdir(parents=True)
    (train / "n500" / "settings.json").write_text(
        json.dumps({"n_validation": 2, "n_outbreaks": 500, "steps": 31})
    )
    (train / "evals").mkdir()
    (train / "evals" / "step-0000000.json").write_text(
        json.dumps({"held_out_loss": 0.4})
    )
    out = train / "n500" / "held-out.json"
    mod.main(["checkpoint", str(train / "n500"), "--device", "cpu"])
    got = json.loads(out.read_text())
    assert got["held_out_loss"] == 0.25
    assert got["held_out_loss_start"] == 0.4
    assert calls["evals"][-1][1] == ["checkpoint"]
    assert calls["evals"][-1][0] == train / "n500" / "finetuned-ckpt"


def test_a_checkpoint_with_no_base_loss_fails_clearly(tmp_path, fake):
    mod, _calls = fake
    where = checkpoint_at(tmp_path / "train", 500, 31)
    with pytest.raises(SystemExit, match="base Chronos-2 loss"):
        mod.main(["checkpoint", str(where), "--device", "cpu"])


def test_base_scores_only_step_0_and_once(tmp_path, fake):
    mod, calls = fake
    train = tmp_path / "train"
    for step, kinds in ((0, ["start"]), (200, ["monitor"])):
        snapshot(train, step, kinds)
    first = checkpoint_at(train, 250, 200)
    mod.main(["base", str(first), "--device", "cpu"])
    assert [c[1] for c in calls["evals"]] == [["start"]]
    assert (train / "evals" / "step-0000000.json").exists()
    assert not (train / "evals" / "step-0000200.json").exists()
    mod.main(["base", str(first), "--device", "cpu"])
    assert len(calls["evals"]) == 1


def test_step_snapshots_are_not_scored(fake):
    # Only checkpoints are scored offloaded.
    mod, _ = fake
    with pytest.raises(SystemExit):
        mod.main(["steps", "n250", "--device", "cpu", "--eval-every", "200"])


def test_evaluation_logs_each_batch(tmp_path, fake, monkeypatch, capsys):
    mod, _ = fake
    train = tmp_path / "train"
    snapshot(train, 0, ["start"])

    def evaluate(*_a, **_k):
        mod.fs.report_batch(1, 2)
        mod.fs.report_batch(2, 2)
        return {"held_out_loss": 0.4}

    monkeypatch.setattr(mod.fe, "evaluate", evaluate)
    mod.main(["base", str(train / "n100"), "--device", "cpu"])
    out = capsys.readouterr().out
    assert "step-0000000: batch 1/2" in out
    assert "step-0000000: batch 2/2" in out
    # Outside an evaluation nothing is logged.
    mod.fs.report_batch(1, 2)
    assert "batch" not in capsys.readouterr().out


def test_base_chronos_2_s_step_0_is_cached_by_model_and_held_out_set(
    tmp_path, fake
):
    mod, calls = fake
    cache = tmp_path / "cache"
    for run in ("a", "b"):
        train = tmp_path / run / "train"
        snapshot(train, 0, ["start"])
        (train / "held-out-blocks.json").write_text('{"outbreaks": [1]}')
        first = checkpoint_at(train, 250, 200)
        mod.main(["base", str(first), "--device", "cpu", "--cache", str(cache)])
        got = json.loads((train / "evals" / "step-0000000.json").read_text())
        assert got["held_out_loss"] == 0.4
    # Scored once: the second run takes the first's from the cache.
    assert [c[1] for c in calls["evals"]] == [["start"]]
    assert len(list(cache.glob("*.json"))) == 1


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_the_runner_writes_the_same_results_over_workers(tmp_path, monkeypatch):
    # End to end on a tiny Chronos-2: step 0 and a checkpoint scored in
    # one process and over three workers write the same results.
    import multiprocessing
    import sys

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork on this platform")
    sys.path.insert(0, str(ROOT / "packages/nowcastfoundation/tests"))
    from chronos_toy import tiny_model
    from test_finetune_offload import blocks, trained

    mod = runner()
    fs = mod.fs
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)
    monkeypatch.setattr(mod.fe, "START_METHOD", "fork")
    monkeypatch.setattr(mod.fe, "load_base", lambda _device: tiny_model())
    out = {}
    for n in (1, 3):
        monkeypatch.setenv(mod.fe.WORKERS_ENV, str(n))
        train = tmp_path / f"w{n}" / "train"
        train.mkdir(parents=True)
        val = blocks(7)
        fs.save_blocks(val, train / fs.HELD_OUT_BLOCKS)
        model = trained()
        request = {"n_validation": len(val), "batch_size": 2}
        fs.save_snapshot(
            model,
            train / "evals" / "step-0000000",
            {**request, "step": 0, "kinds": ["start"]},
        )
        where = train / "n500"
        model.save_pretrained(where / "finetuned-ckpt")
        (where / "settings.json").write_text(
            json.dumps({**request, "steps": 200})
        )
        mod.main(["base", str(where), "--device", "cpu"])
        mod.main(["checkpoint", str(where), "--device", "cpu"])
        out[n] = [
            {
                k: v
                for k, v in json.loads(path.read_text()).items()
                if k != "seconds"
            }
            for path in (
                train / "evals" / "step-0000000.json",
                where / "held-out.json",
            )
        ]
    assert out[1] == out[3]
