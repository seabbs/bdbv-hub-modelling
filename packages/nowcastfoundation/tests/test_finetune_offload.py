"""Evaluation offloaded from the trainer (`finetune.stream --eval offload`)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from chronos_toy import tiny_model, toy_block  # noqa: E402

from nowcastfoundation import series_bias  # noqa: E402
from nowcastfoundation.finetune import evaluate as fe  # noqa: E402
from nowcastfoundation.finetune import stream as fs  # noqa: E402


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)


def blocks(n: int = 4) -> list[dict]:
    """Toy held-out blocks with levels and a source each."""
    return [
        {
            **toy_block(i),
            "levels": ["national", "leaf", "leaf"],
            "source": "daily" if i % 2 else "weekly",
        }
        for i in range(n)
    ]


def trained(bias: bool = True):
    """A tiny Chronos-2 with a LoRA adapter (and bias) moved off zero."""
    model = tiny_model()
    if bias:
        series_bias.attach(model)
    model = fs.lora_model(model, fs.lora_config(bias))
    with torch.no_grad():
        for i, p in enumerate(p for p in model.parameters() if p.requires_grad):
            p.add_(0.01 * (i + 1))
    return model


def test_offload_is_refused_while_early_stopping_can_act():
    fs.check_offload("offload", stop_from=8401, outbreaks=8400)
    fs.check_offload("local", stop_from=100, outbreaks=8400)
    with pytest.raises(ValueError, match="stop"):
        fs.check_offload("offload", stop_from=8400, outbreaks=8400)


def test_evaluation_mode_defaults_to_local():
    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2"]
    )
    assert args.eval == "local"


def test_a_snapshot_is_written_whole(tmp_path):
    model = trained()
    where = tmp_path / "evals" / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100, "kinds": ["monitor"]})
    assert (where / "adapter" / "adapter_model.safetensors").exists()
    assert series_bias.adapter_has_bias(where / "adapter")
    assert json.loads((where / "request.json").read_text())["step"] == 100
    assert sorted(p.name for p in where.parent.iterdir()) == ["step-0000100"]
    # Saving again replaces it.
    fs.save_snapshot(model, where, {"step": 100, "kinds": ["by_source"]})
    assert json.loads((where / "request.json").read_text())["kinds"] == [
        "by_source"
    ]


def test_an_offloaded_step_records_its_rates_and_saves_nothing(tmp_path):
    # Only step 0 and the checkpoints are scored offloaded (#1150), so a
    # step between leaves no snapshot: its history entry is all it keeps.
    model = trained()
    optimiser = fs.make_optimiser(model, 1e-4, 1e-3)
    entry = fs.offloaded_step(model, optimiser, 200)
    assert entry["step"] == 200
    assert entry["lr"] == fs.group_rates(optimiser)
    assert "bias" in entry
    assert not (tmp_path / "evals").exists()


@pytest.mark.usefixtures("short_context")
def test_the_evaluator_scores_a_snapshot_as_the_trainer_would(
    tmp_path, monkeypatch
):
    model = trained()
    val = blocks()
    where = tmp_path / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100})
    base = tiny_model()
    series_bias.attach(base)
    forwards = []
    forward = fs.forward

    def counted(*args, **kwargs):
        forwards.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(fs, "forward", counted)
    got = fe.evaluate(
        where / "adapter",
        base,
        val,
        ["monitor", "by_source", "checkpoint"],
        batch_size=100,
    )
    # One forward pass per batch gives every loss and the coverage.
    from chronos.chronos2.dataset import DatasetMode

    batches = len(list(fs.ByLength(val, 100, DatasetMode.VALIDATION).indices()))
    assert len(forwards) == batches
    monkeypatch.setattr(fs, "forward", forward)

    def near(a, b):
        return a == pytest.approx(b, rel=1e-5)

    assert near(
        got["eval_loss"], fs.held_out_loss(model, val[: fs.ft.EVAL_BLOCKS], 100)
    )
    assert near(
        got["by_source"],
        fs.held_out_by_source(model, val, 100, fs.ft.EVAL_BLOCKS),
    )
    assert near(got["held_out_loss"], fs.held_out_loss(model, val, 100))
    parts = fs.held_out_parts(model, val, 100)
    assert near(got["held_out_loss_nowcast"], parts["nowcast"])
    assert near(
        got["held_out_loss_by_level"],
        fs.held_out_by_level(model, val, fs.LEVEL_BLOCKS),
    )
    assert near(
        got["held_out_loss_by_source"], fs.held_out_by_source(model, val, 100)
    )
    assert near(got["held_out_coverage"], fe.held_out_coverage(model, val, 100))
    assert np.isfinite(got["eval_loss"])
    # Each total comes with its nowcast and forecast parts.
    for name in ("eval_loss", "held_out_loss"):
        assert got[f"{name}_nowcast"] + got[f"{name}_forecast"] == (
            pytest.approx(got[name], rel=1e-6)
        )


@pytest.mark.usefixtures("short_context")
def test_blocks_pack_by_length_into_a_series_token_budget():
    from chronos.chronos2.dataset import DatasetMode

    val = blocks(6)
    data = fs.ByLength(val, 100, DatasetMode.VALIDATION)
    groups = fe.pack(data, budget=2 * fe.block_cost(val[0]))
    # Every block once, each batch of one length within the budget.
    assert sorted(i for g in groups for i in g) == list(range(6))
    assert all(len(g) <= 2 for g in groups)
    assert all(
        len({val[i]["prediction_length"] for i in g}) == 1 for g in groups
    )
    # A block over the budget runs alone.
    assert fe.pack(data, budget=1) == [[i] for i in range(6)]


@pytest.mark.usefixtures("short_context")
def test_packed_batches_give_the_same_losses():
    model = trained()
    val = blocks(6)
    one = fe.held_out_pass(model, val, 100)
    packed = fe.held_out_pass(model, val, 100, budget=10**9)
    for a, b in zip(one, packed, strict=True):
        assert b["loss"] == pytest.approx(a["loss"], rel=1e-5)


@pytest.mark.usefixtures("short_context")
def test_a_batch_out_of_memory_is_split(monkeypatch):
    model = trained()
    val = blocks(4)
    want = fe.held_out_pass(model, val, 100)
    forward = fs.forward

    def tight(model, batch, *args, **kwargs):
        if len(batch["target_idx_ranges"]) > 1:
            raise torch.OutOfMemoryError("too big")
        return forward(model, batch, *args, **kwargs)

    monkeypatch.setattr(fs, "forward", tight)
    got = fe.held_out_pass(model, val, 100, budget=10**9)
    for a, b in zip(want, got, strict=True):
        assert b["loss"] == pytest.approx(a["loss"], rel=1e-5)


@pytest.mark.usefixtures("short_context")
def test_half_precision_is_checked_against_fp32(monkeypatch):
    model = trained()
    val = blocks(4)
    monkeypatch.setenv("PRECISION", "fp32")
    assert fe.precision_check(model, val, 100, "cpu") is None
    monkeypatch.setenv("PRECISION", "bf16")
    got = fe.precision_check(model, val, 100, "cpu")
    assert got["precision"] == "bf16"
    assert got["blocks"] == 4
    assert got["rel"] == pytest.approx(
        abs(got["loss"] - got["loss_fp32"]) / got["loss_fp32"]
    )
    assert got["rel"] < fe.PRECISION_TOLERANCE


def test_held_out_blocks_round_trip_with_their_count(tmp_path):
    val = blocks(3)
    path = tmp_path / "held-out-blocks.pkl"
    fs.save_blocks(val, path)
    got = fe.load_blocks(path, n=3)
    assert len(got) == 3
    assert np.array_equal(got[1]["context"], val[1]["context"], equal_nan=True)
    with pytest.raises(ValueError, match="3"):
        fe.load_blocks(path, n=4)


def _snapshot_run(tmp_path, monkeypatch, n_validation: int) -> Path:
    """A run with a step snapshot and its held-out blocks, base stubbed."""
    monkeypatch.setattr(fe, "load_base", lambda _device: tiny_model())
    train = tmp_path / "train"
    train.mkdir()
    val = blocks()
    fs.save_blocks(val, train / fs.HELD_OUT_BLOCKS)
    where = train / "evals" / "step-0000100"
    fs.save_snapshot(
        trained(),
        where,
        {
            "step": 100,
            "kinds": ["monitor", "by_source"],
            "batch_size": 100,
            "n_validation": n_validation,
        },
    )
    return where


@pytest.mark.usefixtures("short_context")
def test_a_snapshot_is_scored_as_its_request_asks(tmp_path, monkeypatch):
    where = _snapshot_run(tmp_path, monkeypatch, n_validation=4)
    blocks = fe.load_blocks(tmp_path / "train" / fs.HELD_OUT_BLOCKS, 4)
    got = fe.snapshot_result(fe.load_base("cpu"), where, blocks, "cpu")
    assert got["step"] == 100
    assert got["kinds"] == ["monitor", "by_source"]
    assert {"eval_loss", "eval_loss_nowcast", "by_source"} <= set(got)
    assert "held_out_loss" not in got
    out = tmp_path / "result.json"
    fe.write_json(out, got)
    assert json.loads(out.read_text())["step"] == 100
    assert not out.with_name(out.name + ".tmp").exists()


@pytest.mark.usefixtures("short_context")
def test_the_evaluate_rules_runner_gives_the_trainer_s_losses(
    tmp_path, monkeypatch
):
    # WP10 part 2: workflow/scripts/evaluate_offloaded.py, end to end on
    # CPU with the real evaluator, scores step 0 and a checkpoint exactly
    # as the trainer scores the same adapter.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_offloaded",
        Path(__file__).parents[3] / "workflow/scripts/evaluate_offloaded.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    monkeypatch.setattr(runner.fe, "load_base", lambda _device: tiny_model())
    model = trained()
    val = blocks()
    train = tmp_path / "train"
    train.mkdir()
    fs.save_blocks(val, train / fs.HELD_OUT_BLOCKS)
    request = {"n_validation": len(val), "batch_size": 100}
    fs.save_snapshot(
        model,
        train / "evals" / "step-0000000",
        {**request, "step": 0, "kinds": ["start"]},
    )
    fs.save_snapshot(
        model,
        train / "evals" / "step-0000200",
        {**request, "step": 200, "kinds": ["monitor"]},
    )
    where = train / "n500"
    model.save_pretrained(where / "finetuned-ckpt")
    (where / "settings.json").write_text(
        json.dumps({"n_validation": len(val), "batch_size": 100, "steps": 200})
    )
    runner.main(["base", str(where), "--device", "cpu"])
    runner.main(["checkpoint", str(where), "--device", "cpu"])
    # Step snapshots other than step 0 are left unscored.
    assert not (train / "evals" / "step-0000200.json").exists()
    sidecar = json.loads((where / "held-out.json").read_text())
    assert sidecar["held_out_loss"] == pytest.approx(
        fs.held_out_loss(model, val, 100), rel=1e-5
    )
    assert sidecar["held_out_loss_by_source"] == pytest.approx(
        fs.held_out_by_source(model, val, 100), rel=1e-5
    )
    start = json.loads((train / "evals" / "step-0000000.json").read_text())
    assert sidecar["held_out_loss_start"] == start["held_out_loss"]


@pytest.mark.usefixtures("short_context")
@pytest.mark.parametrize("kind", ["start", "checkpoint"])
def test_checkpoints_and_the_start_get_interval_coverage(tmp_path, kind):
    model = trained()
    where = tmp_path / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100})
    base = tiny_model()
    series_bias.attach(base)
    got = fe.evaluate(where / "adapter", base, blocks(), [kind], 100)
    cover = got["held_out_coverage"]
    for prefix in ("", "nowcast_", "forecast_", "source_daily_", "level_leaf_"):
        assert 0 <= cover[f"{prefix}cover50"] <= cover[f"{prefix}cover90"] <= 1
    # Every scored step is a nowcast step or a forecast step.
    assert cover["nowcast_n"] + cover["forecast_n"] == cover["n"] > 0
    assert cover["source_daily_n"] + cover["source_weekly_n"] == cover["n"]


def rolling_toy(seed: int) -> dict:
    """A toy daily+rolling7 block: each place's counts, sums, then lags."""
    block = toy_block(seed)
    x = block["context"]
    n = block["n_targets"]
    sums = np.stack([np.convolve(r, np.ones(7))[: x.shape[1]] for r in x[:n]])
    return {
        **block,
        "context": np.concatenate([x[:n], sums, x[n:]]).astype(np.float32),
        "n_targets": 2 * n,
        "levels": ["national", "leaf", "leaf"] * 2,
        "source": "daily+rolling7",
    }


@pytest.mark.usefixtures("short_context")
def test_a_checkpoint_s_7_day_check_runs_on_its_7_day_blocks(tmp_path):
    # Lead 00:55: one 7-day check per checkpoint (rule seven_day_check),
    # not a second one inside the held-out evaluation.
    model = trained()
    where = tmp_path / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100})
    base = tiny_model()
    series_bias.attach(base)
    val = [*blocks(2), rolling_toy(5), rolling_toy(6)]
    got = fe.evaluate(where / "adapter", base, val, ["checkpoint"], 100)
    assert "held_out_rolling" not in got
    check = fe.seven_day_check(where / "adapter", base, val, 100)
    assert check["n"] > 0
    for key in ("ratio", "inside90", "rough", "rough_truth", "rough_ratio"):
        assert np.isfinite(check[key])


@pytest.mark.usefixtures("short_context")
def test_the_adapter_is_merged_before_scoring(tmp_path, monkeypatch):
    # LoRA's extra low-rank products cost about an eighth of a CPU pass;
    # merged weights give the same losses (#1114).
    model = trained()
    val = blocks()
    where = tmp_path / "step-0000100"
    fs.save_snapshot(model, where, {"step": 100})
    base = tiny_model()
    series_bias.attach(base)
    lora = []
    forward = fs.forward

    def watched(m, *args, **kwargs):
        lora.append(any("lora_A" in n for n, _ in m.named_modules()))
        return forward(m, *args, **kwargs)

    monkeypatch.setattr(fs, "forward", watched)
    got = fe.evaluate(where / "adapter", base, val, ["checkpoint"], 100)
    monkeypatch.setattr(fs, "forward", forward)
    assert lora
    assert not any(lora)
    assert got["held_out_loss"] == pytest.approx(
        fs.held_out_loss(model, val, 100), rel=1e-5
    )


def _mixed(n: int = 4) -> list[dict]:
    """Held-out blocks of two prediction lengths, interleaved."""
    out = []
    for i, block in enumerate(blocks(n)):
        if i % 2:
            block = {
                **block,
                "context": block["context"][:, :-8],
                "prediction_length": block["prediction_length"] - 8,
            }
        out.append(block)
    return out


@pytest.mark.usefixtures("short_context")
def test_predictions_come_in_the_blocks_order_packed_or_not():
    # Batches group blocks by length; each block's quantiles must still
    # be its own, in the order given (validation forecasts, 7-day check).
    model = trained()
    val = _mixed()
    one = fe._predictions(model, val, 100)
    packed = fe._predictions(model, val, 100, budget=10**9)
    for block, got, again in zip(val, one, packed, strict=True):
        assert got[0] is block
        assert again[0] is block
        assert got[1].shape[-1] >= block["prediction_length"]
        np.testing.assert_allclose(again[1], got[1], rtol=1e-4, atol=1e-5)


@pytest.mark.usefixtures("short_context")
def test_predictions_run_at_the_inference_precision(monkeypatch):
    model = trained()
    val = blocks(2)
    seen = []
    forward = fs.forward

    def watched(*args, **kwargs):
        seen.append(torch.is_autocast_enabled("cpu"))
        return forward(*args, **kwargs)

    monkeypatch.setattr(fs, "forward", watched)
    monkeypatch.setenv("PRECISION", "bf16")
    fe._predictions(model, val, 100)
    assert seen
    assert all(seen)


@pytest.mark.usefixtures("short_context")
def test_validation_records_name_their_own_blocks(monkeypatch):
    model = trained()
    val = _mixed(4)
    monkeypatch.setattr(fe, "validation_sample", lambda n, *_: list(range(n)))
    got = fe.validation_forecasts(model, val, 100)
    assert [r["block"] for r in got] == [0, 1, 2, 3]
    # Each record's truth is its own block's horizon.
    for r, block in zip(got, val, strict=True):
        assert len(r["rows"][0]["truth"]) == block["prediction_length"]
