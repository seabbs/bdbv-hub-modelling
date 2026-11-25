"""A training step keeps the bias's rows until backward is done, and is
recomputed when its activations would not fit the device's memory."""

import inspect
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")
pytest.importorskip("peft")

from chronos_toy import batch_of, tiny_model, toy_block  # noqa: E402
from test_series_bias import toy_rows  # noqa: E402

from nowcastfoundation import series_bias  # noqa: E402
from nowcastfoundation.finetune import stream as fs  # noqa: E402


def step_grads(monkeypatch, checkpointed: bool):
    from chronos.chronos2.model import Chronos2EncoderBlock, Chronos2Model

    model = tiny_model()
    series_bias.attach(model)
    model.train()
    module = getattr(model, series_bias.BIAS_MODULE)
    with torch.no_grad():
        for i, p in enumerate(module.parameters()):
            p.fill_(0.1 * (i + 1))
    if checkpointed:
        monkeypatch.setattr(
            Chronos2EncoderBlock, "forward", Chronos2EncoderBlock.forward
        )
        fs.checkpoint_large_batches(0)
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    accepted = inspect.signature(Chronos2Model.forward).parameters
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    return [p.grad.clone() for p in module.parameters()]


def test_a_checkpointed_step_trains_the_bias_as_a_plain_one(monkeypatch):
    plain = step_grads(monkeypatch, False)
    checked = step_grads(monkeypatch, True)
    for a, b in zip(plain, checked, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_a_block_under_the_threshold_is_recomputed_when_asked(monkeypatch):
    # A block too large for memory alone is run again with every encoder
    # block checkpointed (`recompute_every_block`), whatever its series.
    import torch.utils.checkpoint as tc
    from chronos.chronos2.model import Chronos2EncoderBlock, Chronos2Model

    calls: list[int] = []
    real = tc.checkpoint

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tc, "checkpoint", spy)
    monkeypatch.setattr(
        Chronos2EncoderBlock, "forward", Chronos2EncoderBlock.forward
    )
    fs.checkpoint_large_batches(10**9)
    # No series bias, whose time chunks are checkpointed too.
    model = tiny_model()
    model.train()
    batch = batch_of(toy_block())
    accepted = inspect.signature(Chronos2Model.forward).parameters
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert not calls
    with fs.recompute_every_block():
        fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert calls
    calls.clear()
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert not calls


def test_assigned_restores_the_rows_it_replaced():
    model = tiny_model()
    series_bias.attach(model)
    state = getattr(model, series_bias._STATE)
    with series_bias.assigned(model, toy_rows()):
        outer = state["rows"]
        with series_bias.assigned(model, toy_rows()):
            pass
        assert state["rows"] is outer
    assert state["rows"] is None


def spy_checkpoints(monkeypatch, above: int | None = 10**9) -> list[int]:
    """Count checkpoints, encoder blocks checkpointed above `above`."""
    import torch.utils.checkpoint as tc
    from chronos.chronos2.model import Chronos2EncoderBlock

    calls: list[int] = []
    real = tc.checkpoint

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tc, "checkpoint", spy)
    monkeypatch.setattr(
        Chronos2EncoderBlock, "forward", Chronos2EncoderBlock.forward
    )
    fs.checkpoint_large_batches(above)
    return calls


class FakePeak:
    """A meter that reports a set peak, as `CudaPeak` does on a GPU."""

    def __init__(self, peak: int) -> None:
        self.value = peak
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def peak(self) -> int:
        return self.value


def test_a_batch_over_the_budget_is_recomputed_up_front(monkeypatch):
    # A step predicted over the device's total less the trainer's footprint
    # and the margin is recomputed from the start, not after it runs out.
    from chronos.chronos2.model import Chronos2Model

    calls = spy_checkpoints(monkeypatch)
    model = tiny_model()
    model.train()
    batch = batch_of(toy_block())
    accepted = inspect.signature(Chronos2Model.forward).parameters
    series, tokens = batch["context"].shape[0], fs.batch_tokens(batch)
    need = series * tokens * fs.ACTIVATION_BYTES
    total = 8 * 10**9
    margin = int(fs.MEMORY_MARGIN * total)
    roomy = fs.MemoryRule(total, total - margin - need)
    meter = FakePeak(1234)
    fs.train_step(model, batch, accepted, "cpu", 1.0, roomy, meter)
    assert not calls
    assert meter.started == 1
    assert roomy.rows[-1] == {
        "series": series,
        "tokens": tokens,
        "recomputed": False,
        "kept_bias": False,
        "predicted": need,
        "measured": 1234,
        "cost": fs.ACTIVATION_BYTES,
    }
    tight = fs.MemoryRule(total, total - margin - need + 1)
    fs.train_step(model, batch, accepted, "cpu", 1.0, tight, FakePeak(1))
    assert calls
    assert tight.rows[-1]["recomputed"]
    calls.clear()
    # Off CUDA there is no rule: the series threshold alone.
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert not calls
    # The rule does not close a recompute opened around it.
    with fs.recompute_every_block():
        fs.train_step(model, batch, accepted, "cpu", 1.0, roomy, meter)
        assert fs._RECOMPUTING["all"]
    assert roomy.rows[-1]["recomputed"]


def test_a_step_that_runs_out_raises_the_cost_to_recompute_its_size():
    model = tiny_model()
    batch = batch_of(toy_block())
    series, tokens = batch["context"].shape[0], fs.batch_tokens(batch)
    rule = fs.MemoryRule(8 * 10**9, 0)
    assert not rule.needs_recompute(series, tokens)

    def oom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    with (
        mock.patch.object(fs, "forward", oom),
        pytest.raises(RuntimeError, match="out of memory"),
    ):
        fs.train_step(model, batch, {}, "cpu", 1.0, rule, FakePeak(0))
    assert rule.needs_recompute(series, tokens)


def test_the_cost_is_fitted_from_measured_peaks_and_kept_in_the_state():
    rule = fs.MemoryRule(8 * 10**9, 10**9)
    assert rule.cost == fs.ACTIVATION_BYTES
    # Small and recomputed steps do not fit the cost.
    rule.record(10, 10, False, 10**9)
    rule.record(1000, 10, True, 10**12)
    assert rule.measured == 0
    for i in range(fs.FIT_AFTER - 1):
        rule.record(1000, 10, False, 1000 * 10 * (200_000 + i))
    assert rule.cost == fs.ACTIVATION_BYTES
    rule.record(1000, 10, False, 1000 * 10 * 300_000)
    # The most a measured step used per series and token.
    assert rule.cost == 300_000
    again = fs.MemoryRule(8 * 10**9, 10**9, rule.state())
    assert again.cost == rule.cost
    assert again.needs_recompute(2000, 10) == rule.needs_recompute(2000, 10)
    # A step that ran out sets a floor that recomputes its size.
    again.ran_out(2000, 9)
    assert again.needs_recompute(2000, 9)
    assert fs.MemoryRule(1, 0, again.state()).floor == again.floor


def test_the_memory_rows_are_written_and_trimmed_on_resume(tmp_path):
    rule = fs.MemoryRule(8 * 10**9, 10**9)
    for step in (1, 2, 3):
        rule.record(100, 10, False, 5)
        fs.record_memory(tmp_path, step, rule)
    assert not rule.rows
    fs.trim_memory(tmp_path, 2)
    frame = pd.read_csv(tmp_path / fs.MEMORY_FILE)
    assert frame["step"].tolist() == [1, 2]
    assert {"predicted", "measured", "recomputed"} <= set(frame.columns)


def test_batch_tokens_counts_context_patches_reg_and_output_patches():
    batch = {"context": np.zeros((5, 40)), "num_output_patches": 3}
    # Three input patches (40 steps), the reg token and three outputs.
    assert fs.batch_tokens(batch) == 3 + 1 + 3


def test_the_footprint_counts_weights_and_the_trained_parameters_thrice():
    model = torch.nn.Linear(4, 2)
    model.bias.requires_grad_(False)
    weights = (8 + 2) * 4
    assert fs.fixed_footprint(model) == weights + 3 * 8 * 4


@pytest.mark.parametrize(
    ("series", "tokens", "ran_out"),
    [
        # Run runa, n1600 on archie (RTX 2070, 8 GB): blocks under
        # `checkpoint_above` that ran out of memory alone, by series and
        # tokens (trimmed context patches, the reg token and output
        # patches), from the saved chunk blocks.
        (630, 21, True),
        (725, 18, True),
        (720, 17, True),
        (696, 17, True),
        (660, 15, True),
        # Blocks of the same chunks that trained without recomputation.
        (300, 12, False),
        (135, 15, False),
        (500, 10, False),
    ],
)
def test_archie_blocks_that_ran_out_are_recomputed(series, tokens, ran_out):
    # Chronos-2 with its LoRA adapter holds about 0.48 GB.
    rule = fs.MemoryRule(8_161_329_152, 480_000_000)
    assert rule.needs_recompute(series, tokens) is ran_out


def bias_step(monkeypatch, rule, meter=None):
    """Run a bias model's step under `rule`: its grads and checkpoints."""
    from chronos.chronos2.model import Chronos2Model

    calls = spy_checkpoints(monkeypatch)
    model = tiny_model()
    series_bias.attach(model)
    model.train()
    module = getattr(model, series_bias.BIAS_MODULE)
    with torch.no_grad():
        for i, p in enumerate(module.parameters()):
            p.fill_(0.1 * (i + 1))
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    accepted = inspect.signature(Chronos2Model.forward).parameters
    fs.train_step(
        model, batch, accepted, "cpu", 1.0, rule, meter or FakePeak(1)
    )
    grads = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return grads, len(calls), batch


def toy_rule(free: int) -> fs.MemoryRule:
    """A rule for the tiny model with the bias, its budget `free` bytes."""
    total = 8 * 10**9
    margin = int(fs.MEMORY_MARGIN * total)
    shape = fs.BiasShape(layers=2, heads=4, d_model=32)
    return fs.MemoryRule(total, total - margin - free, bias=shape)


def test_a_step_keeps_the_bias_when_it_fits_with_the_same_gradients(
    monkeypatch,
):
    # The series bias's group attention is recomputed in backward only
    # when the step would not fit holding it; the gradients are the same.
    batch = batch_of(toy_block())
    series, tokens = batch["context"].shape[0], fs.batch_tokens(batch)
    roomy = toy_rule(10**9)
    assert roomy.keeps_bias(series, tokens)
    kept, kept_calls, _ = bias_step(monkeypatch, roomy)
    assert kept_calls == 0
    assert roomy.rows[-1]["kept_bias"]
    need = roomy.predict(series, tokens)
    tight = toy_rule(need + roomy.kept_bytes(series, tokens) - 1)
    assert not tight.needs_recompute(series, tokens)
    assert not tight.keeps_bias(series, tokens)
    checked, checked_calls, _ = bias_step(monkeypatch, tight)
    # Only the bias's time chunks are checkpointed, not whole blocks.
    assert checked_calls > 0
    assert not tight.rows[-1]["recomputed"]
    assert not tight.rows[-1]["kept_bias"]
    assert kept.keys() == checked.keys()
    for name in kept:
        torch.testing.assert_close(kept[name], checked[name], rtol=0, atol=0)


def test_a_model_without_the_bias_keeps_nothing_extra():
    rule = fs.MemoryRule(8 * 10**9, 0)
    assert rule.kept_bytes(100, 10) == 0
    assert not rule.keeps_bias(100, 10)


def test_the_bias_held_counts_its_saved_mask_and_group_attention():
    shape = fs.BiasShape(layers=12, heads=12, d_model=768)
    rule = fs.MemoryRule(8 * 10**9, 0, bias=shape)
    series, tokens = 400, 9
    pairs = 12 * 12 * tokens * series**2 * 4
    states = series * tokens * 12 * fs.KEPT_STATES * 768 * 4
    assert rule.kept_bytes(series, tokens) == pairs + states


def test_a_kept_step_fits_the_cost_without_what_the_bias_held():
    rule = toy_rule(10**9)
    held = rule.kept_bytes(1000, 10)
    for _ in range(fs.FIT_AFTER):
        rule.record(1000, 10, False, 1000 * 10 * 300_000 + held, held)
    assert rule.cost == 300_000
    row = rule.rows[-1]
    assert row["predicted"] == row["cost"] * 1000 * 10 + held


def test_a_kept_step_that_runs_out_checkpoints_the_bias_at_its_size():
    rule = toy_rule(10**9)
    series, tokens = 30, 10
    assert rule.keeps_bias(series, tokens)
    rule.ran_out(series, tokens, kept_bias=True)
    assert not rule.keeps_bias(series, tokens)
    # The step itself still runs without recomputing its blocks.
    assert not rule.needs_recompute(series, tokens)
    again = fs.MemoryRule(rule.total, rule.footprint, rule.state(), rule.bias)
    assert not again.keeps_bias(series, tokens)


def test_a_step_that_runs_out_before_the_rule_changes_is_run_again():
    # Out of memory with a cheaper choice than the rule now makes: the
    # same batch is run again whole, so its gradient is unchanged.
    model = tiny_model()
    series_bias.attach(model)
    batch = {**batch_of(toy_block()), "series_rows": toy_rows()}
    series, tokens = batch["context"].shape[0], fs.batch_tokens(batch)
    rule = toy_rule(10**9)

    def oom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    with (
        mock.patch.object(fs, "forward", oom),
        pytest.raises(fs.RunAgainError, match="out of memory"),
    ):
        fs.train_step(model, batch, {}, "cpu", 1.0, rule, FakePeak(0))
    assert not rule.keeps_bias(series, tokens)
    assert not rule.needs_recompute(series, tokens)
    with (
        mock.patch.object(fs, "forward", oom),
        pytest.raises(fs.RunAgainError),
    ):
        fs.train_step(model, batch, {}, "cpu", 1.0, rule, FakePeak(0))
    assert rule.needs_recompute(series, tokens)
    # Recomputed already: nothing cheaper to give up.
    with (
        mock.patch.object(fs, "forward", oom),
        pytest.raises(RuntimeError, match="out of memory") as raised,
    ):
        fs.train_step(model, batch, {}, "cpu", 1.0, rule, FakePeak(0))
    assert not isinstance(raised.value, fs.RunAgainError)


def test_with_no_series_threshold_only_a_recompute_checkpoints(monkeypatch):
    from chronos.chronos2.model import Chronos2Model

    calls = spy_checkpoints(monkeypatch, None)
    model = tiny_model()
    model.train()
    batch = batch_of(toy_block())
    accepted = inspect.signature(Chronos2Model.forward).parameters
    fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert not calls
    with fs.recompute_every_block():
        fs.train_step(model, batch, accepted, "cpu", 1.0)
    assert calls


def test_the_series_threshold_applies_only_without_a_rule():
    assert fs.series_threshold(733, None) == 733
    assert fs.series_threshold(733, fs.MemoryRule(8 * 10**9, 0)) is None
