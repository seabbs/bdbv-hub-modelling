"""A training step keeps the bias's rows until backward is done, and is
recomputed when its activations would not fit the device's free memory."""

import inspect

import numpy as np
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


def spy_checkpoints(monkeypatch) -> list[int]:
    """Count encoder blocks checkpointed, with no series threshold."""
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
    fs.checkpoint_large_batches(10**9)
    return calls


def test_a_batch_over_the_free_memory_is_recomputed_up_front(monkeypatch):
    # A step whose estimated activations exceed the device's free memory
    # less the margin is recomputed from the start, not after it runs out.
    from chronos.chronos2.model import Chronos2Model

    calls = spy_checkpoints(monkeypatch)
    model = tiny_model()
    model.train()
    batch = batch_of(toy_block())
    accepted = inspect.signature(Chronos2Model.forward).parameters
    need = fs.activation_bytes(
        batch["context"].shape[0], fs.batch_tokens(batch)
    )
    total = 8 * 10**9
    margin = int(fs.MEMORY_MARGIN * total)
    roomy = (need + margin, total)
    fs.train_step(model, batch, accepted, "cpu", 1.0, memory=lambda: roomy)
    assert not calls
    tight = (need + margin - 1, total)
    fs.train_step(model, batch, accepted, "cpu", 1.0, memory=lambda: tight)
    assert calls
    calls.clear()
    # Off CUDA there is no device memory to read: the series rule alone.
    fs.train_step(model, batch, accepted, "cpu", 1.0, memory=lambda: None)
    assert not calls
    # The memory rule does not close a recompute opened around it.
    with fs.recompute_every_block():
        fs.train_step(model, batch, accepted, "cpu", 1.0, memory=lambda: roomy)
        assert fs._RECOMPUTING["all"]


def test_batch_tokens_counts_context_patches_reg_and_output_patches():
    batch = {"context": np.zeros((5, 40)), "num_output_patches": 3}
    # Three input patches (40 steps), the reg token and three outputs.
    assert fs.batch_tokens(batch) == 3 + 1 + 3


@pytest.mark.parametrize(
    ("series", "tokens", "ran_out"),
    [
        # Run runa, n1600 on archie (RTX 2070, 8 GB, about 6.9 GB free to
        # the trainer): blocks under `checkpoint_above` that ran out of
        # memory alone, by series and tokens (trimmed context patches, the
        # reg token and output patches), from the saved chunk blocks.
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
    free, total = int(6.9e9), 8_161_329_152
    assert fs.needs_recompute(series, tokens, (free, total)) is ran_out
