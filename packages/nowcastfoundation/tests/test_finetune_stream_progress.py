"""Batch progress of a held-out pass (`finetune.stream.batch_progress`)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("chronos")

from chronos_toy import toy_block  # noqa: E402

from nowcastfoundation.finetune import stream as fs  # noqa: E402


@pytest.fixture
def short_context(monkeypatch):
    # The toy blocks have 48 steps of context, not the trainer's 288.
    monkeypatch.setattr(fs.ft, "CONTEXT", 48)


@pytest.mark.usefixtures("short_context")
def test_each_batch_is_reported_with_the_pass_total():
    from chronos.chronos2.dataset import DatasetMode

    # Nine rows a block, a batch from nine rows: one block a batch.
    data = fs.ByLength(
        [toy_block(i) for i in range(3)], 9, DatasetMode.VALIDATION
    )
    seen = []
    with fs.batch_progress(lambda k, n: seen.append((k, n))):
        got = list(data.indices())
    assert got == [[0], [1], [2]]
    assert seen == [(1, 3), (2, 3), (3, 3)]
    # Outside the context nothing is reported.
    list(data.indices())
    assert len(seen) == 3
