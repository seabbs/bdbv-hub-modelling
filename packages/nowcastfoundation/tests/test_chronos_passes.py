"""Chronos-2 forward passes packed to the device's memory (#1114)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from nowcastfoundation.backends import chronos2

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


def _group(rows: int, width: int = 64) -> dict:
    return {"target": np.ones((rows, width))}


def test_a_token_budget_packs_by_series_times_tokens():
    groups = [_group(10), _group(10), _group(10), _group(300)]
    cost = chronos2.group_cost(groups[0], 16)
    assert cost == 10 * (64 // 16 + 1 + 1)
    got = chronos2._passes(groups, rows=None, tokens=2 * cost, horizon=16)
    assert got == [[0, 1], [2], [3]]
    # Without a token budget, rows as before.
    assert chronos2._passes(groups, rows=20, horizon=16) == [[0, 1], [2], [3]]


def test_the_budget_is_rows_on_the_cpu_and_memory_on_cuda(monkeypatch):
    monkeypatch.delenv("CHRONOS_BATCH", raising=False)
    assert chronos2.pass_budget("cpu") == (256, None)
    monkeypatch.setattr(
        chronos2, "device_memory", lambda: (6 * 2**30, 8 * 2**30)
    )
    rows, tokens = chronos2.pass_budget("cuda")
    assert rows is None
    spare = 6 * 2**30 - int(chronos2.MEMORY_MARGIN * 8 * 2**30)
    assert tokens == spare // chronos2.INFERENCE_BYTES
    # CHRONOS_BATCH set on a machine keeps its rows.
    monkeypatch.setenv("CHRONOS_BATCH", "64")
    assert chronos2.pass_budget("cuda") == (64, None)


class _Pipeline:
    quantiles = (0.1, 0.5, 0.9)

    def __init__(self, most: int) -> None:
        self.most = most
        self.passes: list[int] = []

    def predict_quantiles(
        self, inputs, prediction_length, quantile_levels, **_
    ):
        import torch

        if len(inputs) > self.most:
            raise torch.OutOfMemoryError("too big")
        self.passes.append(len(inputs))
        q = np.zeros((1, prediction_length, len(quantile_levels)))
        return [torch.tensor(q) for _ in inputs], None


def test_a_pass_out_of_memory_is_split(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("CHRONOS_BATCH", "1000")
    pipe = _Pipeline(most=1)
    got = chronos2._predict(
        pipe, [_group(1) for _ in range(4)], 3, np.asarray(pipe.quantiles)
    )
    assert len(got) == 4
    assert all(g is not None for g in got)
    assert pipe.passes == [1, 1, 1, 1]
