"""Chronos-2 loads on the device the workflow resolves (`DEVICE`)."""

from __future__ import annotations

import pytest

pytest.importorskip("chronos")


@pytest.fixture
def loaded(monkeypatch):
    import chronos

    from nowcastfoundation import forecaster as foundation

    seen: dict[str, str] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, device_map=None, **_kw):
            seen["device"] = device_map
            return "pipeline"

    monkeypatch.setattr(chronos, "Chronos2Pipeline", Pipeline)
    foundation._chronos.cache_clear()
    yield foundation, seen
    foundation._chronos.cache_clear()


def test_chronos_loads_on_the_device_setting(loaded, monkeypatch):
    foundation, seen = loaded
    # A device auto-detection never gives, so the setting must be read.
    monkeypatch.setenv("DEVICE", "cuda:1")
    assert foundation._chronos(None) == "pipeline"
    assert seen["device"] == "cuda:1"


def test_auto_or_unset_is_refused_at_load(loaded, monkeypatch):
    # The device is configured, never detected.
    foundation, seen = loaded
    for value in ("auto", None):
        if value is None:
            monkeypatch.delenv("DEVICE", raising=False)
        else:
            monkeypatch.setenv("DEVICE", value)
        foundation._chronos.cache_clear()
        with pytest.raises(ValueError, match=r"DEVICE|auto"):
            foundation._chronos(None)
    assert "device" not in seen


def test_one_device_resolver_for_scoring_and_training():
    from nowcastfoundation.backends import chronos2

    assert chronos2.torch_device("cuda:1") == "cuda:1"
    from nowcastfoundation.finetune import stream

    path = __import__("pathlib").Path(stream.__file__)
    assert "def pick_device" not in path.read_text()
    assert not hasattr(stream, "pick_device")


def test_chronos_batch_and_threads_are_machine_settings(monkeypatch):
    import torch

    from nowcastfoundation import forecaster as foundation

    seen: dict = {"batch": []}

    class Pipeline:
        def predict_quantiles(self, inputs, batch_size=None, **_kw):
            seen["batch"].append(batch_size)
            return [None] * len(inputs), None

    # Each pass is one Chronos-2 batch of its rows, at most CHRONOS_BATCH.
    monkeypatch.setenv("CHRONOS_BATCH", "2")
    monkeypatch.setenv("TORCH_THREADS", "3")
    monkeypatch.setattr(torch, "set_num_threads", lambda n: seen.update(n=n))
    foundation._predict(
        Pipeline(),
        [{"target": [[1.0]]}] * 3,
        2,
        __import__("numpy").array([0.5]),
    )
    assert seen["batch"] == [2, 1]
    foundation.apply_threads()
    assert seen["n"] == 3


def test_the_inter_op_pool_takes_the_job_s_threads_too(monkeypatch):
    # torch's inter-op pool defaults to every core.
    import torch

    from nowcastfoundation.backends import chronos2

    seen = {}
    monkeypatch.setenv("TORCH_THREADS", "2")
    monkeypatch.setattr(
        torch, "set_num_threads", lambda n: seen.update(intra=n)
    )
    monkeypatch.setattr(
        torch, "set_num_interop_threads", lambda n: seen.update(inter=n)
    )
    chronos2.apply_threads()
    assert seen == {"intra": 2, "inter": 2}
