"""The torch device comes from config, never detected."""

import pytest

from nowcastfoundation.backends import chronos2


def test_the_device_is_configured_never_picked(monkeypatch):
    # The workflow config (`forecast.device`, the trainer's DEVICE) or a
    # caller names the device; the backend never detects one.
    monkeypatch.delenv("DEVICE", raising=False)
    with pytest.raises(ValueError, match="DEVICE"):
        chronos2.torch_device()
    with pytest.raises(ValueError, match="auto"):
        chronos2.torch_device("auto")
    monkeypatch.setenv("DEVICE", "cpu")
    assert chronos2.torch_device() == "cpu"
    assert chronos2.torch_device("cuda:1") == "cuda:1"
