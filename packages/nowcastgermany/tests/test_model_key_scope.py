"""A German model's key covers only the code it imports (#1114)."""

import pytest

pytest.importorskip("torch")

from nowcastgermany import checkpoint


def test_the_trainer_is_not_in_a_forecast_model_s_key(monkeypatch):
    seen = []
    real = checkpoint.codekey.digest

    def digest(files, *args):
        seen.extend(str(f) for f in files)
        return real(files, *args)

    monkeypatch.setattr(checkpoint.codekey, "digest", digest)
    assert len(checkpoint.model_key("foundation_zeroshot")) == 12
    assert any(f.endswith("forecaster.py") for f in seen)
    assert not any(f.endswith("finetune/stream.py") for f in seen)
    assert any(f.endswith("germany.toml") for f in seen)
