"""An interim checkpoint exported from a running segment's saved state."""

import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from nowcastfoundation.finetune import interim  # noqa: E402


def _run(tmp_path):
    train = tmp_path / "train"
    (train / "state").mkdir(parents=True)
    weights = {"base_model.model.a.lora_A.weight": torch.ones(2, 3)}
    torch.save(weights, train / "state" / "adapter.pt")
    (train / "state" / "state.json").write_text(
        json.dumps({"step": 48010, "done": 2000, "history": []})
    )
    ck = train / "n1600" / "finetuned-ckpt"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text('{"r": 8}')
    (train / "n1600" / "settings.json").write_text(
        json.dumps({"hash": "abc", "n_outbreaks": 1600, "steps": 37374})
    )
    return train, weights


def test_the_interim_checkpoint_is_the_saved_state_s_adapter(tmp_path):
    train, weights = _run(tmp_path)
    where = interim.export(train)
    assert where == train / "interim" / "s48010"
    got = safetensors.load_file(
        where / "finetuned-ckpt" / "adapter_model.safetensors"
    )
    assert set(got) == set(weights)
    assert torch.equal(
        got["base_model.model.a.lora_A.weight"], torch.ones(2, 3)
    )
    config = where / "finetuned-ckpt" / "adapter_config.json"
    assert json.loads(config.read_text()) == {"r": 8}
    settings = json.loads((where / "settings.json").read_text())
    assert settings["interim"] is True
    assert (settings["steps"], settings["n_outbreaks"]) == (48010, 2000)
    assert settings["hash"].startswith("interim-s48010-")
    assert settings["from_checkpoint"] == "n1600"


def test_an_interim_folder_is_not_a_saved_checkpoint(tmp_path):
    from nowcastfoundation.forecaster import newest_segment

    train, _ = _run(tmp_path)
    interim.export(train)
    assert (
        newest_segment(train) is None or newest_segment(train).name == "n1600"
    )
