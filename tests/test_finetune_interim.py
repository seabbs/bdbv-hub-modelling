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
        json.dumps(
            {
                "step": 48010,
                "done": 2000,
                "trained": [f"o{i}" for i in range(2100)],
                "history": [],
            }
        )
    )
    (train / "consumed.txt").write_text(
        "".join(f"step{i}\n" for i in range(48500))
    )
    ck = train / "n1600" / "finetuned-ckpt"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text('{"r": 8}')
    (train / "n1600" / "settings.json").write_text(
        json.dumps(
            {
                "hash": "abc",
                "n_outbreaks": 1600,
                "steps": 37374,
                "sims_done": 1600,
                "blocks": 50000,
                "tokens": 9,
                "held_out_loss": 0.18,
                "seconds": 1.0,
                "code_commit": "c0ffee",
                "code_sha256": {},
            }
        )
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
    assert settings["from_checkpoint"] == "n1600"


def test_an_interim_folder_is_not_a_saved_checkpoint(tmp_path):
    from nowcastfoundation.forecaster import newest_segment

    train, _ = _run(tmp_path)
    interim.export(train)
    assert (
        newest_segment(train) is None or newest_segment(train).name == "n1600"
    )


def test_the_interim_checkpoint_verifies_and_releases_like_a_saved_one(
    tmp_path,
):
    # Lead 01:50 (Sam): the interim is the hub's fine-tuned model, so it
    # is released through the workflow, named for its settings hash.
    from nowcastfoundation.finetune import verify

    train, _ = _run(tmp_path)
    where = interim.export(train)
    settings = json.loads((where / "settings.json").read_text())
    # Its own outbreaks (in the order trained) and steps, hashed.
    outbreaks = (where / "outbreaks.txt").read_text().split()
    assert outbreaks == [f"o{i}" for i in range(2000)]
    consumed = (where / "consumed.txt").read_text().splitlines()
    assert len(consumed) == 48010
    assert settings["sims_done"] == 2000
    # The last checkpoint's counts and losses are not its own.
    for key in ("blocks", "tokens", "held_out_loss", "seconds"):
        assert key not in settings
    assert verify.settings_problems(settings) == []
    assert verify.problems(where, read=lambda _c, _p: b"") == []
    assert settings["code_commit"] == "c0ffee"
