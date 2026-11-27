"""The real-only fine-tune: held-out dates, settings and its release."""

import json
import subprocess
from pathlib import Path

import pytest

from nowcastfoundation.finetune import realdata as fr
from nowcastfoundation.finetune import verify
from nowcastfoundation.forecaster import REALDATA_PREFIX


def _block(data: str, cut: str, lag: int = 14) -> dict:
    return {
        "data": data,
        "cut": cut,
        "source": "daily",
        "places": ("CD",),
        "lag": lag,
    }


def test_the_last_german_dates_are_held_out():
    days = [f"2021-12-{d:02d}" for d in range(1, 21)]
    blocks = [_block("germany", d) for d in days]
    blocks += [_block("bdbv", "2026-10-07")]
    train, held = fr.split_held_out(blocks, {"germany": 14, "bdbv": 0})
    assert {b["cut"] for b in held} == set(days[-14:])
    assert {b["data"] for b in held} == {"germany"}
    assert len(train) + len(held) == len(blocks)
    assert {b["cut"] for b in train if b["data"] == "germany"} == set(
        days[:-14]
    )


def test_a_block_id_names_its_data_cut_source_places_and_lag():
    got = fr.block_id(_block("bdbv", "2026-10-07", 21))
    assert got == "bdbv:2026-10-07:daily:CD:21"


def test_a_checkpoint_hash_is_the_verified_settings_hash():
    base = {"data": "real", "steps": 1000, "note": "x"}
    settings = fr.checkpoint_settings(base, 500, "abc", 1.5)
    assert settings["data"] == "real"
    assert settings["steps"] == 500
    assert settings["lr_steps"] == 1000
    assert "n_outbreaks" not in settings
    assert settings["hash"] == verify.settings_hash(settings)
    assert fr.is_real(settings)
    assert not fr.is_real({"hash": "abc"})


def test_until_must_be_a_checkpoint():
    fr.check_until(500, [250, 500, 1000])
    with pytest.raises(ValueError, match="not a checkpoint"):
        fr.check_until(300, [250, 500, 1000])


class _Gh:
    """Fake gh recording each command and the assets it would upload."""

    def __init__(self):
        self.commands = []

    def __call__(self, command, **_):
        self.commands.append(command)
        if command[2] == "view":
            return subprocess.CompletedProcess(command, 1, "", "not found")
        return subprocess.CompletedProcess(command, 0, "ok", "")


def test_the_release_is_realdata_ft_with_its_blocks(tmp_path: Path):
    ckpt = tmp_path / "finetuned-ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_text("w")
    (tmp_path / "settings.json").write_text(
        json.dumps({"hash": "abc", "data": "real", "steps": 1000})
    )
    (tmp_path / "blocks.txt").write_text("germany:2021-04-20:daily:CD:14\n")
    gh = _Gh()
    tag = fr.publish(tmp_path, run=gh)
    assert tag == f"{REALDATA_PREFIX}-abc"
    create = next(c for c in gh.commands if c[2] == "create")
    assert create[3] == "realdata-ft-abc"
    assets = {Path(f).name for f in create if Path(f).is_file()}
    assert "blocks.txt" in assets
    assert "outbreaks.txt" not in assets
    assert "--latest=false" in create
