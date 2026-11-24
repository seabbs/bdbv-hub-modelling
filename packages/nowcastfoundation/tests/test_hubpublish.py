"""Publishing a fine-tuned checkpoint to the Hugging Face Hub, offline.

The Hub is never reached: uploads go to a fake `huggingface_hub.HfApi`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from nowcastfoundation import forecaster as fc
from nowcastfoundation import hubpublish as hp
from nowcastfoundation.modelcard import CARD_FILE, NOT_EVALUATED

SETTINGS = {
    "hash": "abc123",
    "series_bias": {"pairs": "same place"},
    "lora": {"r": 8, "alpha": 16},
    "n_outbreaks": 100,
}


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A verified checkpoint folder with its held-out losses."""
    folder = tmp_path / "train" / "n100"
    adapter = folder / "finetuned-ckpt"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (folder / "settings.json").write_text(json.dumps(SETTINGS))
    (folder / "held-out.json").write_text(
        json.dumps({"held_out_loss": 0.3, "held_out_loss_start": 0.7})
    )
    return folder


def test_the_folder_reads_back_as_the_checkpoint(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = hp.build_folder(checkpoint, tmp_path / "hf")
    names = {p.name for p in folder.iterdir()}
    assert names == {
        *fc.ADAPTER_FILES,
        "settings.json",
        fc.SUMS_FILE,
        CARD_FILE,
    }
    # The Hub reader takes it as the checkpoint the folder came from.
    monkeypatch.setattr(
        fc, "_hub_file", lambda _repo, _rev, name: folder / name
    )
    fc.hub_checkpoint.cache_clear()
    got = fc.hub_checkpoint("org/m", "slots-abc123")
    assert got == fc.local_checkpoint(checkpoint)
    assert fc.finetuned_adapter(got) == folder


def test_the_card_reports_what_the_checkpoint_has(
    checkpoint: Path, tmp_path: Path
) -> None:
    scores = pd.DataFrame(
        {
            "study": ["BDBV"],
            "target": ["weekly"],
            "model": ["fine-tuned"],
            "reference": ["randomwalknowcast"],
            "metric": ["CRPS"],
            "relative": [0.9],
        }
    )
    card = (
        hp.build_folder(checkpoint, tmp_path / "hf", scores) / CARD_FILE
    ).read_text()
    assert "0.30" in card
    assert "| BDBV | weekly | fine-tuned |" in card
    # No 7-day check was run for this checkpoint.
    assert card.count(NOT_EVALUATED) == 1


def test_a_dry_run_writes_the_folder_only(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_upload(*_: object) -> str:
        raise AssertionError("a dry run uploaded")

    monkeypatch.setattr(hp, "upload", no_upload)
    out = tmp_path / "huggingface.json"
    assert (
        hp.main([str(checkpoint), str(out), "--repo", "org/m", "--dry-run"])
        == 0
    )
    record = json.loads(out.read_text())
    assert record == {
        "folder": str(checkpoint / "huggingface"),
        "tag": "slots-abc123",
        "dry_run": True,
    }
    # No repository is a dry run too.
    assert hp.main([str(checkpoint), str(out)]) == 0
    assert json.loads(out.read_text())["dry_run"] is True


class _Api:
    """A fake `HfApi` recording its calls."""

    calls: ClassVar[list[tuple[str, tuple, dict]]] = []

    def __init__(self, token: str) -> None:
        self.calls.append(("init", (), {"token": token}))

    def create_repo(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("create_repo", args, kwargs))

    def upload_folder(self, **kwargs: object) -> types.SimpleNamespace:
        self.calls.append(("upload_folder", (), kwargs))
        return types.SimpleNamespace(oid="c0ffee")

    def create_tag(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("create_tag", args, kwargs))


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> type[_Api]:
    _Api.calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = _Api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return _Api


def test_an_upload_is_a_tagged_revision(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: type[_Api],
) -> None:
    monkeypatch.setenv(hp.HF_TOKEN_ENV, "hf_secret")
    out = tmp_path / "huggingface.json"
    assert hp.main([str(checkpoint), str(out), "--repo", "org/m"]) == 0
    record = json.loads(out.read_text())
    assert record["revision"] == "c0ffee"
    assert record["repo"] == "org/m"
    calls = {name: (args, kwargs) for name, args, kwargs in api.calls}
    assert calls["upload_folder"][1]["repo_id"] == "org/m"
    assert calls["create_tag"][1]["tag"] == "slots-abc123"
    assert calls["create_tag"][1]["revision"] == "c0ffee"
    # The token is used, never written out.
    assert calls["init"][1]["token"] == "hf_secret"
    assert "hf_secret" not in out.read_text()
    assert "hf_secret" not in capsys.readouterr().out


def test_an_upload_needs_a_token(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api: type[_Api],
) -> None:
    monkeypatch.delenv(hp.HF_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit):
        hp.main([str(checkpoint), str(tmp_path / "o.json"), "--repo", "org/m"])
    assert not api.calls
