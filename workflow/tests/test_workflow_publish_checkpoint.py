"""The workflow releases a series-bias checkpoint as chronos-slots."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def publish_checkpoint():
    pytest.importorskip("pandas")
    path = ROOT / "workflow" / "scripts" / "publish_checkpoint.py"
    spec = importlib.util.spec_from_file_location("publish_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("bias", "tag"),
    [(True, "chronos-slots-abc"), (False, "chronos-ft-abc")],
)
def test_the_tag_follows_the_series_bias(
    publish_checkpoint, monkeypatch, tmp_path, bias, tag
):
    settings = {"hash": "abc"}
    if bias:
        settings["series_bias"] = {"learning_rate": 1e-3}
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    specs = []
    monkeypatch.setattr(
        publish_checkpoint.fs, "publish", lambda _, spec: specs.append(spec)
    )
    out = tmp_path / "published.json"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), str(out)])
    publish_checkpoint.main()
    assert json.loads(out.read_text())["tag"] == tag


def test_a_release_never_waits_for_its_held_out_losses(
    publish_checkpoint, monkeypatch, tmp_path
):
    # The CPU's held-out losses take hours.
    (tmp_path / "settings.json").write_text(json.dumps({"hash": "abc"}))
    (tmp_path / "held-out.json").write_text('{"held_out_loss": 0.2}')
    calls = []
    monkeypatch.setattr(publish_checkpoint.fs, "publish", lambda *_: None)
    monkeypatch.setattr(
        publish_checkpoint.subprocess, "run", lambda cmd, **_: calls.append(cmd)
    )
    out = tmp_path / "published.json"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), str(out)])
    publish_checkpoint.main()
    assert calls == []


def test_held_out_losses_join_the_release_when_scored(
    publish_checkpoint, monkeypatch, tmp_path
):
    (tmp_path / "held-out.json").write_text('{"held_out_loss": 0.2}')
    (tmp_path / "published.json").write_text(
        json.dumps({"tag": "chronos-slots-abc", "hash": "abc"})
    )
    calls = []
    monkeypatch.setattr(
        publish_checkpoint.subprocess, "run", lambda cmd, **_: calls.append(cmd)
    )
    out = tmp_path / "held-out-released.txt"
    monkeypatch.setattr(
        sys, "argv", ["x", "--held-out", str(tmp_path), str(out)]
    )
    publish_checkpoint.main()
    assert calls[0][:4] == ["gh", "release", "upload", "chronos-slots-abc"]
    assert str(tmp_path / "held-out.json") in calls[0]
    assert out.read_text().strip() == "chronos-slots-abc"


def test_the_main_run_s_checkpoint_is_promoted(
    publish_checkpoint, monkeypatch, tmp_path
):
    (tmp_path / "settings.json").write_text(json.dumps({"hash": "abc"}))
    calls = []

    def promote(where, run):
        calls.append((where, run))
        return {"run": run, "spec": "abc", "tag": "chronos-ft-abc"}

    monkeypatch.setattr(publish_checkpoint.fs, "promote_main", promote)
    out = tmp_path / "main-release.json"
    monkeypatch.setattr(
        sys, "argv", ["x", "--main", "runa", str(tmp_path), str(out)]
    )
    publish_checkpoint.main()
    assert calls == [(tmp_path, "runa")]
    assert json.loads(out.read_text())["tag"] == "chronos-ft-abc"
