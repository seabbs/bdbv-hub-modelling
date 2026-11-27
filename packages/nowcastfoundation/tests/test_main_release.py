"""The fine-tuned models load the main fine-tuning run's checkpoint.

The main run (the paper's `live_run`) names its newest released
checkpoint in the release `chronos-main` (`main.json`). An ablation run's
newer release is never taken. No test reaches GitHub: the releases are
folders under a file URL, and `gh` is a stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nowcastfoundation import forecaster as fc
from nowcastfoundation.finetune import stream as fs

ADAPTER = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"w"}


@pytest.fixture
def releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_release
) -> Path:
    """Releases of two runs: the main run's and a newer ablation's."""
    root = tmp_path / "releases"
    write_release(root, "main4800", ADAPTER, "main4800", "chronos-slots")
    write_release(root, "main1600", ADAPTER, "main1600", "chronos-slots")
    write_release(root, "ablate", ADAPTER, "ablate")
    monkeypatch.setattr(fc, "FINETUNED_URL", root.as_uri())
    monkeypatch.setattr(fc, "RETRY_SECONDS", (0, 0, 0))
    monkeypatch.delenv(fc.FINETUNED_ENV, raising=False)
    fc.resolve_checkpoint.cache_clear()
    yield root
    fc.resolve_checkpoint.cache_clear()


def _point(root: Path, record: dict) -> None:
    folder = root / fc.MAIN_TAG
    folder.mkdir(parents=True, exist_ok=True)
    (folder / fc.MAIN_FILE).write_text(json.dumps(record))


def _config(tmp_path: Path) -> Path:
    conf = tmp_path / "finetune.toml"
    conf.write_text('source = "github"\nhash = "latest"\n')
    return conf


def test_latest_is_the_main_run_s_checkpoint_not_the_newest_release(
    releases: Path, tmp_path: Path
) -> None:
    _point(
        releases,
        {
            "run": "runa",
            "spec": "slots-main4800",
            "tag": "chronos-slots-main4800",
            "n_outbreaks": 4800,
        },
    )
    assert fc.latest_spec() == "slots-main4800"
    assert fc.finetuned_label(_config(tmp_path)) == "slots-main4800"
    got = fc.resolve_checkpoint("latest")
    assert got.hash == "slots-main4800"
    assert got.tag == "chronos-slots-main4800"


@pytest.mark.usefixtures("releases")
def test_no_main_run_checkpoint_fails_rather_than_take_another_run() -> None:
    # Only an ablation's release: no fallback to it.
    with pytest.raises(LookupError, match=fc.MAIN_TAG):
        fc.resolve_checkpoint("latest")


def test_a_record_whose_tag_is_not_its_checkpoint_s_fails(
    releases: Path,
) -> None:
    _point(
        releases,
        {"run": "runa", "spec": "slots-main4800", "tag": "chronos-ft-ablate"},
    )
    with pytest.raises(ValueError, match="chronos-ft-ablate"):
        fc.latest_spec()


def _segment(tmp_path: Path, settings: dict) -> Path:
    where = tmp_path / "n4800"
    where.mkdir(parents=True)
    (where / "settings.json").write_text(json.dumps(settings))
    return where


SETTINGS = {"hash": "main4800", "series_bias": {"lr": 1}, "n_outbreaks": 4800}
BRANCHES = ("main", "next")


def test_promoting_names_the_checkpoint_in_the_main_release(
    tmp_path: Path, fake_releases
) -> None:
    gh = fake_releases({"chronos-slots-main4800": {fc.SUMS_FILE: ""}})
    record = fs.promote_main(
        _segment(tmp_path, SETTINGS), "runa", BRANCHES, run=gh
    )
    assert {k: record[k] for k in ("run", "spec", "tag", "hash")} == {
        "run": "runa",
        "spec": "slots-main4800",
        "tag": "chronos-slots-main4800",
        "hash": "main4800",
    }
    assert record["n_outbreaks"] == 4800
    assert record["endpoint"] == "chronos"
    assert record["commit"] == gh.head
    assert json.loads(gh.releases[fc.MAIN_TAG][fc.MAIN_FILE]) == record
    # A later checkpoint replaces the record in place.
    gh.releases["chronos-slots-next"] = {fc.SUMS_FILE: ""}
    later = _segment(tmp_path / "later", {**SETTINGS, "hash": "next"})
    fs.promote_main(later, "runa", BRANCHES, run=gh)
    main = json.loads(gh.releases[fc.MAIN_TAG][fc.MAIN_FILE])
    assert main["spec"] == "slots-next"


def test_an_unreleased_checkpoint_is_never_promoted(
    tmp_path: Path, fake_releases
) -> None:
    gh = fake_releases({})
    with pytest.raises(RuntimeError, match="chronos-slots-main4800"):
        fs.promote_main(_segment(tmp_path, SETTINGS), "runa", BRANCHES, run=gh)
    assert fc.MAIN_TAG not in gh.releases
