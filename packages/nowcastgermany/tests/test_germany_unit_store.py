"""The fine-tuned German forecasts reach Pages as their release's asset.

A released checkpoint's German unit (`score_germany`) is packed into the
store layout and attached to its own release; Pages downloads the asset
of the checkpoint config/finetune.toml pins. No side branch, no fallback
to another checkpoint.
"""

import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[3]


def _unit(cache: Path, label: str, dates: list[str]) -> None:
    root = cache / "germany-days" / label
    root.mkdir(parents=True)
    for d in dates:
        pd.DataFrame(
            {
                "model": [
                    "foundation",
                    "randomwalknowcast",
                ],
                "target": ["7day", "7day"],
                "origin": [d, d],
                "location": ["DE", "DE"],
                "target_end_date": [d, d],
                "horizon": [0, 0],
                "quantile": [0.5, 0.5],
                "value": [10.0, 11.0],
                "truth": [12.0, 12.0],
                "truth_vintage": ["2022-03-01", "2022-03-01"],
            }
        ).to_csv(root / f"{d}.csv.gz", index=False)


def test_a_pin_names_its_release():
    from nowcastgermany.store import pinned_tag

    assert pinned_tag("slots-08f3982239a3") == "chronos-slots-08f3982239a3"


def test_a_unit_packs_its_fine_tuned_development_forecasts(tmp_path):
    from nowcastgermany.store import ASSET, FORECASTS, pack_unit

    _unit(tmp_path / "cache", "slots-abcdef", ["2021-11-22", "2021-11-23"])
    out = pack_unit(tmp_path / "cache", "slots-abcdef", tmp_path, "2022-02-10")
    assert out.name == ASSET
    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert f"chronos-slots-abcdef/{FORECASTS}" in names
        manifest = json.load(
            tar.extractfile("chronos-slots-abcdef/manifest.json")  # type: ignore[arg-type]
        )
    assert manifest["label"] == "slots-abcdef"
    assert manifest["models"] == ["foundation"]
    assert manifest["origins"] == 2


def test_a_unit_with_held_out_dates_is_refused(tmp_path):
    from nowcastgermany.store import pack_unit

    _unit(tmp_path / "cache", "slots-abcdef", ["2021-11-22", "2022-02-10"])
    with pytest.raises(ValueError, match="held-out"):
        pack_unit(tmp_path / "cache", "slots-abcdef", tmp_path, "2022-02-10")


def test_the_side_branch_and_its_fallback_are_gone():
    import importlib.util

    from nowcastgermany import store

    assert importlib.util.find_spec("nowcastgermany.ft_store") is None
    assert not hasattr(store, "choose_checkpoint")
    assert not hasattr(store, "STORE_BRANCH")
    pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "germany-ft-forecasts" not in pages
    assert "germany-forecasts.tar.gz" in pages


def test_released_units_are_published_by_the_workflow():
    text = (ROOT / "workflow" / "rules" / "publish.smk").read_text()
    assert "rule publish_germany_unit:" in text
    assert "nowcastgermany unit-publish" in text
