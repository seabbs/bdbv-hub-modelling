"""The paper's host-made outputs, published and read back (paperpipe.stored)."""

from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from paperpipe import stored
from paperpipe.flow import PaperFlow

CONFIG = yaml.safe_load(
    (Path(__file__).parents[1] / "workflow" / "config.yaml").read_text()
)


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_each_set_names_its_files_from_the_config() -> None:
    sets = stored.sets(CONFIG, "paper/results")
    assert set(sets) == {"synthetic", "finetune", "simulations"}
    assert (
        sets["synthetic"]["synthetic-dev.csv.gz"]
        == (CONFIG["synthetic"]["forecasts"]["development"])
    )
    assert sets["synthetic"]["pairing-synthetic.csv.gz"].endswith(
        "evidence/pairing/pairing-synthetic.csv.gz"
    )
    assert set(sets["finetune"]) == {
        "validation-curve.csv",
        "comparison.csv",
        "compare-daily.csv",
        "block-combinations.csv",
    }
    sims = sets["simulations"]
    assert {"sim-diseases.csv", "sim-summaries.csv", "sim-realism.csv"} <= set(
        sims
    )
    # The examples' seeds, from the development round only.
    assert all(
        "/r33/sims/" in f for n, f in sims.items() if n.startswith("examples/")
    )
    # Nothing from the held-out round or the paper run's forecasts.
    text = json.dumps(sets)
    assert "/r38/" not in text
    assert "synthetic.csv.gz" not in text.replace("synthetic-dev.csv.gz", "")


def test_the_hash_follows_the_contents_not_the_paths(tmp_path: Path) -> None:
    a = _write(tmp_path / "a" / "x.csv", pd.DataFrame({"v": [1]}))
    b = _write(tmp_path / "b" / "y.csv", pd.DataFrame({"v": [1]}))
    c = _write(tmp_path / "c" / "y.csv", pd.DataFrame({"v": [2]}))
    assert stored.content_hash({"x.csv": a}) == stored.content_hash(
        {"x.csv": b}
    )
    assert stored.content_hash({"x.csv": a}) != stored.content_hash(
        {"x.csv": c}
    )
    assert stored.tag("finetune", {"x.csv": a}).startswith("paper-finetune-")


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"study": ["ebola"], "v": [1]}),
        pd.DataFrame({"study": ["synthetic"], "location": ["33800-0001"]}),
        pd.DataFrame({"seed": [33801]}),
        pd.DataFrame({"study": ["germany"], "origin_date": ["2022-03-01"]}),
    ],
)
def test_held_out_rows_are_never_published(
    tmp_path: Path, frame: pd.DataFrame
) -> None:
    path = _write(tmp_path / "x.csv", frame)
    with pytest.raises(ValueError, match="held out"):
        stored.check_development({"x.csv": path}, CONFIG)


def test_development_rows_pass(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "study": ["synthetic", "germany"],
            "location": ["33300-0001", "DE"],
            "origin_date": ["2026-01-01", "2021-12-01"],
        }
    )
    stored.check_development(
        {"x.csv": _write(tmp_path / "x.csv", frame)}, CONFIG
    )


def test_a_bundle_unpacks_to_its_files_and_overlay(tmp_path: Path) -> None:
    files = {
        "validation-curve.csv": _write(
            tmp_path / "h" / "v.csv", pd.DataFrame({"v": [1]})
        ),
        "comparison.csv": _write(
            tmp_path / "h" / "c.csv", pd.DataFrame({"v": [2]})
        ),
    }
    bundle = stored.bundle("finetune", files, tmp_path / "out")
    with tarfile.open(bundle) as tar:
        names = set(tar.getnames())
    assert {"manifest.json", "validation-curve.csv", "comparison.csv"} <= names
    folder = stored.unpack(bundle, tmp_path / "stored" / "finetune")
    overlay = stored.overlay({"finetune": folder})
    assert overlay["stored"]["finetune_validation"] == str(
        folder / "validation-curve.csv"
    )
    assert overlay["evidence"]["checkpoint_scores"] == {
        "file": str(folder / "comparison.csv")
    }
    # A file the set does not hold stays pending.
    assert "compare_daily" not in overlay["evidence"]


def test_fetch_takes_each_sets_newest_release(tmp_path: Path) -> None:
    releases = [
        {
            "tag_name": "paper-finetune-aaa",
            "created_at": "2026-11-20T00:00:00Z",
        },
        {
            "tag_name": "paper-finetune-bbb",
            "created_at": "2026-11-24T00:00:00Z",
        },
        {"tag_name": "chronos-slots-ccc", "created_at": "2026-11-25T00:00:00Z"},
    ]
    files = {
        "validation-curve.csv": _write(
            tmp_path / "h" / "v.csv", pd.DataFrame({"v": [1]})
        )
    }
    asked = []

    def download(tag: str, dest: Path) -> Path:
        asked.append(tag)
        return stored.bundle("finetune", files, dest)

    overlay = stored.fetch(tmp_path / "stored", releases, download)
    assert asked == ["paper-finetune-bbb"]
    assert "finetune_validation" in overlay["stored"]
    # Sets with no release are left out, so their results stay pending.
    assert "synthetic" not in json.dumps(overlay)


def test_stored_outputs_stand_in_for_the_host_rules() -> None:
    config = copy.deepcopy(CONFIG)
    config["final_run"] = None
    config["rounds_dir"] = None
    config["synthetic"]["simulations"]["development"] = None
    config["stored"] = {
        "finetune_validation": "s/validation-curve.csv",
        "sim_diseases": "s/sim-diseases.csv",
        "sim_summaries": "s/sim-summaries.csv",
        "sim_realism": "s/sim-realism.csv",
        "pairing_synthetic": "s/pairing-synthetic.csv.gz",
        "examples": "s/examples",
    }
    flow = PaperFlow(config, {}, "2026-08-01")
    files = flow.workflow_evidence()
    assert files["finetune_validation"] == "s/validation-curve.csv"
    assert files["sim_diseases"] == "s/sim-diseases.csv"
    assert files["sim_realism"] == "s/sim-realism.csv"
    assert "si_copula" in files
    assert flow.pairing_synthetic_file() == "s/pairing-synthetic.csv.gz"
    examples = flow.example_files()
    assert examples["counts"]
    assert all(f.startswith("s/examples/") for f in examples["counts"])
