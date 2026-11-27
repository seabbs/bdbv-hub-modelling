"""The paper's host-made outputs, published and read back (paperpipe.stored)."""

from __future__ import annotations

import copy
import functools
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from nowcastutils import pointers

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
    assert set(sets) == {"synthetic", "finetune", "simulations", "germany"}
    assert set(sets["germany"]) == {
        "pairing-germany.csv.gz",
        "pairing-germany-rho.csv.gz",
        "germany-forecasts.csv.gz",
    }
    # The development half's table only, never the held-out one.
    assert sets["germany"]["germany-forecasts.csv.gz"] == (
        f"{CONFIG['germany']['results']}/paper-forecasts.csv.gz"
    )
    assert (
        sets["synthetic"]["synthetic-dev.csv.gz"]
        == (CONFIG["synthetic"]["forecasts"]["development"])
    )
    assert sets["synthetic"]["pairing-synthetic.csv.gz"].endswith(
        "evidence/pairing/pairing-synthetic.csv.gz"
    )
    runs = dict.fromkeys([CONFIG["live_run"], *CONFIG["comparisons"].values()])
    assert set(sets["finetune"]) == {
        "comparison.csv",
        "compare-daily.csv",
        "block-combinations.csv",
        *(
            f"progress/{run}/{name}"
            for run in runs
            for name in ("loss.csv", "seven-day.csv", "horizon-scores.csv")
        ),
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
    assert CONFIG["synthetic"]["forecasts"]["paper"] not in text


def test_the_finetune_set_holds_each_run_s_progress_tables() -> None:
    # From the run's reports, without the BDBV check's Ebola rows.
    config = copy.deepcopy(CONFIG)
    config["runs_dir"] = "/r/runs"
    found = stored.sets(config, "paper/results")["finetune"]
    run = config["live_run"]
    assert found[f"progress/{run}/loss.csv"] == (
        f"/r/runs/{run}/reports/progress/loss.csv"
    )
    assert found[f"progress/{run}/horizon-scores.csv"] == (
        f"/r/runs/{run}/reports/horizon-scores.csv"
    )
    assert not any("bdbv" in name for name in found)
    config["runs_dir"] = None
    found = stored.sets(config, "paper/results")["finetune"]
    assert not any(name.startswith("progress/") for name in found)


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
        "progress/runa/loss.csv": _write(
            tmp_path / "h" / "v.csv", pd.DataFrame({"v": [1]})
        ),
        "comparison.csv": _write(
            tmp_path / "h" / "c.csv", pd.DataFrame({"v": [2]})
        ),
    }
    bundle = stored.bundle("finetune", files, tmp_path / "out")
    with tarfile.open(bundle) as tar:
        names = set(tar.getnames())
    assert {"manifest.json", "progress/runa/loss.csv", "comparison.csv"} <= (
        names
    )
    folder = stored.unpack(bundle, tmp_path / "stored" / "finetune")
    overlay = stored.overlay({"finetune": folder})
    assert overlay["stored"]["progress"] == str(folder / "progress")
    assert overlay["evidence"]["checkpoint_scores"] == {
        "file": str(folder / "comparison.csv")
    }
    # A file the set does not hold stays pending.
    assert "compare_daily" not in overlay["evidence"]


def _pointer(gh):  # type: ignore[no-untyped-def]
    return functools.partial(pointers.read, repo="owner/repo", run=gh)


def test_fetch_takes_the_release_each_set_s_pointer_names(
    tmp_path: Path, fake_releases
) -> None:
    # A newer release of the set than its pointer names is never taken.
    gh = fake_releases(
        {
            "paper-finetune-aaa": {stored.ASSET: ""},
            "paper-finetune-bbb": {stored.ASSET: ""},
            "paper-finetune-main": {
                pointers.ASSET: json.dumps({"tag": "paper-finetune-aaa"})
            },
        }
    )
    files = {
        "comparison.csv": _write(
            tmp_path / "h" / "c.csv", pd.DataFrame({"v": [1]})
        )
    }
    asked = []

    def download(tag: str, dest: Path) -> Path:
        asked.append(tag)
        return stored.bundle("finetune", files, dest)

    overlay = stored.fetch(tmp_path / "stored", _pointer(gh), download)
    assert asked == ["paper-finetune-aaa"]
    assert "checkpoint_scores" in overlay["evidence"]
    # A set never promoted is left out, so its results stay pending.
    assert "synthetic" not in json.dumps(overlay)


def test_a_broken_pointer_fails_the_fetch(
    tmp_path: Path, fake_releases
) -> None:
    gh = fake_releases({"paper-germany-main": {pointers.ASSET: "{}"}})
    with pytest.raises(ValueError, match="paper-germany-main"):
        stored.fetch(
            tmp_path / "stored",
            _pointer(gh),
            lambda _tag, dest: dest,
            ["germany"],
        )


def test_promoting_a_set_names_its_release_in_its_pointer(
    tmp_path: Path, fake_releases
) -> None:
    released = tmp_path / "released" / "germany.json"
    released.parent.mkdir()
    released.write_text(
        json.dumps(
            {"set": "germany", "tag": "paper-germany-abc", "hash": "abcdef"}
        )
    )
    gh = fake_releases({"paper-germany-abc": {stored.ASSET: ""}})
    record = stored.promote(released, ("main", "next"), run=gh)
    assert record["tag"] == "paper-germany-abc"
    assert record["hash"] == "abcdef"
    assert record["endpoint"] == "paper-germany"
    assert "release_paper_germany" in record["produced_by"]
    assert pointers.read("paper-germany", "owner/repo", run=gh) == record
    # Not from a configured branch: no promotion.
    gh = fake_releases(
        {"paper-germany-abc": {stored.ASSET: ""}}, branches=("feature",)
    )
    with pytest.raises(RuntimeError, match="main"):
        stored.promote(released, ("main",), run=gh)
    assert "paper-germany-main" not in gh.releases


def test_stored_outputs_stand_in_for_the_host_rules() -> None:
    config = copy.deepcopy(CONFIG)
    config["final_run"] = None
    config["rounds_dir"] = None
    config["synthetic"]["simulations"]["development"] = None
    config["stored"] = {
        "sim_diseases": "s/sim-diseases.csv",
        "sim_summaries": "s/sim-summaries.csv",
        "sim_realism": "s/sim-realism.csv",
        "pairing_synthetic": "s/pairing-synthetic.csv.gz",
        "examples": "s/examples",
    }
    flow = PaperFlow(config, {}, "2026-08-01")
    files = flow.workflow_evidence()
    assert files["sim_diseases"] == "s/sim-diseases.csv"
    assert files["sim_realism"] == "s/sim-realism.csv"
    assert "si_copula" in files
    assert flow.pairing_synthetic_file() == "s/pairing-synthetic.csv.gz"
    examples = flow.example_files()
    assert examples["counts"]
    assert all(f.startswith("s/examples/") for f in examples["counts"])


def test_every_row_of_the_german_set_is_german(tmp_path: Path) -> None:
    # The pairing tables have no `study` column; their dates are German.
    path = _write(
        tmp_path / "rho.csv", pd.DataFrame({"nowcast_date": ["2022-03-01"]})
    )
    with pytest.raises(ValueError, match="held out"):
        stored.check_development({"x.csv": path}, CONFIG, german=True)
    ok = _write(
        tmp_path / "ok.csv", pd.DataFrame({"nowcast_date": ["2022-01-01"]})
    )
    stored.check_development({"x.csv": ok}, CONFIG, german=True)


def test_the_stored_german_pairing_stands_in(tmp_path: Path) -> None:
    folder = tmp_path / "germany"
    folder.mkdir()
    for name in ("pairing-germany.csv.gz", "pairing-germany-rho.csv.gz"):
        (folder / name).write_text("")
    overlay = stored.overlay({"germany": folder})
    assert overlay["stored"]["pairing_germany"].endswith(
        "pairing-germany.csv.gz"
    )
    assert overlay["stored"]["copula_rho"].endswith("rho.csv.gz")
    config = copy.deepcopy(CONFIG)
    config["host_outputs"] = False
    config["synthetic"]["simulations"]["development"] = None
    config["stored"] = {
        **overlay["stored"],
        "pairing_synthetic": "s/pairing-synthetic.csv.gz",
    }
    flow = PaperFlow(config, {}, "2026-08-01")
    files = flow.workflow_evidence()
    assert files["copula_rho"] == overlay["stored"]["copula_rho"]
    assert "si_copula" in files
    assert flow.pairing_germany_file() == overlay["stored"]["pairing_germany"]
    # Without a stored copy Pages leaves the German pairing pending.
    config["stored"] = {}
    files = PaperFlow(config, {}, "2026-08-01").workflow_evidence()
    assert "copula_rho" not in files


def test_the_stored_german_forecasts_stand_in_on_pages(tmp_path: Path) -> None:
    folder = tmp_path / "germany"
    folder.mkdir()
    (folder / "germany-forecasts.csv.gz").write_text("")
    overlay = stored.overlay({"germany": folder})
    path = overlay["stored"]["germany_forecasts"]
    config = copy.deepcopy(CONFIG)
    config["host_outputs"] = False
    config["stored"] = overlay["stored"]
    flow = PaperFlow(config, {}, "2026-08-01")
    assert flow.germany_tables("development")[0] == path
    # A workflow host reads its own benchmark's table.
    config["host_outputs"] = True
    flow = PaperFlow(config, {}, "2026-08-01")
    assert flow.germany_tables("development")[0].endswith(
        "results/germany/paper-forecasts.csv.gz"
    )
