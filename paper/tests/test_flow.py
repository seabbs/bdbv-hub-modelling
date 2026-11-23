"""The paper workflow's paths and arguments (paperpipe.flow)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from paperpipe.flow import HELD_OUT_INCLUDES, PaperFlow

CONFIG = yaml.safe_load(
    (Path(__file__).parents[1] / "workflow" / "config.yaml").read_text()
)


def _flow(**changes: object) -> PaperFlow:
    config = copy.deepcopy(CONFIG)
    for key, value in changes.items():
        block, _, field = key.partition("__")
        if field:
            config[block][field] = value
        else:
            config[block] = value
    return PaperFlow(config, {}, "2026-08-01")


def _w(mode: str, study: str) -> SimpleNamespace:
    return SimpleNamespace(mode=mode, study=study)


def test_load_checks_the_german_split(tmp_path: Path) -> None:
    germany = tmp_path / "germany.toml"
    germany.write_text('[nowcast]\nsplit_date = "2022-01-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    with pytest.raises(ValueError, match="splits at 2022-01-01"):
        PaperFlow.load(config, {}, tmp_path, lambda f: f)


def test_load_expands_the_simulations_and_sets_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "finetune.toml").write_text(
        '[realdata_bdbv]\nrelease = "2026-08-01"\n'
    )
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    # The final run's newest saved checkpoint, from its own folder.
    root = tmp_path / "root"
    segment = root / "runs" / "runa" / "train" / "n800"
    (segment / "finetuned-ckpt").mkdir(parents=True)
    (segment / "settings.json").write_text('{"hash": "abc123"}')
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (segment / "finetuned-ckpt" / name).write_text("")
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(root))
    )
    sims = f"{root}/rounds/r38/sims"
    assert flow.synthetic["simulations"]["paper"] == sims
    # Steps passed the config read the same folders and checkpoint.
    assert config["synthetic"]["simulations"]["paper"] == sims
    # Outputs record the checkpoint's spec; only the models read the folder.
    assert config["finetuned"] == "abc123"
    assert os.environ["BDBV_CHRONOS_FT"] == str(segment)
    assert flow.realdata_after == "2026-08-01"


def test_with_no_saved_checkpoint_the_fine_tuned_models_are_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "finetune.toml").write_text(
        '[realdata_bdbv]\nrelease = "2026-08-01"\n'
    )
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(tmp_path))
    )
    assert config["finetuned"] is None
    assert "BDBV_CHRONOS_FT" not in os.environ


def test_no_script_pins_the_checkpoint() -> None:
    paper = Path(__file__).parents[1]
    assert not (paper / "scripts" / "follow_checkpoint.py").exists()
    assert "finetuned" not in CONFIG
    assert "finetune_run" not in CONFIG
    assert "final_checkpoints" not in CONFIG
    assert CONFIG["final_run"] == "$ROOT/runs/runa"


def test_extra_args_name_only_what_the_entry_sets() -> None:
    entry = {"models": "fm_.*", "suffix": "_x", "backtest": True}
    assert PaperFlow.extra_args(entry) == (
        "--models 'fm_.*' --suffix _x --backtest"
    )
    assert PaperFlow.extra_args({}) == ""


def test_provisional_notes_are_joined_once() -> None:
    extra = [
        {"provisional": "Redo A."},
        {"provisional": "Redo A."},
        {"provisional": "Redo B."},
        {},
    ]
    flow = _flow(germany__extra=extra)
    assert flow.provisional("germany") == "Redo A. Redo B."
    assert flow.note_arg(_w("paper", "germany")) == "--note 'Redo A. Redo B.'"
    assert flow.note_arg(_w("paper", "synthetic")) == ""


def test_germany_reads_its_held_out_half_only_in_a_paper_run() -> None:
    flow = _flow()
    results = CONFIG["germany"]["results"]
    assert flow.germany_tables("development") == [
        f"{results}/paper-forecasts.csv.gz"
    ]
    assert flow.germany_tables("paper") == [
        f"{results}/paper-forecasts.csv.gz",
        f"{results}/held-out/paper-forecasts.csv.gz",
    ]
    assert _flow(germany__real=False).germany_tables("paper") == []


def test_ebola_sources_name_every_input() -> None:
    args = _flow().sources(_w("paper", "ebola"))
    for option in (
        "--hub",
        "--ebola-data",
        "--ebola-config",
        "--ebola-backtest",
        "--ebola-draws",
        "--ebola-draws-models randomwalk randomwalknowcast",
        "--candidates",
        "--store-history",
    ):
        assert option in args
    assert _flow(ebola__hub=None).sources(_w("paper", "ebola")) == ""


def test_truth_arguments_follow_the_config() -> None:
    flow = _flow()
    assert flow.delay_arg(_w("paper", "germany")) == "--delay 40"
    assert flow.delay_arg(_w("paper", "synthetic")) == ""
    assert flow.truth_rule_arg(_w("paper", "ebola")) == "--first-after"
    assert flow.truth_rule_arg(_w("paper", "germany")) == ""


def test_development_publishes_placeholders_for_the_held_out_study() -> None:
    flow = _flow()
    held = flow.held_out_placeholders(_w("development", "germany")).split()
    assert len(held) == len(HELD_OUT_INCLUDES)
    assert all("/ebola-" in path for path in held)
    assert flow.held_out_placeholders(_w("paper", "germany")) == ""


def test_outputs_have_the_ebola_tables_only_in_a_paper_run() -> None:
    flow = _flow()
    dev = flow.outputs(_w("development", "germany"))
    paper = flow.outputs(_w("paper", "germany"))
    assert not any("/ebola-" in path for path in dev)
    assert any("/ebola-" in path for path in paper)
    assert dev[0].endswith("development/evidence.yml")
    assert flow.ebola_real_finetune(_w("development", "germany")) == []


def test_summary_args_name_the_checkpoint_when_set() -> None:
    assert _flow().summary_args() == " --realdata-after 2026-08-01"
    assert _flow(finetuned="abc").summary_args() == (
        "--finetuned abc --realdata-after 2026-08-01"
    )


def test_grid_config_has_both_runs_seeds() -> None:
    grid = _flow().grid_config()
    assert grid["heldout_seeds"][0] == CONFIG["heldout"]["synthetic_seeds"][0]
    assert (
        grid["development_seeds"] == (CONFIG["development"]["synthetic_seeds"])
    )


def test_evidence_files_are_named_by_their_commit() -> None:
    entry = {"ref": "0123456789abcdef", "path": "evidence/x.csv.gz"}
    flow = PaperFlow(CONFIG, {"scale_floor": entry}, "2026-08-01")
    path = flow.evidence_file("scale_floor")
    assert path.endswith("evidence/scale_floor-0123456789ab.csv.gz")
    names = {"scale_floor": "waiting note"}
    assert flow.evidence_args(names) == f"--scale-floor {path}"
    assert flow.evidence_args({}) == ""


def test_evidence_from_a_workflow_output_is_read_where_it_is_written(
    tmp_path: Path,
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "finetune.toml").write_text(
        '[realdata_bdbv]\nrelease = "2026-08-01"\n'
    )
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    entry = {"file": "$SCORE_CACHE/tables/checkpoint-scores.csv"}
    flow = PaperFlow.load(
        config,
        {"checkpoint_scores": entry},
        tmp_path,
        lambda f: f.replace("$SCORE_CACHE", "/s"),
    )
    path = "/s/tables/checkpoint-scores.csv"
    assert flow.evidence_file("checkpoint_scores") == path
    names = {"checkpoint_scores": "scores"}
    assert flow.evidence_args(names) == f"--checkpoint-scores {path}"
    assert flow.pinned() == {}


def test_examples_are_read_from_the_development_round() -> None:
    picks = [["33307-0006", "Multi-wave"], ["33300-0001", "Spread"]]
    flow = _flow(
        synthetic__examples={"picks": picks},
        grid_rounds={"validation_round": 33},
    )
    files = flow.example_files()
    folder = f"{CONFIG['rounds_dir']}/r33/sims"
    assert files == {
        "counts": [
            f"{folder}/counts-33300.csv.gz",
            f"{folder}/counts-33307.csv.gz",
        ],
        "settings": [
            f"{folder}/settings-33300.csv",
            f"{folder}/settings-33307.csv",
        ],
    }


def test_ebola_reads_the_fit_store_history_release() -> None:
    flow = _flow()
    history = flow.store_history()
    assert history.endswith("evidence/fit-store-history.csv.gz")
    assert history in flow.source_files(_w("paper", "ebola"))
    assert f"--store-history {history}" in flow.sources(_w("paper", "ebola"))
    assert _flow(ebola__store_history=None).store_history() is None


def test_the_evidence_reads_the_workflow_runs_comparison() -> None:
    runs = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["runs"]
    # The live run is the final run, by its workflow run name.
    assert CONFIG["live_run"] == Path(CONFIG["final_run"]).name
    assert CONFIG["evidence"]["checkpoint_scores"] == {
        "file": "$ROOT/runs/comparison.csv"
    }
    # Run A's trainer predates the joint tally; no-slots cuts the same
    # blocks, so its tally is Run A's design.
    assert CONFIG["evidence"]["block_combinations"] == {
        "file": "$ROOT/runs/noslots/train/block-combinations.csv"
    }
    assert set(CONFIG["comparisons"]) == {"slots", "delays", "realdata"}
    assert CONFIG["comparisons"]["slots"] in runs
    for name in CONFIG["comparisons"]:
        assert f"compare_{name}" not in CONFIG["evidence"]


def test_the_comparisons_are_passed_by_run_name() -> None:
    flow = _flow()
    assert flow.comparisons_arg() == (
        "--comparison slots=noslots --comparison delays=nodelays "
        "--comparison realdata=realdata"
    )
