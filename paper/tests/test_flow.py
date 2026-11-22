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
    config["finetuned"] = "abc123"
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", "/r")
    )
    assert flow.synthetic["simulations"]["paper"] == "/r/rounds/r38/sims"
    # Steps passed the config read the same folders.
    assert config["synthetic"]["simulations"]["paper"] == "/r/rounds/r38/sims"
    assert os.environ["BDBV_CHRONOS_FT"] == "abc123"
    assert flow.realdata_after == "2026-08-01"


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
        "--store-ref origin/fit-store",
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
    flow = PaperFlow(CONFIG, {"lag_germany": entry}, "2026-08-01")
    path = flow.evidence_file("lag_germany")
    assert path.endswith("evidence/lag_germany-0123456789ab.csv.gz")
    names = {"lag_germany": "waiting note"}
    assert flow.evidence_args(names) == f"--lag-germany {path}"
    assert flow.evidence_args({}) == ""
