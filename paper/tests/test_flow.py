"""The paper workflow's paths and arguments (paperpipe.flow)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from paperpipe.flow import HELD_OUT_INCLUDES, PaperFlow

# The config the workflow passes the paper: the project workflow's, with
# the paper's settings on top (workflow/Snakefile).
CONFIG = {
    **yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    ),
    **yaml.safe_load(
        (Path(__file__).parents[1] / "workflow" / "config.yaml").read_text()
    ),
}


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
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    root = tmp_path / "root"
    # An older saved checkpoint is never what the paper loads.
    older = root / "runs" / "runa" / "train" / "n800"
    (older / "finetuned-ckpt").mkdir(parents=True)
    (older / "settings.json").write_text('{"hash": "abc123"}')
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(root))
    )
    sims = f"{root}/rounds/r38/sims"
    assert flow.synthetic["simulations"]["paper"] == sims
    # Steps passed the config read the same folders and checkpoint.
    assert config["synthetic"]["simulations"]["paper"] == sims
    # The `_ft` models load the final run's final checkpoint, a rule
    # input: its release record, whatever is saved when the flow plans.
    final = CONFIG["finetune"]["prefixes"][-1]
    segment = root / "runs" / "runa" / "train" / f"n{final}"
    assert config["finetuned_record"] == f"{segment}/published.json"
    assert flow.finetuned_env() == f"BDBV_CHRONOS_FT={segment}"
    assert "BDBV_CHRONOS_FT" not in os.environ
    assert flow.realdata_after == "2026-08-01"


def test_with_no_saved_checkpoint_the_fine_tuned_models_are_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    config["final_run"] = None
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(tmp_path))
    )
    assert config["finetuned_record"] is None
    assert flow.finetuned_env() == ""
    assert "BDBV_CHRONOS_FT" not in os.environ


def test_the_final_checkpoint_is_the_runs_own_last_prefix(
    tmp_path: Path,
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    config["runs"] = {"runa": {"prefixes": [100, 3200]}, "other": {}}
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(tmp_path))
    )
    # Only the final run's own record: no other run is an input.
    assert flow.finetuned_inputs() == [
        f"{tmp_path}/runs/runa/train/n3200/published.json"
    ]


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
        "--ebola-draws-models randomwalk randomwalknowcast "
        "foundation_zeroshot_complete foundation_zeroshot "
        "foundation_zeroshot_baselinenowcast",
        "--round-releases",
    ):
        assert option in args
    assert _flow(ebola__hub=None).sources(_w("paper", "ebola")) == ""


CHRONOS = (
    "foundation_zeroshot_complete",
    "foundation_zeroshot",
    "foundation_zeroshot_baselinenowcast",
    "foundation_baselinenowcast",
)
"""The Chronos-2 models the paper's own Ebola backtest runs."""


def test_ebola_runs_every_study_s_chronos_set() -> None:
    # The fine-tuned ragged model is the fit store's (`foundation` is a
    # stored model, which `bdbvhub backtest` never runs); the rest run in
    # the paper's own release backtest.
    models = CONFIG["ebola"]["backtest_models"]
    assert set(models) >= {"randomwalk", "randomwalknowcast", *CHRONOS}
    assert "foundation" not in models


def test_the_ebola_fine_tuned_models_wait_for_the_final_checkpoint() -> None:
    pending = _flow().ebola_draws_models()
    assert "foundation_baselinenowcast" not in pending
    assert {"randomwalknowcast", "foundation_zeroshot"} <= set(pending)
    ready = _flow(finetuned_record="runs/runa/train/n1/published.json")
    assert "foundation_baselinenowcast" in ready.ebola_draws_models()


def test_the_site_backtest_runs_the_walks_only() -> None:
    # The site's own backtest file needs no Chronos-2 run: the paper's
    # draws replace every model it runs itself.
    assert _flow().ebola_site_models() == ["randomwalk", "randomwalknowcast"]


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
    prefixes = "--prefixes 100 200 400 800 1600 3200 4800"
    assert _flow().summary_args() == (
        f" --realdata-after 2026-08-01 {prefixes}"
    )
    assert _flow(finetuned_record="n9/published.json").summary_args() == (
        "--finetuned-record n9/published.json --realdata-after 2026-08-01 "
        f"{prefixes}"
    )


def test_the_summary_takes_the_final_runs_own_checkpoints() -> None:
    runs = {"runa": {"prefixes": [100, 3200]}}
    assert _flow(runs=runs).summary_args().endswith("--prefixes 100 3200")
    # With no final run, the workflow's fine-tuning settings.
    assert (
        _flow(final_run=None, runs=runs)
        .summary_args()
        .endswith("--prefixes 100 200 400 800 1600 3200 4800")
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
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
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


def test_ebola_reads_our_unsubmitted_models_from_the_round_releases() -> None:
    flow = _flow()
    releases = flow.round_releases()
    assert releases.endswith("evidence/round-releases.csv.gz")
    assert releases in flow.source_files(_w("paper", "ebola"))
    sources = flow.sources(_w("paper", "ebola"))
    assert f"--round-releases {releases}" in sources
    assert "store" not in sources
    assert _flow(ebola__round_releases=None).round_releases() is None


def test_the_evidence_reads_the_workflow_runs_comparison() -> None:
    runs = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["runs"]
    # The live run is the final run, by its workflow run name.
    assert CONFIG["live_run"] == Path(CONFIG["final_run"]).name
    assert CONFIG["evidence"]["checkpoint_scores"] == {
        "file": "$ROOT/runs/comparison.csv"
    }
    # The no-slots run writes the joint tally and cuts the same blocks
    # as Run A, so its tally is Run A's design.
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


def test_an_offline_config_leaves_the_workflow_inputs_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    config["germany"]["real"] = False
    config["final_run"] = None
    config["rounds_dir"] = None
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(config, {}, tmp_path, lambda f: f)
    assert config["finetuned_record"] is None
    assert flow.finetuned_inputs() == []
    assert flow.workflow_evidence() == {}
    assert flow.example_files() == {"counts": [], "settings": []}


def test_the_evidence_reads_the_workflow_rules_outputs() -> None:
    flow = _flow()
    files = flow.workflow_evidence()
    rounds = CONFIG["rounds_dir"]
    assert files["sim_diseases"] == f"{rounds}/report/sim-diseases.csv"
    assert files["sim_summaries"].endswith("evidence/sim-summaries.csv")
    assert files["sim_realism"].endswith("evidence/sim-realism.csv")
    args = flow.options(files)
    assert f"--sim-diseases {files['sim_diseases']}" in args
    assert _flow(final_run=None).workflow_evidence().keys() == {
        "si_copula",
        "copula_rho",
        "sim_diseases",
        "sim_summaries",
        "sim_realism",
    }


def test_with_no_development_simulations_the_copula_scores_pend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    germany = tmp_path / "germany.toml"
    split = CONFIG["heldout"]["germany_from"]
    germany.write_text(f'[nowcast]\nsplit_date = "{split}"\n')
    (tmp_path / "packages" / "nowcastfoundation" / "config").mkdir(parents=True)
    (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).write_text('[realdata_bdbv]\nrelease = "2026-08-01"\n')
    config = copy.deepcopy(CONFIG)
    config["germany"]["config"] = str(germany)
    config["synthetic"]["simulations"]["development"] = None
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    flow = PaperFlow.load(
        config, {}, tmp_path, lambda f: f.replace("$ROOT", str(tmp_path))
    )
    assert flow.synthetic["simulations"]["development"] is None
    files = flow.workflow_evidence()
    # The German rho_p needs no simulation; the walks' scores do.
    assert "copula_rho" in files
    assert "si_copula" not in files


def test_the_copula_evidence_reads_real_german_data_only() -> None:
    files = _flow().workflow_evidence()
    assert files["si_copula"].endswith("evidence/si-copula.csv")
    assert files["copula_rho"].endswith("pairing-germany-rho.csv.gz")
    dummy = _flow(germany__real=False).workflow_evidence()
    assert "si_copula" not in dummy
    assert "copula_rho" not in dummy


def test_the_simulations_and_their_ready_file_are_always_inputs(
    tmp_path: Path,
) -> None:
    # Missing simulations are made by the simulate rules, not skipped.
    sims = tmp_path / "r38" / "sims"
    flow = _flow(
        synthetic__simulations={"development": str(sims), "paper": str(sims)}
    )
    files = flow.synthetic_simulations("paper")
    assert f"{tmp_path}/r38/READY" in files
    assert any(f.startswith(f"{sims}/counts-") for f in files)
    assert flow.source_files(_w("paper", "synthetic")) == [
        flow.synthetic["forecasts"]["paper"]
    ]


def test_each_paper_run_has_a_progress_figure_from_its_own_tables() -> None:
    # The live run and each comparison's run, each reading only its own
    # reports.
    flow = _flow()
    runs = flow.progress_runs()
    assert runs == list(
        dict.fromkeys([CONFIG["live_run"], *CONFIG["comparisons"].values()])
    )
    for run in runs:
        files = flow.progress_files(run)
        assert set(files) == {"loss", "seven_day", "bdbv", "horizons"}
        assert all(f"/runs/{run}/reports/" in f for f in files.values())
    outputs = flow.outputs(SimpleNamespace(mode="development", study=""))
    assert flow.progress_figure("development", runs[0]) in outputs


def test_without_runs_a_progress_figure_reads_the_stored_tables(
    tmp_path: Path,
) -> None:
    # Pages has no runs folder: it reads the hosts' published tables
    # (`paperpipe.stored`), which have no BDBV check.
    stored = tmp_path / "progress"
    run = CONFIG["live_run"]
    (stored / run).mkdir(parents=True)
    (stored / run / "loss.csv").write_text("")
    (stored / run / "horizon-scores.csv").write_text("")
    flow = _flow(runs_dir=None, stored={"progress": str(stored)})
    assert flow.progress_files(run) == {
        "loss": f"{stored}/{run}/loss.csv",
        "horizons": f"{stored}/{run}/horizon-scores.csv",
    }
    assert _flow(runs_dir=None).progress_files(run) == {}


def test_only_a_paper_run_s_progress_figure_reads_the_bdbv_check() -> None:
    files = {"loss": "l.csv", "bdbv": "b.csv"}
    assert PaperFlow.progress_args("development", files) == "--loss l.csv"
    assert PaperFlow.progress_args("paper", files) == (
        "--loss l.csv --bdbv b.csv"
    )
