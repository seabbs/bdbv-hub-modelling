"""Runs are config entries: the same rules, each run in its own folder."""

import pytest
from flowlib import localconfig, runs
from test_workflow_dryrun import FT, dry, dry_run

pytest.importorskip("snakemake")

TWO = {
    **FT,
    "runs": {"a": {}, "b": {"prefixes": [250], "outbreaks": 250}},
    "train_runs": ["a", "b"],
}


def test_the_committed_runs_are_run_a_and_its_variants():
    cfg = runs.load_config()
    assert list(cfg["runs"]) == ["runa", "noslots", "nodelays", "realdata"]


def test_only_run_a_trains_until_the_variants_are_due():
    # The variants come after Run A (#261): `finetune` must not start them
    # beside it on the GPU.
    cfg = runs.load_config()
    found = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    assert runs.trained(cfg, found) == ["runa"]


def test_the_no_slots_run_is_run_a_without_the_attention_biases():
    cfg = runs.load_config()
    found = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    a, b = found["runa"].ft, found["noslots"].ft
    assert a["series_bias"]
    assert not b["series_bias"]
    assert {k: v for k, v in a.items() if k != "series_bias"} == {
        k: v for k, v in b.items() if k != "series_bias"
    }
    assert "--series-bias" not in runs.trainer_options(found["noslots"])


def test_a_run_takes_the_finetune_defaults_and_its_overrides():
    cfg = runs.load_config()
    cfg["runs"] = {"x": {"prefixes": [100, 200], "outbreaks": 200}}
    got = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    x = got["x"]
    assert x.ft["prefixes"] == [100, 200]
    assert x.ft["learning_rate"] == cfg["finetune"]["learning_rate"]
    assert x.ft["rounds"] == cfg["grid_rounds"]["rounds"]
    assert x.dir == "/data/runs/x"
    assert x.train == "/data/runs/x/train"
    assert x.prev == {100: None, 200: 100}


def test_the_machine_chooses_where_batches_come_from():
    cfg = runs.load_config()
    got = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": "release"})
    assert got["runa"].ft["sims_from"] == "release"
    assert got["runa"].elsewhere


def test_the_local_config_names_no_run_folder(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("ROOT=/data\n")
    assert "RUN" not in localconfig.read(path)


def test_each_run_trains_in_its_own_folder(tmp_path):
    jobs = dry(tmp_path, "finetune", config=TWO)
    assert jobs["train_segment"] == 3
    out = dry_run(tmp_path, "finetune", config=TWO, quiet=False)
    assert out.returncode == 0, out.stderr
    for run, n in (("a", 250), ("a", 500), ("b", 250)):
        assert f"{tmp_path}/runs/{run}/train/n{n}/settings.json" in out.stdout
    # The rounds are shared: each batch is simulated once for both runs.
    assert jobs["simulate_batch"] == 66


def test_train_runs_picks_the_runs_finetune_trains(tmp_path):
    jobs = dry(tmp_path, "finetune", config={**TWO, "train_runs": ["b"]})
    assert jobs["train_segment"] == 1


def test_a_local_run_folder_is_refused_with_the_fix():
    cfg = runs.load_config()
    with pytest.raises(ValueError, match="runs/<name>"):
        runs.resolve(cfg, {"ROOT": "/data", "RUN": "/data/old"})


def _run(**ft):
    base = {
        "prefixes": [100, 200],
        "series_bias": False,
        "bias_lr": 0.001,
        "layout": "ragged",
        "real_training": None,
    }
    return runs.Run("r", base | ft, "/x/r")


def test_an_option_is_left_out_when_unset():
    assert runs.option("--a", None) == ""
    assert runs.option("--a", 0) == "--a 0"


def test_a_segment_resumes_from_the_previous_prefix_s_state():
    assert runs.resume_options(_run(), 100) == ""
    got = runs.resume_options(_run(), 200)
    assert got == "--prev /x/r/train/n100/state --prev-n 100"


def test_trainer_options_name_only_the_settings_given():
    assert runs.trainer_options(_run()) == ""
    got = runs.trainer_options(
        _run(series_bias=True, loss_horizon_days=7, quiet_min_cases=None)
    )
    assert got == "--loss-horizon-days 7 --series-bias --bias-lr 0.001"


def test_the_no_delays_run_is_run_a_on_complete_counts():
    cfg = runs.load_config()
    found = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    a, b = found["runa"].ft, found["nodelays"].ft
    assert (a["layout"], b["layout"]) == ("ragged", "complete")
    assert {k: v for k, v in a.items() if k != "layout"} == {
        k: v for k, v in b.items() if k != "layout"
    }
    assert "--layout complete" in runs.trainer_options(found["nodelays"])
    assert "--layout" not in runs.trainer_options(found["runa"])


def test_a_variant_trains_only_after_the_runs_finetune_trains(tmp_path):
    # Asked for (by the paper's block combinations, say), a run outside
    # `train_runs` starts only once those runs' final checkpoints exist,
    # so it never takes their GPU.
    target = f"{tmp_path}/runs/b/train/n250/settings.json"
    jobs = dry(tmp_path, target, config={**TWO, "train_runs": ["a"]})
    assert jobs["train_segment"] == 3
    # A run that `finetune` trains waits on no other.
    target = f"{tmp_path}/runs/a/train/n250/settings.json"
    assert (
        dry(tmp_path, target, config={**TWO, "train_runs": ["a"]})[
            "train_segment"
        ]
        == 1
    )


def test_the_real_data_run_is_run_a_with_the_german_blocks():
    cfg = runs.load_config()
    found = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    a, b = found["runa"].ft, found["realdata"].ft
    assert (a["real_training"], b["real_training"]) == (None, "germany")
    assert {k: v for k, v in a.items() if k != "real_training"} == {
        k: v for k, v in b.items() if k != "real_training"
    }
    blocks = "/data/runs/realdata/real-training.pkl"
    assert f"--real-training {blocks}" in runs.trainer_options(
        found["realdata"]
    )
    assert "--real-training" not in runs.trainer_options(found["runa"])


def test_the_real_data_run_cuts_its_german_blocks_before_training(tmp_path):
    target = f"{tmp_path}/runs/realdata/real-training.pkl"
    out = dry_run(tmp_path, target, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "nowcastgermany.finetune_blocks" in out.stdout
    assert "germany_triangle" in out.stdout
