"""Runs are config entries: the same rules, each run in its own folder."""

import re

import pytest
from flowlib import compute, localconfig, runs
from test_workflow_dryrun import FT, _jobs_of, dry, dry_run

pytest.importorskip("snakemake")

TWO = {
    **FT,
    "runs": {"a": {}, "b": {"prefixes": [250], "outbreaks": 250}},
    "train_runs": ["a", "b"],
}


def test_the_committed_runs_are_run_a_its_variants_and_run_b():
    cfg = runs.load_config()
    assert list(cfg["runs"]) == [
        "runa",
        "noslots",
        "nodelays",
        "realdata",
        "realonly",
        "runb",
    ]


def test_run_b_trains_with_the_release_layer():
    # Run B is Run A's recipe with the release layer; the committed
    # config must name the layer's file.
    cfg = runs.load_config()
    found = runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})
    layer = found["runb"].ft["release_layer"]
    assert (
        layer == "packages/nowcastfoundation/config/finetune/release-layer.yaml"
    )
    assert (runs.CONFIG.parents[1] / layer).is_file()
    assert f"--release-layer {layer}" in runs.trainer_options(found["runb"])
    assert "--release-layer" not in runs.trainer_options(found["runa"])
    assert runs.trained(cfg, found) == ["runa", "noslots"]


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


def test_the_release_layer_is_passed_only_when_a_run_names_one():
    # Late release starts and weekly releases drawn at
    # fine-tune time; no run takes it until its entry names the file.
    assert runs.load_config()["finetune"]["release_layer"] is None
    assert "--release-layer" not in runs.trainer_options(_run())
    path = "packages/nowcastfoundation/config/finetune/release-layer.yaml"
    got = runs.trainer_options(_run(release_layer=path))
    assert got == f"--release-layer {path}"


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


def test_a_run_trains_without_waiting_on_any_other(tmp_path):
    # Every run trains on its own: one outside `train_runs`, asked for
    # (by the paper's block combinations, say), plans no other run's
    # segments, and two runs' segments depend on nothing of each other.
    target = f"{tmp_path}/runs/b/train/n250/settings.json"
    jobs = dry(tmp_path, target, config={**TWO, "train_runs": ["a"]})
    assert jobs["train_segment"] == 1
    # Both runs at once: each run's first segment waits on no trainer.
    config = {**TWO, "train_runs": ["a"]}
    out = dry_run(tmp_path, "finetune", target, config=config, extra=["--dag"])
    assert out.returncode == 0, out.stderr
    jobs = dict(re.findall(r'^\s*(\d+)\[label = "([^"]+)"', out.stdout, re.M))
    edges = re.findall(r"^\s*(\d+) -> (\d+)$", out.stdout, re.M)
    trainers = {j for j, label in jobs.items() if label.startswith("train_")}
    assert len(trainers) == 3
    after = {j: set() for j in jobs}
    for a, b in edges:
        after[b].add(a)

    def upstream(j, seen):
        for k in after[j] - seen:
            seen.add(k)
            upstream(k, seen)
        return seen

    firsts = [j for j in trainers if not upstream(j, set()) & trainers]
    assert len(firsts) == 2


def test_the_runs_trainers_queue_in_the_config_s_order(tmp_path):
    # With fewer GPUs than runs, the trained runs go first, in order,
    # then the others; every trainer stays above every other job.
    out = dry_run(
        tmp_path,
        "finetune",
        f"{tmp_path}/runs/b/train/n250/settings.json",
        config={**TWO, "train_runs": ["a"]},
        quiet=False,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    got = {}
    for job in _jobs_of(out, "train_segment"):
        run = re.search(r"run=(\w+)", job["wildcards"])[1]
        got.setdefault(run, set()).add(job["priority"])
    assert min(got["a"]) > max(got["b"])
    assert min(got["b"]) >= compute.TRAIN_PRIORITY
    assert max(got["a"]) < compute.TRAIN_PRIORITY + 1


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
    # The window comes from the German config, as the rule reads it.
    assert "--first-date 2021-11-22 --truth-delay 40" in out.stdout
    assert "germany_triangle" in out.stdout
