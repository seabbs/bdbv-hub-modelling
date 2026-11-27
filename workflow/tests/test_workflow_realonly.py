"""The real-only run: real data alone, the main run's recipe, one target."""

import pytest
from flowlib import runs
from test_workflow_dryrun import _jobs_of, dry, dry_run

pytest.importorskip("snakemake")


def _found():
    cfg = runs.load_config()
    return cfg, runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})


def test_the_real_only_run_is_run_a_s_recipe_on_real_data():
    # The data are the only difference from the main run: its layout,
    # its attention biases and its learning rates.
    cfg, found = _found()
    a, real = found["runa"], found["realonly"]
    assert real.real
    assert not a.real
    assert real.ft["data"] == "real"
    for key in ("series_bias", "layout", "learning_rate", "bias_lr"):
        assert real.ft[key] == a.ft[key], key
    assert real.ft["series_bias"]
    assert real.ft["layout"] == "ragged"
    # Checkpoints at thousandths of the whole training: 1/8 to all.
    assert real.ft["prefixes"][-1] == 1000
    assert "realonly" not in runs.trained(cfg, found)


def test_the_mixed_real_data_run_is_not_the_real_only_run():
    _, found = _found()
    assert found["realdata"].ft["real_training"] == "germany"
    assert not found["realdata"].real
    assert found["realonly"].ft["real_training"] is None


def test_the_real_trainer_takes_the_run_s_recipe():
    _, found = _found()
    opts = runs.real_trainer_options(found["realonly"])
    assert "--series-bias" in opts
    assert "--bias-lr" in opts
    assert "--prefixes 125 250 500 1000" in opts
    # Passes to the main run's steps (Run A's final checkpoint), capped.
    assert "--target-steps 82040" in opts
    assert "--max-passes 20" in opts


def test_one_target_cuts_trains_scores_and_publishes(tmp_path):
    jobs = dry(tmp_path, "finetune_realonly")
    prefixes = runs.load_config()["runs"]["realonly"]["prefixes"]
    assert jobs["realonly_blocks_germany"] == 1
    assert jobs["realonly_blocks_bdbv"] == 1
    assert jobs["train_real"] == len(prefixes)
    # Scored on the German development half as every run's checkpoints.
    assert jobs["score_run_germany"] == len(prefixes)
    assert jobs["run_table"] == 1
    assert jobs["german_report"] == 1
    # The final checkpoint is verified and released as realdata-ft.
    assert jobs["verify_checkpoint"] == 1
    assert jobs["publish_checkpoint"] == 1
    # No simulations.
    for rule in ("train_segment", "fetch_batch", "simulate_batch"):
        assert rule not in jobs


def test_the_trainer_resumes_each_checkpoint_from_the_one_before(tmp_path):
    out = dry_run(tmp_path, "finetune_realonly", quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    shells = [j["shell"] for j in _jobs_of(out, "train_real")]
    assert all("finetune-real" in s for s in shells)
    assert all("--series-bias" in s for s in shells)
    assert any("--until 1000" in s and "n500/state" in s for s in shells)
    blocks = _jobs_of(out, "realonly_blocks_germany")[0]["shell"]
    assert "--realdata" in blocks
