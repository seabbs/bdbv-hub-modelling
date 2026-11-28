"""The real-data run: real data alone, the recipe, one target."""

import pytest
from flowlib import runs
from test_workflow_dryrun import _jobs_of, dry, dry_run

pytest.importorskip("snakemake")


def _found():
    cfg = runs.load_config()
    return cfg, runs.resolve(cfg, {"ROOT": "/data", "SIMS_FROM": ""})


def test_the_real_data_run_is_the_recipe_on_real_data():
    # The data are the only difference from the main run: its layout,
    # its attention biases and its learning rates.
    cfg, found = _found()
    a, real = found["finetuned"], found["realdata"]
    assert real.real
    assert not a.real
    assert real.ft["data"] == "real"
    for key in ("series_bias", "layout", "learning_rate", "bias_lr"):
        assert real.ft[key] == a.ft[key], key
    assert real.ft["series_bias"]
    assert real.ft["layout"] == "ragged"
    # Checkpoints at thousandths of the whole training: 1/8 to all.
    assert real.ft["prefixes"][-1] == 1000
    assert "realdata" not in runs.trained(cfg, found)


def test_the_real_data_run_trains_on_real_data_alone():
    # No simulations and no German blocks mixed into them.
    _, found = _found()
    assert found["realdata"].ft["real_training"] is None
    assert not any(r.ft["real_training"] for r in found.values())


def test_the_real_trainer_takes_the_run_s_recipe():
    _, found = _found()
    opts = runs.real_trainer_options(found["realdata"])
    assert "--series-bias" in opts
    assert "--bias-lr" in opts
    assert "--prefixes 125 250 500 1000" in opts
    # Passes to the main run's steps, capped.
    assert "--target-steps 82040" in opts
    assert "--max-passes 20" in opts


def test_one_target_cuts_trains_scores_and_publishes(tmp_path):
    jobs = dry(tmp_path, "finetune_realdata")
    prefixes = runs.load_config()["runs"]["realdata"]["prefixes"]
    assert jobs["realdata_blocks_germany"] == 1
    assert jobs["realdata_blocks_bdbv"] == 1
    assert jobs["train_real"] == len(prefixes)
    # Scored on the German development half as every run's checkpoints.
    assert jobs["score_run_germany"] == len(prefixes)
    assert jobs["run_table"] == 1
    assert jobs["german_report"] == 1
    # The final checkpoint is verified and released as realdata-ft.
    assert jobs["verify_checkpoint"] == 1
    assert jobs["publish_checkpoint"] == 1
    # No simulations of its own: any are the fine-tuned run's, whose
    # final checkpoint sets its steps
    # (test_the_real_run_trains_after_the_fine_tuned_run).


def test_the_trainer_resumes_each_checkpoint_from_the_one_before(tmp_path):
    out = dry_run(tmp_path, "finetune_realdata", quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    shells = [j["shell"] for j in _jobs_of(out, "train_real")]
    assert all("finetune-real" in s for s in shells)
    assert all("--series-bias" in s for s in shells)
    assert any("--until 1000" in s and "n500/state" in s for s in shells)
    blocks = _jobs_of(out, "realdata_blocks_germany")[0]["shell"]
    assert "--realdata" in blocks


def _final_settings(run):
    return f"{run.train}/n{max(run.ft['prefixes'])}/settings.json"


def test_the_real_run_takes_its_steps_from_the_fine_tuned_run():
    # Its passes aim at the fine-tuned run's steps, read from that run's
    # final checkpoint, so the target is never a hand number.
    _, found = _found()
    real = found["realdata"]
    assert real.ft["steps_from"] == "finetuned"
    assert runs.steps_source(real, found) == _final_settings(found["finetuned"])


def test_the_real_steps_are_the_final_checkpoint_s_when_present(tmp_path):
    _, found = _found()
    settings = tmp_path / "settings.json"
    settings.write_text('{"steps": 91234, "hash": "abc"}')
    opts = runs.real_trainer_options(found["realdata"], str(settings))
    assert "--target-steps 91234" in opts
    # Without the final checkpoint, the config's number.
    missing = str(tmp_path / "none.json")
    opts = runs.real_trainer_options(found["realdata"], missing)
    assert "--target-steps 82040" in opts


def test_the_real_run_trains_after_the_fine_tuned_run(tmp_path):
    # Every real checkpoint takes the fine-tuned run's final checkpoint
    # as an input, so it trains after it, to its steps.
    out = dry_run(tmp_path, "finetune_realdata", quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    _, found = _found()
    final = _final_settings(found["finetuned"]).replace("/data", "")
    shells = [j["shell"] for j in _jobs_of(out, "train_real")]
    assert shells
    assert all(final in s for s in shells)
    segments = _jobs_of(out, "train_segment")
    assert segments
    assert all("run=finetuned" in j["wildcards"] for j in segments)
