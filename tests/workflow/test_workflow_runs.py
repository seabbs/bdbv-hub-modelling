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


def test_the_committed_config_has_one_run():
    cfg = runs.load_config()
    assert list(cfg["runs"]) == ["runa"]


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
