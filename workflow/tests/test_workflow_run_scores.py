"""A run scores its own checkpoints on the German development half.

A machine that only scores (no `finetune` in TARGETS) scores those saved.
"""

import re

import pytest
from test_workflow_dryrun import FT, _checkpoint, _local, dry, score_jobs

pytest.importorskip("snakemake")

CONFIG = {**FT, "score": {"labels": []}}


def _blocks(tmp_path, local_extra=""):
    return score_jobs(
        tmp_path, CONFIG, f"TARGETS=score\n{local_extra}", target="run_scores"
    )


def test_a_run_scores_only_its_saved_checkpoints(tmp_path):
    _local(tmp_path, "TARGETS=score")
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "run_scores", config=CONFIG)
    # n250 and the reference, n500 not yet saved.
    assert jobs["score_germany"] == 1
    assert jobs["score_run_germany"] == 1
    assert jobs["german_report"] == 1
    for rule in ("train_segment", "simulate_batch", "fetch_batch", "seed_list"):
        assert rule not in jobs


def test_a_run_checkpoint_is_scored_from_its_own_segment(tmp_path):
    _checkpoint(tmp_path, 250)
    blocks = _blocks(tmp_path)
    unit = next(b for b in blocks if b.startswith("rule score_run_germany:"))
    run = tmp_path / "runs" / "runa"
    assert f"{run}/scores/germany-days/n250.done" in unit
    assert (
        f"nowcastgermany.checkpoint germany {run}/scores n250 "
        f"{run}/train/n250 " in unit
    )


def test_the_report_reads_the_run_and_the_shared_reference(tmp_path):
    _checkpoint(tmp_path, 250)
    blocks = _blocks(tmp_path)
    report = next(b for b in blocks if b.startswith("rule german_report:"))
    run = tmp_path / "runs" / "runa"
    assert f"{run}/reports/german/index.html" in report
    assert (
        f"german_report.py {run}/scores/germany-days {run}/reports/german "
        in report
    )
    assert "'n250=runa n250'" in report
    assert (
        f"--from reference={tmp_path}/scores/germany-days/reference" in report
    )


def test_a_training_machine_s_report_pulls_the_rest_of_the_run(tmp_path):
    # Where the run trains, its report asks for every checkpoint in reach.
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "run_scores", config=CONFIG)
    assert jobs["train_segment"] == 1
    assert jobs["score_run_germany"] == 2


def test_scoring_keeps_off_the_trainer_device_by_default(tmp_path):
    _checkpoint(tmp_path, 250)
    blocks = _blocks(tmp_path, "DEVICE=cuda\n")
    units = [b for b in blocks if re.match(r"rule score_(run_)?germany:", b)]
    assert len(units) == 2
    for unit in units:
        assert "DEVICE=cpu" in unit
        assert "DEVICE=cuda" not in unit


def test_the_score_target_takes_each_run_scores(tmp_path):
    _local(tmp_path, "TARGETS=score")
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "score", config=CONFIG)
    assert jobs["score_run_germany"] == 1
    assert jobs["german_report"] == 1
