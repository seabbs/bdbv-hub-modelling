"""A segment's trainer log keeps every attempt: a restart appends to it."""

import re
import sys
from pathlib import Path

import pytest
from flowlib.training import Training

sys.path.insert(0, str(Path(__file__).parent))


def _train_jobs(tmp_path):
    pytest.importorskip("snakemake")
    from test_workflow_dryrun import FT, dry_run

    out = dry_run(tmp_path, "finetune", config=FT, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = re.findall(
        r"^rule train_segment:\n(.*?)(?=^rule |\Z)", out.stdout, re.M | re.S
    )
    assert jobs
    return jobs


def test_the_rule_log_is_not_the_trainer_log(tmp_path):
    # Snakemake removes a job's log before it runs, so the trainer's log,
    # which a restart must not lose, is not the rule's.
    for job in _train_jobs(tmp_path):
        log = re.search(r"^    log: (\S+)", job, re.M)[1]
        n = re.search(r"--until (\d+)", job)[1]
        assert not log.endswith(f"/train/n{n}.log")


def test_each_attempt_appends_to_the_trainer_log(tmp_path):
    for job in _train_jobs(tmp_path):
        n = re.search(r"--until (\d+)", job)[1]
        trainer_log = re.escape(f"/train/n{n}.log")
        # Appended to, never written over.
        assert re.search(rf"tee -a \S*{trainer_log}", job)
        assert not re.search(rf"[^>]>\s*\S*{trainer_log}", job)
        assert job.count("tee -a") == 2


def test_a_worker_elsewhere_copies_back_the_log_and_timings(tmp_path):
    training = Training.__new__(Training)
    training.train_dir = lambda run: f"{tmp_path}/{run}/train"

    class Wc:
        run, n = "runa", "800"

    got = Training.segment_copy_back(training, Wc)
    assert f"{tmp_path}/runa/train/n800.log" in got
    assert f"{tmp_path}/runa/train/timings/n800.json" in got
