"""The dashboard's pipeline view: where a run is, and what it waits on."""

import json
import os
import re
import time
from pathlib import Path

from flowlib.dashboard import pipeline

ROOT = Path(__file__).parents[2]
SEEDS = list(range(33400, 33440))


def _job(stamp, rule, jobid, wildcards):
    return (
        f"[{stamp}]\nlocalrule {rule}:\n    jobid: {jobid}\n"
        f"    wildcards: {wildcards}\n    resources: sim_slots=1\n"
    )


def _done(stamp, jobid):
    return f"[{stamp}]\nFinished jobid: {jobid} (Rule: x)\n"


def _failed(stamp, rule, jobid):
    return f"[{stamp}]\nError in rule {rule}:\n    jobid: {jobid}\n"


def _setup(tmp_path, stage, age_s, failed=33431):
    rounds = tmp_path / "rounds"
    sims = rounds / "r34" / "sims"
    sims.mkdir(parents=True)
    run = tmp_path / "run"
    (run / "sims").mkdir(parents=True)
    (run / "seeds.txt").write_text("".join(f"{s}\n" for s in SEEDS))
    for s in SEEDS:
        if s == failed:
            continue
        (sims / f"counts-{s}.csv.gz").write_text("")
        (run / "sims" / f"settings-{s}.csv").write_text(
            "outbreak\n" + "".join(f"{s}-{i}\n" for i in range(20))
        )
    (run / "sims" / "settings-33300.csv").write_text("outbreak\n")
    status = run / "train" / "stream-status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "stage": stage,
                "step": 9889,
                "time": "2026-11-22 16:41:50",
                "split": {"wait_s": 1.0, "cut_s": 120.0, "train_s": 3000.0},
            }
        )
    )
    when = time.time() - age_s
    os.utime(status, (when, when))
    log = tmp_path / "wf" / ".snakemake" / "log" / "2026-11-22T1.snakemake.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        _job(
            "Sun Nov 22 12:00:00 2026",
            "simulate_batch",
            1,
            "round=34, seed=33430",
        )
        + _done("Sun Nov 22 12:20:00 2026", 1)
        + _job(
            "Sun Nov 22 12:00:00 2026",
            "simulate_batch",
            2,
            "round=34, seed=33431",
        )
        + _failed("Sun Nov 22 12:27:00 2026", "simulate_batch", 2)
        + _job("Sun Nov 22 12:30:00 2026", "evaluate_checkpoint", 3, "n=400")
    )
    where = pipeline.Where(
        rounds_dir=rounds,
        rounds=(34,),
        batches=40,
        workflow_dir=tmp_path / "wf",
    )
    return where, run


TRAIN = {"step": 9889, "nxt": 800, "next_step": 19778, "trained": 396.0}


def test_every_stage_names_workflow_rules():
    text = "".join(
        p.read_text() for p in (ROOT / "workflow" / "rules").glob("*.smk")
    )
    rules = set(re.findall(r"^(?:rule|checkpoint) (\w+):", text, re.M))
    for _, _, names in pipeline.STAGES:
        assert set(names) <= rules, set(names) - rules


def test_a_stopped_run_waiting_on_a_failed_batch_is_at_simulate(tmp_path):
    where, run = _setup(tmp_path, "stopped at n400", 3600)
    view = pipeline.build(where, run, TRAIN, alive=None)
    assert view.current == "simulate"
    assert view.waiting == {"waiting_for_batches": [33431], "since": "16:41"}
    by = {s.key: s for s in view.stages}
    assert by["simulate"].progress == {"r34": "39/40"}
    assert by["simulate"].jobs["failed"] == 1
    assert by["simulate"].jobs["mean_job_s"] == 1200
    assert by["simulate"].jobs["to_go"] == 1
    assert by["fetch"].progress == {"batches": "39/40"}
    assert by["evaluate"].jobs["running"] == 1
    assert by["evaluate"].progress == {"evaluate_checkpoint": "n=400"}


def test_a_training_run_is_at_train_with_its_step(tmp_path):
    where, run = _setup(tmp_path, "training n800", 10, failed=None)
    view = pipeline.build(where, run, TRAIN, alive=True)
    assert view.current == "train"
    assert view.waiting == {}
    train = {s.key: s for s in view.stages}["train"]
    assert train.progress["segment"] == "n800"
    assert train.progress["step"] == "9889/19778"
    assert train.progress["simulations"] == "396/800"


def test_the_section_marks_the_current_stage(tmp_path):
    where, run = _setup(tmp_path, "stopped at n400", 3600)
    out = pipeline.section(pipeline.build(where, run, TRAIN, alive=None))
    assert re.search(r'class="stg cur"[^>]*>.*?Simulate', out, re.S)
    assert "Waiting for batches</span> <div class=kv>" in out
    assert "<b>33431</b>" in out
    assert "<span class=kvk>Mean job</span> <b>20 min</b>" in out
