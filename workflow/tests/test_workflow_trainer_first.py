"""The trainer starts the moment it can, whatever else is ready.

Snakemake's scheduler (its MILP) picks the jobs that maximise the sum of
their priorities. A finished segment frees its cores, and the next
segment, a checkpoint's release and its scoring all become ready at
once: their priorities summed must never outweigh the trainer's, or the
GPU waits behind CPU jobs (archie, 25 Nov: verify_checkpoint and
seven_day_check took n1600's freed cores and n3200 never started).
"""

import logging
from dataclasses import dataclass, field

import pytest
from flowlib import compute

pulp = pytest.importorskip("pulp")
milp = pytest.importorskip("snakemake.scheduling.milp")


@dataclass(eq=False)
class _Job:
    """What the scheduler reads of a ready job."""

    name: str
    priority: float
    scheduler_resources: dict
    input: list = field(default_factory=list)


def _job(name, priority, cores, gpu=0):
    return _Job(name, priority, {"_cores": cores, "gpu": gpu})


def _select(jobs, cores, gpu=1):
    scheduler = milp.Scheduler(
        None, milp.SchedulerSettings(), logging.getLogger("test")
    )
    free = {"_cores": cores, "gpu": gpu, "_job_count": 10**6}
    got = scheduler.select_jobs(jobs, jobs, free, {})
    assert got is not None
    return {j.name for j in got}


def test_the_next_segment_beats_its_checkpoint_s_hand_offs_and_scoring():
    # n1600 done: its 4 cores free; ready are n3200 and n1600's quick
    # hand-offs and scoring, which together fit in those cores.
    trainer = _job("train n3200", compute.TRAIN_PRIORITY, 4, gpu=1)
    others = [
        _job("verify n1600", compute.HANDOFF_PRIORITY, 1),
        _job("publish n1600", compute.HANDOFF_PRIORITY, 1),
        _job("seven_day n1600", compute.side_priority(20, 1600), 2),
    ]
    assert _select([trainer, *others], cores=4) == {"train n3200"}


def test_the_trainer_outweighs_a_machine_full_of_other_jobs():
    # Every core of a large machine taken by single-core hand-offs at
    # the highest priority below the trainer's still weighs less.
    cores = 256
    others = [
        _job(f"handoff {i}", compute.HANDOFF_PRIORITY + 1.9, 1)
        for i in range(cores)
    ]
    trainer = _job("train", compute.TRAIN_PRIORITY, cores, gpu=1)
    assert _select([trainer, *others], cores=cores) == {"train"}


def test_the_bound_covers_every_band_below_the_trainer():
    others = max(
        compute.HANDOFF_PRIORITY,
        compute.REFERENCE_PRIORITY,
        compute.CHECKPOINT_PRIORITY,
        compute.BATCH_PRIORITY,
        compute.EVALUATION_PRIORITY,
        compute.NEXT_BATCH_PRIORITY,
    )
    assert 10_000 * (others + 2) < compute.TRAIN_PRIORITY
