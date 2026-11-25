"""The job launcher: cores and memory on a shared machine
(`flowlib.launch`), nothing under a cluster's scheduler."""

import os
from pathlib import Path

import pytest
from flowlib import launch, machine


def test_jobs_take_free_cores_and_never_more_than_their_threads(tmp_path):
    allowed = list(range(8))
    first, held = launch.take(allowed, 2, tmp_path)
    assert first == [0, 1]
    second, held2 = launch.take(allowed, 3, tmp_path)
    assert second == [2, 3, 4]
    # With fewer free than asked, it still runs on that many cores.
    third, held3 = launch.take(allowed, 4, tmp_path)
    assert len(third) == 4
    assert {5, 6, 7} <= set(third)
    for fd in held + held2 + held3:
        os.close(fd)
    # Released when their holders end.
    again, held = launch.take(allowed, 2, tmp_path)
    assert again == [0, 1]
    for fd in held:
        os.close(fd)


def test_a_job_on_all_its_cores_takes_none(tmp_path):
    assert launch.take([0, 1], 2, tmp_path) == ([0, 1], [])
    assert not list(Path(tmp_path).glob("*.lock"))


def test_side_jobs_never_take_the_trainer_s_cores(tmp_path):
    allowed = list(range(8))
    kept = launch.reserved(allowed, 3)
    assert kept == [5, 6, 7]
    side, held = launch.take(launch.unreserved(allowed, 3), 6, tmp_path)
    assert not set(side) & set(kept)
    trainer, held2 = launch.take(kept, 3, tmp_path)
    assert trainer == kept
    for fd in held + held2:
        os.close(fd)


@pytest.mark.parametrize(
    ("cfg", "environ", "on"),
    [
        ({}, {}, True),
        ({"COMPUTE": "single"}, {}, True),
        # An ssh worker runs each job with COMPUTE=single, on its own box.
        ({"COMPUTE": "ssh"}, {}, True),
        ({"COMPUTE": "slurm"}, {}, False),
        ({"COMPUTE": "single"}, {"SLURM_JOB_ID": "7"}, False),
    ],
)
def test_the_launcher_acts_only_on_a_shared_machine(cfg, environ, on):
    assert launch.shared(cfg, environ) is on
    prefix = machine.launch_prefix(cfg, threads=2, mem_mb=4096)
    assert bool(prefix) is (cfg.get("COMPUTE") in (None, "single", "ssh"))


def test_the_prefix_names_the_job_s_threads_or_the_trainer_and_memory():
    got = machine.launch_prefix({}, threads=3, mem_mb=4096)
    assert got == f"{machine.LAUNCH} --threads 3 --mem 4096 "
    got = machine.launch_prefix({}, trainer=True, mem_mb=8192)
    assert got == f"{machine.LAUNCH} --trainer --mem 8192 "


def test_a_job_waits_for_its_memory_and_says_so(capsys):
    free = iter([1000, 3000, 9000])
    slept = []
    launch.wait_for_memory(
        4096,
        headroom_mb=1024,
        meminfo=lambda: (next(free), 64_000),
        sleep=slept.append,
    )
    assert len(slept) == 2
    assert "waiting for 5120 MB" in capsys.readouterr().err


def test_a_job_larger_than_the_machine_never_waits_for_ever(capsys):
    slept = []
    launch.wait_for_memory(
        100_000,
        headroom_mb=0,
        meminfo=lambda: (10, 64_000),
        sleep=slept.append,
    )
    assert slept == []
    assert "more than" in capsys.readouterr().err


def test_forecasting_jobs_run_through_the_launcher():
    from types import SimpleNamespace

    from flowlib import forecast, scoring

    s = scoring.Scoring.__new__(scoring.Scoring)
    s.local = {}
    res = SimpleNamespace(mem_mb=4096)
    got = s._env(forecast.Forecast("cpu", 2), 3, res)
    assert got.startswith(f"{machine.LAUNCH} --threads 3 --mem 4096 ")
    assert "OMP_NUM_THREADS=3" in got
    s.local = {"COMPUTE": "slurm"}
    assert machine.LAUNCH not in s._env(forecast.Forecast("cpu", 2), 3, res)


def test_the_trainer_runs_through_the_launcher():
    text = (
        Path(__file__).parents[2] / "workflow/rules/finetune.smk"
    ).read_text()
    block = text.split("rule train_segment:", 1)[1].split("\nrule ", 1)[0]
    assert "{params.claim} {params.launch}" in block
    assert "trainer=True" in block
