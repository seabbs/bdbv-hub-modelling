"""Each job on as many cores as its threads (`flowlib.cpus`)."""

import os
from pathlib import Path

from flowlib import cpus, machine


def test_jobs_take_free_cores_and_never_more_than_their_threads(tmp_path):
    allowed = list(range(8))
    first, held = cpus.take(allowed, 2, tmp_path)
    assert first == [0, 1]
    second, held2 = cpus.take(allowed, 3, tmp_path)
    assert second == [2, 3, 4]
    # With fewer free than asked, it still runs on that many cores.
    third, held3 = cpus.take(allowed, 4, tmp_path)
    assert len(third) == 4
    assert {5, 6, 7} <= set(third)
    for fd in held + held2 + held3:
        os.close(fd)
    # Released when their holders end.
    again, held = cpus.take(allowed, 2, tmp_path)
    assert again == [0, 1]
    for fd in held:
        os.close(fd)


def test_a_job_on_all_its_cores_takes_none(tmp_path):
    assert cpus.take([0, 1], 2, tmp_path) == ([0, 1], [])
    assert not list(Path(tmp_path).glob("*.lock"))


def test_forecasting_jobs_run_on_their_threads_of_cores():
    from flowlib import forecast, scoring

    s = scoring.Scoring.__new__(scoring.Scoring)
    s.local = {}
    got = s._env(forecast.Forecast("cpu", 2), 3)
    assert got.startswith(f"{machine.CPUS} 3 ")
    assert "OMP_NUM_THREADS=3" in got
