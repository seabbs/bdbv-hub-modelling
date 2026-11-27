"""Worker processes sharing out a job's threads and cores."""

import os

import pytest

from nowcastutils import workers


@pytest.mark.parametrize(
    ("threads", "n", "each"), [(4, 4, 1), (8, 4, 2), (2, 4, 1)]
)
def test_workers_share_the_job_s_threads(threads, n, each) -> None:
    assert workers.worker_threads(threads, n) == each


@pytest.mark.parametrize(
    ("index", "n", "cores"),
    [(0, 2, [10, 11]), (1, 2, [12, 13]), (2, 2, [10, 11]), (0, 8, [10])],
)
def test_each_worker_takes_its_own_of_the_job_s_cores(index, n, cores) -> None:
    assert workers.worker_cores([10, 11, 12, 13], index, n) == cores


def _start(n_tasks, counter, n, parent) -> None:
    # Each worker is given the job's shared counter, its count and its
    # coordinator.
    assert counter.value >= 0
    assert n == min(3, n_tasks)
    assert parent == os.getppid()


def _square(x: int) -> tuple[int, str]:
    return x * x, os.environ["TORCH_THREADS"]


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_pooled_tasks_return_their_results_with_a_share_of_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_THREADS", "6")
    got = dict(
        workers.pooled(_square, [(i,) for i in range(5)], 3, _start, (5,))
    )
    assert got == {(i,): (i * i, "2") for i in range(5)}
    # The coordinator's own threads are as they were.
    assert os.environ["TORCH_THREADS"] == "6"
