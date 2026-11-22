"""The PyRenew reporting step (`PyRenewReporting`), in the simulator.

A batch runs through it end to end with no R. Where the pinned epinowcast
is installed, the same jobs also go through the R engine's step
(`RscriptReporting`): the values each step adds, and the strata and dates
of the counts, must agree.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.simulator import simulate_batch
from nowcastfoundation.simulator.reporting import (
    COUNT_COLUMNS,
    PyRenewReporting,
    RscriptReporting,
)

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"
DRAWS = 20


class Recorder:
    """The PyRenew step, keeping the jobs and results it saw."""

    def __init__(self) -> None:
        self.step = PyRenewReporting()
        self.jobs: list = []
        self.results: list = []

    def report(self, jobs, grid_file, draws):
        out = self.step.report(jobs, grid_file, draws)
        self.jobs += list(jobs)
        self.results += out
        return out

    def software(self):
        return self.step.software()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pyrenew-batch")
    rec = Recorder()
    files = simulate_batch(
        GRID_FILE, 6161, 3, out, reporting=rec, weeks=8, draws=DRAWS
    )
    return files, rec


def test_a_batch_runs_through_the_pyrenew_step(run) -> None:
    files, rec = run
    counts = pd.read_csv(files["counts"])
    assert len(counts) > 0
    assert set(COUNT_COLUMNS) - {"draw"} <= set(counts.columns)
    ok = [r for r in rec.results if r.error is None]
    assert ok
    for r in ok:
        assert list(r.counts.columns) == list(COUNT_COLUMNS)
        assert (r.counts["count"] != 0).all()
        assert set(r.counts["draw"]) <= set(range(1, DRAWS + 1))
        for key in (
            "stratum_location",
            "low_start_days",
            "lambda_hat",
            "sum_p2",
            "sqrt_phi_obs",
        ):
            assert key in r.vals
    assert "pyrenew" in rec.software()


def _needs_enw() -> str | None:
    lib = os.environ.get("ENW_LIB")
    if shutil.which("Rscript") is None:
        return "needs Rscript"
    if not lib or not (Path(lib) / "epinowcast").is_dir():
        return "needs the pinned epinowcast in ENW_LIB"
    return None


@pytest.mark.skipif(_needs_enw() is not None, reason=_needs_enw() or "")
def test_the_steps_agree_on_the_same_jobs(run, monkeypatch) -> None:
    _, rec = run
    # As the simulate rule runs R: this R, then the system's tools (a
    # toolchain earlier on the PATH can break the model's compile).
    rscript = shutil.which("Rscript") or "Rscript"
    path = os.pathsep.join(
        [str(Path(rscript).parent), "/usr/bin", "/bin", os.environ["PATH"]]
    )
    monkeypatch.setenv("PATH", path)
    r_step = RscriptReporting(
        root=ROOT, rscript=rscript, enw_lib=os.environ["ENW_LIB"]
    )
    ok = [
        (j, r)
        for j, r in zip(rec.jobs, rec.results, strict=True)
        if r.error is None
    ]
    jobs = [j for j, _ in ok]
    theirs = r_step.report(jobs, GRID_FILE, DRAWS)
    for (job, ours), r in zip(ok, theirs, strict=True):
        assert r.error is None, (job.id, r.error)
        a, b = ours.vals, r.vals
        assert a["stratum_location"] == list(
            np.atleast_1d(b["stratum_location"])
        )
        assert a["low_start_days"] == int(b["low_start_days"])
        assert a["lambda_hat"] == pytest.approx(float(b["lambda_hat"]))
        assert a["sum_p2"] == pytest.approx(float(b["sum_p2"]), rel=1e-6)
        np.testing.assert_allclose(
            a["sqrt_phi_obs"], np.atleast_1d(b["sqrt_phi_obs"]), rtol=1e-6
        )
        # The same strata and dates. Totals are not compared: each step
        # draws the delay's random effects once per simulation, and with a
        # non-parametric part and no report-date model those change the
        # total (epinowcast adds them to log probabilities, `ref_as_p`);
        # the expectation itself is replayed cell by cell in
        # tests/model/test_reporting_expectation.py.
        assert set(ours.counts["location"]) <= set(a["stratum_location"])
        assert set(r.counts["location"]) <= set(a["stratum_location"])
        # Both within the job's reporting frame.
        first = pd.Timestamp("2026-01-03") + pd.Timedelta(
            days=int(job.vals["offset"])
        )
        last = first + pd.Timedelta(days=len(job.onsets) - 1)
        for counts in (ours.counts, r.counts):
            for col in ("reference_date", "report_date"):
                dates = pd.to_datetime(counts[col])
                assert dates.min() >= first
                assert dates.max() <= last
