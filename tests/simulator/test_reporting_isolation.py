"""A draw whose reporting fails is that draw's error, not its batch's.

The R step reports a pass in one Rscript call; if that call dies (R
killed or out of memory, with no per-job error), each job is reported
again in its own call, so only the jobs that fail alone are redrawn.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.simulator import reporting as rep
from nowcastfoundation.simulator.reporting import (
    PyRenewReporting,
    ReportJob,
    RscriptReporting,
)

FAKE_R = """\
import json, sys
from pathlib import Path
job_dir = Path(sys.argv[3])
jobs = json.loads((job_dir / "jobs.json").read_text())["jobs"]
with open(Path(sys.argv[4]), "a") as log:
    log.write(",".join(j["id"] for j in jobs) + "\\n")
if any(j["id"].startswith("bad") for j in jobs):
    sys.exit(1)
rows = ["id,draw,location,reference_date,report_date,count"]
rows += [f"{j['id']},1,P01,2026-01-03,2026-01-04,3" for j in jobs]
(job_dir / "counts.csv").write_text("\\n".join(rows) + "\\n")
(job_dir / "results.json").write_text(json.dumps(
    [{"id": j["id"], "values": {"x": 1}} for j in jobs]
))
(job_dir / "stack.json").write_text(json.dumps({"r_version": "fake"}))
"""


def _jobs(*ids: str) -> list[ReportJob]:
    return [
        ReportJob(i, 1, {}, {}, np.ones((3, 1)), ["P01"], ["A1"]) for i in ids
    ]


@pytest.fixture
def fake_r(tmp_path: Path) -> tuple[RscriptReporting, Path]:
    script = tmp_path / "fake_r.py"
    script.write_text(FAKE_R)
    calls = tmp_path / "calls.txt"
    rscript = tmp_path / "Rscript"
    rscript.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@" "{calls}"\n'
    )
    rscript.chmod(rscript.stat().st_mode | stat.S_IEXEC)
    return RscriptReporting(rscript=str(rscript), root=tmp_path), calls


def test_a_dead_r_call_fails_only_the_draws_that_fail_alone(
    fake_r: tuple[RscriptReporting, Path], tmp_path: Path
) -> None:
    step, calls = fake_r
    out = step.report(_jobs("a", "bad", "b"), tmp_path / "grid.yaml", 1)
    assert [r.id for r in out] == ["a", "bad", "b"]
    assert out[0].error is None
    assert out[2].error is None
    assert out[1].error is not None
    assert "reporting step failed" in out[1].error
    assert len(out[0].counts) == 1
    assert out[0].vals == {"x": 1}
    assert step.software() == {"r_version": "fake"}
    # The pass, then each job alone.
    assert calls.read_text().split() == ["a,bad,b", "a", "bad", "b"]


def test_a_pass_that_runs_is_one_r_call(
    fake_r: tuple[RscriptReporting, Path], tmp_path: Path
) -> None:
    step, calls = fake_r
    out = step.report(_jobs("a", "b"), tmp_path / "grid.yaml", 1)
    assert all(r.error is None for r in out)
    assert calls.read_text().split() == ["a,b"]


def test_any_error_in_one_pyrenew_draw_is_that_draws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import nowcastrenewal.reporting.simulate as sim

    def fake(*args: object) -> tuple[pd.DataFrame, dict[str, int]]:
        seed = args[5]
        if seed == 2:
            raise RuntimeError("out of memory")
        return pd.DataFrame(columns=list(rep.COUNT_COLUMNS)), {"y": 2}

    monkeypatch.setattr(sim, "report_simulation", fake)
    jobs = _jobs("a", "bad")
    jobs[1].seed = 2
    out = PyRenewReporting().report(jobs, tmp_path / "grid.yaml", 1)
    assert out[0].error is None
    assert out[1].error == "out of memory"
    assert json.dumps(out[0].vals) == '{"y": 2}'
