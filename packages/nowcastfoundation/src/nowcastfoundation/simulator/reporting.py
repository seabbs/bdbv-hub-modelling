# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The simulator's reporting step, behind one interface.

A reporting step takes simulations whose latent step has run (each with
its design, values and every stratum's expected onsets) and returns each
one's reported counts as increments of its daily release triangle, with
the values it added, or why it failed.
`RscriptReporting` runs the R engine's `report_outbreak` (epinowcast) in
one Rscript call per pass; a Python reporting step from PyRenew components
(`nowcastrenewal.reporting`) can take its place by implementing `report`.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

__all__ = [
    "COUNT_COLUMNS",
    "ReportJob",
    "ReportResult",
    "ReportingStep",
    "RscriptReporting",
]

COUNT_COLUMNS = (
    "draw",
    "location",
    "reference_date",
    "report_date",
    "count",
)
"""A report's count columns: one row per non-zero increment."""


@dataclass
class ReportJob:
    """A simulation to report.

    Attributes
    ----------
    id : str
        The simulation's id.
    seed : int
        The reporting step's seed for this simulation.
    design : dict
        Its design (`draws.draw_design`).
    vals : dict
        Its values (`draws.draw_values`), with the latent step's additions
        (`stratum_population`, `attack`).
    onsets : numpy.ndarray
        Expected (ascertained) onsets, days by strata: the days of the
        reporting frame (`7 * (weeks + max_delay // 7 + 4)`, the first
        2026-01-03 plus the `offset`), the strata in epinowcast's `.group`
        order, places first then ages within each place.
    places, ages : list of str
        Each onsets column's place (`P01`, ...) and age (`A1`, ...).
    """

    id: str
    seed: int
    design: dict[str, Any]
    vals: dict[str, Any]
    onsets: np.ndarray
    places: list[str] = field(default_factory=list)
    ages: list[str] = field(default_factory=list)


@dataclass
class ReportResult:
    """A reported simulation.

    Attributes
    ----------
    id : str
        The simulation's id.
    counts : pandas.DataFrame
        `COUNT_COLUMNS` (empty on an error).
    vals : dict
        The job's values with the step's additions.
    error : str or None
        Why the simulation has no report, if it has none.
    """

    id: str
    counts: pd.DataFrame
    vals: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ReportingStep(Protocol):
    """Report simulations' expected onsets (`ReportJob`).

    A step returns one `ReportResult` per job, in the jobs' order, with
    the counts as `COUNT_COLUMNS` (draws 1 to `draws`; locations the
    stratum's place, its age, or `<place>-<age>` when there are both) and
    the job's values with these added, which the settings files and their
    readers use: `stratum_location` (each stratum's location, in the
    onsets' column order), `low_start_days` (days before the outbreak
    first expects one onset a day), `lambda_hat`, `sum_p2` and
    `sqrt_phi_obs` (the observation noise as the step applied it).
    A job that fails gets `error` set rather than raising.
    """

    def report(
        self, jobs: Sequence[ReportJob], grid_file: str | Path, draws: int
    ) -> list[ReportResult]:
        """Return one result per job, in the jobs' order."""
        ...

    def software(self) -> dict[str, Any]:
        """Return what the settings record of the step's software."""
        ...


def _json(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON: {type(x).__name__}")


@dataclass
class RscriptReporting:
    """The R engine's epinowcast reporting step, run through Rscript.

    Attributes
    ----------
    engine : pathlib.Path
        The R engine (`packages/nowcastfoundation/r`), loaded with
        `pkgload`.
    root : pathlib.Path
        The repository root, the step's working directory.
    rscript : str
        The Rscript to run.
    enw_lib : str or None
        The pinned epinowcast's library (`ENW_LIB`); the environment's if
        `None`.
    """

    engine: Path = Path("packages/nowcastfoundation/r")
    root: Path = Path()
    rscript: str = "Rscript"
    enw_lib: str | None = None
    _stack: dict[str, Any] = field(default_factory=dict, repr=False)

    def report(
        self, jobs: Sequence[ReportJob], grid_file: str | Path, draws: int
    ) -> list[ReportResult]:
        """Report the jobs in one Rscript call, in a temporary folder."""
        if not jobs:
            return []
        with tempfile.TemporaryDirectory(prefix="report-") as tmp:
            job_dir = Path(tmp)
            payload = {
                "grid_file": str(Path(grid_file).resolve()),
                "draws": int(draws),
                "jobs": [
                    {
                        "id": j.id,
                        "seed": int(j.seed),
                        "design": j.design,
                        "values": j.vals,
                        "onsets": np.asarray(j.onsets, dtype=float),
                        "places": list(j.places),
                        "ages": list(j.ages),
                    }
                    for j in jobs
                ],
            }
            (job_dir / "jobs.json").write_text(
                json.dumps(payload, default=_json)
            )
            env = dict(os.environ)
            if self.enw_lib is not None:
                env["ENW_LIB"] = str(self.enw_lib)
            env["TMPDIR"] = str(job_dir)
            out = subprocess.run(
                [
                    self.rscript,
                    str(self.engine / "inst" / "report.R"),
                    str(self.engine),
                    str(job_dir),
                ],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
            )
            if out.returncode != 0:
                raise RuntimeError(
                    "the reporting step failed:\n" + out.stderr[-4000:]
                )
            try:
                counts = pd.read_csv(
                    job_dir / "counts.csv",
                    dtype={"id": str, "location": str},
                )
                results = json.loads((job_dir / "results.json").read_text())
                self._stack = json.loads((job_dir / "stack.json").read_text())
            except (OSError, EOFError, ValueError) as e:
                # A truncated output (a full disk, say) despite R's success.
                raise RuntimeError(
                    f"the reporting step's output is unreadable ({e}):\n"
                    + out.stderr[-4000:]
                ) from e
        by_id = {str(i): g for i, g in counts.groupby("id", sort=False)}
        empty = pd.DataFrame(columns=list(COUNT_COLUMNS))
        out_results = []
        for r in results:
            rid = str(r["id"])
            if r.get("error") is not None:
                out_results.append(
                    ReportResult(rid, empty, error=str(r["error"]))
                )
                continue
            rows = by_id.get(rid, empty)
            out_results.append(
                ReportResult(
                    rid,
                    rows.loc[:, list(COUNT_COLUMNS)].reset_index(drop=True),
                    vals=dict(r["values"]),
                )
            )
        return out_results

    def software(self) -> dict[str, Any]:
        """Return R's version, BLAS, LAPACK and epinowcast's fingerprint."""
        return dict(self._stack)
