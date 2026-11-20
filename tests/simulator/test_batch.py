"""The simulator's entry point (`nowcastfoundation.simulator.simulate_batch`).

The batch logic runs with a Python stand-in for the reporting step (each
stratum's expected onsets reported on their reference day), so it needs
neither R nor epinowcast; one test runs the whole simulator with the R
engine's epinowcast step where the pinned epinowcast is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator import draws, simulate_batch
from nowcastfoundation.simulator.reporting import COUNT_COLUMNS, ReportResult

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"


class OnTheDay:
    """Report each stratum's rounded expected onsets on their own day."""

    def __init__(self, fail: set[str] | None = None, scale: float = 1.0):
        self.fail = fail or set()
        self.scale = scale
        self.calls: list[list[str]] = []

    def report(self, jobs, _grid_file, _n_draws):
        self.calls.append([j.id for j in jobs])
        out = []
        for j in jobs:
            outbreak = j.id.split("/")[0]
            if j.id in self.fail or outbreak in self.fail:
                out.append(
                    ReportResult(j.id, pd.DataFrame(), error="stand-in error")
                )
                continue
            on = np.rint(np.asarray(j.onsets) * self.scale).astype(int)
            days, strata = np.nonzero(on)
            dates = pd.Timestamp("2026-01-03") + pd.to_timedelta(days, "D")
            rows = pd.DataFrame(
                {
                    "draw": 1,
                    "location": [f"S{s + 1}" for s in strata],
                    "reference_date": dates.strftime("%Y-%m-%d"),
                    "report_date": dates.strftime("%Y-%m-%d"),
                    "count": on[days, strata],
                }
            )
            out.append(
                ReportResult(
                    j.id,
                    rows[list(COUNT_COLUMNS)],
                    vals={**j.vals, "reported": True},
                )
            )
        return out

    def software(self):
        return {"reporting": "on-the-day"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("batch")
    step = OnTheDay()
    files = simulate_batch(GRID_FILE, 4242, 4, out, reporting=step, weeks=8)
    return files, step


def test_a_batch_writes_its_four_files(run):
    files, _ = run
    assert {"counts", "settings", "latent"} <= set(files)
    counts = pd.read_csv(files["counts"])
    assert tuple(counts.columns) == sb.COUNTS
    latent = pd.read_csv(files["latent"])
    assert tuple(latent.columns) == sb.LATENT
    settings = pd.read_csv(files["settings"])
    assert list(settings["outbreak"]) == [f"4242-{i:04d}" for i in range(1, 5)]
    assert set(counts["outbreak"]) == {f"{o}-1" for o in settings["outbreak"]}
    assert settings.columns.is_unique
    for column in ("tries", "designs", "cases", "seconds", "reporting"):
        assert column in settings
    assert (settings["weeks"] == 8).all()
    assert settings["reported"].all()
    # Cases are the counts' total, at least the floor.
    total = counts.groupby(counts["outbreak"].str[:-2])["count"].sum()
    assert np.allclose(total[settings["outbreak"]], settings["cases"])
    assert (settings["cases"] >= 10).all()


def test_the_latent_file_is_dated_from_the_offset(run):
    files, _ = run
    latent = pd.read_csv(files["latent"])
    settings = pd.read_csv(files["settings"]).set_index("outbreak")
    first = latent.groupby("outbreak").first()
    for outbreak, row in first.iterrows():
        offset = int(settings.loc[outbreak, "offset"])
        start = pd.Timestamp("2026-01-03") + pd.Timedelta(days=offset)
        assert pd.Timestamp(row["date"]) == start + pd.Timedelta(
            days=int(row["day"])
        )


def test_simulations_advance_together_in_passes(run):
    _, step = run
    assert len(step.calls[0]) == 4
    assert all(len(set(c)) == len(c) for c in step.calls)


def test_a_batch_repeats_from_its_seed(tmp_path, run):
    files, _ = run
    again = simulate_batch(
        GRID_FILE, 4242, 4, tmp_path, reporting=OnTheDay(), weeks=8
    )
    a = pd.read_csv(files["counts"])
    b = pd.read_csv(again["counts"])
    pd.testing.assert_frame_equal(a, b)


class FailFirst(OnTheDay):
    """Fail the first `k` draws."""

    def __init__(self, k: int):
        super().__init__()
        self.left = k

    def report(self, jobs, _grid_file, _n_draws):
        self.fail = set()
        for j in jobs:
            if self.left > 0:
                self.fail.add(j.id)
                self.left -= 1
        return super().report(jobs, _grid_file, _n_draws)


def test_a_failing_draw_is_drawn_again(tmp_path):
    files = simulate_batch(
        GRID_FILE, 5151, 1, tmp_path, reporting=FailFirst(2), weeks=8
    )
    failures = pd.read_csv(files["failures"])
    assert len(failures) == 2
    assert (failures["reason"] == "stand-in error").all()
    settings = pd.read_csv(files["settings"])
    assert settings.loc[0, "tries"] == 3
    assert settings.loc[0, "designs"] == 1


def test_a_simulation_with_no_design_stops_the_batch(tmp_path):
    step = OnTheDay(fail={"5150-0001"})
    with pytest.raises(RuntimeError, match="no design in"):
        simulate_batch(GRID_FILE, 5150, 1, tmp_path, reporting=step, weeks=8)
    grid = draws.read_grid(GRID_FILE)
    # One job per try, `tries` per design, `designs` designs (independent
    # strata fail their first stratum, so they take the same calls).
    assert len(step.calls) >= grid["designs"]


def test_too_few_cases_are_drawn_again(tmp_path):
    files = simulate_batch(
        GRID_FILE, 6060, 2, tmp_path, reporting=OnTheDay(scale=1e-3), weeks=8
    )
    failures = pd.read_csv(files["failures"])
    assert (failures["reason"] == "fewer than 10 cases").any()
    settings = pd.read_csv(files["settings"])
    assert (settings["cases"] >= 10).all()


def test_independent_strata_are_simulations_of_their_own():
    grid = draws.read_grid(GRID_FILE)
    b = sb._Batch(grid, 9, 1, 8)
    rng = draws.generator(9, 1)
    for _ in range(500):
        design = draws.draw_design(grid, rng, weeks=8)
        if (
            design["strata_link"] == "independent"
            and design["strata"] == "places_ages"
        ):
            break
    co = b.cell(design, (1, 1), "9-0001")
    job = next(co)
    seen = []
    while True:
        assert job.design["strata"] == "single"
        assert job.design["n_places"] == job.design["n_ages"] == 1
        assert job.design["observation"] == design["observation"]
        assert job.design["weeks"] == 8
        seen.append(job.id)
        outcome = sb._Outcome(
            vals={**job.vals, "stratum_population": [100.0]},
            counts=pd.DataFrame(
                {
                    "draw": [1],
                    "location": ["P01"],
                    "reference_date": ["2026-01-05"],
                    "report_date": ["2026-01-05"],
                    "count": [50],
                }
            ),
            latent=pd.DataFrame({"day": [0], "location": ["all"]}),
            cases=50.0,
        )
        try:
            job = co.send(outcome)
        except StopIteration as stop:
            out, failures = stop.value
            break
    assert not failures
    labels = out.vals["stratum_location"]
    assert seen == [f"9-0001/{lab}" for lab in labels]
    assert set(out.counts["location"]) == set(labels)
    assert all("-" in lab for lab in labels)
    assert out.vals["stratum_population"] == [100.0] * len(labels)
    assert out.cases == 50.0 * len(labels)
    assert len(out.vals["stratum_trajectory"].split(";")) == len(labels)


def test_settings_flatten_vectors_as_the_engine_does():
    row = sb.flat({"a": [1.5], "b": [1, 2], "c": [], "d": True, "e": "x"})
    assert row == {"a": 1.5, "b": "1;2", "c": "", "d": "TRUE", "e": "x"}


def _needs_enw():
    lib = os.environ.get("ENW_LIB")
    if shutil.which("Rscript") is None:
        return "needs Rscript"
    if not lib or not (Path(lib) / "epinowcast").is_dir():
        return "needs the pinned epinowcast in ENW_LIB"
    return None


@pytest.mark.skipif(_needs_enw() is not None, reason=_needs_enw() or "")
def test_the_cli_simulates_through_epinowcast(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "nowcastfoundation.simulator",
            str(GRID_FILE),
            "2",
            "7171",
            str(tmp_path),
        ],
        cwd=ROOT,
        # As the simulate rule runs it: the system's tools first for R's
        # children (the model compiles into its own cache).
        env={
            **os.environ,
            "ENW_WEEKS": "8",
            "ENW_CACHE": str(tmp_path / "enwcache"),
            "RSCRIPT": shutil.which("Rscript") or "Rscript",
            "PATH": "/usr/bin:/bin:" + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    settings = pd.read_csv(tmp_path / "settings-7171.csv")
    counts = pd.read_csv(tmp_path / "counts-7171.csv.gz")
    assert len(settings) == 2
    assert settings["r_version"].str.startswith("R version").all()
    assert settings["enw_fingerprint"].notna().all()
    assert "sqrt_phi_obs" in settings
    # Increments, so revisions can be negative; zeros are not written.
    assert (counts["count"] != 0).all()
    assert (
        pd.to_datetime(counts["report_date"])
        >= pd.to_datetime(counts["reference_date"])
    ).all()
