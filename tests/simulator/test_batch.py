"""The simulator's entry point (`nowcastfoundation.simulator.simulate_batch`).

The batch logic runs with a stand-in for the joint model (each stratum
reporting a few cases on the day); two tests run the whole simulator, the
joint PyRenew model, end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator import draws, simulate_batch
from nowcastrenewal.reporting.simulate import COUNT_COLUMNS
from nowcastrenewal.simulation import outbreak as joint

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"


class OnTheDay:
    """A stand-in for the joint model (`simulate_outbreak`).

    Each stratum reports one case on each of its first `cases` days, on
    the day; the first `fail` calls raise and the next `empty` report
    nothing. Records each call's design.
    """

    def __init__(self, fail: int = 0, empty: int = 0, cases: int = 3):
        self.fail, self.empty, self.cases = fail, empty, cases
        self.calls: list[dict] = []

    def __call__(self, _key, n_days, strata, design, values, draws=1, **_kw):
        self.calls.append(design)
        if self.fail:
            self.fail -= 1
            raise ValueError("stand-in error")
        k = len(strata)
        days = np.arange(n_days)
        path = pd.DataFrame(
            {
                "day": days,
                "infections": 1.0,
                "onsets": 1.0,
                "r": 1.0,
                "reff": 1.0,
            }
        )
        latent = SimpleNamespace(
            path=path, stratum_population=np.full(k, 100.0), attack=0.1
        )
        dates = pd.Timestamp("2026-01-03") + pd.to_timedelta(
            int(values["offset"]) + days[: self.cases], "D"
        )
        rows = [
            {
                "draw": d,
                "location": f"S{s + 1}",
                "reference_date": t.strftime("%Y-%m-%d"),
                "report_date": t.strftime("%Y-%m-%d"),
                "count": 1,
            }
            for d in range(1, draws + 1)
            for s in range(k)
            for t in dates
        ]
        counts = pd.DataFrame(rows, columns=list(COUNT_COLUMNS))
        if self.empty:
            self.empty -= 1
            counts = counts.iloc[:0]
        added = {"stratum_location": [f"S{s + 1}" for s in range(k)]}
        return latent, counts, {**added, "reported": True}


@pytest.fixture
def stand_in(monkeypatch):
    """Put a stand-in in place of the joint model."""

    def use(**kw):
        model = OnTheDay(**kw)
        monkeypatch.setattr(joint, "simulate_outbreak", model)
        return model

    return use


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("batch")
    model = OnTheDay()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(joint, "simulate_outbreak", model)
        files = simulate_batch(GRID_FILE, 4242, 4, out, weeks=8)
    return files, model


@pytest.mark.slow
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
    for column in (
        "tries",
        "designs",
        "cases",
        "seconds",
        "nowcastrenewal_version",
    ):
        assert column in settings
    assert (settings["weeks"] == 8).all()
    assert settings["reported"].all()
    # Cases are the counts' total, none empty.
    total = counts.groupby(counts["outbreak"].str[:-2])["count"].sum()
    assert np.allclose(total[settings["outbreak"]], settings["cases"])
    assert (settings["cases"] > 0).all()


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


def test_a_batch_repeats_from_its_seed(tmp_path, run, stand_in):
    files, _ = run
    stand_in()
    again = simulate_batch(GRID_FILE, 4242, 4, tmp_path, weeks=8)
    a = pd.read_csv(files["counts"])
    b = pd.read_csv(again["counts"])
    pd.testing.assert_frame_equal(a, b)


def test_a_failing_draw_is_drawn_again(tmp_path, stand_in):
    stand_in(fail=2)
    files = simulate_batch(GRID_FILE, 5151, 1, tmp_path, weeks=8)
    failures = pd.read_csv(files["failures"])
    assert len(failures) == 2
    assert (failures["reason"] == "stand-in error").all()
    settings = pd.read_csv(files["settings"])
    assert settings.loc[0, "tries"] == 3
    assert settings.loc[0, "designs"] == 1


@pytest.mark.slow
def test_a_simulation_with_no_design_stops_the_batch(tmp_path, stand_in):
    model = stand_in(fail=10**6)
    with pytest.raises(RuntimeError, match="no design in"):
        simulate_batch(GRID_FILE, 5150, 1, tmp_path, weeks=8)
    grid = draws.read_grid(GRID_FILE)
    # One call per try, `tries` per design, `designs` designs (independent
    # strata fail their first stratum, so they take the same calls).
    assert len(model.calls) >= grid["designs"]


@pytest.mark.slow
def test_only_empty_draws_are_drawn_again(tmp_path, stand_in):
    stand_in(empty=2, cases=1)
    files = simulate_batch(GRID_FILE, 6060, 1, tmp_path, weeks=8)
    failures = pd.read_csv(files["failures"])
    assert len(failures) >= 2
    assert (failures["reason"] == "no cases").all()
    settings = pd.read_csv(files["settings"])
    # Under the old floor of 10 a draw this small was drawn again.
    assert 0 < settings.loc[0, "cases"] < 10


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


def test_the_cli_simulates_through_the_joint_model(tmp_path):
    # The simulate rule's command: each simulation is the joint PyRenew
    # model, with no R.
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
        env={**os.environ, "ENW_WEEKS": "8", "ENW_DRAWS": "2"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    settings = pd.read_csv(tmp_path / "settings-7171.csv")
    counts = pd.read_csv(tmp_path / "counts-7171.csv.gz")
    latent = pd.read_csv(tmp_path / "latent-7171.csv.gz")
    assert len(settings) == 2
    assert settings["nowcastrenewal_version"].notna().all()
    for key in (
        "sqrt_phi_obs",
        "lambda_hat",
        "sum_p2",
        "low_start_days",
        "stratum_location",
        "stratum_population",
        "attack",
    ):
        assert key in settings, key
    assert set(counts.columns) == set(sb.COUNTS)
    assert set(latent.columns) == set(sb.LATENT)
    # Increments, so revisions can be negative; zeros are not written.
    assert (counts["count"] != 0).all()
    assert (
        pd.to_datetime(counts["report_date"])
        >= pd.to_datetime(counts["reference_date"])
    ).all()
    # Two observation draws of each outbreak.
    assert (
        counts["outbreak"].str.rsplit("-", n=1).str[-1].isin(["1", "2"]).all()
    )


def test_the_joint_batch_repeats_from_its_seed(tmp_path):
    a = simulate_batch(GRID_FILE, 9191, 2, tmp_path / "a", weeks=8)
    b = simulate_batch(GRID_FILE, 9191, 2, tmp_path / "b", weeks=8)
    pd.testing.assert_frame_equal(
        pd.read_csv(a["counts"]), pd.read_csv(b["counts"])
    )
