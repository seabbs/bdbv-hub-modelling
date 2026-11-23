"""CI runs a fast tier on every push and the model fits only on request."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
OPT_IN = "full-tests"


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow):
    # YAML reads the `on:` key as True.
    return workflow.get("on", workflow.get(True))


def test_the_fast_tier_runs_no_slow_tests():
    steps = _workflow("test.yml")["jobs"]["test"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "pytest" in runs
    assert "-m slow" not in runs
    assert "bdbvhubrenewal fit" not in runs
    # Across the runner's cores (pytest-xdist).
    assert "-n auto" in runs


@pytest.mark.parametrize("job", ["full", "r-live", "chronos"])
def test_fits_and_replays_run_nightly_on_dispatch_or_when_labelled(job):
    workflow = _workflow("test.yml")
    when = workflow["jobs"][job]["if"]
    for case in ("schedule", "workflow_dispatch", OPT_IN):
        assert case in when
    assert "schedule" in _triggers(workflow)


def test_the_chronos_job_runs_every_torch_gated_test_with_torch():
    # Tests that skip without torch run in a job that has it: the chronos
    # group, CPU only, every file gated on it found, not listed.
    steps = _workflow("test.yml")["jobs"]["chronos"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "--group chronos" in runs
    assert "importorskip" in runs
    assert "pytest" in runs


@pytest.mark.parametrize(
    ("name", "job"),
    [("pages.yml", "prep"), ("baselinenowcast.yml", "fixtures")],
)
def test_backtests_and_fixtures_are_opt_in_on_pull_requests(name, job):
    workflow = _workflow(name)
    assert OPT_IN in workflow["jobs"][job]["if"]
    assert "labeled" in _triggers(workflow)["pull_request"]["types"]


@pytest.mark.parametrize(
    "path",
    # The fit store and the hub submission never cancel a running run.
    sorted(
        p
        for p in WORKFLOWS.glob("*.yml")
        if p.name not in ("fit-store.yml", "submit.yml")
    ),
    ids=lambda p: p.name,
)
def test_runs_off_main_are_superseded_and_main_keeps_its_running_run(path):
    group = yaml.safe_load(path.read_text()).get("concurrency")
    if not isinstance(group, dict) or "cancel-in-progress" not in group:
        pytest.skip("no cancellable concurrency group")
    assert group["cancel-in-progress"] == (
        "${{ github.ref != 'refs/heads/main' }}"
    )


def test_a_pull_request_runs_one_test_job_of_at_most_15_minutes():
    # The workflow and paper tests run inside the fast tier, with
    # Snakemake (the paper group), rather than in jobs of their own.
    assert not (WORKFLOWS / "workflow.yml").exists()
    assert not (WORKFLOWS / "paper-tests.yml").exists()
    job = _workflow("test.yml")["jobs"]["test"]
    assert job["timeout-minutes"] <= 15
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "--group paper pytest" in runs
    paper = [s for s in job["steps"] if s.get("working-directory") == "paper"]
    assert paper
    assert "pytest" in paper[0]["run"]
    assert "--durations" in runs


def test_no_test_runs_in_both_the_full_and_the_chronos_job():
    steps = _workflow("test.yml")["jobs"]["full"]["steps"]
    full = " ".join(s.get("run", "") for s in steps)
    chronos = " ".join(
        s.get("run", "")
        for s in _workflow("test.yml")["jobs"]["chronos"]["steps"]
    )
    grep = "(importorskip|find_spec)"
    assert grep in full
    assert grep in chronos
    assert "--ignore=" in full
