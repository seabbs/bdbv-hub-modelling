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


def test_backtests_are_opt_in_on_pull_requests():
    workflow = _workflow("pages.yml")
    assert OPT_IN in workflow["jobs"]["prep"]["if"]
    assert "labeled" in _triggers(workflow)["pull_request"]["types"]


def test_r_live_checks_the_baselinenowcast_port_at_the_pin():
    # R baselinenowcast runs live in r-live, slow sampler checks too; no
    # fixtures or workflow of their own.
    steps = _workflow("test.yml")["jobs"]["r-live"]["steps"]
    text = yaml.safe_dump(steps)
    assert (
        "epinowcast/baselinenowcast@999d1ad50e27921a4078175d6315bbd655878888"
        in text
    )
    assert "any::zoo" in text
    runs = [s.get("run", "") for s in steps]
    assert any("tests/baselinenowcast" in r and '-m ""' in r for r in runs)
    assert not (WORKFLOWS / "baselinenowcast.yml").exists()
    assert not (ROOT / "tests" / "baselinenowcast" / "fixtures").exists()


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
    keep = "${{ github.ref != 'refs/heads/main' }}"
    if path.name == "test.yml":
        # Tests keeps next's runs too (every merge gets a verdict).
        keep = (
            "${{ github.ref != 'refs/heads/main' && "
            "github.ref != 'refs/heads/next' }}"
        )
    assert group["cancel-in-progress"].strip() == keep


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


def test_every_merge_into_next_gets_a_fast_tier_run():
    # Pull request runs are cancelled when the merge loop deletes the
    # branch, so a push to next runs the fast tier, and a newer merge
    # does not cancel the running one.
    workflow = _workflow("test.yml")
    assert "next" in _triggers(workflow)["push"]["branches"]
    keep = workflow["concurrency"]["cancel-in-progress"]
    assert "refs/heads/next" in keep


def test_the_r_live_job_installs_what_the_live_r_tests_call():
    steps = _workflow("test.yml")["jobs"]["r-live"]["steps"]
    packages = " ".join(
        str(s.get("with", {}).get("packages", "")) for s in steps
    )
    # primarycensored's fitdistdoublecens needs fitdistrplus (a Suggests).
    for name in ("primarycensored", "flexsurv", "fitdistrplus"):
        assert name in packages


def test_r_package_lists_hold_packages_only():
    # pak reads every line of `packages` as a package, so a comment there
    # breaks the install (r-live failed at setup-r-dependencies).
    for path in WORKFLOWS.glob("*.yml"):
        for job in (
            yaml.safe_load(path.read_text()).get("jobs") or {}
        ).values():
            for step in job.get("steps", []):
                listed = str((step.get("with") or {}).get("packages", ""))
                bad = [
                    ln
                    for ln in listed.splitlines()
                    if ln.strip().startswith("#")
                ]
                assert bad == [], f"{path.name}: {bad}"
