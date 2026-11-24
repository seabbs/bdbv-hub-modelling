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


@pytest.mark.parametrize(
    ("name", "job"), [("ci.yml", "package"), ("integration.yml", "integration")]
)
def test_the_fast_tier_runs_no_slow_tests(name, job):
    steps = _workflow(name)["jobs"][job]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "pytest" in runs
    assert "-m slow" not in runs
    assert "bdbvhub fit" not in runs
    # Across the runner's cores (pytest-xdist).
    assert "-n auto" in runs


@pytest.mark.parametrize("job", ["full", "r-live", "chronos"])
def test_fits_and_replays_run_nightly_on_dispatch_or_when_labelled(job):
    workflow = _workflow("heavy.yml")
    when = workflow["jobs"][job]["if"]
    for case in ("schedule", "workflow_dispatch", OPT_IN):
        assert case in when
    assert "schedule" in _triggers(workflow)


def test_the_chronos_job_runs_every_torch_gated_test_with_torch():
    # Tests that skip without torch run in a job that has it: the chronos
    # group, CPU only, every file gated on it found, not listed.
    steps = _workflow("heavy.yml")["jobs"]["chronos"]["steps"]
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
    steps = _workflow("heavy.yml")["jobs"]["r-live"]["steps"]
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
    if path.name == "pages.yml":
        # Pages only loads and renders, so the newest run wins everywhere.
        assert group["cancel-in-progress"] is True
        return
    if path.name in ("ci.yml", "integration.yml"):
        # These keep next's runs too (every merge gets a verdict).
        keep = (
            "${{ github.ref != 'refs/heads/main' && "
            "github.ref != 'refs/heads/next' }}"
        )
    assert group["cancel-in-progress"].strip() == keep


def test_integration_runs_the_tests_of_no_package_in_15_minutes():
    # The workflow, paper and repo tests (those no package owns) and the
    # workflow lint, with Snakemake (the paper group); the packages'
    # tests run in their own CI jobs, so Tests has no fast tier.
    assert not (WORKFLOWS / "workflow.yml").exists()
    assert not (WORKFLOWS / "paper-tests.yml").exists()
    assert "test" not in _workflow("heavy.yml")["jobs"]
    job = _workflow("integration.yml")["jobs"]["integration"]
    assert job["timeout-minutes"] <= 15
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "owners.py --none" in runs
    assert "snakefmt --check" in runs
    assert "snakemake -s workflow/Snakefile --lint" in runs
    assert "--group paper pytest" in runs
    paper = [s for s in job["steps"] if s.get("working-directory") == "paper"]
    assert paper
    assert "pytest" in paper[0]["run"]
    assert "--durations" in runs


def test_no_test_runs_in_both_the_full_and_the_chronos_job():
    steps = _workflow("heavy.yml")["jobs"]["full"]["steps"]
    full = " ".join(s.get("run", "") for s in steps)
    chronos = " ".join(
        s.get("run", "")
        for s in _workflow("heavy.yml")["jobs"]["chronos"]["steps"]
    )
    grep = "(importorskip|find_spec)"
    assert grep in full
    assert grep in chronos
    assert "--ignore=" in full


@pytest.mark.parametrize("name", ["ci.yml", "integration.yml"])
def test_every_merge_into_next_gets_a_fast_tier_run(name):
    # Pull request runs are cancelled when the merge loop deletes the
    # branch, so a push to next runs the fast tier, and a newer merge
    # does not cancel the running one.
    workflow = _workflow(name)
    assert "next" in _triggers(workflow)["push"]["branches"]
    keep = workflow["concurrency"]["cancel-in-progress"]
    assert "refs/heads/next" in keep


def test_the_r_live_job_installs_what_the_live_r_tests_call():
    steps = _workflow("heavy.yml")["jobs"]["r-live"]["steps"]
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


def test_only_main_measures_coverage_and_workers_steal_work():
    # Coverage tracing about doubled the test job's time (next's push
    # runs took up to 14 of 15 min), so only main's runs measure it:
    # its pushes, the schedule and dispatches there. Workers steal the
    # long tail.
    job = _workflow("ci.yml")["jobs"]["package"]
    (tests,) = [s for s in job["steps"] if s.get("name") == "Tests"]
    assert "--dist worksteal" in tests["run"]
    assert "github.ref == 'refs/heads/main'" in tests["run"]
    assert "pull_request" not in tests["run"]
    (upload,) = [s for s in job["steps"] if "codecov" in str(s.get("uses"))]
    assert upload["if"] == "${{ github.ref == 'refs/heads/main' }}"


def test_the_full_tier_has_the_workflow_s_tools():
    # Slow workflow tests (the compute runs) skip without Snakemake.
    steps = _workflow("heavy.yml")["jobs"]["full"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "--group paper pytest -q -m slow" in runs


def test_repo_checks_run_on_every_pull_request():
    # The import contract, the lint of what no package holds and the
    # workflows' own lint, in one short job.
    job = _workflow("ci.yml")["jobs"]["repo"]
    assert "if" not in job
    runs = " ".join(s.get("run", "") for s in job["steps"])
    for check in ("lint-imports", "ruff check tests", "actionlint"):
        assert check in runs, check
    assert not (WORKFLOWS / "lint.yml").exists()


def test_the_heavy_tiers_are_opt_in_in_their_own_workflow():
    # Nightly, dispatch or the label: no push runs them, and Tests is gone.
    assert not (WORKFLOWS / "test.yml").exists()
    workflow = _workflow("heavy.yml")
    assert set(workflow["jobs"]) == {"full", "chronos", "r-live"}
    assert "push" not in _triggers(workflow)


def test_the_submission_ci_has_its_own_ci_job():
    # ci/, its own uv project: lint, types, tests with coverage and the
    # dry run, when ci/ or the workflows it drives change.
    assert not (WORKFLOWS / "submission-ci-tests.yml").exists()
    job = _workflow("ci.yml")["jobs"]["submission"]
    assert job["defaults"]["run"]["working-directory"] == "ci"
    assert "submission" in job["if"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    for check in (
        "ruff check",
        "ruff format --check",
        "mypy",
        "pytest",
        "run_dry.sh",
    ):
        assert check in runs, check
    flags = [s["with"]["flags"] for s in job["steps"] if "codecov" in str(s)]
    assert flags == ["ci"]


def test_repo_checks_fail_on_copied_code_across_the_packages():
    # pylint's symilar over every package's source: any block of 10 or
    # more lines found twice fails the job (Sam: as DRY as possible).
    job = _workflow("ci.yml")["jobs"]["repo"]
    (step,) = [s for s in job["steps"] if s.get("name") == "Duplicate code"]
    assert "symilar -d 10" in step["run"]
    assert "packages" in step["run"]
    assert "duplicates=0 " in step["run"]
