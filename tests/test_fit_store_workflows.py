"""The fit store is kept as Actions artifacts, never on a git branch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
FETCH = "store/artifact.py fetch"
STORE = "${{ env.FIT_STORE }}"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _uploads(job: dict) -> list[dict]:
    return [
        s["with"]
        for s in job["steps"]
        if s.get("uses", "").startswith("actions/upload-artifact")
    ]


def _runs(job: dict) -> str:
    return "\n".join(str(s.get("run", "")) for s in job.get("steps", []))


def _permissions(wf: dict, job: dict) -> dict:
    return job.get("permissions", wf.get("permissions", {}))


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=str)
def test_no_workflow_writes_the_store_to_a_branch(path: Path) -> None:
    text = path.read_text()
    assert "branch.py" not in text
    assert "store_branch" not in text
    assert "FIT_STORE_BRANCH" not in text


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=str)
def test_every_job_that_fetches_the_store_can_read_artifacts(
    path: Path,
) -> None:
    wf = _workflow(path.name)
    for name, job in wf["jobs"].items():
        # archive_candidates.sh fetches the store too (`fit_store fetch`).
        runs = _runs(job)
        if FETCH not in runs and "archive_candidates.sh" not in runs:
            continue
        perms = _permissions(wf, job)
        assert perms.get("actions") in ("read", "write"), (path.name, name)
        text = yaml.safe_dump(job) + yaml.safe_dump(wf.get("env", {}))
        assert "GH_TOKEN: ${{ github.token }}" in text, (path.name, name)


def test_no_store_job_can_push() -> None:
    wf = _workflow("fit-store.yml")
    for name, job in wf["jobs"].items():
        assert _permissions(wf, job).get("contents") != "write", name
    wf = _workflow("site-results.yml")
    for name, job in wf["jobs"].items():
        assert _permissions(wf, job).get("contents") != "write", name


def test_each_store_job_uploads_the_whole_store_in_turn() -> None:
    wf = _workflow("fit-store.yml")
    jobs = wf["jobs"]
    for name in ("store-quick", "store-current", "store-rest"):
        run = _runs(jobs[name])
        # The newest store, then this run's fits merged into it.
        assert run.index(FETCH) < run.index("artifact.py \\\n  merge"), name
        (up,) = _uploads(jobs[name])
        assert up["name"] == STORE, name
        assert up["overwrite"] is True, name
        assert up["retention-days"] == "${{ env.RETENTION }}", name
    # One after another, so none uploads a store without another's fits.
    assert "store-quick" in jobs["store-current"]["needs"]
    assert "fit-rest" in jobs["store-rest"]["needs"]
    assert "store-current" in jobs["fit-rest"]["needs"]


def test_the_bundles_are_their_own_artifacts() -> None:
    (up,) = _uploads(_workflow("fit-store.yml")["jobs"]["backtests"])
    assert up["name"] == f"{STORE}-backtests"
    (up,) = _uploads(_workflow("site-results.yml")["jobs"]["site-results"])
    assert up["name"] == f"{STORE}-site"


@pytest.mark.parametrize("name", ["fit-store.yml", "site-results.yml"])
def test_a_scratch_store_is_kept_a_week_and_the_main_store_90_days(
    name: str,
) -> None:
    env = _workflow(name)["env"]
    assert env["FIT_STORE"] == "${{ inputs.store || 'fit-store' }}"
    assert env["RETENTION"] == "${{ inputs.store && '7' || '90' }}"


def test_site_builds_nothing_until_the_site_results_exist() -> None:
    # Missing or expired site results are pending: the published site
    # stays, and on main Site results is started to make them.
    jobs = _workflow("pages.yml")["jobs"]
    prep = _runs(jobs["prep"])
    assert "exit 1" not in prep
    assert "gh workflow run site-results.yml" in prep
    assert jobs["prep"]["permissions"]["actions"] == "write"
    assert "needs.prep.outputs.ready == 'true'" in jobs["build"]["if"]


def test_the_seed_uploads_the_branch_s_store_once_from_main() -> None:
    wf = _workflow("fit-store-seed.yml")
    (job,) = wf["jobs"].values()
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert "artifact.py latest" in _runs(job)
    names = [u["name"] for u in _uploads(job)]
    assert names == ["fit-store", "fit-store-backtests", "fit-store-site"]


def test_a_round_s_release_keeps_the_fits_it_forecast_from() -> None:
    assets = (
        ROOT / "packages/bdbvsubmission/steps/round_assets.sh"
    ).read_text()
    assert "candidate-* | fit-*)" in assets
    (leg,) = _workflow("fit-candidates.yml")["jobs"].values()
    days = {u["name"]: u["retention-days"] for u in _uploads(leg)}
    # Kept until the round is released, as long as the candidates.
    assert days["fit-${{ matrix.id }}"] == days["candidate-${{ matrix.id }}"]
    submit = _workflow("submit.yml")["jobs"]
    for job in ("baseline", "chronos"):
        (gate,) = _uploads(submit[job])
        assert "/fits" in gate["path"], job
