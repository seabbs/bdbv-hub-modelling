"""The site's results are made by Site results, and Site only assembles them."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
# The steps that score or forecast, which Site no longer runs.
COMPUTE = (
    "compare-hub",
    "nowcastgermany teams",
    "nowcastgermany asof",
    "nowcastgermany paper",
    "bdbvhub report",
    "nowcastgermany stored-load",
    "concat-results",
    "fit-store load",
)


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _jobs(name: str) -> dict:
    return _workflow(name)["jobs"]


def _runs(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job["steps"])


def test_site_runs_no_scoring_or_forecasting() -> None:
    for name, job in _jobs("pages.yml").items():
        for command in COMPUTE:
            assert command not in _runs(job), (name, command)


def test_site_results_makes_and_stores_the_site_results() -> None:
    job = _jobs("site-results.yml")["site-results"]
    runs = _runs(job)
    for command in COMPUTE:
        assert command in runs, command
    stores = [s for s in job["steps"] if "upload-artifact" in s.get("uses", "")]
    assert [s["with"]["name"] for s in stores] == ["${{ env.FIT_STORE }}-site"]
    assert "gh workflow run pages.yml" in runs
    # It waits on no fit, and a dispatch makes the results at once.
    assert "needs" not in job
    assert "workflow_dispatch" in _workflow("site-results.yml")[True]


def test_each_store_remakes_the_site_results_without_the_slow_fits() -> None:
    jobs = _jobs("fit-store.yml")
    assert "site-results" not in jobs
    for name in ("backtests", "store-current", "store-rest"):
        assert "gh workflow run site-results.yml" in _runs(jobs[name]), name
        assert jobs[name]["permissions"]["actions"] == "write", name
    # The backtests are scored before the newest origin's fits.
    assert "store-current" not in jobs["backtests"]["needs"]


def test_site_unpacks_the_stored_site_results() -> None:
    prep = _runs(_jobs("pages.yml")["prep"])
    assert "fetch output/fit-store backtests site" in prep
    assert "tar -xzf output/fit-store/site/latest.tar.gz" in prep
