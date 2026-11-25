"""The site's results are made by Fit store, and Site only assembles them."""

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
    "nowcastgermany ft-load",
    "concat-results",
)


def _jobs(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())["jobs"]


def _runs(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job["steps"])


def test_site_runs_no_scoring_or_forecasting() -> None:
    for name, job in _jobs("pages.yml").items():
        for command in COMPUTE:
            assert command not in _runs(job), (name, command)


def test_fit_store_makes_and_stores_the_site_results() -> None:
    job = _jobs("fit-store.yml")["site-results"]
    runs = _runs(job)
    for command in COMPUTE:
        assert command in runs, command
    assert "push-site" in runs
    assert "gh workflow run pages.yml" in runs
    # After the fits and backtests are stored, so it reads them.
    assert {"store-quick", "store-current", "store-rest", "backtests"} <= set(
        job["needs"]
    )


def test_site_unpacks_the_stored_site_results() -> None:
    prep = _runs(_jobs("pages.yml")["prep"])
    assert "output/fit-store/site/latest.tar.gz" in prep
