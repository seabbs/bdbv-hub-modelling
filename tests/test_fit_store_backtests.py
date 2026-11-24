"""Fit store runs and stores the BDBV backtests that Pages loads."""

from __future__ import annotations

from pathlib import Path

import yaml

FIT_STORE = (
    Path(__file__).parents[1] / ".github" / "workflows" / "fit-store.yml"
)


def _jobs() -> dict:
    return yaml.safe_load(FIT_STORE.read_text())["jobs"]


def test_the_walk_backtests_run_under_the_store_s_write_guard() -> None:
    job = _jobs()["backtest"]
    assert job["if"] == "needs.plan.outputs.write == 'true'"
    cells = [c["cell"] for c in job["strategy"]["matrix"]["include"]]
    assert sorted(cells) == sorted(
        [
            "pairing-bdbv",
            "zones-randomwalk",
            "zones-randomwalknowcast",
            "development-randomwalk",
            "development-randomwalknowcast",
        ]
    )


def test_no_simulated_backtest_is_run() -> None:
    # The recovery and scenario backtests are retired (Sam, 2026-11-24).
    text = FIT_STORE.read_text()
    for word in ("recovery-only", "scenarios-only", "recovery-", "scenario-"):
        assert word not in text, word


def test_the_backtests_are_stored_without_waiting_for_the_fits() -> None:
    # The fits take hours; Pages needs the backtests now (main, 24 Nov).
    job = _jobs()["backtests"]
    assert set(job["needs"]) == {"plan", "backtest"}
    run = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert run.index("fit-store load") < run.index("bdbvhub score")
    assert run.index("push-backtests") < run.index("gh workflow run pages.yml")
    assert job["permissions"]["actions"] == "write"
    assert "backtests" not in _jobs()["pages"]["needs"]
