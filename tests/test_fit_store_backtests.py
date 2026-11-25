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
            "releases-randomwalk",
            "releases-randomwalknowcast",
        ]
    )


def test_the_walks_are_stored_before_the_slow_fits() -> None:
    # The walks' origin forecasts are stored fits (SAMPLE_FREE), in their
    # own quick stage, so the submission need not wait for the renewals.
    jobs = _jobs()
    assert jobs["fit-quick"]["needs"] == "plan"
    assert "quick" in jobs["fit-quick"]["strategy"]["matrix"]
    store = "\n".join(
        str(s.get("run", "")) for s in jobs["store-quick"]["steps"]
    )
    assert "push new keys.json quick.json" in store
    assert "store-quick" in jobs["site-results"]["needs"]


def test_no_simulated_backtest_is_run() -> None:
    # The recovery and scenario backtests are retired (Sam, 2026-11-24).
    text = FIT_STORE.read_text()
    for word in ("recovery-only", "scenarios-only", "recovery-", "scenario-"):
        assert word not in text, word


def test_the_backtests_are_stored_without_waiting_for_the_fits() -> None:
    # The fits take hours; Pages needs the backtests now (main, 24 Nov).
    # Only the walks' quick stage, which they load, comes first.
    job = _jobs()["backtests"]
    assert set(job["needs"]) == {"plan", "backtest", "store-quick"}
    run = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert run.index("fit-store load") < run.index("bdbvhub score")
    assert "push-backtests" in run
    # The site's results, made from them, rebuild Pages.
    assert "backtests" in _jobs()["site-results"]["needs"]


def test_the_zone_backtest_scores_the_stored_zone_runs() -> None:
    # The walks' zone runs are stored fits (#1094): no cell forecasts the
    # zones, and the backtests job scores what `fit-store load` chose.
    job = _jobs()["backtests"]
    run = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert run.index("fit-store load") < run.index("bdbvhub zone-backtest")
    assert "parts-zones" not in run
    assert "zone-backtest --forecast-only" not in FIT_STORE.read_text()
