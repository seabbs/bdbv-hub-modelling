"""The Pages workflow's German held-out half runs every benchmark model."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / ".github" / "workflows" / "pages.yml"


def _jobs() -> dict:
    return yaml.safe_load(PAGES.read_text())["jobs"]


def test_pages_forecasts_nothing_for_the_german_benchmark() -> None:
    # The German forecasts (development and held-out) are stored outputs
    # Pages loads; no job of its own forecasts them (Sam, 2026-11-23).
    jobs = _jobs()
    assert "germany-heldout" not in jobs
    cells = jobs["backtest"]["strategy"]["matrix"]["include"]
    german = [c["cell"] for c in cells if "germany" in c["cell"]]
    assert german == []
    assert (
        "FIT_STORE_BRANCH=fit-store-scratch-fm-chronos" not in PAGES.read_text()
    )


def _build_script() -> str:
    steps = _jobs()["build"]["steps"]
    return "\n".join(str(s.get("run", "")) for s in steps)


def test_the_stored_ft_model_is_not_forecast_by_pages() -> None:
    # Its forecasts come from the checkpoint scorer's store.
    cells = _jobs()["backtest"]["strategy"]["matrix"]["include"]
    live = {c["model"] for c in cells if c.get("germany")}
    assert not live & {
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
    }
    run = _build_script()
    load = run.index("nowcastgermany ft-load")
    assert "germany-ft-forecasts" in run
    assert load < run.index("concat-results --out results/germany")
    assert "parts-germany-ft-store/germany" in run


def test_the_built_german_page_is_checked_for_held_out_dates() -> None:
    run = _build_script()
    check = run.index("germany-site-check")
    assert run.index("mkdocs build") < check


def test_a_paper_dispatch_is_never_deployed() -> None:
    # Its pages hold the held-out half, which waits for the paper.
    jobs = _jobs()
    for name in ("deploy", "release"):
        assert "inputs.paper_target != 'paper'" in jobs[name]["if"], name


def test_each_backtest_is_scored_in_its_own_process() -> None:
    # One `score` of every backtest's parts ran the runner out of memory.
    steps = _jobs()["build"]["steps"]
    (score,) = [s for s in steps if s.get("name") == "Score the backtests"]
    calls = score["run"].split("uv run bdbvhubrenewal score")[1:]
    kinds = ["development", "recovery", "zones", "scenario"]
    assert [[k for k in kinds if f"-{k}-" in c] for c in calls] == [
        [k] for k in kinds
    ]
