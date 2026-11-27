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
    # Pages loads; no job of its own forecasts them.
    jobs = _jobs()
    assert "germany-heldout" not in jobs
    assert "FIT_STORE=fit-store-scratch-fm-chronos" not in PAGES.read_text()


def _build_script() -> str:
    steps = _jobs()["build"]["steps"]
    return "\n".join(str(s.get("run", "")) for s in steps)


def test_our_german_forecasts_are_the_paper_s_published_ones() -> None:
    # Our models' German forecasts are the fine-tuning workflow's units,
    # published for the paper as its paper-germany release
    # (`paperpipe.stored`), stacked with the others where the site's
    # results are made (Site results).
    site = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "site-results.yml").read_text()
    )["jobs"]["site-results"]
    run = "\n".join(str(s.get("run", "")) for s in site["steps"])
    fetch = run.index("paperpipe.stored fetch")
    load = run.index("nowcastgermany stored-load")
    assert "--set germany" in run
    assert fetch < load < run.index("concat-results --out results/germany")
    assert "parts-germany-units/germany" in run


def test_the_built_german_page_is_checked_for_held_out_dates() -> None:
    run = _build_script()
    check = run.index("germany-site-check")
    assert run.index("mkdocs build") < check


def test_a_paper_dispatch_is_never_deployed() -> None:
    # Its pages hold the held-out half, which waits for the paper.
    jobs = _jobs()
    for name in ("deploy", "release"):
        assert "inputs.paper_target != 'paper'" in jobs[name]["if"], name


def test_pages_forecasts_and_scores_nothing() -> None:
    # Pages only loads stored results and renders: the
    # BDBV backtests and their scores come from the fit store.
    jobs = _jobs()
    assert "backtest" not in jobs
    run = "\n".join(
        str(s.get("run", "")) for job in jobs.values() for s in job["steps"]
    )
    for command in (
        "bdbvhub backtest",
        "bdbvhub zone-backtest",
        "pairing-sensitivity",
        "bdbvhub score",
    ):
        assert command not in run, command
    assert "tar -xzf output/fit-store/backtests/latest.tar.gz" in run


def test_pages_cancels_in_progress_on_every_ref() -> None:
    import yaml as _yaml

    workflow = _yaml.safe_load(PAGES.read_text())
    assert workflow["concurrency"]["cancel-in-progress"] is True


def test_pages_builds_without_stored_backtests_yet() -> None:
    # A missing bundle is a warning, not a failed build.
    steps = _jobs()["prep"]["steps"]
    (load,) = [s for s in steps if s.get("name") == "Load the stored backtests"]
    assert "::warning::" in load["run"]
    assert "exit 1" not in load["run"]
