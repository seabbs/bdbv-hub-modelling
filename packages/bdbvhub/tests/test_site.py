"""The docs site's settings (`config/site.toml`): the score baseline."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from bdbvhub.compare import load_comparison_config
from bdbvhub.reporting import report
from bdbvhub.reporting.site import load_site_config, site_baseline, site_models
from nowcastbdbv import zones
from nowcasteval import checks, nowcast_plots, plots, scoresummary

ROOT = Path(__file__).parents[3]


def test_the_site_scores_against_randomwalknowcast() -> None:
    assert load_site_config().baseline == "randomwalknowcast"
    assert site_baseline() == "randomwalknowcast"


def test_a_site_config_sets_the_baseline(tmp_path: Path) -> None:
    path = tmp_path / "site.toml"
    path.write_text('[scores]\nbaseline = "renewal"\n')
    assert load_site_config(path).baseline == "renewal"


def test_reporting_defaults_follow_the_site_baseline() -> None:
    # The site's report defaults to the site's baseline; nowcasteval reads
    # no site config, so its callers name the baseline.
    for fn in (
        report._summaries,
        report._scored_on,
        report._without_baseline,
        report._checks_chart,
    ):
        default = inspect.signature(fn).parameters["baseline"].default
        assert default is None, fn.__qualname__
    for fn in (
        scoresummary.summary_rows,
        nowcast_plots.plot_relative_overall,
        checks.check_summary,
    ):
        default = inspect.signature(fn).parameters["baseline"].default
        assert default is inspect.Parameter.empty, fn.__qualname__
    rel = pd.DataFrame({"model": ["randomwalk", "randomwalknowcast"]})
    assert list(report._without_baseline(rel)["model"]) == ["randomwalk"]


def test_scored_on_names_the_site_baseline() -> None:
    rows = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "randomwalk"],
            "origin": ["a", "a"],
            "truth_vintage": ["b", "b"],
            "pool": ["national", "national"],
            "truth": [5.0, 5.0],
        }
    )
    assert "The randomwalknowcast baseline has 1 national" in (
        report._scored_on(rows)
    )


def test_axis_labels_name_the_baseline() -> None:
    assert plots._metric_label("relative_crps", site_baseline()) == (
        "CRPS relative to randomwalknowcast"
    )
    assert plots._metric_label("crps", None) == "Mean CRPS"
    with pytest.raises(ValueError, match="needs the baseline"):
        plots._metric_label("relative_crps", None)


def test_comparisons_zones_and_history_use_randomwalknowcast() -> None:
    assert load_comparison_config().baseline == "randomwalknowcast"
    assert zones.ZONE_BASELINE == "randomwalknowcast"
    from nowcasteval import score_history as history

    assert history.DEFAULT_BASELINE == "randomwalknowcast"


def test_no_table_filter_defaults_to_randomwalk() -> None:
    source = (
        ROOT / "packages/bdbvhub/src/bdbvhub/reporting/report.py"
    ).read_text()
    assert '{"baseline": "randomwalk"}' not in source


def test_the_site_shows_one_chronos_model() -> None:
    from bdbvhub.reporting.site import site_models

    names = [
        "randomwalk",
        "renewal",
        "foundation",
        "foundation_zeroshot",
        "foundation_zeroshot_complete",
        # A Chronos model added later is hidden too.
        "foundation_de",
    ]
    assert site_models(names) == [
        "randomwalk",
        "renewal",
        "foundation",
    ]


def test_a_site_config_sets_the_shown_models(tmp_path: Path) -> None:
    path = tmp_path / "site.toml"
    path.write_text(
        '[scores]\nbaseline = "randomwalk"\n'
        '[models]\nhide = ["renewal*"]\nshow = ["renewalsp"]\n'
    )
    cfg = load_site_config(path)
    assert cfg.shows("renewalsp")
    assert not cfg.shows("renewal")
    assert cfg.shows("randomwalk")


def test_the_pages_list_only_shown_models() -> None:
    assert "foundation_zeroshot" not in report._page_models()
    assert "randomwalknowcast" in report._page_models()


def test_hidden_models_leave_the_germany_page(tmp_path: Path) -> None:
    from nowcastgermany import reporting as germany

    pd.DataFrame(
        {
            "model": ["foundation_zeroshot", "foundation"],
            "origin": ["2021-12-01", "2021-12-01"],
        }
    ).to_csv(tmp_path / "scores.csv.gz", index=False)
    got = germany._read(tmp_path, ("scores.csv.gz",), site_models)
    assert got is not None
    assert list(got["model"]) == ["foundation"]


def test_hidden_models_leave_the_comparison_rows() -> None:
    rows = pd.DataFrame(
        {
            "model": ["foundation_zeroshot", "renewal", "teamA-weekly"],
            "source": ["real", "real", "hub"],
        }
    )
    assert list(report._site_rows(rows)["model"]) == [
        "renewal",
        "teamA-weekly",
    ]


def test_hidden_models_leave_the_store_note(tmp_path: Path) -> None:
    import json

    from bdbvhub.reporting.store_note import write_store_note

    def row(model: str) -> dict:
        return {
            "model": model,
            "origin": "2026-11-01",
            "key": "k",
            "used": None,
            "commit": None,
            "current": False,
            "failed": False,
        }

    (tmp_path / "fit-store.json").write_text(
        json.dumps([row("foundation_zeroshot"), row("renewal")])
    )
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "`renewal`" in text
    assert "foundation_zeroshot" not in text


def test_hidden_models_leave_the_backtest_explorer() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["foundation_zeroshot_complete", "randomwalknowcast"] * 2,
            "kind": ["real", "real", "simulated", "simulated"],
        }
    )
    got = report._real_forecasts(forecasts, report._page_models())
    assert list(got["model"]) == ["randomwalknowcast"]
