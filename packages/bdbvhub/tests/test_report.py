from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bdbvhub.reporting import report

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"
# nowcast-scores.csv and zone-scores.csv there are score tables without
# samples. Their `log_crps` column is a stand-in set equal to `log_wis`,
# added when the pages moved to CRPS, not a computed score: read it only
# as test input.
"""Score tables for the page tests. They keep no samples, so their
`log_crps` column is a stand-in copied from `log_wis`, not a CRPS score:
read it only as page input."""


@pytest.mark.slow
def test_comparison_page(
    tmp_path: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    from test_hubcompare import _ours, _write_hub

    from bdbvhub.cli import main

    gen, fig = tmp_path / "gen", tmp_path / "fig"
    report._comparison_page(fig, gen, tmp_path / "missing", target)
    assert "compare-hub" in (gen / "comparison-scores.md").read_text()
    _write_hub(tmp_path / "hub")
    # The walks' forecasts of the round, as stored (compare-hub runs no
    # model of ours).
    stored = tmp_path / "stored.csv.gz"
    walks = ("randomwalk", "randomwalknowcast")
    ours = _ours(target, delays, daily, models=walks).drop(columns="source")
    ours.to_csv(stored, index=False)
    out = tmp_path / "out"
    assert (
        main(
            [
                "compare-hub",
                "--hub-dir",
                str(tmp_path / "hub"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out",
                str(out),
                "--data",
                str(ROOT / "tests" / "fixtures" / "data"),
                "--backtest-config",
                str(
                    ROOT
                    / "packages"
                    / "nowcastbdbv"
                    / "config"
                    / "backtest.toml"
                ),
                "--stored",
                str(stored),
            ]
        )
        == 0
    )
    for name in (
        "scores.csv",
        "quantiles.csv.gz",
        "samples.csv.gz",
        "teams.csv",
        "summary.json",
    ):
        assert (out / name).exists(), name
    report._comparison_page(fig, gen, out, target)
    teams = (gen / "comparison-teams.md").read_text()
    scores = (gen / "comparison-scores.md").read_text()
    # Other teams are named as the hub names them.
    assert "teamA-weekly" in teams
    assert "CC0-1.0" in teams
    # The page names the comparison's configured draws per forecast.
    from nowcasteval.backtest import load_config

    n = load_config(
        ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
    ).n_samples
    lead = (gen / "comparison-lead.md").read_text()
    assert f"{n:,} draws" in lead
    assert "as submitted" in lead
    assert "fair" in lead
    assert "teamA-weekly" in scores
    figures = (gen / "comparison-figures.md").read_text()
    assert "2026-10-24" in figures
    assert (fig / "comparison-2026-10-24.svg").exists()
    # The Backtest summaries lead, with the static figure as the fallback,
    # and randomwalk, which we do not submit, starts off.
    # Complete weeks first. The latest release is not offered: these
    # rounds have no score of the baseline against a later release.
    assert "scoresummary-comparison-complete.json" in scores
    assert "scoresummary-comparison-latest.json" not in scores
    assert "comparison-overall.svg" in scores
    assert 'data-view="place"' in (gen / "comparison-place.md").read_text()
    horizon = (gen / "comparison-horizon.md").read_text()
    assert 'data-view="horizon"' in horizon
    assert (fig / "comparison-place.svg").exists()
    data = json.loads(
        (fig / "scoresummary-comparison-complete.json").read_text()
    )
    on = {m["name"]: m["on"] for m in data["models"]}
    assert on["randomwalk (main build)"] is False
    assert on["teamA-weekly"] is True
    assert (
        "BDBV Modeling Hub" in (gen / "comparison-attribution.md").read_text()
    )


def test_lopo_section_hidden_without_results(tmp_path: Path) -> None:
    assert report._lopo_section(tmp_path / "missing.csv") == ""
    scores = _two_model_scores()
    lopo = scores.loc[scores["location"] != "CD"].assign(
        model="renewal", masked=lambda d: d["location"]
    )
    path = tmp_path / "lopo-scores.csv"
    lopo.to_csv(path, index=False)
    text = report._lopo_section(path)
    assert text.startswith("## Leave one province out")
    assert "renewal" in text
    assert "CD-IT" in text


def _scores() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "model": "randomwalk",
            "kind": "real",
            "vintage": "2026-10-09",
            "reference_date": "2026-10-17",
            "location": "CD",
            "target_end_date": ["2026-09-26", "2026-10-17"],
            "horizon": [-3, 0],
            "truth_vintage": "2026-10-24",
            "crps": [10.0, 50.0],
        }
    )
    for col in (
        "dispersion",
        "overprediction",
        "underprediction",
        "wis",
        "log_wis",
        "log_crps",
        "ae_median",
        "coverage_50",
        "coverage_90",
        "bias",
    ):
        base[col] = 0.0
    return base


def _two_model_scores() -> pd.DataFrame:
    rows = []
    for model, crps in (("randomwalk", 10.0), ("renewal", 5.0)):
        for loc in ("CD", "CD-IT"):
            rows.append(_scores().assign(model=model, location=loc, crps=crps))
    return pd.concat(rows, ignore_index=True)


def test_teams_table_links_to_hub_metadata() -> None:
    teams = pd.DataFrame(
        {
            "model": ["teamA-weekly", "epiforecasts-x"],
            "team_name": ["Team A", "Us"],
            "model_name": ["Weekly", "X"],
            "license": ["CC0-1.0", "MIT"],
            "website_url": ["", ""],
        }
    )
    text = report._teams_table(teams, {"x": "epiforecasts-x"})
    assert "| Team " in text
    assert "`teamA-weekly`" in text
    assert "model-metadata/teamA-weekly.yml" in text
    assert "epiforecasts-x" not in text


def test_sampler_section() -> None:
    fits = pd.DataFrame(
        {
            "model": ["renewal", "renewal", "renewalsp"],
            "kind": "real",
            "vintage": ["2026-08-18", "2026-10-09", "2026-10-09"],
            "fit": ["all", "all", "CD-IT"],
            "num_chains": 2,
            "num_warmup": 300,
            "num_samples": 300,
            "seconds": [100.0, 200.0, 50.0],
            "max_rhat": [1.01, 1.2, 1.02],
            "depth_share": [0.9, 1.0, 0.2],
            "divergent_share": [0.0, 0.01, 0.0],
        }
    )
    base = {"num_chains": 2, "num_warmup": 300, "num_samples": 300}
    settings = {
        "renewal": {**base, "target_accept": 0.9},
        "renewalbvd": {**base, "num_samples": 1000, "target_accept": 0.95},
    }
    text = report._sampler_section(fits, settings)
    assert "Stored fits use these settings" in text
    assert (
        "- `renewalbvd`: 2 chains of 300 warm-up and 1000 draws at a "
        "target acceptance of 0.95" in text
    )
    assert "same as" not in text
    thinned = {**base, "num_samples": 1500, "thinning": 2, "target_accept": 0.9}
    assert (
        "2 chains of 300 warm-up and 1500 draws, keeping 1 in 2, at a "
        "target acceptance of 0.9" in report._settings_text(thinned)
    )
    assert "renewalsp" in text
    assert report._sampler_section(fits.iloc[:0], settings) == ""


def test_forecast_tabs() -> None:
    text = report._forecast_tabs(["randomwalk", "renewal"], {"CD": "DRC"})
    assert text.startswith("## Forecasts by place")
    assert '=== "renewal"' in text
    assert "figures/forecasts-renewal-cd.svg#only-light" in text
    assert report._forecast_tabs([], {"CD": "DRC"}) == ""


def test_dropdown_wraps_and_indents() -> None:
    text = report._dropdown("a\n\nb\n")
    assert text == '??? note "Tables"\n\n    a\n\n    b\n'
    assert report._dropdown("  \n") == ""
    assert report._dropdown("x", "More").startswith('??? note "More"')


def test_nowcast_groups() -> None:
    target = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-10-24",
            "location": ["CD", "CD-A", "CD-B", "CD-C", "CD-D"],
            "total": [10, 1, 5, 3, 2],
        }
    )
    groups = report._nowcast_groups(
        target, ["CD", "CD-A", "CD-B", "CD-C", "CD-D", "CD-D-01"]
    )
    assert groups["main"] == ["CD", "CD-B", "CD-C", "CD-D"]
    assert groups["other"] == ["CD-A"]


def test_release_scores_horizon_is_weeks_from_the_reference_date(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "origin": "2026-10-17",
            "target_end_date": ["2026-10-10", "2026-10-17", "2026-10-24"],
        }
    ).to_csv(tmp_path / "nowcast-scores.csv", index=False)
    scores = report._release_scores(tmp_path)
    assert scores["horizon"].tolist() == [-2, -1, 0]
    assert set(scores["vintage"]) == {"2026-10-17"}


def test_summary_covers_only_forecasts_randomwalk_made(tmp_path: Path) -> None:
    scores = report._release_scores(FIXTURES)
    reported = scores.loc[scores["target"] == "reported"]
    gen = tmp_path / "gen"
    gen.mkdir()
    report._horizon_table(gen, reported, ["randomwalk", "randomwalknowcast"])
    text = (gen / "backtest-summary.md").read_text()
    walk = reported.loc[reported["model"] == "randomwalk", "origin"]
    origins = ", ".join(sorted(set(walk.astype(str))))
    assert f"the origins {origins}" in text
    assert "   n |" in text
    assert "nan" not in text.lower()


def test_missing_builds_name_submitted_rounds_without_a_main_build() -> None:
    quantiles = pd.DataFrame(
        {
            "model": ["seabbs-r", "renewalbvd", "seabbs-r"],
            "source": ["submitted", "real", "submitted"],
            "reference_date": ["2026-10-31", "2026-10-31", "2026-11-07"],
        }
    )
    lines = report._missing_builds(quantiles, {"renewalbvd": "seabbs-r"})
    assert len(lines) == 1
    assert "renewalbvd (main build)" in lines[0]
    assert "2026-11-07" in lines[0]


def test_scored_on_counts_pairs_and_forecasts() -> None:
    rows = pd.DataFrame(
        {
            "model": ["randomwalk"] * 3 + ["renewal"],
            "origin": ["a", "a", "b", "a"],
            "truth_vintage": ["b", "c", "c", "b"],
            "pool": ["national", "provinces", "provinces", "national"],
            "truth": [5.0, 0.0, 3.0, 5.0],
        }
    )
    assert report._scored_on(rows, "randomwalk") == (
        "On later releases, the scores cover 3 origin and release pairs.\n"
        "The randomwalk baseline has 1 national and 2 province forecasts.\n"
        "1 of them are of a place and week with no case, where every model's "
        "CRPS is close to 0, so mean CRPS is lower than over places with "
        "cases.\n"
        "The renewal models are not scored at health zones.\n"
    )


def test_checks_chart_has_a_set_per_baseline() -> None:
    def summary(rel: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scale": "zones",
                "split": "flag",
                "group": "none",
                "model": list(rel),
                "n": 4,
                "relative CRPS": list(rel.values()),
                "CRPS": [2.0, 4.0],
                "90% coverage": [0.9, 0.8],
            }
        )

    names = ["randomwalk", "renewal"]
    got = report._checks_chart(
        {
            "randomwalk": summary({"randomwalk": 1.0, "renewal": 2.0}),
            "renewal": summary({"randomwalk": 0.5, "renewal": 1.0}),
        },
        names,
        baseline="randomwalk",
    )
    assert got["baseline"] == 0
    assert [m["name"] for m in got["models"]] == names
    assert got["sets"]["randomwalk|zones|flag"]["rows"][1][3] == 2.0
    assert got["sets"]["renewal|zones|flag"]["rows"][0][3] == 0.5
    json.dumps(got)


def test_by_baseline_stacks_a_table_per_baseline() -> None:
    def make(base: str) -> pd.DataFrame:
        rel = [1.0, 2.0] if base == "a" else [0.5, 1.0]
        return pd.DataFrame({"model": ["a", "b"], "rel_crps": rel})

    got = report._by_baseline(make, ["b", "a"])
    assert list(got.columns) == ["baseline", "model", "rel_crps"]
    assert list(got["baseline"]) == ["b", "b", "a", "a"]
    assert list(got["rel_crps"]) == [0.5, 1.0, 1.0, 2.0]


def test_fit_settings_skip_models_with_no_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    settings = report._fit_settings()
    assert "renewal" in settings
    assert not any(name.startswith("foundation") for name in settings)


def test_complete_target_text_names_models_with_no_complete_score() -> None:
    scores = pd.DataFrame(
        {
            "model": ["randomwalk", "renewal", "randomwalk"],
            "target": ["reported", "reported", "complete"],
        }
    )
    zones = pd.DataFrame({"model": ["randomwalk"], "target": ["complete"]})
    text = report._complete_on(scores, zones, 56)
    assert "at least 56 days after it ends" in text
    assert "Forecasts with such a release so far: 1." in text
    assert "So `renewal` have no complete-week scores yet" in text
    full = report._complete_on(scores.iloc[[0, 2]], zones, 56)
    assert "no complete-week scores" not in full


def test_backtest_and_nowcast_pages_name_the_backtest_draws() -> None:
    from nowcasteval.backtest import load_config

    cfg = load_config(
        ROOT / "packages" / "nowcastbdbv" / "config" / "backtest.toml"
    )
    n, stored = cfg.score_samples, cfg.n_samples
    note = report.draws_note(n, stored)
    assert f"{n:,} draws" in note
    assert f"{stored:,}" in note
    assert "[samples] backtest" in note
    assert "fair" in note
    # The Nowcast page shows the submission's forecasts, at n_samples.
    meta = report._nowcast_meta("2026-11-08", ["renewal"], stored)
    assert f"{stored:,} draws" in meta
    assert "2026-11-08" in meta
    lead = report._release_intro(n, stored)
    assert f"{n:,} draws" in lead


def test_sections_not_stored_yet_say_so(tmp_path: Path) -> None:
    # Before the fit store holds the backtests, every snippet the docs
    # include but nothing wrote says it is not stored yet, so the site
    # still builds; one already written is kept.
    docs = tmp_path / "docs"
    gen = docs / "generated"
    gen.mkdir(parents=True)
    (docs / "backtest.md").write_text(
        '--8<-- "release-headline.md"\n--8<-- "vintages.md"\n'
    )
    (docs / "examples").mkdir()
    (docs / "examples" / "example.md").write_text("x\n")
    (docs / "other.md").write_text('--8<-- "example.md"\n')
    (gen / "vintages.md").write_text("table\n")
    filled = report._not_stored(gen)
    assert filled == ["release-headline.md"]
    assert (gen / "release-headline.md").read_text() == report.NOT_STORED
    assert (gen / "vintages.md").read_text() == "table\n"
    assert not (gen / "example.md").exists()


def test_no_docs_page_embeds_a_backtest_figure_directly() -> None:
    # A figure only the stored backtests make goes through a snippet, so
    # the site builds (mkdocs --strict) before they are stored.
    root = Path(__file__).parents[3]
    text = (root / "docs" / "backtest.md").read_text()
    assert "](figures/scores-by-truth" not in text
    assert '--8<-- "scores-by-truth-figure.md"' in text
