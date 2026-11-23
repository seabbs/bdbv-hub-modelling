import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from paperpipe.evidence import (
    EVIDENCE,
    aggregation_numbers,
    realism_reference,
    write_evidence,
)


def test_aggregation_is_relative_to_the_reference_on_shared_targets() -> None:
    base = {
        "study": "ebola",
        "level": "national",
        "location": "CD",
        "origin_date": "2026-10-04",
        "cover50": 1,
        "cover95": 1,
    }
    rows = [
        {
            **base,
            "model": "randomwalknowcast",
            "option": "reference",
            "target_date": t,
            "wis": 10.0,
            "log_wis": 1.0,
        }
        for t in ("2026-09-27", "2026-10-04")
    ] + [
        {
            **base,
            "model": "fm",
            "option": "joint",
            "target_date": "2026-09-27",
            "wis": 5.0,
            "log_wis": 0.5,
        },
        {
            **base,
            "model": "fm",
            "option": "level",
            "target_date": "2026-09-27",
            "wis": 20.0,
            "log_wis": 2.0,
        },
    ]
    numbers = aggregation_numbers(pd.DataFrame(rows))
    assert numbers["ebola"]["national"]["fm"] == {"joint": 0.5, "level": 2.0}


def _germany_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"kind": k, "statistic": s, "value": v}
            for k, base in (("national", 1.0), ("states", 2.0), ("ages", 3.0))
            for s, v in (("zero_share", base / 10), ("delay_q99", base * 10))
        ]
        + [{"kind": "national", "statistic": "only_germany", "value": 1.0}]
    )


def test_realism_compares_the_simulations_with_germany_only() -> None:
    sims = pd.DataFrame(
        {
            "outbreak": [f"o{i}" for i in range(101)],
            "zero_share": np.linspace(0, 1, 101),
            "delay_q99": np.linspace(0, 100, 101),
        }
    )
    ref = realism_reference(_germany_stats(), sims)
    assert list(ref.columns) == [
        "Germany national",
        "Germany states",
        "Germany ages",
        "sim_q5",
        "sim_q50",
        "sim_q95",
    ]
    assert sorted(ref.index) == ["delay_q99", "zero_share"]
    assert ref.loc["zero_share", "Germany states"] == 0.2
    assert ref.loc["delay_q99", "sim_q95"] == pytest.approx(95.0)


def test_the_realism_step_writes_the_rounds_and_the_reference(tmp_path) -> None:
    from paperpipe.__main__ import main

    germany = tmp_path / "germany.csv"
    _germany_stats().to_csv(germany, index=False)
    rounds = []
    for r in (34, 35):
        path = tmp_path / f"r{r}.csv"
        pd.DataFrame({"outbreak": ["a", "b"], "zero_share": [0.1, 0.3]}).to_csv(
            path, index=False
        )
        rounds.append(str(path))
    sims, ref = tmp_path / "sims.csv", tmp_path / "ref.csv"
    main(
        [
            "realism",
            "--germany",
            str(germany),
            "--out-sims",
            str(sims),
            "--out-reference",
            str(ref),
            *rounds,
        ]
    )
    assert len(pd.read_csv(sims)) == 4
    got = pd.read_csv(ref, index_col=0)
    assert list(got.index) == ["zero_share"]
    assert "BDBV" not in got.columns


def test_missing_evidence_leaves_a_note_and_no_numbers(tmp_path) -> None:
    out = write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    assert "waits on" in (tmp_path / "si" / "sim-diseases.md").read_text()
    assert (tmp_path / "fig" / "sim-realism.png").is_file()
    doc = yaml.safe_load((tmp_path / "e.yml").read_text())
    zero = doc["evidence"]["realism"]["zero_share"]
    assert zero == {
        "sim_q95": "(not yet available)",
        "Germany_states": "(not yet available)",
    }
    assert len(out["evidence"]["waiting"]) == len(EVIDENCE)


def test_a_paper_run_fails_while_evidence_is_missing(tmp_path) -> None:
    from paperpipe.__main__ import main

    with pytest.raises(ValueError, match="waits on"):
        main(
            [
                "evidence",
                "--mode",
                "paper",
                "--si-dir",
                str(tmp_path),
                "--fig-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "e.yml"),
            ]
        )


def test_a_study_without_rows_still_waits(tmp_path) -> None:
    rows = tmp_path / "agg.csv"
    pd.DataFrame(
        {
            "study": "ebola",
            "model": ["randomwalknowcast", "fm"],
            "option": ["reference", "joint"],
            "level": "national",
            "location": "CD",
            "origin_date": "2026-10-04",
            "target_date": "2026-09-27",
            "wis": 1.0,
            "log_wis": 1.0,
            "cover50": 1,
            "cover95": 1,
        }
    ).to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"aggregation_scores": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    assert (tmp_path / "fig" / "aggregation.png").is_file()
    assert not (tmp_path / "si" / "aggregation-cost.md").exists()


def test_checkpoint_panels_show_n_and_grey_thin_ones() -> None:
    from paperpipe.evidence import checkpoint_figure, checkpoint_numbers

    rows = pd.DataFrame(
        {
            "study": ["ebola"] * 3 + ["germany"] * 3,
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "outbreaks": [0, 1000, 2000] * 2,
            "n": [1] * 3 + [400] * 3,
            "rel_wis": [2.0, 1.5, 1.2] * 2,
        }
    )
    fig = checkpoint_figure(rows)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == [
        "ebola, national (n = 1)",
        "germany, national (n = 400)",
    ]
    assert "too few cells" in axes[0].texts[0].get_text()
    assert not axes[1].texts
    numbers = checkpoint_numbers(rows)
    assert numbers["germany"]["national"]["n"] == 400
    assert numbers["ebola"]["national"]["fm_chronos_daily_ragged"] == {
        "zero-shot": 2.0,
        "n1000": 1.5,
        "n2000": 1.2,
    }


def test_the_paper_has_no_scale_floor_test() -> None:
    # Inference applies no scale floor and the code has no switch for
    # one, so there is no test with and without it to quote.

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "workflow" / "config.yaml").read_text()
    )
    assert "scale_floor" not in EVIDENCE
    assert "scale_floor" not in config["evidence"]
    si = (Path(__file__).parents[1] / "si.qmd").read_text()
    assert "evidence.scale_floor" not in si


def test_pair_is_blank_when_either_value_is_missing() -> None:
    from paperpipe.evidence import _pair

    a = pd.Series([0.5, np.nan, 0.5])
    b = pd.Series([0.25, 0.25, np.nan])
    assert _pair(a, b) == ["0.50 / 0.25", "", ""]


def test_checkpoint_models_keep_one_colour_across_panels() -> None:
    from paperpipe.evidence import checkpoint_figure

    rows = pd.DataFrame(
        {
            "study": ["ebola"] * 4 + ["germany"] * 2,
            "level": "national",
            "model": ["a", "a", "b", "b", "b", "b"],
            "outbreaks": [0, 1000] * 3,
            "n": 400,
            "rel_wis": [2.0, 1.5, 1.8, 1.2, 1.1, 0.9],
        }
    )
    fig = checkpoint_figure(rows)
    colours = {
        (ax.get_title(), line.get_label()): line.get_color()
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_label() in {"a, fine-tuned", "b, fine-tuned"}
    }
    assert (
        colours[("ebola, national (n = 400)", "b, fine-tuned")]
        == colours[("germany, national (n = 400)", "b, fine-tuned")]
    )


def test_checkpoint_rows_keep_one_grid_and_the_current_rule() -> None:
    from paperpipe.evidence import CHECKPOINT_RULE, checkpoint_rows

    base = {"level": "national", "model": "fm_chronos_daily_ragged", "n": 9}
    rows = pd.DataFrame(
        [
            {
                **base,
                "study": "germany",
                "checkpoint": "zero-shot",
                "outbreaks": 0,
                "grid": None,
                "rule": CHECKPOINT_RULE,
                "rel_wis": 2.0,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "a",
                "run": "other",
                "outbreaks": 1000,
                "grid": "r3",
                "rule": CHECKPOINT_RULE,
                "rel_wis": 1.5,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "b",
                "run": "one-model",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": 1.2,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "b",
                "run": "one-model",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": "samples-100",
                "rel_wis": 9.0,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "level": "states",
                "checkpoint": "b",
                "run": "one-model",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": None,
                "rel_crps": 0.8,
            },
            {
                **base,
                "study": "held_out",
                "checkpoint": "b",
                "run": "one-model",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": None,
                "rel_crps": None,
            },
        ]
    )
    out = checkpoint_rows(rows, "one-model")
    assert set(out["study"]) == {"germany"}
    assert sorted(out["checkpoint"]) == ["b", "b", "zero-shot"]
    # A row with no relative WIS stays empty, never filled from CRPS.
    assert pd.isna(out.set_index("level").loc["states", "rel_wis"])
    # Old tables, with no run, grid or rule, give the zero-shot rows only.
    old = rows.drop(columns=["run", "grid", "rule", "rel_crps"]).iloc[:2]
    assert list(checkpoint_rows(old, "one-model")["checkpoint"]) == [
        "zero-shot"
    ]


def test_checkpoint_rows_of_other_runs_leave_only_zero_shot() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = pd.DataFrame(
        {
            "study": ["germany", "germany"],
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "checkpoint": ["zero-shot", "6b7aaa8d4f55"],
            "outbreaks": [0, 1000],
            "n": 9,
            "rel_wis": [2.0, 1.5],
        }
    )
    out = checkpoint_rows(rows, "one-model")
    assert list(out["checkpoint"]) == ["zero-shot"]


def test_checkpoint_rows_keep_only_the_live_run() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = pd.DataFrame(
        {
            "study": "germany",
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "checkpoint": ["a", "b", "c"],
            "outbreaks": [1000, 2000, 4000],
            "n": 9,
            "rel_wis": 1.0,
            "grid": ["r4", "r4.1", "r3"],
            "run": ["other", "one-model", "slots-zeroinit"],
        }
    )
    # Only the live run's checkpoints, whatever their grid.
    assert list(checkpoint_rows(rows, "one-model")["checkpoint"]) == ["b"]


def test_joint_evidence_leaves_out_fine_tuned_rows_of_earlier_runs() -> None:
    from paperpipe.evidence import drop_finetuned

    rows = pd.DataFrame(
        {
            "model": [
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged_ft",
                "fm_chronos_daily_ragged_ft_n1000",
                "randomwalknowcast",
            ]
        }
    )
    assert list(drop_finetuned(rows)["model"]) == [
        "fm_chronos_daily_ragged",
        "randomwalknowcast",
    ]


def test_aggregation_options_name_the_lowest_level_not_zones() -> None:
    # Germany's lowest level is states or age groups, not zones.
    from paperpipe.evidence import OPTIONS

    assert not any("zone" in label.lower() for label in OPTIONS.values())


def test_checkpoint_curves_show_the_untuned_models_as_lines() -> None:
    # The fine-tuned checkpoints are the curve; zero-shot Chronos-2 on the
    # ragged layout and on complete data only are horizontal lines, and
    # randomwalknowcast is the line at 1.
    from paperpipe.evidence import checkpoint_figure

    rows = pd.DataFrame(
        {
            "study": "germany",
            "level": "national",
            "model": [
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_complete",
            ],
            "checkpoint": ["zero-shot", "a", "b", "zero-shot"],
            "outbreaks": [0, 1000, 2000, 0],
            "n": 400,
            "rel_wis": [2.0, 1.5, 1.2, 1.7],
        }
    )
    fig = checkpoint_figure(rows)
    ax = next(a for a in fig.axes if a.get_visible())
    lines = {line.get_label(): line for line in ax.get_lines()}
    curve = lines["Fine-tuned (delays)"]
    assert list(curve.get_xdata()) == [1000, 2000]
    zero = lines["Zero-shot (delays)"]
    complete = lines["Zero-shot (complete data only)"]
    assert set(zero.get_ydata()) == {2.0}
    assert set(complete.get_ydata()) == {1.7}
    # One colour per source, the references in other line styles.
    assert zero.get_color() == curve.get_color() == complete.get_color()
    assert len({zero.get_linestyle(), complete.get_linestyle()}) == 2
    assert set(lines["Random walk nowcast"].get_ydata()) == {1}
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Zero-shot (complete data only)" in legend
    assert "Random walk nowcast" in legend


def _comparison() -> pd.DataFrame:
    """Rows of a fine-tuning comparison: two runs scored by set."""
    return pd.DataFrame(
        {
            "set": ["held-out weekly"] * 4 + ["germany national"] * 2,
            "run": [
                "full",
                "full",
                "no delays",
                "no delays",
                "full",
                "no delays",
            ],
            "outbreaks": [1000, 2000, 1000, 2000, 1000, 1000],
            "score": [0.5, 0.4, 0.7, 0.6, 0.9, 1.1],
        }
    )


def test_a_comparison_figure_draws_each_run_by_checkpoint() -> None:
    from paperpipe.evidence import comparison_figure

    fig = comparison_figure(_comparison())
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == [
        "held-out weekly",
        "germany national",
    ]
    lines = {line.get_label(): line for line in axes[0].get_lines()}
    assert list(lines["no delays"].get_xdata()) == [1000, 2000]


def test_the_block_combinations_table_waits_then_shows_its_rows(
    tmp_path,
) -> None:
    write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    table = tmp_path / "si" / "block-combinations.md"
    assert "waits on" in table.read_text()
    rows = tmp_path / "combos.csv"
    pd.DataFrame(
        {"layout": ["daily"], "batched": [True], "share": [0.02]}
    ).to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"block_combinations": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    assert "daily" in table.read_text()


def test_the_daily_nowcasts_have_a_figure(tmp_path) -> None:
    rows = tmp_path / "daily.csv"
    _comparison().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"compare_daily": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    assert (tmp_path / "fig" / "compare-daily.png").is_file()


def test_the_aggregation_figure_has_a_panel_per_study_and_level() -> None:
    from paperpipe.evidence import aggregation_figure

    rows = []
    for study, levels in [
        ("ebola", ["national", "provinces", "zones"]),
        ("germany", ["national", "states", "age groups"]),
    ]:
        for level in levels:
            for model, option in [
                ("randomwalknowcast", "joint"),
                ("fm_chronos_daily_ragged", "joint"),
                ("fm_chronos_daily_ragged", "weekly_row"),
                ("fm_chronos_daily_ragged", "days_copula"),
                ("fm_chronos_daily_ragged", "days_summed"),
            ]:
                rows.append(
                    {
                        "study": study,
                        "model": model,
                        "option": option,
                        "level": level,
                        "location": "X",
                        "origin_date": "2026-10-01",
                        "target_date": "2026-09-26",
                        "wis": 1.0 if model == "randomwalknowcast" else 0.8,
                        "log_wis": 1.0,
                        "cover50": 1,
                        "cover95": 1,
                    }
                )
    fig = aggregation_figure(pd.DataFrame(rows))
    drawn = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(drawn) == 6
    labels = {t.get_text() for t in drawn[0].get_xticklabels()}
    assert {"(a)", "(e) weekly row", "(e) days, copula"} <= labels


def _si_copula() -> pd.DataFrame:
    """Rows of `evaluation.pairing.summarise_pairing`: off and rolling.

    Zones, sums over weeks and per-cell rows of the zones are in the
    rows, as the sensitivity gives them, but not in the figure.
    """
    rows = []
    for study, levels in [
        ("germany", ["national", "states", "ages"]),
        ("synthetic", ["national", "provinces", "zones"]),
    ]:
        for level in levels:
            for variant, value in (("off", 2.0), ("rolling", 1.0)):
                for target, metric in (
                    ("cell", "crps"),
                    ("cell", "cover90"),
                    ("joint", "log_energy"),
                    ("sum_weeks", "crps"),
                ):
                    rows.append(
                        {
                            "study": study,
                            "level": level,
                            "model": "randomwalknowcast",
                            "variant": variant,
                            "target": target,
                            "metric": metric,
                            "value": value,
                            "n": 10,
                        }
                    )
    return pd.DataFrame(rows)


def test_the_copula_figure_has_national_and_provinces_per_study() -> None:
    from paperpipe.evidence import copula_figure

    fig = copula_figure(_si_copula())
    drawn = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    titles = {ax.get_title().split(" (")[0] for ax in drawn}
    assert titles == {
        "Germany, national",
        "Germany, states",
        "Synthetic, national",
        "Synthetic, provinces",
    }
    labels = {t.get_text() for t in drawn[0].get_xticklabels()}
    assert labels == {"CRPS", "90% coverage", "ES (log)"}


def test_the_copula_figure_waits_then_draws(tmp_path) -> None:
    write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    assert (tmp_path / "fig" / "copula-impact.png").is_file()
    rows = tmp_path / "si-copula.csv"
    _si_copula().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"si_copula": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    assert (tmp_path / "fig" / "copula-impact.png").stat().st_size > 20000


def _copula_rho() -> pd.DataFrame:
    """Rows of the walk models' `copula.json`, one per model and origin."""
    return pd.DataFrame(
        {
            "model": ["randomwalknowcast"] * 3 + ["randomwalk"] * 2,
            "origin": ["2026-10-04", "2026-10-11", "2026-10-18"] * 1
            + ["2026-10-04", "2026-10-11"],
            "n_scores": [40, 44, 0, 40, 44],
            "rho_provinces": [0.5, 0.7, 0.7, 0.3, 0.4],
            "n_pairs": [120, 130, 0, 120, 130],
            "fallback": [False, False, True, False, False],
        }
    )


def test_rho_medians_give_rho_p_only_without_fallbacks() -> None:
    from paperpipe.evidence import rho_medians

    out = rho_medians(_copula_rho())
    assert out == {
        "randomwalknowcast": {"rho_p": 0.6, "origins": 3, "fallback": 1},
        "randomwalk": {"rho_p": 0.35, "origins": 2, "fallback": 0},
    }


def test_rho_p_is_quoted_once_its_rows_exist(tmp_path) -> None:
    out = write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    rho = out["evidence"]["rho"]["randomwalknowcast"]
    assert rho["rho_p"] == "(not yet available)"
    rows = tmp_path / "copula-rho.csv"
    _copula_rho().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"copula_rho": rows}
    out = write_evidence(
        paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml"
    )
    assert out["evidence"]["rho"]["randomwalk"]["rho_p"] == 0.35


def test_drop_bdbv_keeps_no_bdbv_score_rows() -> None:
    from paperpipe.evidence import drop_bdbv

    rows = pd.DataFrame(
        {
            "study": ["germany", "ebola", "bdbv", "held_out"],
            "set": ["German national", "BDBV national", "x", "y"],
        }
    )
    assert list(drop_bdbv(rows)["study"]) == ["germany", "held_out"]
    floor = pd.DataFrame(
        {"source": [None, "daily"], "relative_wis": [None, 0.9]}
    )
    assert len(drop_bdbv(floor)) == 1


def test_waiting_notes_name_no_sessions_or_retired_grids() -> None:
    # The notes are rendered into the SI, so they say what each input is.
    code = re.compile(r"\bfm-|germany-chronos2|\br[3-6]\b")
    assert not [v for v in EVIDENCE.values() if code.search(v)]


def test_no_evidence_input_is_bdbv_scores() -> None:
    # The Ebola study is held out, so no development BDBV rows are inputs.
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "workflow" / "config.yaml").read_text()
    )
    assert not [n for n in EVIDENCE if "bdbv" in n]
    assert not [n for n in config["evidence"] if "bdbv" in n]


def test_a_reference_row_counts_once_per_target() -> None:
    base = {
        "study": "germany",
        "level": "national",
        "location": "DE",
        "origin_date": "2026-10-04",
        "target_date": "2026-09-27",
    }
    rows = pd.DataFrame(
        [
            {**base, "model": "randomwalknowcast", "option": o, "wis": w}
            for o, w in (("reference", 10.0), ("joint", 30.0))
        ]
        + [{**base, "model": "fm", "option": "joint", "wis": 5.0}]
    )
    assert aggregation_numbers(rows)["germany"]["national"]["fm"] == {
        "joint": 0.5
    }


def test_no_lag_horizon_evidence_is_read_or_written(tmp_path) -> None:
    assert "lag_germany" not in EVIDENCE
    write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "f",
        tmp_path / "e.yml",
    )
    numbers = yaml.safe_load((tmp_path / "e.yml").read_text())["evidence"]
    assert "lag" not in numbers


def test_the_pairing_step_summarises_both_studies(tmp_path) -> None:
    from paperpipe.__main__ import main

    base = {
        "level": "national",
        "model": "randomwalk",
        "target": "sum",
        "metric": "crps",
        "n": 1,
    }
    paths = []
    for study in ("germany", "synthetic"):
        rows = pd.DataFrame(
            [
                {**base, "study": study, "variant": v, "origin": o, "value": x}
                for v, x in (("off", 2.0), ("rolling", 1.0))
                for o in ("a", "b")
            ]
        )
        path = tmp_path / f"{study}.csv.gz"
        rows.to_csv(path, index=False)
        paths.append(str(path))
    out = tmp_path / "si-copula.csv"
    main(["pairing", *paths, "--out", str(out)])
    got = pd.read_csv(out)
    assert sorted(got["study"].unique()) == ["germany", "synthetic"]
    rolling = got.loc[got["variant"] == "rolling", "relative"]
    assert (rolling == 0.5).all()
