"""The live fine-tuning run's checkpoints in the paper's evidence."""

import pandas as pd
import yaml

from paperpipe.evidence import EVIDENCE, write_evidence

RULE = "own-quantile-wis_crps-1000-common"
MODELS = ("foundation_zeroshot",)


def _scores() -> pd.DataFrame:
    """Checkpoint scores: zero-shot, the live run, no biases, another run."""
    rows = []
    for ck, run, n in [
        ("zero-shot", "zero-shot", 0),
        ("slots-a", "one-model", 125),
        ("slots-b", "one-model", 1000),
        ("plain", "no-slots", 1000),
        ("slots-z", "slots-zeroinit", 500),
    ]:
        for model in MODELS:
            for level in ("national", "states"):
                rows.append(
                    {
                        "study": "germany",
                        "level": level,
                        "model": model,
                        "checkpoint": ck,
                        "run": run,
                        "outbreaks": n,
                        "n": 10,
                        "rule": RULE,
                        "target": "7day",
                        "rel_wis": 3.0 - n / 1000,
                    }
                )
            rows.append(
                {
                    "study": "ebola",
                    "level": "national",
                    "model": model,
                    "checkpoint": ck,
                    "run": run,
                    "outbreaks": n,
                    "n": 10,
                    "rule": RULE,
                    "target": "7day",
                    "rel_crps": 2.0 - n / 1000,
                    "rel_log_crps": 1.5,
                    "last_origin": "2026-11-08",
                }
            )
        if n:
            rows.append(
                {
                    "study": "held_out",
                    "level": "all",
                    "model": "foundation_zeroshot",
                    "checkpoint": ck,
                    "run": run,
                    "outbreaks": n,
                    "rule": RULE,
                    "target": "7day",
                    "loss": 0.4 - n / 10000,
                    "loss_zero_shot": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_the_curve_keeps_zero_shot_and_the_live_run_by_level() -> None:
    from paperpipe.evidence import checkpoint_rows

    out = checkpoint_rows(_scores(), "one-model")
    assert set(out["checkpoint"]) == {"zero-shot", "slots-a", "slots-b"}
    # Each level is kept, for its own panel, and BDBV is held out.
    assert set(out["level"]) == {"national", "states"}
    assert set(out["study"]) == {"germany"}


def test_the_attention_bias_comparison_comes_from_checkpoint_scores() -> None:
    from paperpipe.evidence import COMPARISONS, comparison_rows

    labels = COMPARISONS["slots"][1]
    out = comparison_rows(_scores(), "one-model", "no-slots", labels)
    assert set(out["run"]) == {"attention biases", "no biases"}
    # One set per study and level, not pooled.
    assert {"German national, daily", "German states, daily"} <= set(out["set"])
    assert not any(name.startswith("BDBV") for name in out["set"])
    assert "Validation loss, all" not in set(out["set"])
    assert set(out.columns) >= {"set", "run", "outbreaks", "score"}


def test_the_evidence_has_the_compact_tables_and_labels(tmp_path) -> None:
    rows = tmp_path / "scores.csv"
    _scores().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"checkpoint_scores": rows}
    write_evidence(
        paths,
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
        run="one-model",
        comparisons={"slots": "no-slots"},
    )
    table = (tmp_path / "si" / "checkpoint-table.md").read_text()
    assert "Zero-shot" in table
    assert "slots-z" not in table
    summary = (tmp_path / "si" / "slots-summary.md").read_text()
    assert "no biases" in summary
    assert (tmp_path / "si" / "live-summary.md").is_file()
    assert (tmp_path / "fig" / "compare-slots.png").is_file()
    live = yaml.safe_load((tmp_path / "e.yml").read_text())["evidence"]["live"]
    # BDBV is held out, so the evidence quotes no BDBV checkpoint.
    assert "ebola" not in live
    assert live["germany"]["label"] == (
        "Preliminary: fine-tuned on 1,000 of 8,000 simulations."
    )


def test_the_runs_are_compared_at_matched_sizes() -> None:
    from paperpipe.evidence import slots_summary

    extra = _scores()
    extra = extra.loc[extra["run"] == "no-slots"].assign(
        checkpoint="plain2", outbreaks=2000, rel_crps=0.1, loss=0.01
    )
    out = slots_summary(pd.concat([_scores(), extra]), "one-model", "no-slots")
    # Both runs reached 1,000, so both rows give 1,000, not 2,000.
    assert out.iloc[2]["BDBV CRPS"] == "1.00 (1,000)"
    assert out.iloc[1]["BDBV CRPS"] == "1.00 (1,000)"


def test_the_curve_keeps_the_hub_target_only() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = _scores()
    daily = rows.assign(target="daily", rel_crps=9.0, rel_wis=9.0)
    out = checkpoint_rows(pd.concat([rows, daily]), "one-model")
    assert (out["rel_wis"].dropna() < 9).all()
    assert not out.duplicated(["study", "level", "model", "checkpoint"]).any()


def test_each_comparison_pairs_the_live_run_with_its_variant() -> None:
    from paperpipe.evidence import comparison_rows

    labels = ("with delays", "without delays")
    out = comparison_rows(_scores(), "one-model", "no-slots", labels)
    assert set(out["run"]) == set(labels)
    assert {"German national, daily", "German states, daily"} <= set(out["set"])
    # A variant with no scored checkpoint gives no rows.
    assert comparison_rows(_scores(), "one-model", "nodelays", labels).empty


def test_the_comparisons_wait_on_the_variants_they_name(tmp_path) -> None:
    rows = tmp_path / "scores.csv"
    _scores().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"checkpoint_scores": rows}
    pairs = {"slots": "no-slots", "delays": "nodelays"}
    numbers = write_evidence(
        paths,
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
        run="one-model",
        comparisons=pairs,
    )
    from paperpipe.evidence import COMPARISONS

    waiting = numbers["evidence"]["waiting"]
    assert COMPARISONS["slots"][0] not in waiting
    assert COMPARISONS["delays"][0] in waiting
    assert COMPARISONS["realdata"][0] in waiting
    assert (tmp_path / "fig" / "compare-delays.png").is_file()


def test_the_trainer_s_loss_is_named_the_validation_loss() -> None:
    from paperpipe.evidence import COMPARISONS, comparison_rows

    scores = _scores()
    scores.loc[scores["study"] == "held_out", "level"] = "national"
    labels = COMPARISONS["slots"][1]
    out = comparison_rows(scores, "one-model", "no-slots", labels)
    assert "Validation loss, national" in set(out["set"])
    assert not out["set"].str.startswith("Held-out").any()
