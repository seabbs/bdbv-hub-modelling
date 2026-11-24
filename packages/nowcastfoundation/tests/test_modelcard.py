"""The Hugging Face model card of a fine-tuned checkpoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from nowcastfoundation import modelcard as mc

ROOT = Path(__file__).resolve().parents[3]

SETTINGS = {
    "hash": "08f3982239a3",
    "chronos_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "code_sha256": {"config/finetune/grid.yaml": "d907d9e5bb58" + "0" * 52},
    "lags": "one lag count a block: the largest 99% delay",
    "lora": {"r": 8, "alpha": 16, "modules_to_save": ["series_bias"]},
    "series_bias": {"pairs": "same place", "per": "encoder layer and head"},
    "sources": ["daily", "weekly", "daily+rolling7"],
    "context": 364,
    "loss_horizon_days": 7,
    "learning_rate": 1e-05,
    "batch_size": 256,
    "n_outbreaks": 3100,
    "steps": 71844,
    "code_commit": "014a2bac52a0a1f6af77b2c64ac548d0fda0245c",
    "release_layer": {"late_start": 0.3, "no_lags": 0.05},
}
SUMS = (
    ("adapter_config.json", "a" * 64),
    ("adapter_model.safetensors", "b" * 64),
)
HELD_OUT = {
    "held_out_loss_start": 0.74,
    "held_out_loss": 0.27,
    "held_out_loss_by_level": {"national": 0.15, "leaf": 0.30},
}
SEVEN_DAY = {"ratio": 1.02, "inside90": 0.91, "rough_ratio": 1.1, "n": 500}
SCORES = pd.DataFrame(
    {
        "study": ["Germany", "Germany", "BDBV"],
        "target": ["7-day", "7-day", "weekly"],
        "model": ["fine-tuned", "zero-shot", "fine-tuned"],
        "reference": ["randomwalknowcast"] * 3,
        "metric": ["WIS", "WIS", "CRPS"],
        "relative": [0.82, 0.95, 1.10],
    }
)


def _front(card: str) -> dict:
    head, sep, _ = card.removeprefix("---\n").partition("\n---\n")
    assert sep, "the card starts with a YAML front matter block"
    return yaml.safe_load(head)


def _card(**kw: object) -> str:
    evaluation = mc.CardEvaluation(HELD_OUT, SEVEN_DAY, SCORES)
    args = {
        "settings": SETTINGS,
        "sha256": SUMS,
        "evaluation": evaluation,
        "citation": "@software{x, doi = {10.5281/zenodo.1}}",
        **kw,
    }
    return mc.model_card(**args)  # type: ignore[arg-type]


def test_the_card_carries_the_hub_metadata() -> None:
    front = _front(_card())
    assert front["base_model"] == "amazon/chronos-2"
    assert front["library_name"] == "peft"
    assert front["pipeline_tag"] == "time-series-forecasting"
    assert front["license"] == mc.LICENSE
    assert {"nowcasting", "chronos-2", "lora"} <= set(front["tags"])


def test_the_card_describes_the_checkpoint() -> None:
    card = _card()
    for text in (
        "amazon/chronos-2",
        "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "Apache-2.0",
        "rank 8",
        "alpha 16",
        "series bias",
        "3,100 simulated outbreaks",
        "d907d9e5bb58",
        "one lag count a block",
        "late_start",
        "daily+rolling7",
        "08f3982239a3",
        "a" * 64,
        "014a2bac52a0",
    ):
        assert text in card, text


def test_the_card_reports_the_evaluation() -> None:
    card = _card()
    assert "0.74" in card
    assert "0.27" in card
    assert "national" in card
    assert "1.02" in card
    assert "91%" in card
    # One row per model and study, relative to the reference.
    assert (
        "| Germany | 7-day | fine-tuned | randomwalknowcast | WIS | 0.82 |"
        in (card)
    )
    assert "zenodo.1" in card


def test_a_missing_evaluation_says_so() -> None:
    card = mc.model_card(
        settings=SETTINGS,
        sha256=SUMS,
        evaluation=mc.CardEvaluation(),
        citation="",
    )
    assert card.count(mc.NOT_EVALUATED) == 3
    assert "## Citation" not in card


def test_the_card_states_its_intended_use_and_limits() -> None:
    card = _card()
    assert "## Intended use" in card
    assert "## Limitations" in card


def test_the_citation_is_the_repository_s() -> None:
    bib = mc.citation_bibtex(ROOT / "CITATION.cff")
    assert bib.startswith("@software{")
    assert "10.5281/zenodo.23942616" in bib
    assert "Abbott, Sam" in bib


def test_scores_need_their_columns() -> None:
    with pytest.raises(ValueError, match="relative"):
        mc.CardEvaluation(scores=SCORES.drop(columns="relative"))
