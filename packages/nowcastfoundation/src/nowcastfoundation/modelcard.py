# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The Hugging Face model card of a fine-tuned Chronos-2 checkpoint.

`model_card` writes the card, `README.md` in a Hub repository, from the
checkpoint's `settings.json`, the SHA-256 of its adapter files and its
evaluation (`CardEvaluation`): the held-out losses (`sidecar.SIDECAR`),
the 7-day check (`finetune.evaluate.summarise_seven_day`) and its scores
against the reference models. The card opens with the Hub's metadata
(base model, licence, library, pipeline tag and tags).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

from .backends.chronos2 import CHRONOS_MODEL

__all__ = [
    "CARD_FILE",
    "LICENSE",
    "NOT_EVALUATED",
    "SCORE_COLUMNS",
    "CardEvaluation",
    "citation_bibtex",
    "model_card",
]

CARD_FILE: Final = "README.md"
"""The model card's file name in a Hub repository."""

LICENSE: Final = "apache-2.0"
"""The adapter's licence on the Hub: Chronos-2's own, as the adapter
is applied to its weights."""

BASE_LICENSE: Final = "Apache-2.0"
"""Chronos-2's licence, as its model card states it."""

LIBRARY: Final = "peft"
"""The library that loads the adapter, in the Hub's metadata."""

PIPELINE_TAG: Final = "time-series-forecasting"
"""The Hub's task of the model."""

TAGS: Final = (
    "chronos-2",
    "lora",
    "nowcasting",
    "epidemiology",
    "infectious-disease",
    "time-series",
)
"""The Hub's tags of the model."""

REPOSITORY: Final = "https://github.com/seabbs/bdbv-hub-modelling"
"""The code that trained and evaluates the checkpoint."""

NOT_EVALUATED: Final = "Not evaluated for this checkpoint."
"""The line of an evaluation section with nothing to report."""

SCORE_COLUMNS: Final = (
    "study",
    "target",
    "model",
    "reference",
    "metric",
    "relative",
)
"""The columns of `CardEvaluation.scores`: each model's score relative to
the reference model's (below 1 is better), by study and target."""


@dataclass(frozen=True)
class CardEvaluation:
    """What the card reports of a checkpoint's evaluation.

    Attributes
    ----------
    held_out : mapping
        The held-out losses (`sidecar.SIDECAR`): `held_out_loss`,
        `held_out_loss_start` (base Chronos-2's) and
        `held_out_loss_by_level`, each optional.
    seven_day : mapping
        The 7-day check (`finetune.evaluate.summarise_seven_day`).
    scores : pandas.DataFrame or None
        Scores relative to a reference model, `SCORE_COLUMNS`.
    """

    held_out: Mapping[str, Any] = field(default_factory=dict)
    seven_day: Mapping[str, Any] = field(default_factory=dict)
    scores: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Check the scores' columns."""
        if self.scores is not None:
            missing = [c for c in SCORE_COLUMNS if c not in self.scores]
            if missing:
                raise ValueError(
                    f"The card's scores lack {', '.join(missing)}."
                )


def citation_bibtex(path: Path | str) -> str:
    """Return a BibTeX entry of the repository from its `CITATION.cff`.

    Parameters
    ----------
    path : pathlib.Path or str
        The `CITATION.cff` file.

    Returns
    -------
    str
    """
    cff = yaml.safe_load(Path(path).read_text())
    authors = " and ".join(
        f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]
    )
    doi = cff.get("doi", "")
    fields = {
        "author": authors,
        "title": f"{{{cff['title']}}}",
        "year": str(cff.get("date-released", "2026"))[:4],
        "publisher": "Zenodo" if doi else "",
        "doi": doi,
        "url": f"https://doi.org/{doi}" if doi else cff["repository-code"],
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@software{{abbott_bdbv_hub_modelling,\n{body}\n}}"


def _front_matter() -> str:
    """Return the Hub's metadata block of the card."""
    meta = {
        "base_model": CHRONOS_MODEL,
        "base_model_relation": "adapter",
        "library_name": LIBRARY,
        "pipeline_tag": PIPELINE_TAG,
        "license": LICENSE,
        "tags": list(TAGS),
    }
    return "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n"


def _adapter(settings: Mapping[str, Any]) -> list[str]:
    """Return the lines on the base model and the adapter."""
    lora = settings.get("lora", {})
    lines = [
        "## Base model and adapter",
        "",
        f"- Base model: [{CHRONOS_MODEL}](https://huggingface.co/"
        f"{CHRONOS_MODEL}) at revision `{settings.get('chronos_revision')}`"
        f", licence {BASE_LICENSE}.",
        f"- LoRA adapter: rank {lora.get('r')}, alpha {lora.get('alpha')}"
        + (
            f", also training {', '.join(lora['modules_to_save'])}."
            if lora.get("modules_to_save")
            else "."
        ),
    ]
    bias = settings.get("series_bias")
    if bias:
        parts = "; ".join(f"{k}: {v}" for k, v in bias.items())
        lines.append(
            "- A learnt series bias on Chronos-2's group attention between "
            f"related series ({parts})."
        )
    lines.append(
        f"- Trained with learning rate {settings.get('learning_rate')}, "
        f"batch size {settings.get('batch_size')} and "
        f"{_count(settings.get('steps'))} steps."
    )
    return lines


def _count(value: Any) -> str:
    """Return a whole number with thousands separators, or as it is."""
    return f"{value:,}" if isinstance(value, int) else str(value)


def _training(settings: Mapping[str, Any]) -> list[str]:
    """Return the lines on the training data."""
    grid = settings.get("code_sha256", {}).get("config/finetune/grid.yaml")
    lines = [
        "## Training data",
        "",
        f"- {_count(settings.get('n_outbreaks'))} simulated outbreaks from "
        "the joint renewal and reporting simulator"
        + (f", grid version `{grid[:12]}`." if grid else "."),
        f"- Lag rule: {settings.get('lags')}.",
    ]
    layer = settings.get("release_layer")
    lines.append(
        "- Release layer: "
        + (
            "; ".join(f"{k} {v}" for k, v in layer.items())
            if isinstance(layer, Mapping)
            else "none"
        )
        + "."
    )
    lines.append(
        f"- Context of {settings.get('context')} steps; the loss covers "
        f"{settings.get('loss_horizon_days')} days after the cut-off."
    )
    lines.append(
        f"- Trained from commit `{str(settings.get('code_commit'))[:12]}` "
        f"of [{REPOSITORY.removeprefix('https://github.com/')}]"
        f"({REPOSITORY})."
    )
    return lines


def _layout(settings: Mapping[str, Any]) -> list[str]:
    """Return the lines on the input layout the adapter expects."""
    sources = ", ".join(f"`{s}`" for s in settings.get("sources", []))
    return [
        "## Expected input",
        "",
        "One block per reporting triangle, every place of the system in "
        "the block:",
        "",
        "- a target row per place of its counts by reference date "
        "(daily, or one step per period);",
        "- for the sources with period totals, a second target row per "
        "place of its 7-day (or 3-day) totals;",
        "- each place's lag series (the counts reported by each delay) as "
        "past and future covariates.",
        "",
        f"Sources trained on: {sources}.",
        "The `nowcastfoundation` package builds the block "
        "(`forecaster.layout_block`) and loads the adapter "
        "(`forecaster.finetuned_adapter`).",
    ]


def _number(value: Any) -> str:
    """Return a loss or ratio to 2 decimal places."""
    return f"{float(value):.2f}"


def _held_out(held_out: Mapping[str, Any]) -> list[str]:
    """Return the held-out loss section."""
    lines = ["### Held-out loss", ""]
    if "held_out_loss" not in held_out:
        return [*lines, NOT_EVALUATED]
    loss = _number(held_out["held_out_loss"])
    start = held_out.get("held_out_loss_start")
    lines.append(
        f"Weighted quantile loss on held-out simulated blocks: {loss}"
        + (f" (base Chronos-2: {_number(start)})." if start else ".")
    )
    by_level = held_out.get("held_out_loss_by_level") or {}
    if by_level:
        lines += ["", "| Level | Loss |", "|---|---|"]
        lines += [f"| {k} | {_number(v)} |" for k, v in by_level.items()]
    return lines


def _seven_day(check: Mapping[str, Any]) -> list[str]:
    """Return the 7-day check section."""
    lines = ["### 7-day check", ""]
    if not check:
        return [*lines, NOT_EVALUATED]
    return [
        *lines,
        "On held-out blocks with 7-day rows: the median of the 7-day row "
        f"over the sum of the daily medians is {_number(check['ratio'])}, "
        f"{float(check['inside90']):.0%} of the daily sums fall in the "
        "7-day row's 90% interval, and the 7-day median is "
        f"{_number(check['rough_ratio'])} times as rough as the truth "
        f"({_count(check.get('n'))} steps).",
    ]


def _scores(scores: pd.DataFrame | None) -> list[str]:
    """Return the table of scores relative to the reference models."""
    lines = ["### Scores against reference models", ""]
    if scores is None or scores.empty:
        return [*lines, NOT_EVALUATED]
    head = "| Study | Target | Model | Reference | Metric | Relative |"
    lines += [
        "Each model's score over the reference model's: below 1 is better.",
        "",
        head,
        "|---|---|---|---|---|---|",
    ]
    for row in scores[list(SCORE_COLUMNS)].itertuples(index=False):
        study, target, model, reference, metric, relative = row
        lines.append(
            f"| {study} | {target} | {model} | {reference} | {metric} | "
            f"{_number(relative)} |"
        )
    return lines


def _checksums(
    settings: Mapping[str, Any], sha256: Sequence[tuple[str, str]]
) -> list[str]:
    """Return the checkpoint's identity: settings hash and file sums."""
    return [
        "## Files",
        "",
        f"Settings hash `{settings.get('hash')}`. SHA-256 of the adapter "
        "files, also in `sha256sums.txt`:",
        "",
        *(f"- `{name}`: `{sha}`" for name, sha in sha256),
    ]


_USE: Final = (
    "## Intended use",
    "",
    "Nowcasts and short forecasts of infectious disease counts from data "
    "still being reported: a reporting triangle of counts by reference "
    "and report date, in one or more places, with daily or weekly "
    "releases. It gives quantiles of the final counts.",
    "",
    "## Limitations",
    "",
    "- Trained on simulated outbreaks only; real reporting may differ "
    "from anything the simulator draws.",
    "- Checked on two real case studies (German COVID-19 "
    "hospitalisations and Bundibugyo virus disease in the Democratic "
    "Republic of the Congo); other settings are untested.",
    "- Forecasts beyond the trained horizon and inputs outside the "
    "expected layout are out of scope.",
    "- Not a substitute for epidemiological judgement in public health "
    "decisions.",
)


def model_card(
    settings: Mapping[str, Any],
    sha256: Sequence[tuple[str, str]],
    evaluation: CardEvaluation,
    citation: str,
) -> str:
    """Return the model card of a fine-tuned checkpoint.

    Parameters
    ----------
    settings : mapping
        The checkpoint's `settings.json`.
    sha256 : sequence of (str, str)
        The SHA-256 of each adapter file, by name
        (`forecaster.Checkpoint.sha256`).
    evaluation : CardEvaluation
        What the card reports of its evaluation; each missing part says
        `NOT_EVALUATED`.
    citation : str
        The BibTeX to cite (`citation_bibtex`); empty leaves the section
        out.

    Returns
    -------
    str
        The card, Markdown with the Hub's YAML metadata first.
    """
    lines = [
        f"# Chronos-2 fine-tuned for nowcasting (`{settings.get('hash')}`)",
        "",
        f"A LoRA adapter of {CHRONOS_MODEL} fine-tuned on simulated "
        "outbreaks to nowcast and forecast counts from reporting "
        "triangles.",
        "",
        *_adapter(settings),
        "",
        *_training(settings),
        "",
        *_layout(settings),
        "",
        "## Evaluation",
        "",
        *_held_out(evaluation.held_out),
        "",
        *_seven_day(evaluation.seven_day),
        "",
        *_scores(evaluation.scores),
        "",
        *_USE,
        "",
        *_checksums(settings, sha256),
    ]
    if citation:
        lines += ["", "## Citation", "", "```bibtex", citation, "```"]
    return _front_matter() + "\n" + "\n".join(lines) + "\n"
