"""Docs pages say what the code does now, without PR history in prose."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
# This repo's own issues and PRs; links to other projects' issues are sources.
PR_REF = re.compile(
    r"(?<![\w/])#\d{2,}\b|bdbv-hub-modelling/(?:issues|pull)/\d+"
)
PROSE = sorted(
    str(path.relative_to(ROOT))
    for pattern in (
        "docs/**/*.md",
        "README.md",
        "packages/*/README.md",
        "paper/*.md",
        "paper/*.qmd",
    )
    for path in ROOT.glob(pattern)
)


@pytest.mark.parametrize("page", PROSE)
def test_pages_carry_no_pr_or_issue_numbers(page: str) -> None:
    hits = [
        line
        for line in (ROOT / page).read_text().splitlines()
        if PR_REF.search(line)
    ]
    assert not hits, hits


def test_foundation_cites_package_paths() -> None:
    text = (DOCS / "methods" / "foundation.md").read_text()
    assert "forecasters.finetune" not in text
    assert "nowcastfoundation.finetune.training_block" in text


def test_foundation_says_daily_ragged_reads_the_rolling_row() -> None:
    text = (DOCS / "methods" / "foundation.md").read_text()
    assert "The daily source is summed to weeks" not in text
    assert "rolling" in text


def test_baselines_does_not_say_places_are_independent() -> None:
    text = (DOCS / "methods" / "baselines.md").read_text()
    assert "Places are sampled independently" not in text
