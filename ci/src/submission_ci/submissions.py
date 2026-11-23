"""Write the table of what each round submitted under each hub id."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from submission_ci.policy import Policy

COLUMNS: tuple[str, ...] = ()


def rows(
    state: dict[str, Any], ledger: Path, policy: Policy
) -> list[dict[str, str]]:
    """Return one row per round and hub id."""
    del state, ledger, policy
    return []


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    del argv
    return 1
