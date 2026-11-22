"""The simulator's code files and grid, read from the grid config."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
ENGINE = (
    "packages/nowcastfoundation/src/nowcastfoundation/simulator",
    "packages/nowcastrenewal/src/nowcastrenewal/latent",
    "packages/nowcastrenewal/src/nowcastrenewal/components",
    "packages/nowcastrenewal/src/nowcastrenewal/reporting",
    "packages/nowcastrenewal/src/nowcastrenewal/simulation",
)
"""The simulator's code: its entry point and draws, and the joint model
(the latent step, the reporting and the shared components)."""


def read_grid(grid_file: Path | str) -> dict:
    """Return the grid config (`config/finetune/grid.yaml`)."""
    return yaml.safe_load((ROOT / grid_file).read_text())


def grid_id(grid_file: Path | str) -> str:
    """Return the grid file's identifier, as each manifest row records it.

    The first 12 hex digits of the file's SHA-256.
    """
    return hashlib.sha256((ROOT / grid_file).read_bytes()).hexdigest()[:12]
