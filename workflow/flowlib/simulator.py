"""The simulator's pins and engine files, read from the grid config."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
ENGINE = (
    "packages/nowcastfoundation/src/nowcastfoundation/simulator",
    "packages/nowcastrenewal/src/nowcastrenewal/latent",
    "packages/nowcastfoundation/r",
)
"""The simulator's code: its entry point and draws, the latent step and
the R engine's reporting step."""


def read_grid(grid_file: Path | str) -> dict:
    """Return the grid config (`config/finetune/grid.yaml`)."""
    return yaml.safe_load((ROOT / grid_file).read_text())


def epinowcast_sha(grid_file: Path | str) -> str:
    """Return the epinowcast commit the grid's simulator is pinned to.

    Raises `ValueError` if the grid sets no `epinowcast_sha`.
    """
    sha = str(read_grid(grid_file).get("epinowcast_sha") or "")
    if not sha:
        raise ValueError(f"{grid_file} sets no epinowcast_sha")
    return sha


def grid_id(grid_file: Path | str) -> str:
    """Return the grid file's identifier, as each manifest row records it.

    The first 12 hex digits of the file's SHA-256.
    """
    return hashlib.sha256((ROOT / grid_file).read_bytes()).hexdigest()[:12]
