"""The simulator's code files and its grid file's identifier."""

from __future__ import annotations

import hashlib
from pathlib import Path

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


def grid_id(grid_file: Path | str) -> str:
    """Return the grid file's identifier, as each manifest row records it.

    The first 12 hex digits of the file's SHA-256.
    """
    return hashlib.sha256((ROOT / grid_file).read_bytes()).hexdigest()[:12]
