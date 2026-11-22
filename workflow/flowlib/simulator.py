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


def jax_cache_env(folder: str) -> str:
    """Return the environment that shares JAX's compiles in `folder`.

    Each batch compiles the model's operations for its simulations'
    shapes; a persistent cache lets the next batch, on this host, read
    those it shares rather than compile them again. The counts do not
    change. Every compile is kept, however small, as most are. JAX's own
    cap is left off, as it scans the whole folder on every write; the
    simulate rule empties the folder once it is over `JAX_CACHE_GB`
    (workflow/scripts/trim_jax_cache.sh). Batches running at once share
    it: an entry read while another batch writes it fails to load and
    is compiled again. Empty `folder`: no cache.
    """
    if not folder:
        return ""
    return " ".join(
        [
            f"JAX_COMPILATION_CACHE_DIR={folder}",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0",
        ]
    )


def grid_id(grid_file: Path | str) -> str:
    """Return the grid file's identifier, as each manifest row records it.

    The first 12 hex digits of the file's SHA-256.
    """
    return hashlib.sha256((ROOT / grid_file).read_bytes()).hexdigest()[:12]
