# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""An offloaded checkpoint's held-out losses, kept beside its settings.

The workflow's `evaluate_checkpoint` rule writes them as `SIDECAR`
(workflow/flowlib/evaluations.py has the layout); readers take a
checkpoint's settings through `with_sidecar`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

HELD_OUT_BLOCKS: Final = "held-out-blocks.pkl"
"""The held-out blocks an offloaded run saves for its evaluator; must
match `HELD_OUT_BLOCKS` in nowcastfoundation.finetune.stream (tested)."""

SIDECAR: Final = "held-out.json"
"""The sidecar's file name, beside `settings.json` and in the release."""

RESULT_ONLY: Final = ("kinds", "device", "seconds", "n_outbreaks", "step")
"""Fields of an evaluation's result about the run, not losses."""


def with_sidecar(
    settings: Mapping[str, Any], sidecar: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return `settings` with the sidecar's losses added, settings first."""
    if not sidecar:
        return dict(settings)
    losses = {k: v for k, v in sidecar.items() if k not in RESULT_ONLY}
    return {**losses, **settings}


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def start_loss(train: Path | str) -> float | None:
    """Return base Chronos-2's held-out loss, step 0's or a checkpoint's."""
    train = Path(train)
    got = _read(train / "evals" / "step-0000000.json")
    if isinstance(got, dict) and "held_out_loss" in got:
        return float(got["held_out_loss"])
    # A checkpoint evaluated in the trainer, or an earlier sidecar.
    for path in sorted(
        [*train.glob("n*/settings.json"), *train.glob(f"n*/{SIDECAR}")]
    ):
        s = _read(path)
        if isinstance(s, dict) and s.get("held_out_loss_start") is not None:
            return float(s["held_out_loss_start"])
    return None
