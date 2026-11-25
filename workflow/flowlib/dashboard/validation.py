"""Validation forecasts: checkpoints' forecasts of held-out blocks.

A run's train folder holds base Chronos-2's forecasts of a fixed sample
of its held-out (validation) blocks (`evaluations.VALIDATION_BASE`) and
each checkpoint's of the same blocks (`n<N>/` +
`evaluations.VALIDATION_FILE`), each block with its rows' history and
truth (nowcastfoundation.finetune.evaluate.forecast_record). The loss
section's card lists a run's blocks and models (`index`) and fetches one
block at a time (`block`). Only these files are read.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from flowlib import evaluations

from . import logs
from .runs import Run

FILE = evaluations.VALIDATION_FILE
BASE_FILE = evaluations.VALIDATION_BASE

BASE_LABEL = "Zero-shot"
"""Base Chronos-2's label in the card."""

_SEEN: dict[tuple[str, int], dict[str, Any]] = {}


def _load(path: Path) -> dict[str, Any] | None:
    """Return a forecasts file, read again only when it changes."""
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _SEEN:
        try:
            with gzip.open(path, "rt") as f:
                _SEEN[key] = json.load(f)
        except (OSError, ValueError):
            return None
    return _SEEN[key]


def files(run: Run) -> list[tuple[str, int, Path]]:
    """Return a run's forecasts files as (key, size, path), base first."""
    if run.folder is None:
        return []
    train = Path(run.folder) / "train"
    out = []
    if (train / BASE_FILE).is_file():
        out.append(("base", 0, train / BASE_FILE))
    ckpts = sorted(
        (p for p in train.glob(f"n*/{FILE}")),
        key=lambda p: logs._size(p.parent.name),
    )
    out += [(p.parent.name, logs._size(p.parent.name), p) for p in ckpts]
    return out


def index(run: Run) -> dict[str, Any]:
    """Return a run's models and the blocks they forecast."""
    found = [(k, n, _load(p)) for k, n, p in files(run)]
    found = [(k, n, d) for k, n, d in found if d]
    models = [
        {"key": k, "label": BASE_LABEL if k == "base" else k, "n": n}
        for k, n, _ in found
    ]
    blocks = []
    if found:
        for rec in found[0][2].get("blocks", []):
            blocks.append(
                {
                    "block": rec["block"],
                    "source": rec.get("source", ""),
                    "cut": rec.get("cut"),
                    "mix": rec.get("mix", ""),
                    "rows": [r["name"] for r in rec.get("rows", [])],
                }
            )
    return {"models": models, "blocks": blocks}


def block(run: Run, i: int) -> dict[str, Any] | None:
    """Return one block's rows and each model's quantiles of them."""
    rows: list[dict[str, Any]] | None = None
    nowcast = 0
    q: dict[str, list[Any]] = {}
    for key, _, path in files(run):
        data = _load(path)
        rec = next(
            (b for b in (data or {}).get("blocks", []) if b["block"] == i),
            None,
        )
        if rec is None:
            continue
        if rows is None:
            rows = [
                {k: r[k] for k in ("name", "level", "history", "truth")}
                for r in rec["rows"]
            ]
            nowcast = rec.get("nowcast", 0)
        q[key] = [r["q"] for r in rec["rows"]]
    if rows is None:
        return None
    return {"block": i, "nowcast": nowcast, "rows": rows, "q": q}
