"""Offloaded held-out evaluation: where its results are and who reads them.

With `finetune.eval: offload` the trainer saves an adapter snapshot and a
request at step 0 (`train/evals/step-0000000/`) and writes checkpoints
without their held-out losses (`finetune-stream`). The `evaluate` rules
(rules/evaluate.smk, run by workflow/scripts/evaluate_offloaded.py) score
them: step 0's result is `train/evals/step-0000000.json` and a
checkpoint's is the sidecar
`n<N>/held-out.json`, beside its `settings.json`, which is a protected
output of its segment. Readers take both through `checkpoint_settings`
and `merged_history`, so a run evaluated either way reads the same.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nowcastfoundation.sidecar import (
    RESULT_ONLY,
    SIDECAR,
    start_loss,
    with_sidecar,
)

__all__ = [
    "QUICK_BASE",
    "QUICK_CHECK",
    "VALIDATION_BASE",
    "VALIDATION_FILE",
    "checkpoint_settings",
    "level_points",
    "merged_history",
    "pending",
    "start_loss",
    "step_of",
    "step_results",
    "validation_curve",
    "write_validation_curve",
]


QUICK_CHECK = "quick-check.json"
"""A checkpoint's quick check (rule quick_check), beside its settings:
the loss, its parts and the coverage of a fixed sample of the held-out
blocks (`finetune.evaluate.quick_check`)."""

QUICK_BASE = "evals/quick-check-base.json"
"""Base Chronos-2's quick check, in the run's train folder (rule
quick_check_base)."""

QUICK_LOSSES = {
    "loss": "eval_loss",
    "loss_nowcast": "eval_loss_nowcast",
    "loss_forecast": "eval_loss_forecast",
}
"""A quick check's losses as the history's subset losses."""


VALIDATION_FILE = "validation-forecasts.json.gz"
"""A checkpoint's forecasts of a sample of its run's held-out blocks
(rule validation_forecasts), beside its settings."""

VALIDATION_BASE = "validation-forecasts-base.json.gz"
"""Base Chronos-2's forecasts of the same blocks, in the run's train
folder (rule validation_forecasts_base)."""


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_settings(where: Path | str) -> dict[str, Any]:
    """Return a checkpoint's settings with its sidecar's losses added."""
    where = Path(where)
    settings = _read(where / "settings.json")
    sidecar = _read(where / SIDECAR)
    out = with_sidecar(
        settings if isinstance(settings, dict) else {},
        sidecar if isinstance(sidecar, dict) else None,
    )
    quick = _read(where / QUICK_CHECK)
    if isinstance(quick, dict):
        out["quick_check"] = quick
    return out


def quick_results(train: Path | str) -> dict[int, dict[str, Any]]:
    """Return the quick checks' losses as subset losses, by step.

    Base Chronos-2's at step 0 and each checkpoint's at its step.
    """
    train = Path(train)
    found = [(0, _read(train / QUICK_BASE))]
    for where in sorted(train.glob(f"n*/{QUICK_CHECK}")):
        settings = _read(where.parent / "settings.json")
        if isinstance(settings, dict) and "steps" in settings:
            found.append((int(settings["steps"]), _read(where)))
    return {
        step: {QUICK_LOSSES[k]: v for k, v in got.items() if k in QUICK_LOSSES}
        for step, got in found
        if isinstance(got, dict)
    }


def step_results(train: Path | str) -> dict[int, dict[str, Any]]:
    """Return each evaluated step's losses, by step."""
    out = {}
    for path in sorted(Path(train).glob("evals/step-*.json")):
        got = _read(path)
        if not isinstance(got, dict):
            continue
        step = got.get("step", step_of(path.with_suffix("")))
        out[int(step)] = {k: v for k, v in got.items() if k not in RESULT_ONLY}
    return out


def merged_history(
    history: Sequence[Mapping[str, Any]], train: Path | str
) -> list[dict[str, Any]]:
    """Return the trainer's history with each step's offloaded losses.

    With the quick checks' losses as the subset losses at step 0 and at
    each checkpoint (`quick_results`).
    """
    results = step_results(train)
    for step, quick in quick_results(train).items():
        results[step] = {**results.get(step, {}), **quick}
    out = []
    for entry in history:
        entry = dict(entry)
        if "hash" not in entry and entry.get("step") in results:
            entry = {**results.pop(entry["step"]), **entry}
        out.append(entry)
    out += [{**v, "step": k} for k, v in results.items()]
    return sorted(out, key=lambda h: h.get("step", 0))


def level_points(
    history: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[tuple[int, Mapping[str, float]]]:
    """Return validation loss by level at the start and at each checkpoint.

    Against simulations trained: 0 for step 0's `by_level` in the
    history, and each checkpoint's simulations, from `(n, settings)` with
    its sidecar's losses. Empty with no start.
    """
    start = next((h for h in history if h.get("step") == 0), None)
    if start is None or not start.get("by_level"):
        return []
    return [(0, start["by_level"])] + [
        (n, s["held_out_loss_by_level"])
        for n, s in checkpoints
        if s.get("held_out_loss_by_level")
    ]


def validation_curve(
    train: Path | str, run: str, prefixes: Iterable[int]
) -> list[dict[str, Any]]:
    """Return a run's validation loss by level as rows, start and checkpoints.

    One row per point and level (`blocks`), against simulations trained
    (`outbreaks_seen`): the paper's fine-tune curve.
    """
    train = Path(train)
    got = _read(train / "history.json")
    hist = merged_history(got if isinstance(got, list) else [], train)
    cks = [(n, checkpoint_settings(train / f"n{n}")) for n in sorted(prefixes)]
    return [
        {
            "run": run,
            "blocks": k,
            "outbreaks_seen": n,
            "validation_loss": float(v),
        }
        for n, by in level_points(hist, cks)
        for k, v in by.items()
    ]


def write_validation_curve(
    train: Path | str, run: str, prefixes: Iterable[int], out: Path | str
) -> None:
    """Write `validation_curve` as a CSV."""
    rows = validation_curve(train, run, prefixes)
    names = ["run", "blocks", "outbreaks_seen", "validation_loss"]
    with Path(out).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def step_of(snapshot: Path) -> int:
    """Return a step snapshot's step, from its name `step-<k>`."""
    return int(snapshot.name.removeprefix("step-"))


def pending(
    train: Path | str, after: int = -1, upto: int | None = None
) -> list[Path]:
    """Return the step snapshots with no result yet, in step order.

    Only those with `after` < step <= `upto`: one segment's, so two
    segments' evaluations never score the same snapshot.
    """
    evals = Path(train) / "evals"
    return [
        d
        for d in sorted(evals.glob("step-*"))
        if d.is_dir()
        and not d.name.endswith(".tmp")
        and (d / "request.json").exists()
        and not d.with_name(d.name + ".json").exists()
        and after < step_of(d) <= (upto if upto is not None else step_of(d))
    ]


def check_report(
    where: Path | str, run: str, n: int, prefixes: Iterable[int]
) -> tuple[str, str]:
    """Return whether checkpoint n<n> beats zero-shot, as a report and a line.

    On the held-out simulations only: BDBV is held out for the paper.
    The line is an ALERT when it does not, naming the next checkpoint.
    """
    s = checkpoint_settings(where)
    ok = s["held_out_loss"] < s["held_out_loss_start"]
    text = (
        f"{run} n{n} {'beats' if ok else 'does NOT beat'} zero-shot: "
        f"held-out {s['held_out_loss']:.4f} vs "
        f"{s['held_out_loss_start']:.4f}"
    )
    after = [p for p in sorted(prefixes) if p > n]
    nxt = f", check before n{after[0]}" if after else ""
    return text, text if ok else f"ALERT {text}; training goes on{nxt}"
