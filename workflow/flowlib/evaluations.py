"""Offloaded held-out evaluation: where its results are and who reads them.

With `finetune.eval: offload` the trainer saves an adapter snapshot and a
request at step 0 and each evaluation step (`train/evals/step-<k>/`) and
writes checkpoints without their held-out losses
(scripts/finetune_stream.py). The `evaluate` rules (rules/evaluate.smk,
run by workflow/scripts/evaluate_offloaded.py) score them: each step's
result is `train/evals/step-<k>.json` and a checkpoint's is the sidecar
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

from nowcastfoundation.sidecar import RESULT_ONLY, SIDECAR, with_sidecar

__all__ = [
    "checkpoint_settings",
    "heldout_curve",
    "level_points",
    "merged_history",
    "pending",
    "start_loss",
    "step_of",
    "step_results",
    "write_heldout_curve",
]


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
    return with_sidecar(
        settings if isinstance(settings, dict) else {},
        sidecar if isinstance(sidecar, dict) else None,
    )


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
    """Return the trainer's history with each step's offloaded losses."""
    results = step_results(train)
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
    """Return held-out loss by level at the start and at each checkpoint.

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


def heldout_curve(
    train: Path | str, run: str, prefixes: Iterable[int]
) -> list[dict[str, Any]]:
    """Return a run's held-out loss by level as rows, start and checkpoints.

    One row per point and level (`blocks`), against simulations trained
    (`outbreaks_seen`): the paper's fine-tune curve.
    """
    train = Path(train)
    got = _read(train / "history.json")
    hist = merged_history(got if isinstance(got, list) else [], train)
    cks = [(n, checkpoint_settings(train / f"n{n}")) for n in sorted(prefixes)]
    return [
        {"run": run, "blocks": k, "outbreaks_seen": n, "heldout_loss": float(v)}
        for n, by in level_points(hist, cks)
        for k, v in by.items()
    ]


def write_heldout_curve(
    train: Path | str, run: str, prefixes: Iterable[int], out: Path | str
) -> None:
    """Write `heldout_curve` as a CSV."""
    rows = heldout_curve(train, run, prefixes)
    names = ["run", "blocks", "outbreaks_seen", "heldout_loss"]
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
