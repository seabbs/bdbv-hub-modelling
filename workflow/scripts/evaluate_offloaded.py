"""Score an offloaded run's snapshots and checkpoints (rules/evaluate.smk).

    python workflow/scripts/evaluate_offloaded.py steps TRAIN/n<N> \
        --eval-every K [--prev TRAIN/n<M>]
    python workflow/scripts/evaluate_offloaded.py base TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py checkpoint TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py seven-day TRAIN/n<N>

`steps` scores segment n<N>'s step snapshots, `base` step 0 alone (base
Chronos-2, the start loss), `checkpoint` its adapter, `seven-day` its
7-day check (`SEVEN_DAY`, beside its settings), with
the trainer's own losses (nowcastfoundation.finetune.evaluate); the files are as
`flowlib.evaluations` says.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import evaluations  # noqa: E402

from nowcastfoundation import series_bias  # noqa: E402
from nowcastfoundation.backends.chronos2 import torch_device  # noqa: E402
from nowcastfoundation.finetune import evaluate as fe  # noqa: E402
from nowcastfoundation.finetune import stream as fs  # noqa: E402


def missing(train: Path, after: int, upto: int, every: int) -> list[str]:
    """Return the segment's due steps with neither snapshot nor result.

    Each step after `after` up to `upto` that is a multiple of `every` is
    due.
    """
    evals = train / "evals"

    def found(step: int) -> bool:
        name = f"step-{step:07d}"
        return (evals / name / "request.json").exists() or (
            evals / f"{name}.json"
        ).exists()

    due = range((after // every + 1) * every, upto + 1, every)
    return [f"step-{k:07d}" for k in due if not found(k)]


def steps(
    where: Path,
    prev: Path | None,
    device: str,
    every: int,
) -> None:
    """Score segment `where`'s step snapshots that have no result yet.

    The segment's steps are those after checkpoint `prev`'s (after step
    0 for the first, which `base_step` scores alone, so no step waits
    for it) up to its own. A due step with neither snapshot nor result
    fails the job: its snapshots were not copied in, and scoring nothing
    must not mark the segment done.
    """
    train = where.parent
    upto = json.loads((where / "settings.json").read_text())["steps"]
    after = (
        json.loads((prev / "settings.json").read_text())["steps"]
        if prev is not None
        else 0
    )
    gone = missing(train, after, upto, every)
    if gone:
        raise SystemExit(
            f"{where}: no step snapshot or result for {', '.join(gone[:5])}"
            f"{' ...' if len(gone) > 5 else ''} in {train / 'evals'}"
            " (were the trainer's snapshots copied in?)"
        )
    score_snapshots(train, evaluations.pending(train, after, upto), device)


def base_step(where: Path, device: str) -> None:
    """Score step 0 (base Chronos-2) alone, unless it has a result.

    It scores every held-out block, and only a checkpoint's evaluation
    needs it (the start loss); the step evaluations never wait for it.
    """
    train = where.parent
    score_snapshots(train, evaluations.pending(train, -1, 0), device)


def logging_passes(label: str) -> Any:
    """Log each pass of an evaluation as `label` (fe.pass_progress)."""

    def report(name: str, seconds: float) -> None:
        fs.log(f"{label}: {name} in {seconds:.0f} s")

    return fe.pass_progress(report)


def score_snapshots(train: Path, todo: Sequence[Path], device: str) -> None:
    """Score step snapshots `todo`, newest first, writing each result."""
    if not todo:
        fs.log(f"{train}: no step snapshots to score")
        return
    todo = sorted(todo, key=evaluations.step_of, reverse=True)
    base = fe.load_base("cpu")
    blocks: dict[Any, list[dict]] = {}
    for snap in todo:
        n = json.loads((snap / "request.json").read_text()).get("n_validation")
        if n not in blocks:
            blocks[n] = fe.load_blocks(train / fs.HELD_OUT_BLOCKS, n)
        with logging_passes(snap.name):
            result = fe.snapshot_result(base, snap, blocks[n], device)
        fe.write_json(snap.with_name(snap.name + ".json"), result)
        kinds = ", ".join(result["kinds"])
        fs.log(f"{snap.name}: {kinds} in {result['seconds']:.0f} s")


def checkpoint(where: Path, device: str) -> None:
    """Score a checkpoint's adapter and write its held-out sidecar."""
    settings = json.loads((where / "settings.json").read_text())
    blocks = fe.load_blocks(
        where.parent / fs.HELD_OUT_BLOCKS, settings.get("n_validation")
    )
    try:
        with logging_passes(where.name):
            got = fe.write_checkpoint_sidecar(
                where,
                fe.load_base("cpu"),
                blocks,
                device,
                settings.get("batch_size", 256),
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    fs.log(f"{where.name}: held-out loss {got['held_out_loss']:.4f}")


SEVEN_DAY = "seven-day.json"
"""A checkpoint's 7-day check (`seven_day`), beside its settings."""


def seven_day(where: Path, device: str) -> None:
    """Write a checkpoint's 7-day check on its run's held-out blocks."""
    settings = json.loads((where / "settings.json").read_text())
    blocks = fe.load_blocks(
        where.parent / fs.HELD_OUT_BLOCKS, settings.get("n_validation")
    )
    base = fe.load_base(device)
    series_bias.attach_for(base, where / "finetuned-ckpt")
    t0 = time.perf_counter()
    check = fe.seven_day_check(
        where / "finetuned-ckpt", base, blocks, settings.get("batch_size", 256)
    )
    seconds = time.perf_counter() - t0
    fe.write_json(
        where / SEVEN_DAY,
        {"seven_day": check, "device": device, "seconds": seconds},
    )
    fs.log(f"{where.name}: 7-day check {check} in {seconds:.0f} s")


def main(argv: Sequence[str] | None = None) -> None:
    """Run `steps` or `checkpoint`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=("steps", "base", "checkpoint", "seven-day"))
    p.add_argument("where", type=Path)
    p.add_argument("--prev", type=Path)
    # The rule passes `forecast.device` (flowlib.forecast); never detected.
    p.add_argument("--device", required=True)
    p.add_argument(
        "--eval-every",
        type=int,
        help="the trainer's --eval-every: each multiple must be scored",
    )
    args = p.parse_args(argv)
    device = torch_device(args.device)
    if args.what == "steps":
        if args.eval_every is None:
            p.error("steps needs --eval-every")
        steps(args.where, args.prev, device, args.eval_every)
    elif args.what == "base":
        base_step(args.where, device)
    elif args.what == "seven-day":
        seven_day(args.where, device)
    else:
        checkpoint(args.where, device)


if __name__ == "__main__":
    main()
