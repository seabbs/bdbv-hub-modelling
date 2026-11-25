"""Score an offloaded run's step 0 and checkpoints (rules/evaluate.smk).

    python workflow/scripts/evaluate_offloaded.py base TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py checkpoint TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py seven-day TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py validation TRAIN/n<N>
    python workflow/scripts/evaluate_offloaded.py validation-base TRAIN/n<N>

`base` scores step 0 alone (base Chronos-2, the start loss), `checkpoint`
segment n<N>'s adapter, `seven-day` its 7-day check (`SEVEN_DAY`, beside
its settings), `validation` its forecasts of a sample of the held-out
blocks and `validation-base` base Chronos-2's of the same blocks
(`evaluations.VALIDATION_FILE`, `VALIDATION_BASE`), with the trainer's
own losses
(nowcastfoundation.finetune.evaluate); the files are as
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


def base_step(where: Path, device: str) -> None:
    """Score step 0 (base Chronos-2) alone, unless it has a result.

    It scores every held-out block; a checkpoint's evaluation needs it
    for the start loss.
    """
    train = where.parent
    score_snapshots(train, evaluations.pending(train, -1, 0), device)


def logging_batches(label: str) -> Any:
    """Log each batch of an evaluation as `label` (fs.batch_progress)."""
    start = time.perf_counter()

    def report(k: int, n: int) -> None:
        fs.log(f"{label}: batch {k}/{n} ({time.perf_counter() - start:.0f} s)")

    return fs.batch_progress(report)


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
        with logging_batches(snap.name):
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
        with logging_batches(where.name):
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
    with logging_batches(f"{where.name} 7-day"):
        check = fe.seven_day_check(
            where / "finetuned-ckpt",
            base,
            blocks,
            settings.get("batch_size", 256),
        )
    seconds = time.perf_counter() - t0
    fe.write_json(
        where / SEVEN_DAY,
        {"seven_day": check, "device": device, "seconds": seconds},
    )
    fs.log(f"{where.name}: 7-day check {check} in {seconds:.0f} s")


def validation(where: Path, device: str, base_only: bool = False) -> None:
    """Write a checkpoint's (or base Chronos-2's) validation forecasts.

    Of the sample of the run's held-out blocks that
    `fe.validation_sample` picks; `where` is a checkpoint's folder, whose
    settings give the run's batch size and validation blocks.
    """
    import gzip

    settings = json.loads((where / "settings.json").read_text())
    blocks = fe.load_blocks(
        where.parent / fs.HELD_OUT_BLOCKS, settings.get("n_validation")
    )
    model = fe.load_base(device)
    if base_only:
        out, label = where.parent / evaluations.VALIDATION_BASE, "base"
    else:
        from peft import PeftModel

        series_bias.attach_for(model, where / "finetuned-ckpt")
        model = PeftModel.from_pretrained(model, where / "finetuned-ckpt")
        out, label = where / evaluations.VALIDATION_FILE, where.name
    model.eval()
    t0 = time.perf_counter()
    with logging_batches(f"{label} validation forecasts"):
        records = fe.validation_forecasts(
            model, blocks, settings.get("batch_size", 256)
        )
    seconds = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(
            {
                "model": label,
                "n_outbreaks": 0 if base_only else settings.get("n_outbreaks"),
                "device": device,
                "seconds": seconds,
                "blocks": records,
            },
            f,
        )
    tmp.replace(out)
    fs.log(f"{label}: {len(records)} validation forecasts in {seconds:.0f} s")


WHAT = ("base", "checkpoint", "seven-day", "validation", "validation-base")
"""The subcommands."""


def main(argv: Sequence[str] | None = None) -> None:
    """Run one of `WHAT`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=WHAT)
    p.add_argument("where", type=Path)
    # The rule passes its scoring device (flowlib.compute); never detected.
    p.add_argument("--device", required=True)
    args = p.parse_args(argv)
    device = torch_device(args.device)
    if args.what == "base":
        base_step(args.where, device)
    elif args.what == "seven-day":
        seven_day(args.where, device)
    elif args.what in ("validation", "validation-base"):
        validation(args.where, device, args.what == "validation-base")
    else:
        checkpoint(args.where, device)


if __name__ == "__main__":
    main()
