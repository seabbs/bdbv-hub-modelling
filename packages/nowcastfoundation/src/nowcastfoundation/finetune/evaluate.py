# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score an adapter snapshot of `scripts/finetune_stream.py --eval offload`.

Scores the trainer's saved held-out blocks with its own loss functions,
so the losses are those the trainer would have logged.

    uv run --group chronos python scripts/finetune_evaluate.py \
        TRAIN/evals/step-0000100 --train TRAIN --out result.json

Kinds: `start` (step 0), `monitor` (the logged held-out loss),
`by_source` and `checkpoint` (a checkpoint's held-out settings).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from nowcastfoundation import series_bias
from nowcastfoundation.backends.chronos2 import torch_device

KINDS = ("start", "monitor", "by_source", "checkpoint")


def load_blocks(path: Path, n: int | None = None) -> list[dict]:
    """Return the trainer's held-out blocks, checking their count."""
    blocks = fs.load_blocks(Path(path))
    if n is not None and len(blocks) != n:
        raise ValueError(
            f"{path} holds {len(blocks)} held-out blocks, the run {n}"
        )
    return blocks


def evaluate(
    adapter: Path,
    base,
    blocks: Sequence[dict],
    kinds: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Return the losses `kinds` ask for of `base` with `adapter`.

    `base` is a base Chronos-2 model, with the series bias attached if
    the adapter holds one.
    """
    from peft import PeftModel

    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    val = list(blocks)
    out: dict[str, Any] = {}
    if "start" in kinds or "checkpoint" in kinds:
        parts = fs.held_out_parts(model, val, batch_size)
        out["held_out_loss"] = parts["loss"]
        out |= fs.loss_parts("held_out_loss", parts)
    if "start" in kinds or "checkpoint" in kinds:
        out["held_out_coverage"] = held_out_coverage(model, val, batch_size)
    if "start" in kinds:
        out["by_level"] = fs.held_out_by_level(model, val, fs.LEVEL_BLOCKS)
        out["by_source"] = fs.held_out_by_source(
            model, val, batch_size, ft.EVAL_BLOCKS
        )
    if "monitor" in kinds:
        parts = fs.held_out_parts(model, val[: ft.EVAL_BLOCKS], batch_size)
        out["eval_loss"] = parts["loss"]
        out |= fs.loss_parts("eval_loss", parts)
    if "by_source" in kinds:
        out["by_source"] = fs.held_out_by_source(
            model, val, batch_size, ft.EVAL_BLOCKS
        )
    if "checkpoint" in kinds:
        out["held_out_loss_by_level"] = fs.held_out_by_level(
            model, val, fs.LEVEL_BLOCKS
        )
        out["held_out_loss_by_source"] = fs.held_out_by_source(
            model, val, batch_size
        )
    return out


INTERVAL_LEVELS = (0.05, 0.25, 0.75, 0.95)
"""The ends of the 90% and 50% central intervals `held_out_coverage`
scores."""


def _covered(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    """Return each target step's 50% and 90% hits, step kind, source, level.

    A step is a nowcast step up to its block's cut-off
    (`finetune_stream.nowcast_steps`), as the loss's parts split them.
    """
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    blocks = list(blocks)
    data = fs.ByLength(
        blocks,
        batch_size,
        DatasetMode.VALIDATION,
        series_bias.has_bias(model),
    )
    accepted = inspect.signature(Chronos2Model.forward).parameters
    levels = np.asarray(model.chronos_config.quantiles, dtype=float)
    # Each interval end is the model's quantile at that level, linear
    # between its two nearest levels where it has none.
    upper = np.clip(
        np.searchsorted(levels, INTERVAL_LEVELS), 1, len(levels) - 1
    )
    weight = np.clip(
        (np.asarray(INTERVAL_LEVELS) - levels[upper - 1])
        / (levels[upper] - levels[upper - 1]),
        0.0,
        1.0,
    )[None, :, None]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for indices in data.indices():  # type: ignore[no-untyped-call]
            batch = data._build_batch(indices)
            out = fs.forward(model, batch, accepted)
            raw = out.quantile_preds.float().cpu().numpy()
            preds = (1 - weight) * raw[:, upper - 1] + weight * raw[:, upper]
            target = batch["future_target"].float().cpu().numpy()
            ranges = batch["target_idx_ranges"]
            for i, (lo, hi) in zip(indices, ranges, strict=True):
                block = blocks[i]
                n_steps = int(block["prediction_length"])
                d = fs.nowcast_steps(block)
                names = block.get("levels", ())
                for r in range(lo, hi):
                    y = target[r, :n_steps]
                    q = preds[r, :, :n_steps]
                    ok = ~np.isnan(y)
                    rows.append(
                        {
                            "source": block.get("source", ""),
                            "level": names[r - lo]
                            if r - lo < len(names)
                            else "",
                            "nowcast": np.arange(n_steps)[ok] < d,
                            "in50": ((y >= q[1]) & (y <= q[2]))[ok],
                            "in90": ((y >= q[0]) & (y <= q[3]))[ok],
                        }
                    )
    return rows


def held_out_coverage(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> dict[str, float]:
    """Return the held-out 50% and 90% interval coverage of target steps.

    Over every step, then the nowcast and forecast steps, then by source
    and by level (`<prefix>cover50`, `<prefix>cover90`, `<prefix>n`).
    """
    rows = _covered(model, blocks, batch_size)
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in rows:
        for part in ("nowcast", "forecast"):
            keep = row["nowcast"] if part == "nowcast" else ~row["nowcast"]
            groups.setdefault(f"{part}_", []).append(
                (row["in50"][keep], row["in90"][keep])
            )
        hits = (row["in50"], row["in90"])
        groups.setdefault("", []).append(hits)
        for key in ("source", "level"):
            if row[key]:
                groups.setdefault(f"{key}_{row[key]}_", []).append(hits)
    out: dict[str, float] = {}
    for prefix, parts in groups.items():
        in50 = np.concatenate([a for a, _ in parts])
        in90 = np.concatenate([b for _, b in parts])
        out[f"{prefix}n"] = len(in50)
        out[f"{prefix}cover50"] = float(in50.mean()) if len(in50) else np.nan
        out[f"{prefix}cover90"] = float(in90.mean()) if len(in90) else np.nan
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    # A snapshot folder (adapter/ and request.json) or a checkpoint's
    # finetuned-ckpt folder.
    p.add_argument("snapshot", type=Path)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    # Checked against KINDS by `evaluate`.
    p.add_argument("--kinds", nargs="+")
    p.add_argument(
        "--batch-size", type=int, default=ft.workflow_finetune("batch_size")
    )
    p.add_argument("--device", default="auto")
    return p


def load_base(device: str):
    """Return base Chronos-2 on `device`."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        fs.CHRONOS_MODEL, revision=fs.CHRONOS_REVISION, device_map=device
    ).model


def main(argv: Sequence[str] | None = None) -> None:
    """Score a snapshot and write its losses as JSON."""
    args = build_parser().parse_args(argv)
    snapshot = args.snapshot
    request_file = snapshot / "request.json"
    request = (
        json.loads(request_file.read_text()) if request_file.exists() else {}
    )
    adapter = snapshot / "adapter" if request else snapshot
    kinds = args.kinds or request.get("kinds")
    if not kinds:
        raise SystemExit(f"no kinds asked for {snapshot}")
    batch_size = request.get("batch_size", args.batch_size)
    blocks = load_blocks(
        args.train / fs.HELD_OUT_BLOCKS, request.get("n_validation")
    )
    device = torch_device(args.device)
    base = load_base(device)
    series_bias.attach_for(base, adapter)
    tick = time.perf_counter()
    losses = evaluate(adapter, base, blocks, kinds, batch_size)
    result = {
        **{k: request[k] for k in ("step", "n_outbreaks") if k in request},
        **losses,
        "kinds": list(kinds),
        "device": device,
        "seconds": time.perf_counter() - tick,
    }
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(args.out)
    fs.log(f"{snapshot.name}: {', '.join(kinds)} in {result['seconds']:.0f} s")


if __name__ == "__main__":
    main()
