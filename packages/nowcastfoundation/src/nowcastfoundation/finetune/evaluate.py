# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Score adapter snapshots and checkpoints of an offloaded run.

Scores the trainer's saved held-out blocks with its own loss functions,
so the losses are those the trainer would have logged.
workflow/scripts/evaluate_offloaded.py runs it (rules/evaluate.smk).

Kinds: `start` (step 0), `monitor` (the logged held-out loss),
`by_source` and `checkpoint` (a checkpoint's held-out settings).
"""

from __future__ import annotations

import copy
import inspect
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nowcastfoundation import series_bias, sidecar
from nowcastfoundation.finetune import ROLLING
from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import stream as fs
from nowcastfoundation.forecaster import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    rolling_totals,
)

KINDS = ("start", "monitor", "by_source", "checkpoint")


def load_blocks(path: Path, n: int | None = None) -> list[dict[str, Any]]:
    """Return the trainer's held-out blocks, checking their count."""
    blocks = fs.load_blocks(Path(path))
    if n is not None and len(blocks) != n:
        raise ValueError(
            f"{path} holds {len(blocks)} held-out blocks, the run {n}"
        )
    return blocks


def evaluate(
    adapter: Path,
    base: Any,
    blocks: Sequence[dict[str, Any]],
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
        out["held_out_rolling"] = held_out_rolling(model, val, batch_size)
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
    (`finetune.stream.nowcast_steps`), as the loss's parts split them.
    """
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
    for block, raw, target in _predictions(model, blocks, batch_size):
        preds = (1 - weight) * raw[:, upper - 1] + weight * raw[:, upper]
        n_steps = int(block["prediction_length"])
        d = fs.nowcast_steps(block)
        names = block.get("levels", ())
        for r in range(len(raw)):
            y = target[r, :n_steps]
            q = preds[r, :, :n_steps]
            ok = ~np.isnan(y)
            rows.append(
                {
                    "source": block.get("source", ""),
                    "level": names[r] if r < len(names) else "",
                    "nowcast": np.arange(n_steps)[ok] < d,
                    "in50": ((y >= q[1]) & (y <= q[2]))[ok],
                    "in90": ((y >= q[0]) & (y <= q[3]))[ok],
                }
            )
    return rows


def _predictions(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> Iterator[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    """Yield each held-out block, its quantiles and its future targets.

    The quantiles are the model's own levels, `[rows, Q, steps]`, and
    the targets `[rows, steps]`, for the block's target rows.
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
    with torch.no_grad():
        for indices in data.indices():  # type: ignore[no-untyped-call]
            batch = data._build_batch(indices)
            out = fs.forward(model, batch, accepted)
            raw = out.quantile_preds.float().cpu().numpy()
            target = batch["future_target"].float().cpu().numpy()
            ranges = batch["target_idx_ranges"]
            for i, (lo, hi) in zip(indices, ranges, strict=True):
                yield blocks[i], raw[lo:hi], target[lo:hi]


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


ROLLING_PARTS = ("gap", "scale", "rough", "rough_daily_sum", "rough_truth")
"""What `rolling_consistency` sums over a block's places and steps."""


def rolling_consistency(
    block: dict[str, Any], median: np.ndarray
) -> dict[str, float] | None:
    """Return how far a block's rolling-sum rows are from its daily rows.

    A `ROLLING` source's block has each place's daily row, then its
    rolling-sum row. The rolling row's median should be the rolling sum
    of the daily row's medians, the context's last days included. The
    sums over places and steps: `gap`, the absolute difference between
    the two; `scale`, the sum of the daily rows' rolling sums; and the
    roughness (absolute second differences) of the rolling row's median
    (`rough`), of the daily medians' rolling sum (`rough_daily_sum`) and
    of the true rolling sums (`rough_truth`). A rolling row much rougher
    than its daily sum copies the daily row's day-of-week cycle.

    Parameters
    ----------
    block : dict
        A held-out block, its target rows first.
    median : numpy.ndarray
        The model's median of each target row's future steps,
        `[targets, steps]`.

    Returns
    -------
    dict or None
        The sums (`ROLLING_PARTS`) and `n`, the steps compared, or None
        for a block of a source without rolling sums.
    """
    window = ROLLING.get(str(block.get("source", "")))
    if window is None:
        return None
    n = int(block["n_targets"]) // 2
    h = int(block["prediction_length"])
    context = np.asarray(block["context"], dtype=float)
    future = context.shape[1] - h
    past = np.nan_to_num(context[:n, future - (window - 1) : future])
    daily = np.concatenate([past, median[:n, :h]], axis=1)
    own = rolling_totals(daily, window)[:, window - 1 :]
    row = median[n : 2 * n, :h]
    truth = context[n : 2 * n, future:]

    def rough(x: np.ndarray) -> float:
        return float(np.nansum(np.abs(np.diff(x, 2, axis=1))))

    return {
        "gap": float(np.abs(row - own).sum()),
        "scale": float(np.abs(own).sum()),
        "rough": rough(row),
        "rough_daily_sum": rough(own),
        "rough_truth": rough(truth),
        "n": float(n * h),
    }


def summarise_rolling(
    parts: Sequence[tuple[str, dict[str, float] | None]],
) -> dict[str, float]:
    """Return each source's rolling check over its blocks' sums.

    `<source>_gap` is the summed gap over the summed scale; each
    roughness is per unit of that scale, so the three compare;
    `<source>_n` counts the steps.
    """
    by: dict[str, dict[str, float]] = {}
    for source, part in parts:
        if part is None:
            continue
        total = by.setdefault(source, dict.fromkeys([*ROLLING_PARTS, "n"], 0.0))
        for key, value in part.items():
            total[key] += value
    out: dict[str, float] = {}
    for source, total in sorted(by.items()):
        scale = total["scale"] or np.nan
        out[f"{source}_n"] = int(total["n"])
        for key in ("gap", "rough", "rough_daily_sum", "rough_truth"):
            out[f"{source}_{key}"] = float(total[key] / scale)
    return out


def held_out_rolling(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> dict[str, float]:
    """Return the rolling check (`rolling_consistency`) by source.

    Only the blocks of sources with rolling sums (`ROLLING`) are
    predicted; the median is the model's 0.5 level.
    """
    rolling = [b for b in blocks if b.get("source") in ROLLING]
    if not rolling:
        return {}
    levels = list(model.chronos_config.quantiles)
    mid = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    if abs(levels[mid] - 0.5) > 1e-9:
        raise ValueError(f"the model has no 0.5 quantile: {levels}")
    return summarise_rolling(
        [
            (str(block["source"]), rolling_consistency(block, raw[:, mid]))
            for block, raw, _ in _predictions(model, rolling, batch_size)
        ]
    )


def load_base(device: str) -> Any:
    """Return base Chronos-2 on `device`."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    ).model


def score(
    base: Any,
    adapter: Path,
    blocks: list[dict[str, Any]],
    kinds: Sequence[str],
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], float]:
    """Return the losses `kinds` ask for of `adapter`, and the seconds.

    Scores a copy of base Chronos-2 on `device` with the adapter (and its
    series bias) attached, then frees it.
    """
    model = copy.deepcopy(base).to(device)
    series_bias.attach_for(model, adapter)
    tick = time.perf_counter()
    losses = evaluate(adapter, model, blocks, kinds, batch_size)
    del model
    fs.free_memory()
    return losses, time.perf_counter() - tick


def snapshot_result(
    base: Any, snapshot: Path, blocks: list[dict[str, Any]], device: str
) -> dict[str, Any]:
    """Return a step snapshot's losses, as its `request.json` asks."""
    request = json.loads((snapshot / "request.json").read_text())
    kinds = request["kinds"]
    losses, seconds = score(
        base,
        snapshot / "adapter",
        blocks,
        kinds,
        request.get("batch_size", 256),
        device,
    )
    return {
        **{k: request[k] for k in ("step", "n_outbreaks") if k in request},
        **losses,
        "kinds": list(kinds),
        "device": device,
        "seconds": seconds,
    }


def write_checkpoint_sidecar(
    where: Path,
    base: Any,
    blocks: list[dict[str, Any]],
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Score checkpoint `where` on the held-out blocks; write its sidecar.

    The sidecar (`sidecar.SIDECAR`, beside its settings) holds the
    checkpoint's held-out losses and base Chronos-2's (`sidecar.start_loss`).

    Raises
    ------
    ValueError
        If the run has no base Chronos-2 loss to put beside them.
    """
    start = sidecar.start_loss(where.parent)
    if start is None:
        raise ValueError(
            f"{where}: no base Chronos-2 loss: neither evals/step-0000000.json "
            "nor a locally evaluated checkpoint's held_out_loss_start"
        )
    losses, seconds = score(
        base,
        where / "finetuned-ckpt",
        blocks,
        ["checkpoint"],
        batch_size,
        device,
    )
    out = {
        "held_out_loss_start": start,
        **losses,
        "device": device,
        "seconds": seconds,
    }
    write_json(where / sidecar.SIDECAR, out)
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON beside `path` and move it into place."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.rename(path)
