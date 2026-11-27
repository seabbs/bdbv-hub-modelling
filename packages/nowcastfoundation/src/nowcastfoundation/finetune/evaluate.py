# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Score adapter snapshots and checkpoints of an offloaded run.

Scores the trainer's saved held-out blocks with its own loss functions,
so the losses are those the trainer would have logged.
workflow/scripts/evaluate_offloaded.py runs it (rules/evaluate.smk).

Kinds: `start` (step 0), `monitor` (the logged held-out loss),
`by_source` and `checkpoint` (a checkpoint's held-out settings).

On the CPU the one-pass evaluation splits its batches over
`EVALUATE_WORKERS` worker processes (`evaluate_workers`, `Split`), each
with its share of the job's threads and cores
(`nowcastutils.workers.pooled`). The batches are those one process runs,
so the results are the same however many workers run them, up to float
rounding from the workers' torch thread counts. A worker
exits when its coordinator does, which stops its workers on SIGTERM.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import os
import time
from collections.abc import Callable, Sequence
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
from nowcastutils import workers

KINDS = ("start", "monitor", "by_source", "checkpoint")

WORKERS_ENV = "EVALUATE_WORKERS"
"""The environment variable with the worker processes a held-out pass
splits its batches over (the rule's, `flowlib.scoring`)."""


def evaluate_workers(device: str) -> int:
    """Return the worker processes a held-out pass on `device` splits over.

    `EVALUATE_WORKERS` (`WORKERS_ENV`) on the CPU, else one: off the CPU
    the pass runs in its own process, which holds the device.
    """
    if device != "cpu":
        return 1
    return max(1, int(os.environ.get(WORKERS_ENV) or 1))


@dataclasses.dataclass(frozen=True)
class Split:
    """The worker processes a held-out pass splits its batches over.

    Each loads the held-out blocks from `blocks` (`load_blocks`, checked
    against `n`) and base Chronos-2 with the adapter scored, from
    `loader(device)` (`load_base` when None), which the workers import
    by name.
    """

    workers: int
    blocks: Path
    n: int | None = None
    loader: Callable[[str], Any] | None = None


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
    split: Split | None = None,
) -> dict[str, Any]:
    """Return the losses `kinds` ask for of `base` with `adapter`.

    `base` is a base Chronos-2 model, with the series bias attached if
    the adapter holds one. One forward pass over the blocks the kinds
    need (`held_out_pass`) gives every loss and the coverage, each as the
    trainer's own functions give it (`finetune.stream`). With `split`,
    over its worker processes (`split_pass`).
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    model = merged(base, adapter)
    val = list(blocks)
    full = "start" in kinds or "checkpoint" in kinds
    need = range(len(val)) if full else _subset_needs(val, kinds)
    device = str(next(model.parameters()).device)
    rows = [val[i] for i in need]
    if split is not None and split.workers > 1:
        passed = split_pass(
            split, adapter, list(need), rows, batch_size, device
        )
    else:
        passed = held_out_pass(model, rows, batch_size, eval_budget(device))
    scored = dict(zip(need, passed, strict=True))
    got = [scored.get(i) for i in range(len(val))]
    out: dict[str, Any] = {}
    if full:
        out |= _losses("held_out_loss", got)
        out["held_out_coverage"] = summarise_coverage(
            [row for r in got if r for row in r["cover"]]
        )
    if "start" in kinds:
        out["by_level"] = _by_level(val, got, fs.LEVEL_BLOCKS)
        out["by_source"] = _by_source(val, got, ft.EVAL_BLOCKS)
    if "monitor" in kinds:
        out |= _losses("eval_loss", got[: ft.EVAL_BLOCKS])
    if "by_source" in kinds:
        out["by_source"] = _by_source(val, got, ft.EVAL_BLOCKS)
    if "checkpoint" in kinds:
        out["held_out_loss_by_level"] = _by_level(val, got, fs.LEVEL_BLOCKS)
        out["held_out_loss_by_source"] = _by_source(val, got, None)
    if full:
        check = precision_check(model, val, batch_size, device)
        if check is not None:
            out["precision_check"] = check
    return out


def _subset_needs(
    val: Sequence[dict[str, Any]], kinds: Sequence[str]
) -> list[int]:
    """Return the blocks `monitor` and `by_source` score, in order."""
    need = set(range(min(len(val), ft.EVAL_BLOCKS)))
    if "by_source" in kinds:
        for idx in _source_blocks(val, ft.EVAL_BLOCKS).values():
            need |= set(idx)
    return sorted(need)


def _source_blocks(
    val: Sequence[dict[str, Any]], n: int | None
) -> dict[str, list[int]]:
    """Return each source's first `n` blocks (all with None), by source."""
    by: dict[str, list[int]] = {}
    for i, block in enumerate(val):
        by.setdefault(block["source"], []).append(i)
    return {k: v[:n] for k, v in sorted(by.items())}


def _losses(
    name: str, got: Sequence[dict[str, Any] | None]
) -> dict[str, float]:
    """Return loss `name` and its nowcast and forecast parts over `got`.

    Chronos-2's loss is a mean over rows, so the loss of many batches is
    their rows' sum over their count, as the trainer's weighted mean.
    """
    rows = sum(r["n"] for r in got if r)
    out = {name: sum(float(r["loss"].sum()) for r in got if r) / rows}
    for part in ("nowcast", "forecast"):
        out[f"{name}_{part}"] = (
            sum(float(r[part].sum()) for r in got if r) / rows
        )
    return out


def _by_source(
    val: Sequence[dict[str, Any]],
    got: Sequence[dict[str, Any] | None],
    n: int | None,
) -> dict[str, float]:
    """Return the loss of each source's first `n` blocks (all with None)."""
    return {
        source: _losses("loss", [got[i] for i in idx])["loss"]
        for source, idx in _source_blocks(val, n).items()
    }


def _by_level(
    val: Sequence[dict[str, Any]],
    got: Sequence[dict[str, Any] | None],
    n: int,
) -> dict[str, float]:
    """Return the loss by level of the first `n` blocks.

    As `finetune.stream.held_out_by_level`: a block's loss at a level is
    its rows of that level's, the other rows' targets masked, over all
    its rows; the levels' losses are means over blocks.
    """
    losses: dict[str, list[float]] = {}
    for block, r in zip(val[:n], got[:n], strict=False):
        if r is None:
            continue
        levels = np.array(block["levels"])
        mine = r["loss"][: len(levels)]
        for level in sorted(set(levels)):
            losses.setdefault(level, []).append(
                float(mine[levels == level].sum()) / r["n"]
            )
    return {k: float(np.mean(v)) for k, v in losses.items()}


def merged(base: Any, adapter: Path) -> Any:
    """Return `base` with `adapter`'s LoRA merged into its weights.

    For scoring: the same outputs as the adapter on top, to float
    rounding, without LoRA's extra low-rank products in every layer
    (about an eighth of a pass on a CPU). The series bias, saved whole,
    stays as a module. `base` is changed in place.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.eval()
    return model


def row_losses(
    model: Any,
    quantile_preds: Any,
    future_target: Any,
    future_target_mask: Any,
    patched_future_covariates_mask: Any,
    loc_scale: Any,
    num_output_patches: int,
) -> Any:
    """Return Chronos-2's loss of each row, whose mean is its batch loss.

    `Chronos2Model._compute_loss` stopped before its mean over the batch:
    each row's quantile loss, masked, mean over the horizon and summed
    over the quantile levels.
    """
    import torch
    from einops import rearrange

    batch = future_target.shape[0]
    patch = model.chronos_config.output_patch_size
    target, _ = model.instance_norm(future_target, loc_scale)
    target = target.unsqueeze(1).to(model.device)
    mask = (
        future_target_mask.unsqueeze(1).to(model.device)
        if future_target_mask is not None
        else ~torch.isnan(target)
    )
    target = torch.where(mask > 0.0, target, 0.0)
    if quantile_preds.shape[-1] > target.shape[-1]:
        pad = (*target.shape[:-1], quantile_preds.shape[-1] - target.shape[-1])
        target = torch.cat([target, torch.zeros(pad).to(target)], dim=-1)
        mask = torch.cat([mask, torch.zeros(pad).to(mask)], dim=-1)
    quantiles = rearrange(model.quantiles, "q -> 1 q 1")
    loss = 2 * torch.abs(
        (target - quantile_preds)
        * ((target <= quantile_preds).float() - quantiles)
    )
    known = rearrange(
        patched_future_covariates_mask,
        "b n p -> b 1 (n p)",
        b=batch,
        n=num_output_patches,
        p=patch,
    )
    loss = loss * (mask.float() * (1 - known))
    return loss.mean(dim=-1).sum(dim=-1)


class _RowLosses:
    """Chronos-2's loss by row, and its nowcast and forecast parts.

    While open, each loss Chronos-2 computes is also kept by row
    (`row_losses`): over every step, the steps before `steps[row]`
    (nowcast) and those from it (forecast), as
    `finetune.stream._LossSplit` splits it.
    """

    def __init__(self) -> None:
        self.steps: Any = None
        self.rows: dict[str, np.ndarray] = {}

    def __enter__(self) -> Any:
        import torch
        from chronos.chronos2.model import Chronos2Model

        self.original = original = Chronos2Model._compute_loss

        def compute(model: Any, **kwargs: Any) -> Any:
            loss = original(model, **kwargs)
            target = kwargs["future_target"]
            mask = kwargs["future_target_mask"]
            mask = ~torch.isnan(target) if mask is None else mask > 0
            at = torch.arange(target.shape[-1], device=mask.device)
            steps = torch.as_tensor(self.steps, device=mask.device)
            now = at[None, :] < steps[:, None]
            kind = target.device.type
            for part, keep in (
                ("loss", torch.ones_like(now)),
                ("nowcast", now),
                ("forecast", ~now),
            ):
                kwargs["future_target_mask"] = (mask & keep).float()
                with torch.autocast(device_type=kind, enabled=False):
                    got = row_losses(
                        model,
                        **{
                            **kwargs,
                            "quantile_preds": kwargs["quantile_preds"].float(),
                        },
                    )
                self.rows[part] = got.double().cpu().numpy()
            return loss

        Chronos2Model._compute_loss = compute
        return self

    def __exit__(self, *exc: Any) -> None:
        from chronos.chronos2.model import Chronos2Model

        Chronos2Model._compute_loss = self.original


def block_cost(block: dict[str, Any]) -> int:
    """Return a block's series x tokens at most, before trimming.

    Its rows over the context patches, the reg token and the output
    patches, as `stream.batch_tokens` counts a batch's.
    """
    tokens = (
        -(-ft.CONTEXT // fs.INPUT_PATCH)
        + 1
        + -(-int(block["prediction_length"]) // ft.OUTPUT_PATCH)
    )
    return int(block["context"].shape[0]) * tokens


def pack(data: Any, budget: int) -> list[list[int]]:
    """Return batches of blocks of one length within `budget` each.

    Within each prediction length (`stream.ByLength`), the largest blocks
    first, each batch filled while its series x tokens (`block_cost`)
    stay within `budget`; a block over it runs alone.
    """
    groups: list[list[int]] = []
    for length in data.data:
        idx = [i for i, n in enumerate(data.lengths) if n == length]
        costs = {
            i: block_cost(
                {
                    "context": data.inputs[i]["context"],
                    "prediction_length": length,
                }
            )
            for i in idx
        }
        group: list[int] = []
        used = 0
        for i in sorted(idx, key=lambda k: -costs[k]):
            if group and used + costs[i] > budget:
                groups.append(group)
                group, used = [], 0
            group.append(i)
            used += costs[i]
        if group:
            groups.append(group)
    return groups


def plan(data: Any, budget: int | None) -> list[list[int]]:
    """Return a held-out pass's batches of `data` (`fs.ByLength`).

    The trainer's batches (`ByLength.plan`) without a `budget`, else
    blocks packed to it (`pack`).
    """
    return data.plan() if budget is None else pack(data, budget)


def eval_budget(device: str) -> int | None:
    """Return a held-out batch's series x tokens on `device`, or None.

    On CUDA the backend's pass budget (`backends.chronos2.pass_budget`),
    so a batch fills the device; elsewhere None, for the trainer's
    batches (`stream.ByLength`), as larger ones gain nothing on a CPU.
    """
    from nowcastfoundation.backends.chronos2 import pass_budget

    if not device.startswith("cuda"):
        return None
    return pass_budget(device)[1]


def _each_batch(
    model: Any,
    blocks: Sequence[dict[str, Any]],
    batch_size: int,
    use: Callable[[list[int], dict[str, Any], Any], None],
    budget: int | None = None,
    precision: str | None = None,
    before: Callable[[list[int]], None] | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    data: Any = None,
) -> None:
    """Run each batch of held-out blocks through the model once.

    Batches are `groups`, by default `plan`'s of `data` (the blocks'
    `stream.ByLength`, built if not given); a batch that runs out of
    memory is split in two. Each forward runs at the inference precision
    (`precision`, by default `PRECISION`;
    `backends.chronos2.inference_precision`), after `before(indices)`,
    and `use(indices, batch, output)` takes its result.
    """
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    from nowcastfoundation.backends.chronos2 import autocast

    if data is None:
        data = fs.ByLength(
            list(blocks),
            batch_size,
            DatasetMode.VALIDATION,
            series_bias.has_bias(model),
        )
    if groups is None:
        groups = plan(data, budget)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    device = str(next(model.parameters()).device)
    training = model.training
    model.eval()

    def run(indices: list[int]) -> None:
        batch = data._build_batch(indices)
        if before is not None:
            before(indices)
        try:
            with autocast(device, precision):
                got = fs.forward(model, batch, accepted)
        except torch.OutOfMemoryError:
            if len(indices) == 1:
                raise
            del batch
            fs.free_memory()
            half = len(indices) // 2
            run(indices[:half])
            run(indices[half:])
            return
        use(indices, batch, got)

    with torch.no_grad():
        for k, indices in enumerate(groups, start=1):
            run(list(indices))
            fs.report_batch(k, len(groups))
    model.train(training)


def held_out_pass(
    model: Any,
    blocks: Sequence[dict[str, Any]],
    batch_size: int,
    budget: int | None = None,
    precision: str | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    data: Any = None,
) -> list[dict[str, Any]]:
    """Return each block's row losses and interval hits, from one pass.

    Each batch (`_each_batch`; only `groups`, if given, whose blocks
    alone have entries) is run through the model once: its loss by
    row (`row_losses`, in fp32) over every step, the nowcast steps and
    the forecast steps, and its target rows' 50% and 90% interval hits
    (`_block_hits`). A block's entry holds `n`, its rows, `loss`,
    `nowcast` and `forecast`, by row, and `cover`. Row losses are means
    over rows, so the batching changes no result.
    """
    blocks = list(blocks)
    at = at_levels(model, INTERVAL_LEVELS)
    out: list[dict[str, Any]] = [{} for _ in blocks]
    with _RowLosses() as split:

        def before(indices: list[int]) -> None:
            split.steps = np.concatenate(
                [
                    np.full(
                        int(blocks[i]["context"].shape[0]),
                        fs.nowcast_steps(blocks[i]),
                    )
                    for i in indices
                ]
            )

        def use(indices: list[int], batch: dict[str, Any], got: Any) -> None:
            raw = got.quantile_preds.float().cpu().numpy()
            target = batch["future_target"].float().cpu().numpy()
            start = 0
            for i, (lo, hi) in zip(
                indices, batch["target_idx_ranges"], strict=True
            ):
                n = int(blocks[i]["context"].shape[0])
                if lo != start:
                    raise ValueError("a block's rows are not where expected")
                out[i] = {
                    "n": n,
                    **{k: v[start : start + n] for k, v in split.rows.items()},
                    "cover": _block_hits(
                        blocks[i], at(raw[lo:hi]), target[lo:hi]
                    ),
                }
                start += n

        _each_batch(
            model,
            blocks,
            batch_size,
            use,
            budget,
            precision,
            before,
            groups,
            data,
        )
    return out


_WORKER: dict[str, Any] = {}
"""A worker process's model, blocks and their batches (`_start_worker`)."""


def _start_worker(
    blocks: Path,
    n: int | None,
    need: Sequence[int],
    adapter: Path,
    device: str,
    batch_size: int,
    loader: Callable[[str], Any],
    counter: Any,
    n_workers: int,
    parent: int,
) -> None:
    """Set up a held-out pass's worker: its cores, model and blocks.

    Base Chronos-2 on `device` (`loader`) with `adapter` (and its series
    bias) merged, as `score` loads it, and the held-out blocks `need` of those
    at `blocks`, in the batches one process builds. It exits when its
    coordinator, process `parent`, does.
    """
    from chronos.chronos2.dataset import DatasetMode

    from nowcastfoundation.backends.chronos2 import apply_threads

    workers.watch_parent(parent)
    workers.pin_worker(counter, n_workers)
    apply_threads()
    val = load_blocks(blocks, n)
    rows = [val[i] for i in need]
    model = loader(device)
    series_bias.attach_for(model, adapter)
    model = merged(model, adapter)
    data = fs.ByLength(
        rows, batch_size, DatasetMode.VALIDATION, series_bias.has_bias(model)
    )
    _WORKER.update(model=model, rows=rows, data=data, batch_size=batch_size)


def _worker_batch(group: tuple[int, ...]) -> list[dict[str, Any]]:
    """Return a batch's blocks' `held_out_pass` entries, in a worker."""
    got = held_out_pass(
        _WORKER["model"],
        _WORKER["rows"],
        _WORKER["batch_size"],
        groups=[list(group)],
        data=_WORKER["data"],
    )
    return [got[i] for i in group]


def split_pass(
    split: Split,
    adapter: Path,
    need: Sequence[int],
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    """Return `held_out_pass` of `rows` over `split`'s worker processes.

    `rows` are the held-out blocks `need`, scored with `adapter` on
    `device`. The batches are those one process runs (`plan`), the
    costliest (`block_cost`) given out first, so the entries are the same
    however many workers run them, up to float rounding. Each batch is
    reported as it is done (`fs.report_batch`).
    """
    from chronos.chronos2.dataset import DatasetMode

    rows = list(rows)
    data = fs.ByLength(rows, batch_size, DatasetMode.VALIDATION)
    groups = plan(data, eval_budget(device))
    groups.sort(key=lambda g: -sum(block_cost(rows[i]) for i in g))
    out: list[dict[str, Any]] = [{} for _ in rows]
    tasks = [(tuple(g),) for g in groups]
    results = workers.pooled(
        _worker_batch,
        tasks,
        split.workers,
        _start_worker,
        (
            split.blocks,
            split.n,
            list(need),
            adapter,
            device,
            batch_size,
            split.loader or load_base,
        ),
    )
    for k, ((group,), got) in enumerate(results, start=1):
        for i, entry in zip(group, got, strict=True):
            out[i] = entry
        fs.report_batch(k, len(groups))
    return out


PRECISION_BLOCKS = 20
"""Held-out blocks `precision_check` scores at fp32 too."""

PRECISION_TOLERANCE = 0.005
"""The relative difference in held-out loss between half precision and
fp32 that `precision_check` accepts (0.5%)."""


def spread(blocks: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Return `n` blocks spread evenly over `blocks`, or all of them."""
    blocks = list(blocks)
    if n >= len(blocks):
        return blocks
    return [blocks[i * len(blocks) // n] for i in range(n)]


def precision_check(
    model: Any,
    blocks: Sequence[dict[str, Any]],
    batch_size: int,
    device: str,
) -> dict[str, Any] | None:
    """Return the held-out loss at half precision against fp32, or None.

    On `PRECISION_BLOCKS` blocks spread over the set, when the inference
    precision on `device` is not fp32; `rel` is their relative
    difference, which `PRECISION_TOLERANCE` bounds.
    """
    from nowcastfoundation.backends.chronos2 import inference_precision

    precision = inference_precision(device)
    if precision == "fp32":
        return None
    sample = spread(blocks, PRECISION_BLOCKS)
    half = _losses("loss", held_out_pass(model, sample, batch_size))["loss"]
    full = _losses(
        "loss", held_out_pass(model, sample, batch_size, precision="fp32")
    )["loss"]
    rel = abs(half - full) / full
    if rel > PRECISION_TOLERANCE:
        fs.log(
            f"held-out loss at {precision} is {rel:.2%} off fp32, over"
            f" {PRECISION_TOLERANCE:.1%}"
        )
    return {
        "precision": precision,
        "blocks": len(sample),
        "loss": half,
        "loss_fp32": full,
        "rel": rel,
    }


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
    at = at_levels(model, INTERVAL_LEVELS)
    rows: list[dict[str, Any]] = []
    for block, raw, target in _predictions(model, blocks, batch_size):
        rows += _block_hits(block, at(raw), target)
    return rows


def _block_hits(
    block: dict[str, Any], preds: np.ndarray, target: np.ndarray
) -> list[dict[str, Any]]:
    """Return a block's target rows' interval hits (`_covered`).

    `preds` are its rows' quantiles at `INTERVAL_LEVELS`, `target` their
    future values.
    """
    rows = []
    n_steps = int(block["prediction_length"])
    d = fs.nowcast_steps(block)
    names = block.get("levels", ())
    for r in range(len(preds)):
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


def at_levels(
    model: Any, wanted: Sequence[float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function giving the model's quantiles at levels `wanted`.

    From `[rows, Q, steps]` at the model's own levels to `[rows,
    len(wanted), steps]`: each the model's quantile at that level, linear
    between its two nearest levels where it has none.
    """
    levels = np.asarray(model.chronos_config.quantiles, dtype=float)
    upper = np.clip(np.searchsorted(levels, wanted), 1, len(levels) - 1)
    weight = np.clip(
        (np.asarray(wanted) - levels[upper - 1])
        / (levels[upper] - levels[upper - 1]),
        0.0,
        1.0,
    )[None, :, None]

    def at(raw: np.ndarray) -> np.ndarray:
        out: np.ndarray = (1 - weight) * raw[:, upper - 1] + weight * raw[
            :, upper
        ]
        return out

    return at


def _predictions(
    model: Any,
    blocks: Sequence[dict[str, Any]],
    batch_size: int,
    budget: int | None = None,
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    """Return each held-out block, its quantiles and its future targets.

    In the blocks' order, however they are batched (`_each_batch`). The
    quantiles are the model's own levels, `[rows, Q, steps]`, and the
    targets `[rows, steps]`, for the block's target rows.
    """
    blocks = list(blocks)
    out: list[Any] = [None] * len(blocks)

    def use(indices: list[int], batch: dict[str, Any], got: Any) -> None:
        raw = got.quantile_preds.float().cpu().numpy()
        target = batch["future_target"].float().cpu().numpy()
        for i, (lo, hi) in zip(
            indices, batch["target_idx_ranges"], strict=True
        ):
            out[i] = (blocks[i], raw[lo:hi], target[lo:hi])

    _each_batch(model, blocks, batch_size, use, budget)
    return out


def held_out_coverage(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> dict[str, float]:
    """Return the held-out 50% and 90% interval coverage of target steps.

    Over every step, then the nowcast and forecast steps, then by source
    and by level (`<prefix>cover50`, `<prefix>cover90`, `<prefix>n`).
    """
    return summarise_coverage(_covered(model, blocks, batch_size))


def summarise_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return the coverage of target rows' interval hits (`_covered`)."""
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


SEVEN_DAY_LEVELS = (0.05, 0.5, 0.95)
"""The levels the 7-day check reads: the 90% interval's ends and the
median."""

SEVEN_DAY_BLOCKS = 100
"""Held-out blocks with 7-day rows each checkpoint's 7-day check scores
(`seven_day_check`)."""


def seven_day_parts(
    block: dict[str, Any],
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Return a block's 7-day check, step by step, or None without 7-day rows.

    A `ROLLING` source's block has each place's daily row, then its
    rolling-sum (7-day) row. The daily medians' rolling sums (the
    context's last days included) are set against the 7-day row: their
    ratio (`ratio`) and whether they fall in its 90% interval (`inside`).
    `rough` and `rough_truth` are each place's mean absolute second
    difference of the log (plus one) of the 7-day median and of the
    truth: a 7-day median rougher than the truth carries a weekday cycle
    the 7-day sums do not have. Steps with no truth (past the horizon)
    are left out.

    Parameters
    ----------
    block : dict
        A held-out block, its target rows first.
    median, lower, upper : numpy.ndarray
        The model's median and 90% interval ends of each target row's
        future steps, `[targets, steps]`.
    """
    window = ROLLING.get(str(block.get("source", "")))
    if window is None:
        return None
    n = int(block["n_targets"]) // 2
    h = int(block["prediction_length"])
    context = np.asarray(block["context"], dtype=float)
    future = context.shape[1] - h
    truth = context[n : 2 * n, future:]
    keep = np.isfinite(truth).all(axis=0)
    past = np.nan_to_num(context[:n, future - (window - 1) : future])
    daily = np.concatenate([past, median[:n, :h]], axis=1)
    sums = rolling_totals(daily, window)[:, window - 1 :][:, keep]
    row = median[n : 2 * n, :h][:, keep]
    lo, hi = lower[n : 2 * n, :h][:, keep], upper[n : 2 * n, :h][:, keep]

    def rough(x: np.ndarray) -> np.ndarray:
        if x.shape[1] < 3:
            return np.full(x.shape[0], np.nan)
        diffs = np.diff(np.log1p(np.maximum(x, 0)), 2, axis=1)
        out: np.ndarray = np.abs(diffs).mean(axis=1)
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sums > 0, row / sums, np.nan)
    return {
        "ratio": ratio.ravel(),
        "inside": ((lo <= sums) & (sums <= hi)).ravel(),
        "rough": rough(row),
        "rough_truth": rough(truth[:, keep]),
    }


def summarise_seven_day(
    parts: Sequence[dict[str, np.ndarray] | None],
) -> dict[str, float]:
    """Return the 7-day check over blocks' steps and places.

    `ratio`, the median of the 7-day row over its daily sums;
    `inside90`, the share of daily sums in the row's 90% interval;
    `rough` and `rough_truth`, the mean roughness of the 7-day median
    and of the truth, and `rough_ratio` the first over the second (1 is
    as smooth as the truth); `n`, the steps compared. Empty with no
    7-day rows.
    """
    found = [p for p in parts if p is not None and len(p["inside"])]
    if not found:
        return {}

    def joined(key: str) -> np.ndarray:
        return np.concatenate([p[key] for p in found])

    rough, truth = (
        np.nanmean(joined("rough")),
        np.nanmean(joined("rough_truth")),
    )
    return {
        "ratio": float(np.nanmedian(joined("ratio"))),
        "inside90": float(joined("inside").mean()),
        "rough": float(rough),
        "rough_truth": float(truth),
        "rough_ratio": float(rough / truth) if truth > 0 else float("nan"),
        "n": len(joined("inside")),
    }


def seven_day_check(
    adapter: Path,
    base: Any,
    blocks: Sequence[dict[str, Any]],
    batch_size: int,
    most: int = SEVEN_DAY_BLOCKS,
) -> dict[str, float]:
    """Return a checkpoint's 7-day check (`seven_day_parts`) on its blocks.

    The first `most` held-out blocks with 7-day rows are predicted with
    `base` and `adapter`, as `evaluate` loads them; the levels are read
    as `at_levels` gives them.
    """
    rolling = [b for b in blocks if b.get("source") in ROLLING][:most]
    if not rolling:
        return {}
    model = merged(base, adapter)
    at = at_levels(model, SEVEN_DAY_LEVELS)
    parts = []
    budget = eval_budget(str(next(model.parameters()).device))
    for block, raw, _ in _predictions(model, rolling, batch_size, budget):
        q = at(raw)
        parts.append(seven_day_parts(block, q[:, 1], q[:, 0], q[:, 2]))
    return summarise_seven_day(parts)


VALIDATION_LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""The levels a validation forecast keeps, the dashboard's seven."""

VALIDATION_BLOCKS = 20
"""Held-out blocks in a run's validation forecasts (`validation_sample`)."""

VALIDATION_ROWS = 12
"""Target rows a validation forecast keeps of a block, national first."""

VALIDATION_TAIL = 28
"""Context steps before the horizon a validation forecast keeps."""


def validation_sample(
    n_blocks: int, most: int = VALIDATION_BLOCKS
) -> list[int]:
    """Return the held-out blocks a run's validation forecasts show.

    `most` indices spread evenly over the blocks, the first included;
    the same for every checkpoint of a run.
    """
    if n_blocks <= most:
        return list(range(n_blocks))
    return sorted({int(i * n_blocks / most) for i in range(most)})


def _values(x: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in x]


def forecast_record(
    index: int,
    block: dict[str, Any],
    q: np.ndarray,
    target: np.ndarray,
    tail: int = VALIDATION_TAIL,
) -> dict[str, Any]:
    """Return one held-out block's forecast, for the dashboard.

    `q` is the block's quantiles at `VALIDATION_LEVELS`, `[rows, 7,
    steps]`, and `target` its truth, `[rows, steps]`. Each kept row
    (`VALIDATION_ROWS`, national rows first) has its name and level, its
    last `tail` context steps before the horizon, its truth and its
    quantiles over the block's prediction length.
    """
    h = int(block["prediction_length"])
    n = int(block["n_targets"])
    context = np.asarray(block["context"], dtype=float)
    past = context[:n, context.shape[1] - h - tail : context.shape[1] - h]
    levels = list(block.get("levels", ()))
    places = list(block.get("places", ()))
    targets = list(block.get("target_places", range(n)))

    def name(r: int) -> str:
        i = targets[r] if r < len(targets) else r
        return str(places[i]) if i < len(places) else f"row {r}"

    def level(r: int) -> str:
        return str(levels[r]) if r < len(levels) else ""

    order = sorted(range(n), key=lambda r: level(r) != "national")
    return {
        "block": int(index),
        "source": str(block.get("source", "")),
        "cut": block.get("cut_date"),
        "mix": str(block.get("mix", "")),
        "nowcast": int(block.get("nowcast", 0)),
        "rows": [
            {
                "name": name(r),
                "level": level(r),
                "history": _values(past[r]),
                "truth": _values(target[r, :h]),
                "q": [_values(q[r, k, :h]) for k in range(q.shape[1])],
            }
            for r in order[:VALIDATION_ROWS]
        ],
    }


def validation_forecasts(
    model: Any, blocks: Sequence[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    """Return `model`'s forecasts of the run's sampled held-out blocks.

    The blocks `validation_sample` picks, each as `forecast_record`
    gives it. `model` is base Chronos-2, with an adapter for a
    checkpoint.
    """
    picks = validation_sample(len(blocks))
    chosen = [blocks[i] for i in picks]
    at = at_levels(model, VALIDATION_LEVELS)
    out = []
    for i, (block, raw, target) in zip(
        picks,
        _predictions(
            model,
            chosen,
            batch_size,
            eval_budget(str(next(model.parameters()).device)),
        ),
        strict=True,
    ):
        out.append(forecast_record(i, block, at(raw), target))
    return out


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
    split: Split | None = None,
) -> tuple[dict[str, Any], float]:
    """Return the losses `kinds` ask for of `adapter`, and the seconds.

    Scores a copy of base Chronos-2 on `device` with the adapter (and its
    series bias) attached, then frees it; with `split`, over its worker
    processes.
    """
    model = copy.deepcopy(base).to(device)
    series_bias.attach_for(model, adapter)
    tick = time.perf_counter()
    losses = evaluate(adapter, model, blocks, kinds, batch_size, split)
    del model
    fs.free_memory()
    return losses, time.perf_counter() - tick


def snapshot_result(
    base: Any,
    snapshot: Path,
    blocks: list[dict[str, Any]],
    device: str,
    split: Split | None = None,
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
        split,
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
    split: Split | None = None,
) -> dict[str, Any]:
    """Score checkpoint `where` on the held-out blocks; write its sidecar.

    The sidecar (`sidecar.SIDECAR`, beside its settings) holds the
    checkpoint's held-out losses and base Chronos-2's (`sidecar.start_loss`).
    With `split`, the pass runs over its worker processes.

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
        split,
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
