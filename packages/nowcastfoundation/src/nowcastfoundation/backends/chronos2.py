# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""The Chronos-2 backend: device, precision and batched quantiles.

`_predict` gives groups of series (`nowcastfoundation.backends.Series`)
to a Chronos-2 pipeline in forward passes packed to the device's memory
(`pass_budget`), with the series bias's places and parents where the
model has it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from nowcastfoundation import series_bias
from nowcastfoundation.backends import Series

__all__ = [
    "CHRONOS_KEYS",
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "PRECISIONS",
    "apply_threads",
    "chronos_batch",
    "inference_precision",
    "torch_device",
]

CHRONOS_MODEL: Final = "amazon/chronos-2"
"""The Hugging Face checkpoint."""

CHRONOS_REVISION: Final = "29ec3766d36d6f73f0696f85560a422f50e8498c"
"""The checkpoint's commit, pinned so runs are reproducible."""


def torch_device(device: str | None = None) -> str:
    """Return the torch device Chronos-2 runs on, to train or predict.

    `device`, by default `DEVICE`: the workflow config names it
    (`forecast.device` for forecasting jobs, the trainer's `DEVICE`), and
    a caller outside the workflow passes one or sets `DEVICE`. It is
    never detected here.

    Raises
    ------
    ValueError
        If no device is given or set, or it is `auto`.
    """
    device = device or os.environ.get("DEVICE")
    if not device:
        raise ValueError(
            "no torch device: set DEVICE (cpu, cuda or mps; the workflow's "
            "`forecast.device`) or pass one"
        )
    if device == "auto":
        raise ValueError(
            "device auto is not allowed: name cpu, cuda or mps (the "
            "workflow config's `forecast.device`)"
        )
    return device


PRECISIONS: Final = ("fp32", "fp16", "bf16", "auto")
"""Inference precisions (`PRECISION`). Training always runs in fp32."""


def inference_precision(device: str, precision: str | None = None) -> str:
    """Return the precision Chronos-2 forecasts in on `device`.

    `precision`, by default `PRECISION`, `auto` if unset, so every
    forecast takes its device's: fp32 on a CPU, where stored fits and
    submissions run. `auto` takes each device's
    half precision: fp16 on CUDA GPUs below compute capability 8 (fp16
    tensor cores, no bf16), bf16 on compute capability 8 and above and on
    Apple GPUs, fp32 on the CPU (bf16 ran about 5 times slower on a CPU
    without native bf16). fp16 moved relative WIS by under 0.06% on a GPU
    without bf16.

    Raises
    ------
    ValueError
        If the precision is not one of `PRECISIONS`.
    """
    precision = precision or os.environ.get("PRECISION") or "auto"
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r} is not one of {PRECISIONS}")
    if precision != "auto":
        return precision
    kind = device.split(":")[0]
    if kind == "cuda":
        import torch

        major, _ = torch.cuda.get_device_capability(device)
        return "bf16" if major >= 8 else "fp16"
    return "bf16" if kind == "mps" else "fp32"


def autocast(device: str, precision: str | None = None) -> Any:
    """Return the context Chronos-2 forecasts in.

    Autocast at half precision (`inference_precision` of `precision`), or
    none at fp32.
    """
    precision = inference_precision(device, precision)
    if precision == "fp32":
        return contextlib.nullcontext()
    import torch

    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def apply_threads() -> None:
    """Set torch's CPU threads to `TORCH_THREADS`, if set (the rule's)."""
    n = os.environ.get("TORCH_THREADS", "").strip()
    if n:
        import torch

        torch.set_num_threads(int(n))


def chronos_batch() -> int:
    """Return the series in one Chronos-2 forward (`CHRONOS_BATCH`).

    256 by default, Chronos-2's own, which fits an 8 GB GPU.
    """
    return int(os.environ.get("CHRONOS_BATCH") or 256)


CHRONOS_KEYS: Final = ("target", "past_covariates", "future_covariates")
"""The parts of a group Chronos-2 reads; the rest name its rows' places."""


def _chronos_input(group: Series) -> dict[str, Any]:
    """Return the parts of a group Chronos-2 reads (`CHRONOS_KEYS`)."""
    return {k: group[k] for k in CHRONOS_KEYS if k in group}


def _group_rows(group: Series) -> np.ndarray:
    """Return each of a group's rows' place and parent, `[2, rows]`.

    In Chronos-2's row order: the targets, then the covariates by sorted
    name, past-only ones before known-future ones. Places are numbered in
    the group's `places` order; a target row's place is its
    `target_places` entry, one row per place by default.

    Raises
    ------
    ValueError
        If a covariate names no place, or the target rows do not match
        their places.
    """
    past = list(group.get("past_covariates", {}))
    known = set(group.get("future_covariates", {}))
    n_targets = np.atleast_2d(group["target"]).shape[0]
    named = dict.fromkeys(n.rsplit("_lag_", 1)[0] for n in past)
    places = list(group.get("places") or named)
    target_places = list(group.get("target_places", range(len(places))))
    if len(target_places) != n_targets:
        raise ValueError(
            f"{len(target_places)} target places for {n_targets} targets"
        )
    rows = sorted(n for n in past if n not in known)
    rows += sorted(n for n in past if n in known)
    at = {p: i for i, p in enumerate(places)}
    try:
        owner = [at[n.rsplit("_lag_", 1)[0]] for n in rows]
    except KeyError as err:
        raise ValueError(f"covariate of no target place: {err}") from err
    place = np.concatenate([target_places, owner]).astype(int)
    parent = np.asarray(series_bias.parents_of(places), dtype=int)
    return np.stack([place, parent[place]])


def _schema(group: Series) -> tuple[Any, ...]:
    """Return what groups predicted in one pass must share.

    Chronos-2 takes one schema a call: the same number of targets and the
    same covariate names.
    """
    return (
        np.atleast_2d(group["target"]).shape[0],
        tuple(sorted(group.get("past_covariates", {}))),
        tuple(sorted(group.get("future_covariates", {}))),
    )


def _n_rows(group: Series) -> int:
    """Return a group's rows in Chronos-2's batch: targets and covariates."""
    n = np.atleast_2d(group["target"]).shape[0]
    return int(n + len(group.get("past_covariates", {})))


INFERENCE_BYTES: Final = 60_000
"""GPU memory a Chronos-2 forward pass holds per series and token, with no
gradients: a twelfth of a training step's (finetune.stream's
`ACTIVATION_BYTES`), as inference keeps one layer's activations, not
every layer's. A pass that still runs out is split (`_predict`)."""

MEMORY_MARGIN: Final = 0.1
"""Share of the device's memory a pass leaves free, for the allocator's
fragmentation."""


def device_memory() -> tuple[int, int] | None:
    """Return the CUDA device's free and total bytes, or None off CUDA.

    Free counts what PyTorch caches but does not use, as a pass can take
    it.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free + cached, total


def pass_budget(device: str) -> tuple[int | None, int | None]:
    """Return a forward pass's most rows, or most series x tokens.

    `CHRONOS_BATCH` rows where a machine sets it; on CUDA the free memory
    less `MEMORY_MARGIN` of the total over `INFERENCE_BYTES`, so a pass
    fills the device; elsewhere `chronos_batch()` rows, as larger passes
    gain nothing on a CPU.
    """
    if os.environ.get("CHRONOS_BATCH") or not device.startswith("cuda"):
        return chronos_batch(), None
    memory = device_memory()
    if memory is None:
        return chronos_batch(), None
    free, total = memory
    spare = free - int(MEMORY_MARGIN * total)
    return None, max(1, spare // INFERENCE_BYTES)


OUTPUT_PATCH: Final = 16
"""Steps in one of Chronos-2's input and output patches."""


def group_cost(group: Series, horizon: int) -> int:
    """Return a group's series x tokens in a forward pass.

    Its rows over its context patches, the reg token and its output
    patches.
    """
    width = int(np.atleast_2d(group["target"]).shape[-1])
    tokens = -(-width // OUTPUT_PATCH) + 1 + -(-horizon // OUTPUT_PATCH)
    return _n_rows(group) * tokens


def _passes(
    groups: Sequence[Series],
    rows: int | None,
    tokens: int | None = None,
    horizon: int = 0,
) -> list[list[int]]:
    """Return the groups of each forward pass, by index.

    Groups of one schema share passes of at most `rows` rows, or, with
    `tokens`, of at most that many series x tokens (`group_cost`), in
    order; a group over the budget is a pass of its own.
    """
    by: dict[tuple[Any, ...], list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(_schema(g), []).append(i)
    out: list[list[int]] = []
    for idx in by.values():
        chunk: list[int] = []
        used = 0
        for i in idx:
            n = (
                group_cost(groups[i], horizon)
                if tokens is not None
                else _n_rows(groups[i])
            )
            budget = tokens if tokens is not None else rows or 1
            if chunk and used + n > budget:
                out.append(chunk)
                chunk, used = [], 0
            chunk.append(i)
            used += n
        out.append(chunk)
    return out


def _predict(
    pipeline: Any, groups: Sequence[Series], horizon: int, levels: np.ndarray
) -> list[Any]:
    """Return Chronos-2's quantiles at `levels` for each group.

    Groups of one schema share forward passes packed to the device
    (`pass_budget`, `_passes`), each pass exactly one of Chronos-2's
    batches; a pass that runs out of memory is split in two. With the
    series bias each pass carries its groups' rows' places and parents,
    kept apart (`series_bias.batch_rows`), as in held-out losses.
    """
    import torch

    model = getattr(pipeline, "model", None)
    bias = model is not None and series_bias.has_bias(model)
    device = torch_device()
    out: list[Any] = [None] * len(groups)

    def run(chunk: list[int]) -> None:
        part = [groups[i] for i in chunk]
        rows = (
            series_bias.batch_rows([_group_rows(g) for g in part])
            if bias
            else None
        )
        try:
            with (
                (
                    series_bias.assigned(model, rows)
                    if bias
                    else contextlib.nullcontext()
                ),
                autocast(device),
            ):
                quantiles, _ = pipeline.predict_quantiles(
                    [_chronos_input(g) for g in part],
                    batch_size=sum(_n_rows(g) for g in part),
                    prediction_length=horizon,
                    quantile_levels=levels.tolist(),
                )
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            half = len(chunk) // 2
            run(chunk[:half])
            run(chunk[half:])
            return
        for i, q in zip(chunk, quantiles, strict=True):
            out[i] = q

    rows, tokens = pass_budget(device)
    for chunk in _passes(groups, rows, tokens, horizon):
        run(chunk)
    return out
