"""The LoRA update, from the adapters the trainer writes at each checkpoint.

Each adapted module's update is `(alpha / r) B @ A`. From it: its size
(Frobenius norm) by encoder layer and kind of module, by projection and
for the output layer, over checkpoints; how much it moved since the
previous checkpoint; and at the last checkpoint its effective rank and
the sizes of A and B. page/lora.js draws them as one grid of equal
panels. No scoring: only the trainer's own files are read.
"""

from __future__ import annotations

import functools
import json
import math
import re
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flowlib.dashboard import logs
from flowlib.dashboard.runs import Run
from nowcasteval import labels

LORA = re.compile(r"\.lora_([AB])\.weight$")
"""A LoRA matrix, by side."""

BLOCK = re.compile(r"encoder\.block\.(\d+)\.layer\.(\d+)\..*?\.(\w+)$")
"""An encoder block's sublayer and projection: 0 time, 1 group attention."""

KINDS = {"0": "time attention", "1": "group attention"}
"""Encoder sublayers by index (`Chronos2EncoderBlock.layer`)."""

OUTPUT = "output layer"
PROJECTIONS = ("q", "k", "v", "o")
"""The attention projections LoRA adapts, in the panel's order."""

COLOURS = {
    "time attention": labels.CATEGORICAL[0],
    "group attention": labels.CATEGORICAL[1],
    OUTPUT: labels.CATEGORICAL[2],
    **{p: labels.CATEGORICAL[3 + i] for i, p in enumerate(PROJECTIONS)},
}
"""Kinds then projections, in the categorical order, so no two share."""

EXPLAINERS = (
    "lorasize",
    "loraproj",
    "lorachange",
    "lorarank",
    "loraab",
)
"""The explainer keys (`page/explain.json`) the panels use."""

ASSET = Path(__file__).with_name("page") / "lora.js"


@functools.cache
def _script() -> str:
    """Return page/lora.js, read once a process (as `page._asset`)."""
    return ASSET.read_text()


def _matrices(path: Path) -> dict[str, tuple[list[int], array]]:
    """Return a safetensors file's float32 LoRA matrices."""
    out = {}
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        for name, info in head.items():
            if name == "__metadata__" or not LORA.search(name):
                continue
            if info.get("dtype") != "F32":
                continue
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            out[name] = (info["shape"], array("f", f.read(b - a)))
    return out


def _gram(shape: list[int], values: array, rows: bool) -> np.ndarray:
    """Return `M M^T` (`rows`) or `M^T M` of a rank-r LoRA matrix.

    A is r x in, so A A^T is r x r; B is out x r, so B^T B is r x r.
    """
    mat = _mat(shape, values)
    return mat @ mat.T if rows else mat.T @ mat


def _mat(shape: list[int], values: array) -> np.ndarray:
    return np.frombuffer(values, dtype=np.float32).reshape(shape).astype(float)


def _factors(ckpt: Path) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Return each module's scale (alpha / r), B and A, by its name."""
    config = json.loads((ckpt / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    mats = _matrices(ckpt / "adapter_model.safetensors")
    out = {}
    for name, (shape, values) in mats.items():
        other = mats.get(name.replace("lora_A", "lora_B"))
        if "lora_A" not in name or other is None:
            continue
        out[name.replace(".lora_A.weight", "")] = (
            scale,
            _mat(*other),
            _mat(shape, values),
        )
    return out


def effective_rank(values: Sequence[float]) -> float:
    """Return the effective rank of singular values.

    The exponential of the entropy of their squares' shares: 1 for one
    direction, r for r equal ones, 0 for no update.
    """
    sq = np.asarray(values, dtype=float) ** 2
    total = sq.sum()
    if total <= 0:
        return 0.0
    p = sq[sq > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


@dataclass(frozen=True)
class Module:
    """One adapted module's update at one checkpoint."""

    kind: str
    layer: int | None
    proj: str
    size: float
    """Frobenius norm of the scaled B A."""
    a: float
    """Frobenius norm of A."""
    b: float
    """Frobenius norm of B."""
    rank: float
    """Effective rank of the scaled B A (`effective_rank`)."""


def _where(name: str) -> tuple[str, int | None, str] | None:
    """Return a module's kind, encoder layer and projection, if shown."""
    match = BLOCK.search(name)
    if match:
        kind = KINDS.get(match.group(2))
        return (kind, int(match.group(1)), match.group(3)) if kind else None
    return (OUTPUT, None, OUTPUT) if name.endswith("output_layer") else None


def _module(where, scale: float, b: np.ndarray, a: np.ndarray) -> Module:
    # B A = Qb (Rb Ra^T) Qa^T: its singular values are Rb Ra^T's (r x r).
    rb = np.linalg.qr(b, mode="r")
    ra = np.linalg.qr(a.T, mode="r")
    sv = scale * np.linalg.svd(rb @ ra.T, compute_uv=False)
    return Module(
        *where,
        size=float(math.sqrt(max(float((sv**2).sum()), 0.0))),
        a=float(np.linalg.norm(a)),
        b=float(np.linalg.norm(b)),
        rank=effective_rank(sv),
    )


_SEEN: dict[tuple[str, float], dict[str, Module]] = {}
_MOVED: dict[tuple[str, float, str, float], dict[str, float]] = {}


def _key(ckpt: Path) -> tuple[str, float]:
    return str(ckpt), (ckpt / "adapter_model.safetensors").stat().st_mtime


def modules(ckpt: Path | str) -> dict[str, Module]:
    """Return each shown module's update at a checkpoint, by name."""
    ckpt = Path(ckpt)
    key = _key(ckpt)
    if key not in _SEEN:
        out = {}
        for name, (scale, b, a) in _factors(ckpt).items():
            where = _where(name)
            if where:
                out[name] = _module(where, scale, b, a)
        _SEEN[key] = out
    return _SEEN[key]


def changes(old: Path | str, new: Path | str) -> dict[str, float]:
    """Return each module's squared change of update between checkpoints.

    `||P1 - P0||^2 = ||P1||^2 + ||P0||^2 - 2 <P1, P0>`, with
    `<B1 A1, B0 A0> = sum((B1^T B0) * (A1 A0^T))`, all r x r.
    """
    old, new = Path(old), Path(new)
    key = (*_key(old), *_key(new))
    if key not in _MOVED:
        before, after = _factors(old), _factors(new)
        out = {}
        for name, (s1, b1, a1) in after.items():
            if name not in before:
                continue
            s0, b0, a0 = before[name]
            b1, b0 = s1 * b1, s0 * b0
            n1 = float(((b1.T @ b1) * (a1 @ a1.T)).sum())
            n0 = float(((b0.T @ b0) * (a0 @ a0.T)).sum())
            cross = float(((b1.T @ b0) * (a1 @ a0.T)).sum())
            out[name] = max(n1 + n0 - 2 * cross, 0.0)
        _MOVED[key] = out
    return _MOVED[key]


def _adapters(run: Run) -> list[tuple[int, Path]]:
    """Return each saved checkpoint's simulations and adapter, by size."""
    if run.folder is None:
        return []
    out = []
    for name, settings, _ in logs.checkpoints(run.folder):
        ckpt = Path(run.folder) / "train" / name / "finetuned-ckpt"
        if (ckpt / "adapter_model.safetensors").is_file():
            n = int(settings.get("n_outbreaks") or logs._size(name))
            out.append((n, ckpt))
    return sorted(out, key=lambda x: x[0])


def _by_layer(mods: Mapping[str, Module], kind: str, n: int) -> list[list[str]]:
    """Return each of `n` layers' module names of one kind."""
    out: list[list[str]] = [[] for _ in range(n)]
    for name, m in mods.items():
        if m.kind == kind and m.layer is not None:
            out[m.layer].append(name)
    return out


def _quad(xs: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in xs))


def run_payload(run: Run) -> dict[str, Any] | None:
    """Return one run's LoRA views, or None without a saved adapter."""
    ckpts = _adapters(run)
    if not ckpts:
        return None
    mods = [modules(c) for _, c in ckpts]
    every = {name: m for ms in mods for name, m in ms.items()}
    layers = 1 + max(
        (m.layer for m in every.values() if m.layer is not None), default=-1
    )
    kinds = list(KINDS.values())
    size: dict[str, list] = {k: [] for k in kinds}
    change: dict[str, list] = {k: [] for k in kinds}
    proj: dict[str, list] = {p: [] for p in PROJECTIONS}
    output = []
    for i, ms in enumerate(mods):
        sizes = {name: m.size for name, m in ms.items()}
        moved = changes(ckpts[i - 1][1], ckpts[i][1]) if i else None
        for kind in kinds:
            groups = _by_layer(ms, kind, layers)
            now = [_quad([sizes[x] for x in g]) for g in groups]
            size[kind].append(now)
            if moved is None:
                change[kind].append(None)
                continue
            # The change over the layer's update now: 1 is as large.
            change[kind].append(
                [
                    math.sqrt(sum(moved.get(x, 0.0) for x in g)) / s
                    if g and s > 0
                    else None
                    for g, s in zip(groups, now, strict=True)
                ]
            )
        for p in PROJECTIONS:
            xs = [m.size for m in ms.values() if m.proj == p]
            proj[p].append(_quad(xs) / math.sqrt(len(xs)) if xs else None)
        out_layer = [m.size for m in ms.values() if m.kind == OUTPUT]
        output.append(out_layer[0] if out_layer else None)
    last = mods[-1]
    rank, a, b = {}, {}, {}
    for kind in kinds:
        groups = _by_layer(last, kind, layers)
        rank[kind] = [
            float(np.mean([last[x].rank for x in g])) if g else None
            for g in groups
        ]
        a[kind] = [_quad([last[x].a for x in g]) if g else None for g in groups]
        b[kind] = [_quad([last[x].b for x in g]) if g else None for g in groups]
    return {
        "key": run.key,
        "label": run.label,
        "color": run.colour,
        "n": [n for n, _ in ckpts],
        "layers": layers,
        "size": size,
        "output": output,
        "proj": proj,
        "change": change,
        "rank": rank,
        "a": a,
        "b": b,
    }


def payload(run_list: Sequence[Run]) -> dict[str, Any]:
    """Return every run's LoRA views, the runs with none and the colours."""
    out, missing = [], []
    for r in run_list:
        got = run_payload(r)
        if got is None:
            missing.append({"key": r.key, "label": r.label})
        else:
            out.append(got)
    return {"runs": out, "missing": missing, "col": COLOURS}


def section(
    run_list: Sequence[Run], tip: Callable[[str], str] = lambda _: ""
) -> str:
    """Return the LoRA section: one grid of equal panels (page/lora.js).

    `tip` gives the info marker of an explainer key (`page.tip`).
    """
    data = json.dumps(payload(run_list), separators=(",", ":"))
    tips = json.dumps({k: tip(k) for k in EXPLAINERS})
    return (
        "<div id=lorahead class=biashead></div>"
        "<div class=biasgrid id=loragrid></div>"
        f'<script type="application/json" id=lodata>{data}</script>'
        f'<script type="application/json" id=lotips>{tips}</script>'
        f"<script>{_script()}</script>"
    )
