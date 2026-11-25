"""Each job's memory, from what it used last time.

A rule's `mem_mb` is `from_benchmark(<its benchmark path>, <minimum>)`:
the peak RSS (`max_rss`) of the job's own last Snakemake benchmark, with
`MARGIN` on top, or, before it has one, the most any job of the rule
alike in all but the `like` wildcards used, or else the rule's minimum.
Each retry after a job runs out of memory doubles it (the profiles'
`retries`). So the scheduler, on one machine or a cluster, holds each job
to what it needs, and no machine's figures are written in the config.
"""

from __future__ import annotations

import csv
import glob
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

MARGIN = 1.25
"""The share over a job's last peak its memory takes."""

STEP = 256
"""The MB a job's memory is rounded up to."""


def peak_mb(path: Path | str) -> float | None:
    """Return a benchmark's last `max_rss` (MB), or None."""
    try:
        with Path(path).open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return None
    if not rows:
        return None
    try:
        return float(rows[-1]["max_rss"])
    except (KeyError, TypeError, ValueError):
        return None


def padded(peak: float) -> int:
    """Return `peak` (MB) with `MARGIN`, rounded up to `STEP`."""
    return int(math.ceil(peak * MARGIN / STEP) * STEP)


def _fields(wildcards: Any) -> dict[str, str]:
    """Return a job's wildcards by name."""
    if hasattr(wildcards, "items"):
        return {k: str(v) for k, v in wildcards.items()}
    return {k: str(v) for k, v in vars(wildcards).items()}


def from_benchmark(
    pattern: str,
    minimum: int,
    like: Iterable[str] | Iterable[Iterable[str]] = (),
) -> Callable[..., int]:
    """Return a rule's `mem_mb`: its job's last peak, or the rule's.

    Parameters
    ----------
    pattern : str
        The rule's benchmark path, with its wildcards.
    minimum : int
        The least the rule's jobs take (MB).
    like : iterable of str, or of iterables of str
        The wildcards a job's peak may be taken across before it has its
        own (another checkpoint, another code key); several sets are
        tried in order, the first with a peak taken.

    Returns
    -------
    callable
        `mem_mb(wildcards, attempt)`, doubling at each attempt.
    """
    given = list(like)
    sets = (
        [tuple(given)]
        if all(isinstance(x, str) for x in given)
        else [tuple(x) for x in given]
    )

    def mem_mb(wildcards: Any, attempt: int = 1) -> int:
        fields = _fields(wildcards)
        own = peak_mb(pattern.format(**fields))
        peaks = [own] if own is not None else []
        for names in sets:
            if peaks or not names:
                break
            starred = {k: ("*" if k in names else v) for k, v in fields.items()}
            found = (
                peak_mb(p)
                for p in glob.glob(pattern.format(**starred))  # noqa: PTH207
            )
            peaks = [p for p in found if p is not None]
        base = max([minimum, *(padded(p) for p in peaks)])
        return base * 2 ** (max(1, attempt) - 1)

    return mem_mb
