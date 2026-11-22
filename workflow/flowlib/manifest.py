"""A round's manifest, one row per batch.

Each batch gets its row in a file of its own, and the round's
`manifest.csv` joins them in seed order. The columns must match the
manifests of rounds already simulated, and a row a manifest already
holds is kept as it is.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

HEADER = (
    "round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,"
    "settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished"
)


def _sha(path: Path, optional: bool = False) -> str:
    """Return a file's SHA-256; empty for a missing `optional` file."""
    if optional and not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last(log: Path, name: str) -> str:
    """Return the last `name` value in a log, empty if it records none."""
    found = re.findall(rf"{name} ([0-9.]+)", log.read_text())
    return found[-1] if found else ""


def row(
    sims: Path | str,
    *,
    round_: int,
    seed: int,
    kind: str,
    grid: str,
    commit: str,
    draws: int,
    batch: int,
    host: str,
    finished: str,
) -> str:
    """Return a batch's manifest row (no newline).

    The memory and time are the last `/usr/bin/time` record in the
    batch's log, so a retried batch gives its last attempt's. Only the
    failures file may be missing; a missing counts, settings or log file
    raises `FileNotFoundError`.
    """
    sims = Path(sims)
    log = sims / "logs" / f"{seed}.log"
    fields = [
        round_,
        seed,
        kind,
        grid,
        commit,
        draws,
        batch,
        _sha(sims / f"counts-{seed}.csv.gz"),
        _sha(sims / f"settings-{seed}.csv"),
        _sha(sims / f"failures-{seed}.csv", optional=True),
        _last(log, "peak_rss_kb"),
        _last(log, "wall_s"),
        host,
        finished,
    ]
    return ",".join(str(f) for f in fields)


def existing(manifest: Path | str, round_: int, seed: int) -> str | None:
    """Return the row a manifest already holds for a batch, if any."""
    manifest = Path(manifest)
    if not manifest.is_file():
        return None
    start = f"{round_},{seed},"
    for line in manifest.read_text().splitlines():
        if line.startswith(start):
            return line
    return None


def join(rows: Iterable[Path | str], out: Path | str) -> None:
    """Write the batches' row files as one manifest, in seed order."""
    lines = [Path(r).read_text().strip() for r in rows]
    lines.sort(key=lambda x: int(x.split(",")[1]))
    Path(out).write_text("\n".join([HEADER, *lines]) + "\n")


def seeds(manifest: Path | str) -> set[int]:
    """Return the batches a round's manifest lists (none if it is absent).

    A host appends a batch's row only once its files are in place, so a
    listed batch can be copied.
    """
    path = Path(manifest)
    if not path.exists():
        return set()
    rows = path.read_text().splitlines()[1:]
    return {int(r.split(",")[1]) for r in rows if r.strip()}


def stamp() -> str:
    """Return the time now in UTC, as the manifests write it."""
    return f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}"


def commit() -> str:
    """Return the commit the workflow runs at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def new_row(
    sims: Path | str,
    *,
    round_: int,
    seed: int,
    grid: str,
    draws: int,
    batch: int,
    host: str,
) -> str:
    """Return a grid block batch's row, at this commit and time."""
    return row(
        sims, round_=round_, seed=seed, kind="block", grid=grid,
        commit=commit(), draws=draws, batch=batch, host=host,
        finished=stamp(),
    )  # fmt: skip


def record(
    out: Path | str,
    old: Path | str,
    sims: Path | str,
    *,
    round_: int,
    seed: int,
    mine: bool,
    grid: str,
    draws: int,
    batch: int,
    host: str,
) -> None:
    """Write a batch's row to `out`: the row manifest `old` holds, or a new one.

    A batch simulated on another host (not `mine`) must arrive with its
    row.
    """
    line = existing(old, round_, seed)
    if line is None and not mine:
        raise ValueError(f"batch {seed} arrived without its manifest row")
    if line is None:
        line = new_row(
            sims, round_=round_, seed=seed, grid=grid, draws=draws,
            batch=batch, host=host,
        )  # fmt: skip
    Path(out).write_text(f"{line}\n")


def write_seeds(seeds: Iterable[int], path: Path | str) -> None:
    """Write seeds one a line."""
    Path(path).write_text("".join(f"{s}\n" for s in seeds))


def mark_ready(
    seeds: Iterable[int], seeds_file: Path | str, ready_file: Path | str
) -> None:
    """Write a round's seed list, then READY with the time."""
    write_seeds(seeds, seeds_file)
    Path(ready_file).write_text(f"{stamp()}\n")
