"""A training run's state: its log, checkpoints and trainer status.

A run's folder is `finetune-stream`'s output folder: its logs,
`train/n<N>.log` per segment (a line per logged step),
`train/n<N>/settings.json` per checkpoint and the stream status
(`train/stream-status.json`). Everything here only reads.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from nowcastfoundation.sidecar import SIDECAR, with_sidecar

SEVEN_DAY = "seven-day.json"
"""A checkpoint's 7-day check, beside its settings
(workflow/scripts/evaluate_offloaded.py `SEVEN_DAY`)."""

STEP = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] step (\d+): "
    r"loss ([\d.eE+-]+), held-out \([^)]*\) "
    r"(?:([\d.eE+-]+)|left to the evaluator)"
    r"(?: \(nowcast ([\d.eE+-]+), forecast ([\d.eE+-]+)\))?"
    r"(?:, by source ((?:[\w+]+ [\d.eE+-]+(?:, |$))+))?"
)
"""A logged step of the trainer: time, step and loss, then its held-out
part, which the dashboard does not read (validation is the full set at
each checkpoint, flowlib.evaluations). The bias relations' means and
ranges after it are read with `BIAS`."""

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[\w+\] (.*)$")
"""Any line of the trainer: its time and message."""

NUM = r"[+-]?[\d.]+(?:e[+-]?\d+)?"
BIAS = re.compile(rf"(\w+?)_? ({NUM}) \[({NUM}), ({NUM})\]")
"""One bias relation's mean and range in a step's `bias` list."""

CHUNK = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[train\] outbreaks (\d+)-(\d+): "
    r"\d+ blocks, \d+ series, (\d+) steps"
)
"""A chunk the trainer plans: its first and last simulation and steps."""

CHUNK_TIME = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] chunk time: "
    r"wait ([\d.]+) s, cut ([\d.]+) s, train ([\d.]+) s, eval ([\d.]+) s"
)
"""A chunk's time when it ends: waiting for batches, cutting blocks,
training and evaluating, in seconds."""

RESUME = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[train\] resumed at step (\d+), "
    r"outbreaks (\d+)-, batch (\d+)"
)
"""Where a (re)started trainer picks up: the step, the first simulation
of the chunk it is in and the steps of that chunk already done."""

SAVED = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] checkpoint (n\d+): hash"
)
"""A checkpoint the trainer saves: its time and name."""

PUBLISHED = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] published "
)
"""A checkpoint's release (`n<N>/publish.log`): its time."""

SPLIT = ("wait", "cut", "train", "eval")
"""The parts of a chunk's time, as the trainer logs them."""


def _float(x: str | None) -> float | None:
    return float(x) if x is not None else None


class Log:
    """A log file, read incrementally: whole lines, new ones appended."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.offset = 0
        self._rows: dict[int, dict[str, Any]] = {}
        self._chunks: dict[int, tuple[int, int]] = {}
        self._times: dict[int, dict[str, Any]] = {}
        self._open: tuple[int, int, int, int] | None = None
        self._resume: tuple[int, int, int] | None = None
        self._start: dict[int, int] = {}
        self._next: int | None = None
        self._saved: dict[str, float] = {}

    def _update(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.offset, self._rows, self._chunks = 0, {}, {}
            self._times, self._open = {}, None
            self._resume, self._start, self._next = None, {}, None
            self._saved = {}
        if size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].decode("utf-8", "replace").splitlines():
            r = RESUME.match(line)
            if r:
                at, first, done = map(int, r.groups())
                self._resume = (at, first, done)
                continue
            c = CHUNK.match(line)
            if c:
                a, b, n = map(int, c.groups())
                self._chunks[a - 1] = (b, n)
                # The chunk's first step: the resumed step less the
                # chunk's steps done before it, or where the last ended.
                before = 0
                if self._resume and self._resume[1] == a:
                    before = self._resume[2]
                    self._start[a] = self._resume[0] - before
                elif self._next is not None:
                    self._start[a] = self._next
                self._open = (a, b, n, before)
                self._next = self._start[a] + n if a in self._start else None
                self._resume = None
                continue
            sv = SAVED.match(line)
            if sv:
                t = datetime.strptime(sv[1], "%Y-%m-%d %H:%M:%S")
                self._saved[sv[2]] = t.timestamp()
                continue
            ct = CHUNK_TIME.match(line)
            if ct:
                # The time ends the chunk planned last; one with no plan
                # in this log (it began before a restart) is left out.
                if self._open:
                    a, b, n, before = self._open
                    t = datetime.strptime(ct[1], "%Y-%m-%d %H:%M:%S")
                    self._times[a] = {
                        "t": t.timestamp(),
                        "first": a,
                        "sims": b - a + 1,
                        "steps": n,
                        "before": before,
                        **{
                            k: float(v)
                            for k, v in zip(SPLIT, ct.groups()[1:], strict=True)
                        },
                    }
                    self._open = None
                continue
            m = STEP.match(line)
            if not m:
                continue
            g = m.groups()
            cut = line.find(", bias ")
            biased = line[cut + 7 :] if cut >= 0 else ""
            t = datetime.strptime(g[0], "%Y-%m-%d %H:%M:%S").timestamp()
            self._rows[int(g[1])] = {
                "t": t,
                "step": int(g[1]),
                "loss": _float(g[2]),
                "bias": {
                    k: {"mean": float(v), "min": float(lo), "max": float(hi)}
                    for k, v, lo, hi in BIAS.findall(biased)
                },
            }

    def rows(self) -> list[dict[str, Any]]:
        """Return the logged steps so far, in step order."""
        self._update()
        return [self._rows[k] for k in sorted(self._rows)]

    def chunks(self) -> dict[int, tuple[int, int]]:
        """Return the planned chunks: simulations before -> (last, steps).

        A chunk planned twice (a resumed segment) keeps its newest line.
        """
        self._update()
        return dict(self._chunks)

    def chunk_times(self) -> list[dict[str, Any]]:
        """Return each finished chunk's simulations, steps and time split.

        `steps` is the chunk's planned steps and `before` those done
        before a restart resumed it, which its time leaves out. A chunk
        logged twice (a resumed segment) keeps its newest time.
        """
        self._update()
        return [self._times[k] for k in sorted(self._times)]

    def starts(self) -> dict[int, int]:
        """Return the planned chunks' first steps, where the log says.

        From the step a restart resumed at, less the chunk's steps done
        before it, and on from there: a log written again by a restart
        has lost the chunks before it.
        """
        self._update()
        return dict(self._start)

    def saved(self) -> dict[str, float]:
        """Return the time each checkpoint was saved, by name."""
        self._update()
        return dict(self._saved)


class RunLog:
    """A run's logged steps from every log it writes, each read as a `Log`.

    A run writes `train/n<N>.log`, one per segment. New segment logs are
    found on each read, a file reached by two names (a link) is read
    once, and a step logged twice keeps its newest row.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        self._logs: dict[Path, Log] = {}

    def _paths(self) -> list[Path]:
        return sorted(
            self.folder.glob("train/n*.log"), key=lambda p: _size(p.name)
        )

    def _each(self) -> list[Log]:
        out = []
        for path in self._paths():
            real = path.resolve()
            if real not in self._logs:
                self._logs[real] = Log(real)
            if self._logs[real] not in out:
                out.append(self._logs[real])
        return out

    def chunks(self) -> dict[int, tuple[int, int]]:
        """Return every log's planned chunks (`Log.chunks`)."""
        merged: dict[int, tuple[int, int]] = {}
        for log in self._each():
            merged |= log.chunks()
        return merged

    def starts(self) -> dict[int, int]:
        """Return every log's chunks' first steps (`Log.starts`)."""
        merged: dict[int, int] = {}
        for log in self._each():
            merged |= log.starts()
        return merged

    def saved(self) -> dict[str, float]:
        """Return every log's checkpoint save times (`Log.saved`)."""
        merged: dict[str, float] = {}
        for log in self._each():
            merged |= log.saved()
        return merged

    def chunk_times(self) -> list[dict[str, Any]]:
        """Return every log's finished chunks (`Log.chunk_times`).

        Each names its segment (the log's `n<N>`); a chunk in two logs
        keeps the newest.
        """
        merged: dict[int, dict[str, Any]] = {}
        for log in self._each():
            for row in log.chunk_times():
                old = merged.get(row["first"])
                if old is None or row["t"] >= old["t"]:
                    merged[row["first"]] = row | {"segment": log.path.stem}
        return [merged[k] for k in sorted(merged)]

    def last_line(self) -> tuple[float, str] | None:
        """Return the time and text of the newest log's last line.

        The newest log is the one written last, the running segment's
        `train/n<N>.log`: its last line may be any of the trainer's lines,
        not only a logged step, so a checkpoint's evaluation or a wait
        for simulations does not look stale.
        """
        paths = [p for p in self._paths() if p.exists()]
        if not paths:
            return None
        newest = max(paths, key=lambda p: (p.stat().st_mtime, _size(p.name)))
        with newest.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            m = STAMP.match(line)
            if m:
                t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp()
                return t, m[2]
        return None

    def rows(self) -> list[dict[str, Any]]:
        """Return the logged steps of every log, in step order."""
        merged: dict[int, dict[str, Any]] = {}
        for log in self._each():
            for row in log.rows():
                if (
                    row["step"] not in merged
                    or row["t"] >= merged[row["step"]]["t"]
                ):
                    merged[row["step"]] = row
        return [merged[k] for k in sorted(merged)]


def timed(row: dict[str, Any]) -> tuple[float, float]:
    """Return the steps and simulations a chunk's logged time covers.

    A chunk a restart resumed part way (`before` steps done) timed only
    the rest of its steps, and that share of its simulations.
    """
    steps = row["steps"] - row.get("before", 0)
    share = steps / row["steps"] if row["steps"] else 0.0
    return steps, row["sims"] * share


def throughput(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the rates of finished chunks, over the chunks' own time.

    Steps a minute and simulations an hour (`timed`) over the summed
    wait, cut, train and evaluation time (`SPLIT`), so a restart or a
    pause between chunks does not count; the GPU is busy training and
    evaluating. None with no chunk.
    """
    total = sum(r[k] for r in rows for k in SPLIT)
    if not rows or total <= 0:
        return None
    steps = sum(timed(r)[0] for r in rows)
    sims = sum(timed(r)[1] for r in rows)
    return {
        "chunks": len(rows),
        "steps_per_min": steps / total * 60,
        "sims_per_hour": sims / total * 3600,
        "gpu_busy": sum(r["train"] + r["eval"] for r in rows) / total,
        "share": {k: sum(r[k] for r in rows) / total for k in SPLIT},
    }


class JsonCache:
    """JSON files, read again only when their size or time changes."""

    def __init__(self) -> None:
        self._seen: dict[str, tuple[tuple[int, int], Any]] = {}

    def get(self, path: Path | str) -> Any:
        """Return a JSON file's content, or None if it is missing or bad."""
        try:
            st = Path(path).stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        p = str(path)
        if p not in self._seen or self._seen[p][0] != key:
            try:
                value = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                value = None
            self._seen[p] = (key, value)
        return self._seen[p][1]


CACHE = JsonCache()


def released(folder: Path | str, name: str) -> float | None:
    """Return when a checkpoint was released, from its `publish.log`."""
    try:
        text = (Path(folder) / "train" / name / "publish.log").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = PUBLISHED.match(line)
        if m:
            t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
            return t.timestamp()
    return None


def _size(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


CHECKPOINT_DIR = re.compile(r"^n(\d+)$")
"""A checkpoint's folder name: `n<simulations>`, nothing else."""


def _checkpoint_dirs(folder: Path | str) -> list[tuple[int, Path]]:
    """Return the run's checkpoint folders and their sizes, by size.

    Only folders named `n<digits>`: logs and kept copies beside them
    (`n3200.log.pin-014a2bac5`) are not checkpoints.
    """
    out = []
    for d in Path(folder).glob("train/n*"):
        m = CHECKPOINT_DIR.match(d.name)
        if m and d.is_dir():
            out.append((int(m.group(1)), d))
    return sorted(out)


def checkpoints(folder: Path | str) -> list[tuple[str, dict, float | None]]:
    """Return each saved checkpoint's name, settings and time, by size."""
    out = []
    for _, d in _checkpoint_dirs(folder):
        s = CACHE.get(d / "settings.json")
        if isinstance(s, dict):
            side = CACHE.get(d / SIDECAR)
            s = with_sidecar(s, side if isinstance(side, dict) else None)
            # The 7-day check (rule seven_day_check), where scored.
            check = CACHE.get(d / SEVEN_DAY)
            if isinstance(check, dict):
                s = {**s, "seven_day": check.get("seven_day") or {}}
            latest = d / SIDECAR if side else d / "settings.json"
            try:
                mtime = latest.stat().st_mtime
            except OSError:
                mtime = None
            out.append((d.name, s, mtime))
    return out


def prefixes(folder: Path | str, default: Sequence[int]) -> list[int]:
    """Return the run's checkpoint sizes: the saved ones with `default`."""
    saved = {n for n, _ in _checkpoint_dirs(folder)}
    return sorted((saved - {0}) | set(default))


FRESH_S = 600
"""Seconds within which a stream status shows the trainer running: it is
written every 10 steps, and the log tile turns red after 10 minutes."""


def trainer_state(folder: Path | str) -> tuple[str, bool | None]:
    """Return a description of the trainer process and whether it runs.

    It runs if the stream status (`train/stream-status.json`) was
    written in the last `FRESH_S` seconds. A finished or stopped segment
    is neither running nor dead.
    """
    path = Path(folder) / "train" / "stream-status.json"
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        age = None
    if age is not None and age <= FRESH_S:
        return f"alive (status {age / 60:.0f} min ago)", True
    stage = str((CACHE.get(path) or {}).get("stage", ""))
    if stage == "finished" or stage.startswith("stopped at"):
        return stage, None
    if age is None:
        return "unknown", None
    return "DEAD (no recent status)", False


def waiting_batches(folder: Path | str, n: int) -> list[int]:
    """Return the batches segment `n` needs that the run does not have.

    As the workflow picks them: the run's seeds (`seeds.txt`) in order
    until they hold `n` outbreaks, and one spare. A batch here counts its
    outbreaks (`sims/settings-<seed>.csv`); a missing one counts as the
    size of those here (20 if none), so a stopped run can say which
    batch it waits on. Held-out batches are not counted.
    """
    folder = Path(folder)
    try:
        seeds = [int(x) for x in (folder / "seeds.txt").read_text().split()]
    except OSError:
        return []
    sizes = {}
    for seed in seeds:
        path = folder / "sims" / f"settings-{seed}.csv"
        if path.is_file():
            with path.open() as f:
                sizes[seed] = max(0, sum(1 for _ in f) - 1)
    full = max(sizes.values(), key=list(sizes.values()).count) if sizes else 20
    total, need = 0, []
    for seed in seeds:
        if total >= n:
            need.append(seed)
            break
        total += sizes.get(seed, full)
        need.append(seed)
    return [s for s in need if s not in sizes]
