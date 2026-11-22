"""The run as a pipeline: its stages, where it is, and what it waits on.

Simulate rounds, fetch the batches into the run, cut blocks and views,
train a segment, evaluate it on the held-out blocks and score it on the
German data, then the next segment. Each stage's progress comes from the
files it writes (round folders, the run's batches, the trainer's status
file) and its timings from this machine's newest Snakemake log, read as
the monitor reads it (`flowlib.runstatus`).
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flowlib import runstatus
from flowlib.dashboard import logs

STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "simulate",
        "Simulate rounds",
        ("simulate_batch", "batch_record", "round_manifest", "sim_qc"),
    ),
    ("fetch", "Fetch batches", ("fetch_batch", "seed_list")),
    ("cut", "Cut blocks and views", ()),
    ("train", "Train segment", ("train_segment",)),
    (
        "evaluate",
        "Held-out evaluation",
        ("evaluate_steps", "evaluate_checkpoint", "checkpoint_check"),
    ),
    ("score", "German scoring", ("score_run_germany", "run_scores")),
)
"""Each stage: its key, title and the workflow rules that run it."""


@dataclass(frozen=True)
class Where:
    """Where this machine's workflow keeps what the stages write."""

    rounds_dir: Path | None = None
    rounds: Sequence[int] = ()
    batches: int = 40
    # The checkout whose Snakemake runs the workflow (its .snakemake/log).
    workflow_dir: Path | None = None


@dataclass
class Stage:
    """One stage as shown."""

    key: str
    title: str
    progress: str = ""
    timing: str = ""
    running: int = 0


@dataclass
class View:
    """The pipeline: its stages, the current one and what it waits on."""

    stages: list[Stage] = field(default_factory=list)
    current: str = ""
    waiting: str = ""


def _dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m"


def _jobs(where: Where) -> list[dict[str, Any]]:
    if where.workflow_dir is None:
        return []
    log = runstatus.newest_log(where.workflow_dir)
    return runstatus.parse_log(log.read_text(errors="replace")) if log else []


def _timing(jobs: Sequence[Mapping[str, Any]]) -> tuple[str, float | None]:
    """Return a stage's job counts and mean time, and that mean (s)."""
    took = [
        (
            datetime.fromisoformat(j["ended"])
            - datetime.fromisoformat(j["started"])
        ).total_seconds()
        for j in jobs
        if j["state"] == "done" and j["started"] and j["ended"]
    ]
    mean = sum(took) / len(took) if took else None
    running = sum(j["state"] == "running" for j in jobs)
    failed = sum(j["state"] == "failed" for j in jobs)
    parts = [
        f"{running} running" if running else "",
        f"{len(took)} done" + (f", {_dur(mean)} each" if mean else "")
        if took
        else "",
        f"{failed} failed" if failed else "",
    ]
    return "; ".join(p for p in parts if p), mean


def _simulated(where: Where) -> dict[int, int]:
    if where.rounds_dir is None:
        return {}
    return {
        r: len(
            list(
                Path(where.rounds_dir, f"r{r}", "sims").glob("counts-*.csv.gz")
            )
        )
        for r in where.rounds
    }


def _in_rounds(where: Where, seed: int) -> bool:
    if where.rounds_dir is None:
        return False
    return any(Path(where.rounds_dir).glob(f"r*/sims/counts-{seed}.csv.gz"))


def _clock(text: str) -> str:
    try:
        return datetime.fromisoformat(str(text)).strftime("%H:%M")
    except ValueError:
        return ""


def build(
    where: Where,
    folder: Path | None,
    train: Mapping[str, Any],
    alive: bool | None,
) -> View:
    """Return the pipeline of the run in `folder`.

    `train` holds the live status: `step`, `nxt` (the next checkpoint),
    `next_step` (its step) and `trained` (simulations so far); `alive`
    is whether the trainer runs (`logs.trainer_state`).
    """
    jobs = _jobs(where)
    status = (
        logs.CACHE.get(Path(folder) / "train" / "stream-status.json") or {}
        if folder
        else {}
    )
    view = View()
    for key, title, rules in STAGES:
        mine = [j for j in jobs if j["rule"] in rules]
        timing, mean = _timing(mine)
        stage = Stage(
            key,
            title,
            timing=timing,
            running=sum(j["state"] == "running" for j in mine),
        )
        view.stages.append(stage)
        if key == "simulate":
            done = _simulated(where)
            stage.progress = " · ".join(
                f"r{r} {n}/{where.batches}" for r, n in done.items()
            )
            left = len(done) * where.batches - sum(done.values())
            if left and mean:
                eta = left * mean / max(stage.running, 1)
                stage.timing += f"; {left} to go, ETA {_dur(eta)}"
        elif key == "fetch" and folder:
            seeds = Path(folder, "seeds.txt")
            total = len(seeds.read_text().split()) if seeds.is_file() else 0
            have = len(list(Path(folder, "sims").glob("settings-*.csv")))
            stage.progress = f"{have} of {total} training batches in the run"
        elif key == "cut" and status:
            split = status.get("split") or {}
            if split.get("cut_s") is not None:
                stage.progress = f"{_dur(split['cut_s'])} cutting this segment"
        elif key == "train" and train.get("nxt"):
            stage.progress = (
                f"n{train['nxt']}: step {train.get('step') or 0} of "
                f"{train.get('next_step')}"
            )
            if train.get("trained") is not None:
                stage.progress += (
                    f"; {min(train['trained'], train['nxt']):.0f} of "
                    f"{train['nxt']} simulations"
                )
            if status.get("stage"):
                stage.timing = str(status["stage"])
        if not stage.progress:
            stage.progress = "; ".join(
                f"{j['rule']} {j['wildcards']}".strip()
                for j in mine
                if j["state"] == "running"
            )
    stage_text = str(status.get("stage", "")).lower()
    since = _clock(status.get("waiting_since") or status.get("time") or "")
    missing = (
        logs.waiting_batches(folder, int(train["nxt"]))
        if folder and not alive and train.get("nxt")
        else []
    )
    if missing:
        what = "batches" if len(missing) > 1 else "batch"
        view.waiting = f"waiting for {what} {', '.join(map(str, missing))}"
        view.waiting += f" since {since}" if since else ""
        absent = any(not _in_rounds(where, s) for s in missing)
        view.current = "simulate" if absent else "fetch"
    elif alive:
        if status.get("waiting_since"):
            view.waiting = f"trainer waiting for batches since {since}"
            view.current = "fetch"
        elif "cut" in stage_text or "block" in stage_text:
            view.current = "cut"
        else:
            view.current = "train"
    else:
        running = {s.key for s in view.stages if s.running}
        view.current = next(
            (k for k in ("evaluate", "score", "simulate") if k in running), ""
        )
    return view


def section(view: View) -> str:
    """Return the pipeline as html.

    The stage cards in order, the current one marked, and what the run
    waits on.
    """
    cards = []
    for s in view.stages:
        cls = (
            "stg cur"
            if s.key == view.current
            else ("stg act" if s.running else "stg")
        )
        cards.append(
            f'<div class="{cls}"><div class=k>{html.escape(s.title)}</div>'
            f"<div>{html.escape(s.progress) or '&nbsp;'}</div>"
            f"<div class=s>{html.escape(s.timing)}</div></div>"
        )
    wait = (
        f"<div class=waitnote>{html.escape(view.waiting)}</div>"
        if view.waiting
        else ""
    )
    return (
        f"<div class=pipe>{'<span class=arr>→</span>'.join(cards)}</div>{wait}"
    )
