"""A run's fine-tuning progress as tables, for its SI figure.

Rule `run_progress` (rules/evaluate.smk) writes three tables of one run
from its own files, read as they stand, so a run still training shows the
checkpoints it has saved:

- `loss_rows`: the trainer's training loss by step, the held-out loss by
  source at each evaluated step, the held-out loss by level at the start
  and at each saved checkpoint, and each checkpoint's step and outbreaks;
- `seven_day_rows`: each checkpoint's 7-day check (rule
  `seven_day_check`);
- `bdbv_rows`: each released checkpoint's BDBV check, relative CRPS by
  model (rule `ft_check_page`), where its page is built.

The paper's `progress_<run>` rule plots them
(`paperpipe.progress.progress_figure`).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from flowlib.evaluations import checkpoint_settings, merged_history

__all__ = [
    "BDBV_COLUMNS",
    "LOSS_COLUMNS",
    "SEVEN_DAY_COLUMNS",
    "bdbv_rows",
    "loss_rows",
    "seven_day_rows",
    "write_progress",
]

LOSS_COLUMNS: Final = ("kind", "series", "step", "outbreaks", "loss")
SEVEN_DAY_COLUMNS: Final = (
    "outbreaks",
    "ratio",
    "inside90",
    "rough_ratio",
    "n",
)
BDBV_COLUMNS: Final = ("outbreaks", "model", "relative_crps")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _segments(train: Path) -> list[tuple[int, Path]]:
    """Return the run's saved segments `n<N>`, smallest first."""
    found = [
        (int(d.name[1:]), d)
        for d in Path(train).glob("n*")
        if d.name[1:].isdigit() and (d / "settings.json").is_file()
    ]
    return sorted(found)


def loss_rows(train: Path | str, prefixes: Iterable[int]) -> list[dict]:
    """Return a run's losses by step, in `LOSS_COLUMNS`.

    `kind` is `training` (`series` all), `validation` (by source, at each
    evaluated step), `validation by level` (at the start and at each saved
    checkpoint of `prefixes`) or `checkpoint` (each saved checkpoint's step
    and outbreaks, with no loss). Empty with no history.
    """
    train = Path(train)
    got = _read(train / "history.json")
    if not isinstance(got, list):
        return []
    hist = merged_history(got, train)
    out: list[dict] = []

    def row(kind: str, series: str, step: int, loss: float | None, n=None):
        out.append(
            {
                "kind": kind,
                "series": series,
                "step": int(step),
                "outbreaks": n,
                "loss": None if loss is None else float(loss),
            }
        )

    for h in hist:
        if "loss" in h and "step" in h:
            row("training", "all", h["step"], h["loss"])
    for h in hist:
        if isinstance(h.get("by_source"), dict):
            for source, loss in h["by_source"].items():
                row("validation", source, h.get("step", 0), loss)
    start = next((h for h in hist if h.get("step") == 0), {})
    for level, loss in (start.get("by_level") or {}).items():
        row("validation by level", level, 0, loss, 0)
    for n in sorted(prefixes):
        settings = checkpoint_settings(train / f"n{n}")
        step = settings.get("steps")
        if step is None:
            continue
        for level, loss in (
            settings.get("held_out_loss_by_level") or {}
        ).items():
            row("validation by level", level, step, loss, n)
    for h in hist:
        if "hash" in h and "n_outbreaks" in h:
            row(
                "checkpoint",
                f"n{h['n_outbreaks']}",
                h["step"],
                None,
                h["n_outbreaks"],
            )
    return out


def seven_day_rows(train: Path | str) -> list[dict]:
    """Return each checked checkpoint's 7-day check, by outbreaks."""
    out = []
    for n, where in _segments(Path(train)):
        got = _read(where / "seven-day.json")
        check = (got or {}).get("seven_day") if isinstance(got, dict) else None
        if not check:
            continue
        out.append(
            {"outbreaks": n} | {k: check.get(k) for k in SEVEN_DAY_COLUMNS[1:]}
        )
    return out


def bdbv_rows(train: Path | str, ft_check: Path | str) -> list[dict]:
    """Return each released checkpoint's BDBV check: relative CRPS by model.

    A checkpoint's check is its page's `overall.csv` under `ft_check`
    (`bdbvhub.reporting.ftcheck_report`), named by its release hash;
    checkpoints with no page yet are left out.
    """
    out = []
    for n, where in _segments(Path(train)):
        got = _read(where / "published.json")
        tag = got.get("hash") if isinstance(got, dict) else None
        if not tag:
            continue
        ckpt = f"slots-{tag}"
        path = (
            Path(ft_check) / ckpt / f"bdbv-ft-check-{ckpt}-data" / "overall.csv"
        )
        if not path.is_file():
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                out.append(
                    {
                        "outbreaks": n,
                        "model": r["model"],
                        "relative_crps": float(r["relative_crps"]),
                    }
                )
    return out


def _write(rows: Sequence[Mapping], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_progress(
    train: Path | str,
    prefixes: Iterable[int],
    ft_check: Path | str,
    loss: Path | str,
    seven_day: Path | str,
    bdbv: Path | str,
) -> None:
    """Write a run's three progress tables, each with its header."""
    _write(loss_rows(train, prefixes), LOSS_COLUMNS, Path(loss))
    _write(seven_day_rows(train), SEVEN_DAY_COLUMNS, Path(seven_day))
    _write(bdbv_rows(train, ft_check), BDBV_COLUMNS, Path(bdbv))
