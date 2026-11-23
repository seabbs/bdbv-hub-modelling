"""Backtest and scoring time, from the workflow's own records.

Two records, one table (`summarise`):

- `job`: a rule's `benchmark:` file, `<root>/benchmarks/<rule>/<key>.tsv`
  (Snakemake's wall seconds and peak RSS in MB);
- `model_date`: `timings.csv` beside a backtest's forecasts or in a
  German unit's folder, seconds by model and date from
  `nowcastgermany.evaluation.germany_forecasts` (stage `forecast`, and
  `prefetch` for Chronos-2's batched pass over every date of a call).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

COLUMNS = (
    "kind",
    "rule",
    "key",
    "model",
    "origin",
    "stage",
    "seconds",
    "max_rss",
    "path",
)
"""The summary's columns."""


def _jobs(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("benchmarks/*/*.tsv")):
        bench = pd.read_csv(path, sep="\t")
        for _, row in bench.iterrows():
            rows.append(
                {
                    "kind": "job",
                    "rule": path.parent.name,
                    "key": path.stem,
                    "seconds": float(row["s"]),
                    "max_rss": pd.to_numeric(row["max_rss"], errors="coerce"),
                    "path": str(path),
                }
            )
    return rows


def _cells(root: Path) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(root.rglob("timings.csv")):
        cells = pd.read_csv(path, dtype={"origin": str}, keep_default_na=False)
        if cells.empty:
            continue
        frames.append(
            cells.assign(
                kind="model_date",
                key=path.parent.name,
                path=str(path),
                seconds=pd.to_numeric(cells["seconds"]),
            )
        )
    return frames


def summarise(roots: Iterable[Path | str]) -> pd.DataFrame:
    """Return every job's and every model and date's seconds under `roots`."""
    jobs: list[dict[str, object]] = []
    cells: list[pd.DataFrame] = []
    for root in map(Path, roots):
        jobs += _jobs(root)
        cells += _cells(root)
    parts = [pd.DataFrame(jobs), *cells]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(parts, ignore_index=True).reindex(columns=list(COLUMNS))
