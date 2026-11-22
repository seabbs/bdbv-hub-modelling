"""Gather backtest and scoring time into one CSV (flowlib.benchmarks).

    python workflow/scripts/benchmarks.py OUT ROOT...

Each ROOT is searched for rules' `benchmarks/<rule>/<key>.tsv` and for
`timings.csv` files (seconds by model and date).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import benchmarks


def main(argv: list[str]) -> None:
    """Write the summary of every ROOT to OUT."""
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    table = benchmarks.summarise(argv[1:])
    table.to_csv(out, index=False)
    jobs = table.query("kind == 'job'")
    print(f"{len(jobs)} jobs, {len(table) - len(jobs)} model-date cells: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
