# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Simulate one batch from the grid config.

    python -m nowcastfoundation.simulator GRID N SEED OUT

ENW_LIB (the pinned epinowcast's library), ENW_DRAWS (observation draws
per simulation), ENW_WEEKS (a pilot's fixed length) and RSCRIPT as the
simulate rule sets them.
"""

from __future__ import annotations

import argparse
import os

from .batch import simulate_batch


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m nowcastfoundation.simulator"
    )
    parser.add_argument("grid", help="the grid config file")
    parser.add_argument("n", type=int, help="simulations")
    parser.add_argument("seed", type=int, help="the batch's seed")
    parser.add_argument("out", help="the folder the files go to")
    args = parser.parse_args(argv)
    weeks = os.environ.get("ENW_WEEKS")
    simulate_batch(
        args.grid,
        args.seed,
        args.n,
        args.out,
        draws=int(os.environ.get("ENW_DRAWS", "1")),
        rscript=os.environ.get("RSCRIPT", "Rscript"),
        enw_lib=os.environ.get("ENW_LIB"),
        weeks=int(weeks) if weeks else None,
    )


if __name__ == "__main__":
    main()
