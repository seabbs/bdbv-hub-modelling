"""Check the built site's German page names no held-out nowcast date.

    uv run germany-site-check SITE [--config config/germany.toml]

Run from the repo root. SITE is the built site (`mkdocs build`,
`_site`). Reads the German page and its summary and explorer JSON
(`reporting.held_out_leaks`)
and exits non-zero, listing them, if any names a date on or after the
split date in config/germany.toml (or after it, in the page's prose).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nowcastgermany.evaluation import load_germany_config
from nowcastgermany.reporting import held_out_leaks


def main(argv: list[str] | None = None) -> int:
    """Check SITE's German page."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("site", type=Path, nargs="?", default=Path("_site"))
    parser.add_argument(
        "--config", type=Path, default=Path("config/germany.toml")
    )
    args = parser.parse_args(argv)
    site = args.site
    split = load_germany_config(args.config).split_date
    files = [
        site / "germany" / "index.html",
        site / "figures" / "scoresummary-germany-latest.json",
        site / "figures" / "explorer-germany.json",
    ]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"Not built: {', '.join(missing)}", file=sys.stderr)
        return 1
    leaks = held_out_leaks(files, split)
    for leak in leaks:
        print(f"::error::held-out date on the German page: {leak}")
    if not leaks:
        print(f"The German page names no nowcast date from {split} on.")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
