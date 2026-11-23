"""Import every module of a workspace package installed on its own.

Run in an environment that holds the package's wheel and only what its
`pyproject.toml` declares (the per-package CI job builds one). Each module
is imported in a new interpreter. Modules in `KNOWN` still import a
package above them or an undeclared dependency at import time; the job
fails if one of them starts to import (remove it from `KNOWN`) or if any
other module fails.

Usage: python tests/packages/package_alone.py <package> <src folder>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KNOWN: dict[str, set[str]] = {
    # Upward imports (bdbvhub, nowcastbaselines; the import-linter
    # exceptions) and scoringrules, which the next refit declares.
    "nowcasteval": {
        "nowcasteval.backtest",
        "nowcasteval.checks",
        "nowcasteval.coupling",
        "nowcasteval.marginals",
        "nowcasteval.nowcast",
        "nowcasteval.nowcast_plots",
        "nowcasteval.pairing",
        "nowcasteval.pairing_plots",
        "nowcasteval.plots",
        "nowcasteval.ppc",
        "nowcasteval.scoresummary",
        "nowcasteval.scoring",
    },
    # Upward imports of bdbvhub.
    "nowcastfoundation": {"nowcastfoundation.simulate"},
    "nowcastgermany": {
        "nowcastgermany.cli",
        "nowcastgermany.evaluation",
        "nowcastgermany.pairing",
        "nowcastgermany.ft_store",
        "nowcastgermany.reporting",
    },
    # scoringrules, through nowcasteval.scoring.
    "bdbvhub": {
        "bdbvhub.cli",
        "bdbvhub.compare",
        "bdbvhub.reporting.hubtables",
        "bdbvhub.reporting.report",
        "bdbvhub.reporting.store_note",
        "bdbvhub.reporting.zone_nowcast",
        "bdbvhub.store.archive",
        "bdbvhub.store.digest",
        "bdbvhub.store.store",
        "bdbvhub.zones",
    },
}
"""Modules that fail to import alone, by package, as of 2026-11-22."""


def modules(src: Path) -> list[str]:
    """Return the dotted modules under a package's source folder."""
    out = []
    for path in sorted(src.rglob("*.py")):
        parts = path.relative_to(src.parent).with_suffix("").parts
        out.append(".".join(parts).removesuffix(".__init__"))
    return out


def failures(names: list[str]) -> dict[str, str]:
    """Import each module in a new interpreter; return the failing ones."""
    out = {}
    for name in names:
        run = subprocess.run(
            [sys.executable, "-I", "-c", f"import {name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            lines = run.stderr.strip().splitlines()
            out[name] = lines[-1] if lines else "failed"
    return out


def main(argv: list[str]) -> int:
    """Check one package; print what changed and return the exit code."""
    package, src = argv[0], Path(argv[1])
    failed = failures(modules(src))
    known = KNOWN.get(package, set())
    new = {m: e for m, e in failed.items() if m not in known}
    fixed = sorted(known - set(failed))
    for name, error in sorted(new.items()):
        print(f"FAIL {name}: {error}")
    for name in fixed:
        print(f"NOW IMPORTS {name}: remove it from KNOWN")
    print(f"{package}: {len(failed)} known failures, {len(new)} new")
    return 1 if new or fixed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
