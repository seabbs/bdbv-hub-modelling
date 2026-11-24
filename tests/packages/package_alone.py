"""Import every module of a workspace package installed on its own.

Run in an environment that holds the package's wheel and only what its
`pyproject.toml` declares (the per-package CI job builds one). Each module
is imported in a new interpreter, and the job fails if any module does
not import: it uses a package above it or a dependency it does not
declare.

Usage: python tests/packages/package_alone.py <package> <src folder>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    """Check one package; print each failing module and return 1 if any."""
    package, src = argv[0], Path(argv[1])
    failed = failures(modules(src))
    for name, error in sorted(failed.items()):
        print(f"FAIL {name}: {error}")
    print(f"{package}: {len(failed)} modules fail to import")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
