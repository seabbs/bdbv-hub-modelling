"""List the packages whose CI jobs a change runs.

Each package's job runs its own `packages/<name>/tests`; the root
`tests/` (workflow, repo and cross-package checks) runs in integration.
A change runs the jobs of the packages whose source it touches and of
every package that imports one of them (declared dependencies and the
import-linter contract's listed upward imports), so each change is
checked by every package's tests that can see it. A changed test runs
its own package's job only. A change to the shared test setup or the
lock file runs every job.

Usage: git diff --name-only ... | python .github/scripts/legs.py
           (a JSON list of packages)
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def layers() -> list[str]:
    """Workspace packages from the lowest layer to the highest."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contract = config["tool"]["importlinter"]["contracts"][0]
    out: list[str] = []
    for layer in reversed(contract["layers"]):
        out += [name.strip() for name in layer.split("|")]
    return out


SHARED = (
    "pyproject.toml",
    "uv.lock",
    "conftest.py",
    ".github/scripts/",
    ".github/workflows/tests.yml",
)
"""Files every package's job depends on."""


def importers() -> dict[str, set[str]]:
    """Each package and the packages that import it."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contract = config["tool"]["importlinter"]["contracts"][0]
    order = layers()
    out: dict[str, set[str]] = {name: set() for name in order}
    for name in order:
        project = tomllib.loads(
            (ROOT / "packages" / name / "pyproject.toml").read_text()
        )["project"]
        for dep in project["dependencies"]:
            dep_name = re.split(r"[\[<>=; ]", dep, maxsplit=1)[0]
            if dep_name in out:
                out[dep_name].add(name)
    for rule in contract.get("ignore_imports", []):
        source, target = (s.strip().split(".")[0] for s in rule.split("->"))
        out[target].add(source)
    return out


def legs(changed: list[str]) -> list[str]:
    """Return the packages whose jobs a change runs, in layer order."""
    order = layers()
    if any(f == s or f.startswith(s) for f in changed for s in SHARED):
        return order
    source: set[str] = set()
    tests: set[str] = set()
    for f in changed:
        parts = f.split("/")
        if parts[0] == "packages" and len(parts) > 2 and parts[1] in order:
            (tests if parts[2] == "tests" else source).add(parts[1])
    touched = set(source)
    by = importers()
    todo = list(touched)
    while todo:
        for user in by[todo.pop()] - touched:
            touched.add(user)
            todo.append(user)
    # A changed test runs in its own package's job only.
    touched |= tests
    return [name for name in order if name in touched]


def main() -> int:
    """Print, as JSON, the packages the changed files on stdin reach."""
    changed = [line.strip() for line in sys.stdin if line.strip()]
    print(json.dumps(legs(changed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
