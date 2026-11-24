"""List the test files each workspace package owns.

A test file belongs to the highest package (in the import-linter layers
of pyproject.toml) it imports, so each file runs in one package's CI
job: the job of the package whose change it checks last. Files that
import no workspace package (the workflow, the paper, the repo checks)
belong to none and run in the integration job.

`legs` gives the packages a change touches: those whose files changed
and every package that imports one of them (declared dependencies and
the contract's listed upward imports), so a change is checked by every
package's tests that can see it, plus the owners of changed tests. A
change to the shared test setup or the lock file touches every package.

Usage: python tests/packages/owners.py <package>   (files, one a line)
       python tests/packages/owners.py --none      (files of no package)
       git diff --name-only ... | python tests/packages/owners.py --legs
           (a JSON list of packages)
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
TESTS = ROOT / "tests"


def layers() -> list[str]:
    """Workspace packages from the lowest layer to the highest."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contract = config["tool"]["importlinter"]["contracts"][0]
    out: list[str] = []
    for layer in reversed(contract["layers"]):
        out += [name.strip() for name in layer.split("|")]
    return out


def imported(path: Path, packages: set[str]) -> set[str]:
    """The workspace packages a file imports, anywhere in it."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        found |= {n.split(".")[0] for n in names} & packages
    return found


def owners() -> dict[str, str | None]:
    """Each test file (relative to the root) and the package that owns it."""
    order = layers()
    rank = {name: i for i, name in enumerate(order)}
    out: dict[str, str | None] = {}
    for path in sorted(TESTS.rglob("test_*.py")):
        found = imported(path, set(order))
        top = max(found, key=rank.__getitem__) if found else None
        out[path.relative_to(ROOT).as_posix()] = top
    return out


SHARED = (
    "pyproject.toml",
    "uv.lock",
    "tests/conftest.py",
    "tests/packages/",
    ".github/workflows/ci.yml",
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
    """The packages whose jobs a change to `changed` runs, in layer order."""
    order = layers()
    if any(f == s or f.startswith(s) for f in changed for s in SHARED):
        return order
    owned = owners()
    touched = {
        f.split("/")[1]
        for f in changed
        if f.startswith("packages/") and f.split("/")[1] in order
    }
    by = importers()
    todo = list(touched)
    while todo:
        for user in by[todo.pop()] - touched:
            touched.add(user)
            todo.append(user)
    # A changed test runs in its owner's job only.
    touched |= {p for f in changed if (p := owned.get(f))}
    return [name for name in order if name in touched]


def main(argv: list[str]) -> int:
    (which,) = argv
    if which == "--legs":
        changed = [line.strip() for line in sys.stdin if line.strip()]
        print(json.dumps(legs(changed)))
        return 0
    want = None if which == "--none" else which
    if want is not None and want not in layers():
        print(f"unknown package {which}", file=sys.stderr)
        return 2
    print("\n".join(f for f, p in owners().items() if p == want))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
