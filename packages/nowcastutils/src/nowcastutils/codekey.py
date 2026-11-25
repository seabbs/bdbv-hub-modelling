"""Code keys: the repository's modules a job imports, and their digest.

A job's results depend on the code it runs, not on every file of the
packages it touches. `module_files` follows the imports of its entry
modules through the repository's own packages (`packages/*/src`), read
from their source without running them, so a change to a module the job
never imports leaves its key, and its results, as they are. `digest`
hashes the files found.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[3]
"""The repository's `packages` folder, whose modules a key follows."""


def _origin(name: str) -> Path | None:
    """Return the file of module `name` if it is the repository's."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    path = Path(spec.origin).resolve()
    return path if path.is_relative_to(PACKAGES) else None


def _imported(path: Path, name: str) -> list[str]:
    """Return the modules a module's source may import, by name.

    Every `import a.b` and `from a import b` anywhere in it, functions
    included; `from a import b` names both `a` and `a.b`, as `b` may be
    either a module or a name in `a`.
    """
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                up = parts[: len(parts) - node.level + 1]
                base = ".".join([*up, base] if base else up)
            out.append(base)
            out += [f"{base}.{alias.name}" for alias in node.names]
    return out


def _with_parents(name: str) -> list[str]:
    """Return a module and its parent packages, which importing it runs."""
    parts = name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def module_files(entries: Iterable[str]) -> list[str]:
    """Return the repository's files that modules `entries` import.

    Transitively, through the repository's own packages only; each
    module's parent packages are in too. Sorted paths.
    """
    seen: dict[str, Path] = {}
    todo = [p for e in entries for p in _with_parents(e)]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _origin(name)
        if path is None:
            continue
        seen[name] = path
        for imported in _imported(path, name):
            todo += [p for p in _with_parents(imported) if p not in seen]
    return sorted({str(p) for p in seen.values()})


def digest(files: Sequence[str | Path], root: str | Path | None = None) -> str:
    """Return a 12-character digest of files' names and contents.

    Names relative to `root` when given (by default `PACKAGES`' parent),
    so checkouts at other paths give the same key.
    """
    base = Path(root) if root is not None else PACKAGES.parent
    h = hashlib.sha256()
    for f in sorted(Path(f).resolve() for f in files):
        name = f.relative_to(base) if f.is_relative_to(base) else Path(f.name)
        h.update(name.as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]
