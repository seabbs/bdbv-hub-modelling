"""The packages import one way and each installs and imports alone."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _contract() -> dict:
    return CONFIG["tool"]["importlinter"]


def test_the_contract_covers_every_workspace_package() -> None:
    packages = {p.name for p in ROOT.glob("packages/*/src/*") if p.is_dir()}
    contract = _contract()
    assert set(contract["root_packages"]) == packages
    layers = {
        name.strip()
        for layer in contract["contracts"][0]["layers"]
        for name in layer.split("|")
    }
    assert layers == packages


def test_the_packages_keep_the_contract() -> None:
    lint = shutil.which("lint-imports")
    assert lint is not None
    run = subprocess.run(
        [lint, "--no-cache"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # Every listed exception is still an import, so the list only shrinks.
    assert "No matches for ignored import" not in run.stdout + run.stderr


def test_ci_installs_and_imports_every_package_alone() -> None:
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "packages.yml").read_text()
    )
    matrix = workflow["jobs"]["alone"]["strategy"]["matrix"]["package"]
    packages = {p.name for p in ROOT.glob("packages/*/src/*") if p.is_dir()}
    assert set(matrix) == packages


STUDIES = ("nowcastgermany",)
"""Study packages, which build on the generic ones and never the reverse
(Sam, 2026-11-23)."""

STUDY_FREE = ("nowcastfoundation",)
"""Packages whose modules import no study yet, function-level imports
included; the rest follow one PR at a time, then an import-linter
contract replaces this list."""


def test_these_packages_import_no_study() -> None:
    found = []
    for package in STUDY_FREE:
        for path in sorted((ROOT / "packages" / package / "src").rglob("*.py")):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                words = line.split()
                if words[:1] in (["import"], ["from"]) and any(
                    words[1].split(".")[0] == s for s in STUDIES
                ):
                    found.append(f"{path.relative_to(ROOT)}:{n}")
    assert found == []
