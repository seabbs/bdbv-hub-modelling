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
