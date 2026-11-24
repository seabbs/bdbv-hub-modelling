"""The packages import one way and each installs and imports alone."""

from __future__ import annotations

import shutil
import subprocess
import sys
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


def test_ci_checks_each_package_in_its_own_job() -> None:
    # One CI job per package: installed alone with every module imported,
    # ruff, format, mypy and the tests it owns (tests/packages/owners.py).
    import yaml

    sys.path.insert(0, str(ROOT / "tests" / "packages"))
    from owners import layers

    packages = {p.name for p in ROOT.glob("packages/*/src/*") if p.is_dir()}
    assert set(layers()) == packages
    workflows = ROOT / ".github" / "workflows"
    job = yaml.safe_load((workflows / "ci.yml").read_text())["jobs"]["package"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    for check in (
        "package_alone.py",
        "ruff check",
        "ruff format --check",
        "mypy",
        "owners.py",
        "pytest",
    ):
        assert check in runs, check
    assert not (workflows / "packages.yml").exists()


def test_a_change_runs_the_jobs_of_every_package_that_can_see_it() -> None:
    sys.path.insert(0, str(ROOT / "tests" / "packages"))
    from owners import layers, legs

    assert legs(["docs/index.md"]) == []
    assert legs(["uv.lock"]) == layers()
    assert legs(["packages/bdbvhub/tests/test_zones.py"]) == ["bdbvhub"]
    seen = legs(["packages/nowcastbaselines/src/nowcastbaselines/chain.py"])
    assert {"nowcastbaselines", "nowcastrenewal", "bdbvhub"} <= set(seen)
    assert "nowcastutils" not in seen


STUDIES = ("nowcastgermany",)
"""Study packages, which build on the generic ones and never the reverse
(Sam, 2026-11-23)."""

STUDY_FREE = ("nowcastfoundation", "nowcasteval")
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


def test_a_package_with_its_own_tests_has_none_left_under_tests() -> None:
    # Packages move their tests to packages/<name>/tests one at a time;
    # once moved, no root test is theirs.
    sys.path.insert(0, str(ROOT / "tests" / "packages"))
    from owners import owners

    moved = {p.parent.name for p in ROOT.glob("packages/*/tests")}
    assert {
        "nowcastutils",
        "nowcasteval",
        "nowcastbaselines",
        "nowcastrenewal",
        "nowcastfoundation",
        "nowcastgermany",
        "nowcastbdbv",
        "bdbvhub",
    } <= moved
    left = [
        f
        for f, owner in owners().items()
        if f.startswith("tests/") and owner in moved
    ]
    assert left == []
