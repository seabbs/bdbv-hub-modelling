"""Check hubValidations (R) gives the results recorded in the manifest.

Skipped when Rscript or hubValidations is not installed. Set
``R_CACHE_ROOTPATH`` if R cannot write to its default cache.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from hv_fixtures import BUNDLE, FIXTURES, METADATA_DIRS, R_VALIDATOR


def _have_hubvalidations() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = !requireNamespace("
            "'hubValidations', quietly = TRUE))",
        ],
        check=False,
        capture_output=True,
    )
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _have_hubvalidations(), reason="Rscript/hubValidations not available"
)


@pytest.fixture(scope="module")
def r_results(hub_path: Path, tmp_path_factory: Any) -> dict[str, set[str]]:
    out = tmp_path_factory.mktemp("r") / "r.json"
    md = [a for d in METADATA_DIRS for a in ("--metadata-dir", str(d))]
    cmd = [
        "Rscript",
        str(R_VALIDATOR),
        "--hub",
        str(hub_path),
        *md,
        "--skip-window",
        "--json",
        str(out),
        *[str(BUNDLE / f["path"]) for f in FIXTURES],
    ]
    subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=900
    )
    res: dict[str, set[str]] = defaultdict(set)
    for x in json.loads(out.read_text()):
        res.setdefault(x["file"], set())
        if x["status"] in ("fail", "error"):
            res[x["file"]].add(x["check"])
    return res


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda f: f["path"])
def test_r_matches_manifest(
    fx: dict[str, Any], r_results: dict[str, set[str]]
) -> None:
    got = r_results[str(BUNDLE / fx["path"])]
    assert sorted(got) == fx["r_failures"]


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


@pytest.mark.parametrize("checkout", ["full", "partial"])
def test_custom_checks_run_with_their_scripts_and_source(
    hub_path: Path, tmp_path: Path, checkout: str
) -> None:
    # The hub's custom checks (hub-config/validations.yml, since 2026-11-16)
    # source scripts from src/ and read the new file from the original
    # checkout at a base commit. Without them every submission failed R
    # validation (main dry run 45296907564, 2026-11-24).
    hub = tmp_path / "hub"
    shutil.copytree(hub_path, hub)
    custom = BUNDLE / "fixtures" / "custom-check"
    shutil.copy(custom / "hub-config" / "validations.yml", hub / "hub-config")
    shutil.copytree(custom / "src", hub / "src")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "hub"],
        ["config", "uploadpack.allowFilter", "true"],
    ):
        _git("-C", str(hub), *args)
    if checkout == "partial":
        # As Submit's actions/checkout makes it: shallow, without blobs
        # outside the sparse paths (main dry run 45379275554, where a
        # shared clone of it failed and the check had no source).
        clone = tmp_path / "hub-clone"
        _git("init", "-q", str(clone))
        _git("-C", str(clone), "remote", "add", "origin", hub.as_uri())
        _git("-C", str(clone), "fetch", "-q", "--depth=1",
             "--filter=blob:none", "origin", "HEAD")  # fmt: skip
        _git("-C", str(clone), "sparse-checkout", "set", "--no-cone",
             "hub-config", "model-metadata", "src")  # fmt: skip
        _git("-C", str(clone), "checkout", "-q", "FETCH_HEAD")
        hub = clone
    out = tmp_path / "r.json"
    sub = (
        BUNDLE
        / "fixtures/constructed/valid_basic/testteam-testmodel"
        / "2026-10-31-testteam-testmodel.csv"
    )
    md = [a for d in METADATA_DIRS for a in ("--metadata-dir", str(d))]
    subprocess.run(
        [
            "Rscript",
            str(R_VALIDATOR),
            "--hub",
            str(hub),
            *md,
            "--skip-window",
            "--json",
            str(out),
            str(sub),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    checks = {x["check"]: x["status"] for x in json.loads(out.read_text())}
    assert checks["probe"] == "pass"
    # The checkout is left as it was.
    status = subprocess.run(
        ["git", "-C", str(hub), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
