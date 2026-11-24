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


def test_custom_checks_run_with_their_scripts_and_source(
    hub_path: Path, tmp_path: Path
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
    ):
        subprocess.run(["git", "-C", str(hub), *args], check=True)
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


def test_custom_checks_run_on_a_partial_sparse_hub_checkout(
    hub_path: Path, tmp_path: Path
) -> None:
    # The CI checks the hub out with actions/checkout's sparse-checkout: a
    # blob-less partial clone. A `git clone --shared` of it dies on the
    # missing blobs, which left the custom check without its source
    # (main dry run 45379275554, 2026-11-24).
    full = tmp_path / "full"
    shutil.copytree(hub_path, full)
    custom = BUNDLE / "fixtures" / "custom-check"
    shutil.copy(custom / "hub-config" / "validations.yml", full / "hub-config")
    shutil.copytree(custom / "src", full / "src")
    (full / "model-output" / "other").mkdir(parents=True, exist_ok=True)
    (full / "model-output" / "other" / "x.csv").write_text("a\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["commit", "-qm", "hub"],
        ["config", "uploadpack.allowFilter", "true"],
    ):
        subprocess.run([*git, "-C", str(full), *args], check=True)
    hub = tmp_path / "hub"
    subprocess.run(
        [*git, "clone", "-q", "--filter=blob:none", "--no-checkout",
         f"file://{full}", str(hub)],
        check=True,
    )  # fmt: skip
    for args in (
        ["sparse-checkout", "set", "hub-config", "model-metadata", "src"],
        ["checkout", "-q"],
    ):
        subprocess.run([*git, "-C", str(hub), *args], check=True)
    assert not (hub / "model-output" / "other" / "x.csv").exists()
    out = tmp_path / "r.json"
    sub = (
        BUNDLE
        / "fixtures/constructed/valid_basic/testteam-testmodel"
        / "2026-10-31-testteam-testmodel.csv"
    )
    md = [a for d in METADATA_DIRS for a in ("--metadata-dir", str(d))]
    r = subprocess.run(
        ["Rscript", str(R_VALIDATOR), "--hub", str(hub), *md,
         "--skip-window", "--json", str(out), str(sub)],
        check=False, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    checks = {x["check"]: x["status"] for x in json.loads(out.read_text())}
    assert checks["probe"] == "pass", (checks, r.stderr[-2000:])
    listed = subprocess.run(
        ["git", "-C", str(hub), "worktree", "list"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    assert len(listed.strip().splitlines()) == 1
