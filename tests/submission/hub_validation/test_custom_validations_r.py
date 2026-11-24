"""hubValidations runs the hub's own custom checks (hub-config/validations.yml).

Since the hub's #142 its validations.yml sources checks in src/ that read
the PR's checkout and base commit (FORECAST_SOURCE_HUB, PR_BASE_SHA).
Our validator copies src/ into its temporary hub and stands up that
checkout, so a file the hub accepts is not ruled out by an R error.
Skipped when Rscript or hubValidations is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hv_fixtures import BUNDLE, R_VALIDATOR
from test_agreement_r import _have_hubvalidations

pytestmark = pytest.mark.skipif(
    not _have_hubvalidations(), reason="Rscript/hubValidations not available"
)

SNAPSHOT = BUNDLE / "fixtures" / "hub-snapshot-rt"

CHECK = """
check_source <- function(hub_path, file_path) {
  src <- Sys.getenv("FORECAST_SOURCE_HUB")
  sha <- Sys.getenv("PR_BASE_SHA")
  ok <- nzchar(src) && grepl("^[0-9a-f]{40}$", sha) &&
    file.exists(file.path(src, "model-output", file_path))
  hubValidations::capture_check_cnd(
    check = ok, file_path = file_path,
    msg_subject = "The PR checkout", msg_attribute = "available."
  )
}
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_the_hubs_custom_checks_see_the_pr_checkout(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    shutil.copytree(SNAPSHOT, hub)
    (hub / "hub-config" / "validations.yml").write_text(
        "default:\n  validate_model_file:\n    source_check:\n"
        "      fn: check_source\n      source: src/validations/check.R\n"
    )
    (hub / "src" / "validations").mkdir(parents=True)
    (hub / "src" / "validations" / "check.R").write_text(CHECK)
    _git(hub, "init", "-q")
    _git(hub, "add", ".")
    _git(hub, "commit", "-q", "-m", "hub")
    sub = tmp_path / "out" / "seabbs-renewal"
    sub.mkdir(parents=True)
    rows = [
        "reference_date,target,target_end_date,location,output_type,"
        "output_type_id,value"
    ] + [
        f"2026-11-28,wk inc confirmed cases,2026-11-21,CD,sample,{s},10"
        for s in range(1, 101)
    ]
    (sub / "2026-11-28-seabbs-renewal.csv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "r.json"
    r = subprocess.run(
        [
            "Rscript",
            str(R_VALIDATOR),
            "--hub",
            str(hub),
            "--no-metadata",
            "--skip-window",
            "--json",
            str(out),
            str(sub / "2026-11-28-seabbs-renewal.csv"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    checks = {x["check"]: x["status"] for x in json.loads(out.read_text())}
    assert checks.get("source_check") == "pass", (checks, r.stdout, r.stderr)
    # The detached checkout is removed afterwards.
    listed = subprocess.run(
        ["git", "-C", str(hub), "worktree", "list"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert len(listed.strip().splitlines()) == 1
