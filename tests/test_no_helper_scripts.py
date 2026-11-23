"""WP4.3c (#343): no helper scripts outside the packages and the workflow.

Each former `scripts/` or `tools/` file is a workflow rule, a package
command, a test helper beside its tests, or gone.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVED = {
    "scripts/finetune_stream.py": "finetune-stream",
    "scripts/finetune_verify.py": "finetune-verify",
    "scripts/sim_qc.py": "sim-qc",
    "scripts/check_germany_site.py": "germany-site-check",
    "scripts/germany_ft_store.py": "germany-ft-store",
    "scripts/score_history.py": "score-history",
    "scripts/package_alone.py": "tests/packages/package_alone.py",
    "scripts/baselinenowcast_fixtures.R": "tests/baselinenowcast/r/cases.R",
    "scripts/snapshot_fixtures.R": "tests/baselinenowcast/r/cases.R",
    "scripts/check_fixtures.py": "deleted",
    "scripts/cache_bust.py": "docs/hooks/cache_bust.py",
    "scripts/build_diagrams.sh": "deleted",
    "scripts/build_zone_centres.py": (
        "packages/nowcastbdbv/data-raw/build_zone_centres.py"
    ),
    "scripts/hub_imputation_fit.R": "tests/model/test_hub_imputation_delays.py",
    "scripts/fit-store.sh": "packages/bdbvhub/src/bdbvhub/store/branch.py",
    "scripts/results-release.sh": (
        "packages/bdbvhub/src/bdbvhub/submission/release.py"
    ),
    "scripts/paper.sh": "workflow/scripts/paper_run.py",
    "scripts/validate-hub.R": "tests/submission/hub_validation/validate_hub.R",
    "scripts/validate_with_hubvalidations.R": (
        "tests/submission/hub_validation/validate_with_hubvalidations.R"
    ),
    "tools/README.md": "deleted",
    "tools/paper_local.sh": "deleted",
    "tools/run_target.sh": "deleted",
}
"""Each former helper and where it went."""


def test_scripts_and_tools_are_gone() -> None:
    assert not (ROOT / "scripts").exists()
    assert not (ROOT / "tools").exists()


def test_moved_helpers_are_gone() -> None:
    left = [old for old in MOVED if (ROOT / old).exists()]
    assert not left


def test_moved_files_are_where_they_went() -> None:
    files = [new for new in MOVED.values() if "/" in new]
    assert [f for f in files if not (ROOT / f).is_file()] == []


def test_commands_are_console_scripts() -> None:
    text = "\n".join(
        p.read_text() for p in (ROOT / "packages").glob("*/pyproject.toml")
    )
    names = [n for n in MOVED.values() if "/" not in n and n != "deleted"]
    assert [n for n in names if f"\n{n} = " not in text] == []


def test_no_path_names_a_moved_helper() -> None:
    hits = []
    for path in [
        *ROOT.glob(".github/workflows/*.yml"),
        *ROOT.glob("workflow/rules/*.smk"),
        *ROOT.glob("taskfiles/*.yml"),
        *ROOT.glob("ci/steps/*.sh"),
        ROOT / "Taskfile.yml",
        ROOT / "mkdocs.yml",
    ]:
        text = path.read_text()
        hits += [f"{path.name}: {old}" for old in MOVED if old in text]
    assert hits == []


def test_panoptes_database_is_ignored_not_committed() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", ".panoptes.db"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert tracked == ""
    assert ".panoptes.db" in (ROOT / ".gitignore").read_text().split()
