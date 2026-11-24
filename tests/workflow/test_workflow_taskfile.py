"""Each Taskfile lives beside its code; the root only includes them."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

pytestmark = pytest.mark.skipif(not shutil.which("task"), reason="needs task")

# Each task before the projects had Taskfiles, and its task now.
MOVED = {
    "setup": "setup",
    "test": "test",
    "test-slow": "test-slow",
    "test-all": "test-all",
    "lint": "lint",
    "format": "format",
    "docs": "docs",
    "package-docs": "package-docs",
    "serve": "serve",
    "getting-started": "getting-started",
    "paper": "paper",
    "paper:targets": "paper:targets",
    "paper:render": "paper",
    "data": "nowcastbdbv:data",
    "data-bvd": "nowcastbdbv:data-bvd",
    "backtest": "bdbvhub:backtest",
    "backtest-lopo": "bdbvhub:backtest-lopo",
    "backtest-zones": "bdbvhub:backtest-zones",
    "results": "bdbvhub:results",
    "score-history": "nowcasteval:score-history",
    "germany": "nowcastgermany:benchmark",
    "compare-hub": "bdbvhub:compare-hub",
    "plots": "bdbvhub:plots",
    "simulate": "workflow:simulate",
    "push": "workflow:push",
    "release": "workflow:release",
    "finetune": "workflow:finetune",
    "publish": "workflow:publish",
    "score": "workflow:score",
    "land": "workflow:land",
    "dashboard": "workflow:dashboard",
    "dashboard:static": "workflow:dashboard-static",
    "all": "workflow:all",
    "tick": "workflow:tick",
    "run": "workflow:run",
    "monitor": "workflow:monitor",
    "dag": "workflow:dag",
    "rulegraph": "workflow:rulegraph",
    "progress": "workflow:progress",
    "report": "workflow:report",
    "workflow:check": "workflow:check",
    "nowcast": "bdbvhub:nowcast",
    "metadata": "bdbvhub:metadata",
    "validate-hub": "bdbvhub:validate",
    "submission-preflight": "ci:preflight",
    "ci-test": "ci:test",
    "ci-dry": "ci:dry",
    "ci-fit": "bdbvhub:ci-fit",
    "fit-store": "bdbvhub:fit-store",
    "fit-store-plan": "bdbvhub:fit-store-plan",
}

# The named dry runs.
DRY = ["dry:tick", "dry:paper", "dry:submit", "dry:fit-store"]

# Each included Taskfile's namespace and its folder (Sam, 24 Nov).
LOCATIONS = {
    "nowcastutils": "packages/nowcastutils",
    "nowcasteval": "packages/nowcasteval",
    "nowcastbaselines": "packages/nowcastbaselines",
    "nowcastrenewal": "packages/nowcastrenewal",
    "nowcastfoundation": "packages/nowcastfoundation",
    "nowcastgermany": "packages/nowcastgermany",
    "nowcastbdbv": "packages/nowcastbdbv",
    "bdbvhub": "packages/bdbvhub",
    "workflow": "workflow",
    "ci": "ci",
    "paper": "paper",
}


def _tasks() -> dict[str, dict]:
    out = subprocess.run(
        ["task", "--list-all", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {t["name"]: t for t in json.loads(out)["tasks"]}


def test_every_task_has_a_home():
    tasks = _tasks()
    missing = sorted(new for new in MOVED.values() if new not in tasks)
    assert missing == []


def test_the_dry_runs_are_named():
    tasks = _tasks()
    assert [name for name in DRY if name not in tasks] == []


def test_the_root_keeps_only_top_level_tasks():
    tasks = _tasks()
    top = {name for name in tasks if ":" not in name}
    assert top == {new for new in MOVED.values() if ":" not in new}


def test_the_root_includes_each_taskfile_from_its_folder():
    import yaml

    root = yaml.safe_load((ROOT / "Taskfile.yml").read_text())
    got = {
        ns: (inc["taskfile"], inc["dir"])
        for ns, inc in root["includes"].items()
    }
    assert got == {
        ns: (f"{folder}/Taskfile.yml", folder)
        for ns, folder in LOCATIONS.items()
    }
    assert not (ROOT / "taskfiles").exists()


@pytest.mark.parametrize("namespace", sorted(LOCATIONS))
def test_each_taskfile_exists_and_holds_its_namespace(namespace):
    folder = ROOT / LOCATIONS[namespace]
    assert (folder / "Taskfile.yml").is_file()
    mine = [t for n, t in _tasks().items() if n.startswith(f"{namespace}:")]
    assert mine
    for task in mine:
        assert task["location"]["taskfile"] == str(folder / "Taskfile.yml")


@pytest.mark.parametrize("namespace", sorted(LOCATIONS))
def test_each_folder_lists_its_tasks_on_its_own(namespace):
    out = subprocess.run(
        ["task", "--list-all"],
        cwd=ROOT / LOCATIONS[namespace],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


def test_contributing_has_the_old_to_new_table():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    for old, new in MOVED.items():
        if old != new:
            assert f"| `{old}` | `{new}` |" in text, old


@pytest.mark.parametrize(
    ("where", "task"),
    [
        (".", "workflow:monitor"),
        ("workflow", "monitor"),
        (".", "ci:preflight"),
        ("ci", "preflight"),
        (".", "nowcastutils:test"),
        ("packages/nowcastutils", "test"),
        (".", "bdbvhub:fit-store-plan"),
        ("packages/bdbvhub", "fit-store-plan"),
    ],
)
def test_a_task_runs_at_the_repository_root_from_anywhere(where, task):
    # The commands and the `sh:` vars alike: a dry run resolves the vars
    # (localconfig.py, the hub config, the ci project) and prints the
    # commands, and makes no folder of its own.
    before = {p for p in ROOT.rglob("home") if ".venv" not in p.parts}
    out = subprocess.run(
        ["task", "--dry", task],
        cwd=ROOT / where,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    after = {p for p in ROOT.rglob("home") if ".venv" not in p.parts}
    assert after == before
