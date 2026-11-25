"""Previewing the paper from the Site build's inputs (fetch_build.py)."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

PAPER = Path(__file__).parents[1]
ROOT = PAPER.parent
FETCH = PAPER / "scripts" / "fetch_build.py"


def _bundle(path: Path, files: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_fetch_unpacks_the_bundle_into_build(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "paper-build.tar.gz",
        {
            "build/results.yml": "a: 1\n",
            "build/figures/main/x.png": "png",
        },
    )
    paper = tmp_path / "paper"
    (paper / "build").mkdir(parents=True)
    (paper / "build" / "stale.txt").write_text("old")
    out = subprocess.run(
        [
            sys.executable,
            str(FETCH),
            "--url",
            bundle.as_uri(),
            "--paper",
            str(paper),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert (paper / "build" / "results.yml").read_text() == "a: 1\n"
    assert (paper / "build" / "figures" / "main" / "x.png").is_file()
    # The bundle replaces the folder whole.
    assert not (paper / "build" / "stale.txt").exists()


def test_fetch_refuses_paths_outside_build(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bad.tar.gz", {"../escape.txt": "x"})
    paper = tmp_path / "paper"
    paper.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            str(FETCH),
            "--url",
            bundle.as_uri(),
            "--paper",
            str(paper),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert not (tmp_path / "escape.txt").exists()


def test_the_preview_task_fetches_then_previews() -> None:
    tasks = yaml.safe_load((PAPER / "Taskfile.yml").read_text())["tasks"]
    cmds = [str(c) for c in tasks["preview"]["cmds"]]
    assert "scripts/fetch_build.py" in cmds[0]
    assert cmds[-1].startswith("quarto preview main.qmd")


def test_pages_publishes_the_build_it_rendered() -> None:
    pages = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
    job = pages["jobs"]["paper-build"]
    assert job["needs"] == "build"
    run = "\n".join(s.get("run", "") for s in job["steps"])
    assert "gh release upload paper-build" in run
    assert "--clobber" in run
