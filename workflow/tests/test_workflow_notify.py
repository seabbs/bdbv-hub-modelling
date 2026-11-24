"""Mailbox lines and the evidence branch guard."""

import pytest
from flowlib import notify


def test_lines_are_posted_only_when_asked(tmp_path):
    box = tmp_path / "box.md"
    cfg = {"MAILBOX": str(box), "HOST_NAME": "h", "NOTIFY": ""}
    notify.post("round 4 ready", cfg)
    assert not box.exists()
    notify.post("round 4 ready", {**cfg, "NOTIFY": "1"})
    line = box.read_text()
    assert line.endswith(" h workflow: round 4 ready\n")
    assert line[:4].isdigit()


def test_land_refuses_main_and_other_branches():
    notify.check_branch("checkpoint-scoring", "checkpoint-scoring")
    with pytest.raises(SystemExit, match="main"):
        notify.check_branch("main", "main")
    with pytest.raises(SystemExit, match="checkpoint-scoring"):
        notify.check_branch("snakemake-workflow", "checkpoint-scoring")


def test_land_commits_the_tables_and_checks_they_reached_origin(
    tmp_path, monkeypatch
):
    import subprocess

    def git(*args, cwd=tmp_path / "work"):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=cwd, check=True, capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip

    git("init", "-q", "--bare", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), "work", cwd=tmp_path)
    git("checkout", "-q", "-b", "evidence")
    git("commit", "-q", "--allow-empty", "-m", "start")
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("x\n")
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    head = notify.land(
        (tmp_path / "a.csv", tmp_path / "b.csv"), "ev.csv", "ck.csv", "evidence"
    )
    assert head == git("rev-parse", "origin/evidence")
    assert git("show", "--name-only", "--format=", "HEAD").split() == [
        "ck.csv",
        "ev.csv",
    ]
