"""The usable CPU count (`nowcastutils.devices.usable_cpus`)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nowcastutils.devices import usable_cpus

ROOT = Path(__file__).parents[3]


def test_usable_cpus_follow_the_affinity_where_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: {0, 3}, raising=False
    )
    assert usable_cpus() == 2


def test_usable_cpus_fall_back_to_the_host_without_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As on macOS and Windows, which have no `os.sched_getaffinity`.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert usable_cpus() == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert usable_cpus() == 1


def test_only_the_helper_reads_the_affinity() -> None:
    found = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "sched_getaffinity",
            "--",
            ":(glob)packages/*/src/**",
            "workflow",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert found == ["packages/nowcastutils/src/nowcastutils/devices.py"]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_the_helper_type_checks_on_every_platform(platform: str) -> None:
    # mypy checks `os` as the platform has it: macOS has no affinity.
    pytest.importorskip("mypy")
    got = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--platform",
            platform,
            "--no-incremental",
            "packages/nowcastutils/src/nowcastutils/devices.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert got.returncode == 0, got.stdout
