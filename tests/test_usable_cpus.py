"""The usable CPU count (`nowcastutils.devices.usable_cpus`)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from nowcastutils.devices import usable_cpus

ROOT = Path(__file__).parents[1]


def test_usable_cpus_follow_the_affinity_where_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: {0, 3}, raising=False
    )
    assert usable_cpus() == 2


def test_usable_cpus_fall_back_to_the_host_without_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As on macOS and Windows, which have no `os.sched_getaffinity`.
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
            "packages",
            "workflow",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert found == ["packages/nowcastutils/src/nowcastutils/devices.py"]
