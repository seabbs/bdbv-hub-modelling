"""Optional notification lines, and the guard on where evidence is pushed.

The workflow needs neither. A line is appended to `MAILBOX` only when
both `NOTIFY` and `MAILBOX` are set in the local config, and a line that
cannot be written is a warning on stderr, never a failure of the rule
that posted it.
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def post(message: str, cfg: Mapping[str, str]) -> None:
    """Append `<time> <host> workflow: <message>` to the mailbox."""
    if not (cfg.get("NOTIFY") and cfg.get("MAILBOX")):
        return
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{stamp} {cfg.get('HOST_NAME', '')} workflow: {message}\n"
    try:
        with Path(cfg["MAILBOX"]).expanduser().open("a") as f:
            f.write(line)
    except OSError as err:
        print(
            f"notify: could not write {cfg['MAILBOX']}: {err}", file=sys.stderr
        )


def check_branch(current: str, allowed: str) -> None:
    """Raise unless evidence is landed from the branch it belongs to."""
    if current in ("main", "master"):
        raise SystemExit(f"never land evidence on {current}")
    if current != allowed:
        raise SystemExit(
            f"land runs on the evidence branch {allowed}, not {current}"
        )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def land(
    tables: tuple[Path | str, Path | str],
    evidence: str,
    checkpoints: str,
    branch: str,
) -> str:
    """Commit the score tables on the evidence branch, push, return the head.

    Raises unless the checkout is on `branch` and the commit reached
    origin.
    """
    check_branch(_git("rev-parse", "--abbrev-ref", "HEAD"), branch)
    shutil.copy(tables[0], evidence)
    shutil.copy(tables[1], checkpoints)
    _git("add", evidence, checkpoints)
    if _git("diff", "--cached", "--name-only"):
        _git("commit", "-q", "-m", "evidence: checkpoint scores")
    _git("push", "-q", "origin", branch)
    _git("fetch", "-q", "origin", branch)
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", f"origin/{branch}"):
        raise SystemExit(f"{head} is not on origin/{branch}")
    return head
