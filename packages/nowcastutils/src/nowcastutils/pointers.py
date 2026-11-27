# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Stable releases that name each key workflow endpoint's current version.

Each version of an endpoint keeps its own release (a content-hashed or
settings-hashed tag). The endpoint's stable release `<endpoint>-main`
holds one asset, `main.json`, naming the version that machines and CI
read: its tag, hash, the commit and step that produced it, and when.
The workflow rule that produces the endpoint promotes it (`promote`),
from a commit on one of the configured branches only and only once the
version's release exists. Readers take the record (`read`) and fail if
it is missing; they never guess the newest release of a prefix.

`promote` and `read` run `gh` (and `git`) through `run`, as
`subprocess.run`, so tests pass a stand-in.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["ASSET", "SUFFIX", "promote", "published_commit", "read", "tag"]

ASSET = "main.json"
"""Each pointer release's one asset: the current version's record."""

SUFFIX = "-main"
"""A pointer release's tag is its endpoint's name and this."""

Run = Callable[..., "subprocess.CompletedProcess[str]"]


def tag(endpoint: str) -> str:
    """Return the tag of an endpoint's pointer release."""
    return f"{endpoint}{SUFFIX}"


def published_commit(branches: Iterable[str], run: Run = subprocess.run) -> str:
    """Return HEAD's commit if it is on one of the remote `branches`.

    Each branch is fetched first (a failed fetch leaves the last one), so
    a commit pushed since still counts. A detached HEAD, as on a
    training host or in CI, counts when its commit is on such a branch.

    Raises
    ------
    RuntimeError
        If no branch is configured or HEAD is on none of them.
    """
    names = list(branches)
    if not names:
        raise RuntimeError("no branch is configured to promote from")
    head = run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    for branch in names:
        run(["git", "fetch", "-q", "origin", branch], capture_output=True)
        seen = run(
            ["git", "merge-base", "--is-ancestor", head, f"origin/{branch}"],
            capture_output=True,
        )
        if seen.returncode == 0:
            return head
    raise RuntimeError(
        f"HEAD ({head or 'unknown'}) is on none of {', '.join(names)}, "
        "so nothing is promoted; promote from a pushed commit of those."
    )


def _gh(run: Run, repo: str, *command: str) -> subprocess.CompletedProcess[str]:
    done: subprocess.CompletedProcess[str] = run(
        ["gh", "release", *command, "-R", repo],
        capture_output=True,
        text=True,
    )
    return done


def promote(
    endpoint: str,
    target: str,
    *,
    hash: str,
    produced_by: str,
    branches: Iterable[str],
    repo: str,
    title: str,
    needs: str | None = None,
    extra: Mapping[str, Any] | None = None,
    run: Run = subprocess.run,
) -> dict[str, Any]:
    """Name release `target` the current version of `endpoint`.

    Writes `ASSET` to the release `tag(endpoint)`, made on first use and
    replaced in place after.

    Parameters
    ----------
    endpoint : str
        The endpoint, such as `chronos` or `paper-germany`.
    target : str
        The version's own release tag.
    hash : str
        The version's hash (of its settings or contents).
    produced_by : str
        The workflow step that made the version.
    branches : iterable of str
        The branches a promotion may run from (`published_commit`).
    repo : str
        The GitHub repository, `owner/name`.
    title : str
        The pointer release's title.
    needs : str, optional
        An asset the version's release must hold, which readers load.
    extra : mapping, optional
        More fields for the record.
    run : callable
        Runs a command, as `subprocess.run`.

    Returns
    -------
    dict
        The record written.

    Raises
    ------
    RuntimeError
        If HEAD is on no configured branch, the version's release (or
        its `needs` asset) is missing, or `gh` fails.
    """
    commit = published_commit(branches, run)
    view = _gh(run, repo, "view", target, "--json", "assets")
    if view.returncode != 0:
        raise RuntimeError(f"{target} is not released, so it is not promoted")
    assets = {a["name"] for a in json.loads(view.stdout)["assets"]}
    if needs is not None and needs not in assets:
        raise RuntimeError(f"{target} has no {needs}, so it is not promoted")
    record = {
        "endpoint": endpoint,
        "tag": target,
        "hash": hash,
        "commit": commit,
        "produced_by": produced_by,
        "promoted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    pointer = tag(endpoint)
    notes = (
        f"The current {endpoint}: {target}, made by {produced_by} at "
        f"{commit[:12]}. Readers load the release {ASSET} names; each "
        "version keeps its own release. The workflow updates this from "
        "its configured branches only.\n\nThis was opened by a bot. "
        "Please ping @seabbs for any questions."
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ASSET
        path.write_text(json.dumps(record, indent=1) + "\n")
        steps: list[tuple[str, ...]]
        if _gh(run, repo, "view", pointer, "--json", "assets").returncode:
            steps = [
                ("create", pointer, "--latest=false", "--title", title,
                 "--notes", notes, str(path)),
            ]  # fmt: skip
        else:
            steps = [
                ("upload", pointer, str(path), "--clobber"),
                ("edit", pointer, "--notes", notes),
            ]
        for step in steps:
            done = _gh(run, repo, *step)
            if done.returncode != 0:
                raise RuntimeError(
                    f"could not {step[0]} {pointer}: {done.stderr}"
                )
    return record


def read(endpoint: str, repo: str, run: Run = subprocess.run) -> dict[str, Any]:
    """Return the record of `endpoint`'s current version.

    Raises
    ------
    LookupError
        If the pointer release or its `ASSET` is missing: the endpoint
        was never promoted.
    ValueError
        If the record names no release.
    RuntimeError
        If `gh` fails otherwise.
    """
    pointer = tag(endpoint)
    done = _gh(run, repo, "download", pointer, "-p", ASSET, "-O", "-")
    if done.returncode != 0:
        if "not found" in done.stderr or "no assets" in done.stderr:
            raise LookupError(
                f"{endpoint} was never promoted: the release {pointer} of "
                f"{repo} has no {ASSET}."
            )
        raise RuntimeError(f"could not read {pointer}: {done.stderr}")
    record: dict[str, Any] = json.loads(done.stdout)
    target = str(record.get("tag") or "")
    if not target or target == pointer:
        raise ValueError(f"{ASSET} of {pointer} names no release: {record}")
    return record
