"""The fit store branch (`bdbvhub.store.store`): guard, fetch and push.

    fit-store-branch guard EVENT REF SCRATCH
        print `true` if the run may write a store, `false` if it only
        plans, and fail if SCRATCH is not a scratch branch name
    fit-store-branch fetch DIR
        copy the branch's files into DIR (empty if there is no branch),
        with the commit fetched in DIR/.commit, and fail if the remote
        cannot be read
    fit-store-branch push DIR KEYS PLAN
        add DIR's planned fits to the branch, prune it and push
    fit-store-branch push-backtests BUNDLE COMMIT
        store BUNDLE as backtests/latest.tar.gz, made at COMMIT, and push
    fit-store-branch push-site BUNDLE COMMIT KEY
        store BUNDLE as site/latest.tar.gz, made at COMMIT from inputs
        with hash KEY, and push

KEYS is the JSON list of current keys, `<model>/<origin>/<key>`, and
PLAN the matrix of planned fits, as `fit-store plan --keys --out` write
them. Only planned files are copied, each fit with its zone run beside it
(`<key>.zones.csv.gz`, a walk's). For each pair in KEYS the push keeps
the current key's fit and the newest good fit of older code, and deletes
the rest with their zone runs. FIT_STORE_BRANCH names the branch (default
fit-store). Run in a checkout whose `origin` holds the branch.

Standard library and git only, so a CI job with no forecast
environment runs it as `python3 -I <this file> ...`.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

SCRATCH_PREFIX = "fit-store-scratch"
"""The prefix a scratch store branch must have."""

BOT = (
    "github-actions[bot]",
    "41898282+github-actions[bot]@users.noreply.github.com",
)
"""The name and email the store's commits are made as."""

ZONES_SUFFIX = ".zones.csv.gz"
"""A fit's zone run beside it (`bdbvhub.store.store.ZONES_SUFFIX`)."""

PUSH_TRIES = 5
"""Times a push is tried, refetching the branch each time."""


def _branch() -> str:
    return os.environ.get("FIT_STORE_BRANCH") or "fit-store"


def _git(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, check=check
    )


def _error(message: str) -> int:
    print(f"::error::{message}", file=sys.stderr)
    return 1


def guard(event: str, ref: str, scratch: str = "") -> int:
    """Print whether this run may write a store; fail on a bad SCRATCH."""
    if scratch and not scratch.startswith(SCRATCH_PREFIX):
        return _error(f"store_branch must start with {SCRATCH_PREFIX}")
    if event == "pull_request":
        write = False
    else:
        write = bool(scratch) or ref == "refs/heads/main"
    print("true" if write else "false")
    return 0


def _fetch_tip(branch: str) -> bool:
    return (
        _git(
            "fetch", "-q", "--depth", "1", "origin", branch, check=False
        ).returncode
        == 0
    )


def fetch(where: Path) -> int:
    """Copy the branch's files into WHERE, with the commit in `.commit`."""
    branch = _branch()
    where.mkdir(parents=True, exist_ok=True)
    # Exit 2 means the branch does not exist: an empty store, not an error.
    found = _git(
        "ls-remote", "-q", "--exit-code", "--heads", "origin", branch,
        check=False,
    ).returncode  # fmt: skip
    if found == 2:
        return 0
    if found != 0:
        return _error(f"could not read {branch} from origin")
    if not _fetch_tip(branch):
        return _error(f"could not fetch {branch}")
    archive = _git("archive", "FETCH_HEAD").stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(where, filter="data")
    tip = _git("rev-parse", "FETCH_HEAD").stdout.decode()
    (where / ".commit").write_text(tip)
    return 0


def prune(
    work: Path, new: Path, keys: Sequence[str], plan: Sequence[dict[str, str]]
) -> list[str]:
    """Copy NEW's planned fits into WORK and prune it; return those stored.

    For each `<model>/<origin>/<key>` in KEYS, WORK keeps that key's
    file and the newest good fit of older code, and loses the rest. A
    fit's zone run goes with it.
    """
    stored = []
    for fit in plan:
        stem = f"{fit['model']}/{fit['origin']}/{fit['key']}"
        name = f"{stem}.json.gz"
        if (new / name).is_file():
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(new / name, work / name)
            zones = f"{stem}{ZONES_SUFFIX}"
            if (new / zones).is_file():
                shutil.copyfile(new / zones, work / zones)
            stored.append(stem)
    for entry in keys:
        model, origin, key = entry.split("/")
        older = []
        for path in (work / model / origin).glob("*.json.gz"):
            if path.name == f"{key}.json.gz":
                continue
            meta = json.loads(gzip.decompress(path.read_bytes()))["meta"]
            older.append((meta["status"] == "ok", meta["fitted_at"], path))
        good = sorted(f for f in older if f[0])
        keep = good[-1][2] if good else None
        for _, _, path in older:
            if path != keep:
                path.unlink()
                path.with_name(
                    path.name.removesuffix(".json.gz") + ZONES_SUFFIX
                ).unlink(missing_ok=True)
    return stored


def _commit(stage: Callable[[Path], str]) -> int:
    """Stage files on the branch with STAGE, commit and push them.

    STAGE changes a checkout of the branch's tip and returns the commit
    message. A push that loses a race refetches and stages again. One the
    remote refuses (`[remote rejected]`: a file over GitHub's size limit,
    a rule) fails at once. A failed push shows git's reason.
    """
    branch = _branch()
    orphan = f"{branch}-new"
    for attempt in range(1, PUSH_TRIES + 1):
        work = Path(tempfile.mkdtemp(prefix="fit-store."))
        work.rmdir()
        if _fetch_tip(branch):
            _git("worktree", "add", "-q", "--detach", str(work), "FETCH_HEAD")
        else:
            _git("worktree", "add", "-q", "--detach", str(work))
            _git("switch", "-q", "--orphan", orphan, cwd=work)
        try:
            msg = stage(work)
            _git("add", "-A", cwd=work)
            if (
                _git(
                    "diff", "--cached", "--quiet", cwd=work, check=False
                ).returncode
                == 0
            ):
                print("Nothing to store.")
                return 0
            _git(
                "-c", f"user.name={BOT[0]}", "-c", f"user.email={BOT[1]}",
                "commit", "-q", "-m", msg, cwd=work,
            )  # fmt: skip
            pushed = _git(
                "push", "-q", "origin", f"HEAD:refs/heads/{branch}",
                cwd=work, check=False,
            )  # fmt: skip
        finally:
            _git("worktree", "remove", "--force", str(work), check=False)
            _git("branch", "-q", "-D", orphan, check=False)
        if pushed.returncode == 0:
            print(msg)
            return 0
        reason = pushed.stderr.decode(errors="replace").strip()
        print(reason, file=sys.stderr)
        if "[remote rejected]" in reason:
            break
        if attempt < PUSH_TRIES:
            time.sleep(attempt * 5)
    return _error(f"could not push to {branch}")


def push(new: Path, keys_file: Path, plan_file: Path) -> int:
    """Add NEW's planned fits to the branch, prune it and push it."""
    new, keys_file, plan_file = (
        p.resolve() for p in (new, keys_file, plan_file)
    )
    keys = json.loads(keys_file.read_text())
    plan = json.loads(plan_file.read_text())["include"]
    return _commit(
        lambda work: "Store " + " ".join(prune(work, new, keys, plan))
    )


BACKTESTS = "backtests"
"""The branch folder of the stored backtests, which `prune` never
touches: `latest.tar.gz` (their results and outputs, as Pages unpacks
them) and `latest.json` (the commit that made them)."""


def push_backtests(bundle: Path, commit: str) -> int:
    """Store BUNDLE as the branch's backtests, made at COMMIT."""
    bundle = bundle.resolve()

    def stage(work: Path) -> str:
        folder = work / BACKTESTS
        folder.mkdir(exist_ok=True)
        shutil.copyfile(bundle, folder / "latest.tar.gz")
        meta = {"commit": commit}
        (folder / "latest.json").write_text(json.dumps(meta) + "\n")
        return f"Store the backtests of {commit}"

    return _commit(stage)


SITE = "site"
"""The branch folder of the site's results (site-results.yml),
which `prune` never touches: `latest.tar.gz` (the results and docs files
Pages unpacks) and `latest.json` (the commit that made them and the hash
of their inputs, so unchanged inputs are not made again)."""


def push_site(bundle: Path, commit: str, key: str) -> int:
    """Store BUNDLE as the branch's site results, made at COMMIT."""
    bundle = bundle.resolve()

    def stage(work: Path) -> str:
        folder = work / SITE
        folder.mkdir(exist_ok=True)
        shutil.copyfile(bundle, folder / "latest.tar.gz")
        meta = {"commit": commit, "key": key}
        (folder / "latest.json").write_text(json.dumps(meta) + "\n")
        return f"Store the site results of {commit}"

    return _commit(stage)


USAGE = (
    "usage: fit-store-branch guard EVENT REF SCRATCH | fetch DIR"
    " | push DIR KEYS PLAN | push-backtests BUNDLE COMMIT"
    " | push-site BUNDLE COMMIT KEY"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one action from the command line; return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    action, rest = (args[0], args[1:]) if args else ("", [])
    if action == "guard" and len(rest) in (2, 3):
        return guard(*rest)
    if action == "fetch" and len(rest) == 1:
        return fetch(Path(rest[0]))
    if action == "push" and len(rest) == 3:
        return push(*(Path(r) for r in rest))
    if action == "push-backtests" and len(rest) == 2:
        return push_backtests(Path(rest[0]), rest[1])
    if action == "push-site" and len(rest) == 3:
        return push_site(Path(rest[0]), rest[1], rest[2])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
