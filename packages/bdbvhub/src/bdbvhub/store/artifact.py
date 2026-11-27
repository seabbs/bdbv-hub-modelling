"""The fit store as GitHub Actions artifacts (`bdbvhub.store.store`).

    fit-store-artifact guard EVENT REF SCRATCH
        print `true` if the run may write a store, `false` if it only
        plans, and fail if SCRATCH is not a scratch store name
    fit-store-artifact fetch DIR [BUNDLE ...]
        unpack the newest store into DIR (left empty if there is none),
        with its version in DIR/.commit, and each BUNDLE (`backtests`,
        `site`) into DIR/BUNDLE; fail if the artifacts cannot be read
    fit-store-artifact merge DIR NEW KEYS PLAN
        add NEW's planned fits to the store in DIR and prune it
    fit-store-artifact latest
        print the newest store's version (blank if there is none)

The store is the newest unexpired artifact named FIT_STORE (default
`fit-store`) made by a run in this repository; for the main store, by a
run on main. Each write fetches it, merges its fits and uploads the
whole store again under the same name (`actions/upload-artifact`), so
the newest artifact is the store. Its bundles are the artifacts
`<FIT_STORE>-backtests` and `<FIT_STORE>-site`, each `latest.tar.gz`
with `latest.json` beside it. Artifacts expire: a store that has
expired reads as empty, so its fits are made again, and a missing
bundle as not yet made.

Until the first store artifact is uploaded, `fetch` reads the git
branch of the store's name, which held the store before, so nothing is
refitted; each bundle's parts there are joined into `latest.tar.gz`.

KEYS is the JSON list of current keys, `<model>/<origin>/<key>`, and
PLAN the matrix of planned fits, as `fit-store plan --keys --out` write
them. Only planned files are copied, each fit with its zone run beside it
(`<key>.zones.csv.gz`, a walk's). For each pair in KEYS the merge keeps
the current key's fit and the newest good fit of older code, and deletes
the rest with their zone runs.

Standard library, `gh` and git only, so a CI job with no forecast
environment runs it as `python3 -I <this file> ...` with GH_TOKEN set.
GITHUB_REPOSITORY names the repository (default: `gh repo view`'s).
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
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MAIN = "fit-store"
"""The main store's name, written from main only."""

SCRATCH_PREFIX = "fit-store-scratch"
"""The prefix a scratch store's name must have."""

BUNDLES = ("backtests", "site")
"""The bundles stored beside the fits, each its own artifact."""

ZONES_SUFFIX = ".zones.csv.gz"
"""A fit's zone run beside it (`bdbvhub.store.store.ZONES_SUFFIX`)."""

VERSION_FILE = ".commit"
"""The fetched store's version (`bdbvhub.store.store.COMMIT_FILE`)."""


class StoreError(Exception):
    """The artifacts could not be read."""


def _name() -> str:
    return os.environ.get("FIT_STORE") or MAIN


def _tries() -> int:
    return int(os.environ.get("FIT_STORE_TRIES") or 3)


def _error(message: str) -> int:
    print(f"::error::{message}", file=sys.stderr)
    return 1


def _warn(message: str) -> None:
    print(f"::warning::{message}", file=sys.stderr)


def guard(event: str, ref: str, scratch: str = "") -> int:
    """Print whether this run may write a store; fail on a bad SCRATCH."""
    if scratch and not scratch.startswith(SCRATCH_PREFIX):
        return _error(f"store must start with {SCRATCH_PREFIX}")
    if event == "pull_request":
        write = False
    else:
        write = bool(scratch) or ref == "refs/heads/main"
    print("true" if write else "false")
    return 0


def _gh(*args: str, out: Any = subprocess.PIPE) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args], stdout=out, stderr=subprocess.PIPE, check=False
    )


def _repo() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    got = _gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    if got.returncode != 0:
        raise StoreError("could not tell the repository; set GITHUB_REPOSITORY")
    return got.stdout.decode().strip()


def _trusted(art: dict[str, Any], store: str) -> bool:
    """Whether ART may be read as STORE's: made here, on main for MAIN."""
    run = art.get("workflow_run") or {}
    here = run.get("head_repository_id") == run.get("repository_id")
    if store.startswith(SCRATCH_PREFIX):
        return here
    return here and run.get("head_branch") == "main"


def newest(repo: str, name: str, store: str) -> dict[str, Any] | None:
    """Return the newest unexpired artifact NAME of STORE, or None."""
    got = _gh("api", f"repos/{repo}/actions/artifacts?name={name}&per_page=100")
    if got.returncode != 0:
        raise StoreError(f"could not read the artifacts named {name}")
    arts = [
        a
        for a in json.loads(got.stdout)["artifacts"]
        if a["name"] == name and not a["expired"] and _trusted(a, store)
    ]
    return max(arts, key=lambda a: a["created_at"]) if arts else None


def version(art: dict[str, Any]) -> str:
    """Return ART's version: its digest, else its id."""
    digest = art.get("digest") or ""
    return digest.removeprefix("sha256:") or str(art["id"])


def _download(repo: str, art: dict[str, Any], where: Path) -> bool:
    with tempfile.TemporaryFile() as tmp:
        got = _gh(
            "api", f"repos/{repo}/actions/artifacts/{art['id']}/zip", out=tmp
        )
        if got.returncode != 0:
            return False
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(where)
    return True


def _fetch_one(repo: str, name: str, store: str, where: Path) -> str | None:
    """Unpack the newest artifact NAME into WHERE; return its version.

    An artifact replaced while it downloads (a write in progress) is
    looked up again.
    """
    for attempt in range(1, _tries() + 1):
        try:
            art = newest(repo, name, store)
        except StoreError:
            if attempt == _tries():
                raise
        else:
            if art is None:
                return None
            if _download(repo, art, where):
                return version(art)
            if attempt == _tries():
                raise StoreError(f"could not download the artifact {name}")
        time.sleep(attempt * 5)
    return None


def _from_branch(store: str, where: Path) -> bool:
    """Copy the git branch STORE into WHERE, if it exists, parts joined."""
    found = subprocess.run(
        ["git", "ls-remote", "-q", "--exit-code", "--heads", "origin", store],
        capture_output=True, check=False,
    ).returncode  # fmt: skip
    if found != 0:
        return False
    fetched = subprocess.run(
        ["git", "fetch", "-q", "--depth", "1", "origin", store],
        capture_output=True, check=False,
    )  # fmt: skip
    if fetched.returncode != 0:
        raise StoreError(f"could not fetch the branch {store}")
    archive = subprocess.run(
        ["git", "archive", "FETCH_HEAD"], capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(where, filter="data")
    for bundle in BUNDLES:
        parts = sorted((where / bundle).glob("latest.tar.gz.*"))
        if parts:
            with (where / bundle / "latest.tar.gz").open("wb") as out:
                for part in parts:
                    out.write(part.read_bytes())
                    part.unlink()
    tip = subprocess.run(
        ["git", "rev-parse", "FETCH_HEAD"], capture_output=True, check=True
    ).stdout.decode()
    (where / VERSION_FILE).write_text(tip)
    print(f"Read the store from the branch {store}, as no artifact holds it.")
    return True


def fetch(where: Path, bundles: Sequence[str] = ()) -> int:
    """Unpack the newest store into WHERE and each of BUNDLES beside it."""
    store = _name()
    where.mkdir(parents=True, exist_ok=True)
    try:
        repo = _repo()
        found = _fetch_one(repo, store, store, where)
        if found is not None:
            (where / VERSION_FILE).write_text(found + "\n")
            for bundle in BUNDLES:
                shutil.rmtree(where / bundle, ignore_errors=True)
        elif _from_branch(store, where):
            for bundle in BUNDLES:
                if bundle not in bundles:
                    shutil.rmtree(where / bundle, ignore_errors=True)
        else:
            _warn(f"no store artifact {store}: the store is empty")
        for bundle in bundles:
            name = f"{store}-{bundle}"
            tmp = Path(tempfile.mkdtemp(prefix="fit-store."))
            if _fetch_one(repo, name, store, tmp) is None:
                shutil.rmtree(tmp)
                if not (where / bundle).is_dir():
                    _warn(f"no artifact {name}: its results are not yet made")
                continue
            shutil.rmtree(where / bundle, ignore_errors=True)
            shutil.move(tmp, where / bundle)
    except StoreError as err:
        return _error(str(err))
    return 0


def latest() -> int:
    """Print the newest store's version, blank if there is none."""
    store = _name()
    try:
        art = newest(_repo(), store, store)
    except StoreError as err:
        return _error(str(err))
    print(version(art) if art else "")
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


def merge(work: Path, new: Path, keys_file: Path, plan_file: Path) -> int:
    """Add NEW's planned fits to the store in WORK and prune it."""
    work.mkdir(parents=True, exist_ok=True)
    keys = json.loads(keys_file.read_text())
    plan = json.loads(plan_file.read_text())["include"]
    stored = prune(work, new, keys, plan)
    print("Store " + " ".join(stored) if stored else "Nothing to store.")
    return 0


USAGE = (
    "usage: fit-store-artifact guard EVENT REF SCRATCH"
    " | fetch DIR [BUNDLE ...] | merge DIR NEW KEYS PLAN | latest"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one action from the command line; return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    action, rest = (args[0], args[1:]) if args else ("", [])
    if action == "guard" and len(rest) in (2, 3):
        return guard(*rest)
    if (
        action == "fetch"
        and len(rest) >= 1
        and all(b in BUNDLES for b in rest[1:])
    ):
        return fetch(Path(rest[0]), rest[1:])
    if action == "merge" and len(rest) == 4:
        return merge(*(Path(r) for r in rest))
    if action == "latest" and not rest:
        return latest()
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
