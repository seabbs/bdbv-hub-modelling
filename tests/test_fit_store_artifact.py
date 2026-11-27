from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "packages/bdbvhub/src/bdbvhub/store/artifact.py"
)
"""Run by path with `-I`, as the CI jobs with no forecast environment do."""

REPO = "owner/repo"

FAKE_GH = '''#!/usr/bin/env python3
"""A fake `gh api` serving the artifacts in $FAKE_GH/artifacts.json."""
import json, os, sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

root = Path(os.environ["FAKE_GH"])
with (root / "calls.txt").open("a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
if (root / "fail").exists() or sys.argv[1] != "api":
    sys.exit(1)
url = urlparse(sys.argv[2])
arts = json.loads((root / "artifacts.json").read_text())
if url.path.endswith("/zip"):
    art = url.path.split("/")[-2]
    if not (root / f"{art}.zip").exists():
        sys.exit(1)
    sys.stdout.buffer.write((root / f"{art}.zip").read_bytes())
else:
    name = parse_qs(url.query)["name"][0]
    found = [a for a in arts if a["name"] == name]
    print(json.dumps({"total_count": len(found), "artifacts": found}))
'''


def _run(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(MODULE), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def gh(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A fake `gh` on PATH serving artifacts from a folder."""
    server = tmp_path / "server"
    (server / "bin").mkdir(parents=True)
    (server / "artifacts.json").write_text("[]")
    fake = server / "bin" / "gh"
    fake.write_text(FAKE_GH)
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{server / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH": str(server),
        "GITHUB_REPOSITORY": REPO,
    }
    env.pop("FIT_STORE", None)
    return server, env


def _serve(
    server: Path,
    id_: int,
    name: str,
    files: dict[str, bytes],
    *,
    branch: str = "main",
    created: str = "2026-11-01T00:00:00Z",
    expired: bool = False,
    fork: bool = False,
) -> None:
    """Serve an artifact NAME holding FILES, made by a run on BRANCH."""
    arts = json.loads((server / "artifacts.json").read_text())
    arts.append(
        {
            "id": id_,
            "name": name,
            "expired": expired,
            "created_at": created,
            "digest": f"sha256:{id_:064x}",
            "workflow_run": {
                "id": 1000 + id_,
                "head_branch": branch,
                "repository_id": 1,
                "head_repository_id": 2 if fork else 1,
            },
        }
    )
    (server / "artifacts.json").write_text(json.dumps(arts))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, data in files.items():
            z.writestr(path, data)
    (server / f"{id_}.zip").write_bytes(buf.getvalue())


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


@pytest.mark.parametrize(
    ("event", "ref", "scratch", "write"),
    [
        ("push", "refs/heads/main", "", "true"),
        ("schedule", "refs/heads/main", "", "true"),
        ("workflow_dispatch", "refs/heads/branch", "", "false"),
        ("workflow_dispatch", "refs/heads/branch", "fit-store-scratch", "true"),
        ("workflow_dispatch", "refs/heads/main", "fit-store-scratch-a", "true"),
        ("pull_request", "refs/pull/1/merge", "", "false"),
    ],
)
def test_guard_writes_main_and_scratch_stores_only(
    event: str, ref: str, scratch: str, write: str
) -> None:
    out = _run("guard", event, ref, scratch)
    assert out.returncode == 0
    assert out.stdout.strip() == write


@pytest.mark.parametrize("scratch", ["main", "fit-store", "gh-pages", "x"])
def test_guard_refuses_other_store_names(scratch: str) -> None:
    out = _run("guard", "workflow_dispatch", "refs/heads/branch", scratch)
    assert out.returncode != 0


def test_fetch_unpacks_the_newest_store_made_on_main(
    tmp_path: Path, gh: tuple[Path, dict[str, str]]
) -> None:
    server, env = gh
    _serve(server, 1, "fit-store", {"m/o/old.json.gz": b"old"})
    _serve(
        server, 3, "fit-store", {"m/o/new.json.gz": b"new"},
        created="2026-11-03T00:00:00Z",
    )  # fmt: skip
    # Newer, but from a branch, a fork or expired: never the main store.
    _serve(
        server, 4, "fit-store", {"m/o/branch.json.gz": b"x"},
        branch="other", created="2026-11-04T00:00:00Z",
    )  # fmt: skip
    _serve(
        server, 5, "fit-store", {"m/o/fork.json.gz": b"x"},
        fork=True, created="2026-11-05T00:00:00Z",
    )  # fmt: skip
    _serve(
        server, 6, "fit-store", {"m/o/gone.json.gz": b"x"},
        expired=True, created="2026-11-06T00:00:00Z",
    )  # fmt: skip
    got = tmp_path / "got"
    out = _run("fetch", str(got), cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr
    assert _files(got) == [".commit", "m/o/new.json.gz"]
    # The store's version, which `fit-store unpack` reports.
    assert (got / ".commit").read_text().strip() == f"{3:064x}"


def test_fetch_reads_a_scratch_store_from_any_branch(
    tmp_path: Path, gh: tuple[Path, dict[str, str]]
) -> None:
    server, env = gh
    _serve(server, 1, "fit-store", {"m/o/main.json.gz": b"x"})
    _serve(
        server, 2, "fit-store-scratch-a", {"m/o/scratch.json.gz": b"x"},
        branch="feature",
    )  # fmt: skip
    got = tmp_path / "got"
    out = _run(
        "fetch",
        str(got),
        cwd=tmp_path,
        env={**env, "FIT_STORE": "fit-store-scratch-a"},
    )
    assert out.returncode == 0, out.stderr
    assert _files(got) == [".commit", "m/o/scratch.json.gz"]


def test_fetch_adds_the_named_bundles_in_their_folders(
    tmp_path: Path, gh: tuple[Path, dict[str, str]]
) -> None:
    server, env = gh
    _serve(server, 1, "fit-store", {"m/o/k.json.gz": b"x"})
    _serve(
        server, 2, "fit-store-backtests",
        {"latest.tar.gz": b"b", "latest.json": b"{}"},
    )  # fmt: skip
    _serve(server, 3, "fit-store-site", {"latest.tar.gz": b"s"})
    got = tmp_path / "got"
    out = _run("fetch", str(got), "backtests", cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr
    assert _files(got) == [
        ".commit",
        "backtests/latest.json",
        "backtests/latest.tar.gz",
        "m/o/k.json.gz",
    ]


def test_fetch_leaves_a_missing_or_expired_store_empty(
    tmp_path: Path, gh: tuple[Path, dict[str, str]]
) -> None:
    # An expired store is refitted and its bundles remade, never a
    # failure: the folder is left empty and a warning says so.
    server, env = gh
    _serve(server, 1, "fit-store", {"m/o/k.json.gz": b"x"}, expired=True)
    got = tmp_path / "got"
    out = _run("fetch", str(got), "backtests", "site", cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr
    assert _files(got) == []
    assert "::warning::" in out.stderr


def test_fetch_fails_when_the_artifacts_cannot_be_listed(
    tmp_path: Path,
    gh: tuple[Path, dict[str, str]],
) -> None:
    # An empty store would rule out every fitted candidate as "not in the
    # fit store", so a failed lookup must fail the step instead.
    server, env = gh
    (server / "fail").touch()
    got = tmp_path / "got"
    out = _run(
        "fetch", str(got), cwd=tmp_path, env={**env, "FIT_STORE_TRIES": "1"}
    )
    assert out.returncode != 0
    assert "could not read" in out.stderr
    assert not any(got.iterdir())


def test_latest_prints_the_store_s_version(
    gh: tuple[Path, dict[str, str]],
) -> None:
    server, env = gh
    out = _run("latest", env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
    _serve(server, 7, "fit-store", {"m/o/k.json.gz": b"x"})
    out = _run("latest", env=env)
    assert out.stdout.strip() == f"{7:064x}"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_fetch_reads_the_old_store_branch_until_a_store_is_uploaded(
    tmp_path: Path, gh: tuple[Path, dict[str, str]]
) -> None:
    # The fit-store branch held the store before the artifacts did. Until
    # the first store artifact is uploaded, its fits are read, so nothing
    # is refitted, and its bundles' parts are joined.
    server, env = gh
    _git("init", "-q", "--bare", "remote.git", cwd=tmp_path)
    _git("clone", "-q", "remote.git", "clone", cwd=tmp_path)
    clone = tmp_path / "clone"
    for key, value in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", key, value, cwd=clone)
    (clone / "m" / "o").mkdir(parents=True)
    (clone / "m" / "o" / "k.json.gz").write_bytes(b"fit")
    (clone / "site").mkdir()
    (clone / "site" / "latest.tar.gz.000").write_bytes(b"01")
    (clone / "site" / "latest.tar.gz.001").write_bytes(b"23")
    (clone / "site" / "latest.json").write_text("{}")
    _git("add", "-A", cwd=clone)
    _git("commit", "-q", "-m", "store", cwd=clone)
    _git("push", "-q", "origin", "HEAD:fit-store", cwd=clone)
    got = tmp_path / "got"
    out = _run("fetch", str(got), "site", cwd=clone, env=env)
    assert out.returncode == 0, out.stderr
    assert _files(got) == [
        ".commit",
        "m/o/k.json.gz",
        "site/latest.json",
        "site/latest.tar.gz",
    ]
    assert (got / "site" / "latest.tar.gz").read_bytes() == b"0123"
    # Once a store artifact exists, the branch is not read.
    _serve(server, 1, "fit-store", {"m/o/new.json.gz": b"x"})
    got = tmp_path / "again"
    out = _run("fetch", str(got), cwd=clone, env=env)
    assert out.returncode == 0, out.stderr
    assert _files(got) == [".commit", "m/o/new.json.gz"]


def _fit(root: Path, name: str, status: str, fitted_at: str) -> None:
    path = root / f"{name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"status": status, "fitted_at": fitted_at}
    path.write_bytes(gzip.compress(json.dumps({"meta": meta}).encode()))


def _merge(tmp_path: Path, store: Path, new: Path, keys, plan) -> None:
    (tmp_path / "keys.json").write_text(json.dumps(keys))
    include = [
        dict(zip(("model", "origin", "key"), p.split("/"), strict=True))
        for p in plan
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"include": include}))
    out = _run(
        "merge",
        str(store),
        str(new),
        str(tmp_path / "keys.json"),
        str(tmp_path / "plan.json"),
    )
    assert out.returncode == 0, out.stderr


def test_merge_copies_planned_fits_and_prunes(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = tmp_path / "first"
    _fit(first, "m/o/old1", "ok", "2026-11-01")
    _fit(first, "m/o/old2", "ok", "2026-11-02")
    _fit(first, "m/o/bad", "failed", "2026-11-03")
    _fit(first, "n/o/other", "ok", "2026-11-01")
    _fit(first, "m/o/stray", "ok", "2026-11-04")
    _merge(
        tmp_path, store, first, [],
        ["m/o/old1", "m/o/old2", "m/o/bad", "n/o/other"],
    )  # fmt: skip
    second = tmp_path / "second"
    _fit(second, "m/o/cur", "ok", "2026-11-05")
    _merge(tmp_path, store, second, ["m/o/cur"], ["m/o/cur"])
    # The current key, the newest good older fit and another model's fit.
    assert _files(store) == [
        "m/o/cur.json.gz",
        "m/o/old2.json.gz",
        "n/o/other.json.gz",
    ]


def test_a_fit_s_zone_file_is_stored_and_pruned_with_it(
    tmp_path: Path,
) -> None:
    # A walk's zone run sits beside its fit (`store.zones_path`).
    store = tmp_path / "store"
    for key, fitted_at in (
        ("one", "2026-11-01"),
        ("two", "2026-11-02"),
        ("three", "2026-11-03"),
    ):
        new = tmp_path / key
        _fit(new, f"m/o/{key}", "ok", fitted_at)
        (new / "m" / "o" / f"{key}.zones.csv.gz").write_bytes(b"zones")
        _merge(tmp_path, store, new, [f"m/o/{key}"], [f"m/o/{key}"])
    # The current key and the newest good older fit, each with its zones.
    assert _files(store) == [
        "m/o/three.json.gz",
        "m/o/three.zones.csv.gz",
        "m/o/two.json.gz",
        "m/o/two.zones.csv.gz",
    ]
