"""The paper's host-made outputs, published as releases and read back.

The workflow's hosts make what the Site build cannot: the synthetic
study's forecasts, the fine-tuning runs' progress tables and checkpoint
scores, and the simulation summaries. Each set of them is published as a
release, `paper-<set>-<hash>`, whose hash is of the files' contents, so a
set is published once per version (rules `release_paper_<set>`). Only
development outputs are published: `publish` refuses a held-out synthetic
seed, a held-out German date or an Ebola row. Each published version then
becomes the set's current one: the release `paper-<set>-main` names it in
`main.json` (`promote`, rules `promote_paper_<set>`, target
`paper_inputs`; `nowcastutils.pointers`).

The Site build reads each set's `paper-<set>-main`, fetches the release it
names and writes a config overlay that points the paper workflow at its
files (`fetch`), so they are inputs there, never jobs. A set never promoted
is left out with a warning, and what reads it stays pending; a pointer that
names no release fails.

    python -m paperpipe.stored publish SET --heldout JSON --file NAME=PATH...
        --record OUT
    python -m paperpipe.stored promote --released RECORD --record OUT
        --branches BRANCH...
    python -m paperpipe.stored fetch OUT_DIR --overlay OVERLAY [--set SET...]

Site results fetches the `germany` set alone (`--set germany`): our
German development forecasts from the fine-tuning workflow's units
(`germany-units.csv.gz`, `nowcastgermany units`), so the site and the
paper show the same ones.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from nowcastutils import pointers

from paperpipe.config import seed_of

PREFIX = "paper-"
"""Every set's release tags start `paper-<set>-`."""

SETS = ("synthetic", "finetune", "simulations", "germany")
"""Every set, in the order `fetch` reads them."""

ASSET = "paper-inputs.tar.gz"
"""Each release's one asset."""

REPO = "seabbs/bdbv-hub-modelling"
"""The repository the releases are read from, unless GITHUB_REPOSITORY."""

# Each stored file and the config entry it stands in for.
STORED = {
    "pairing-synthetic.csv.gz": ("stored", "pairing_synthetic"),
    "sim-diseases.csv": ("stored", "sim_diseases"),
    "sim-summaries.csv": ("stored", "sim_summaries"),
    "sim-realism.csv": ("stored", "sim_realism"),
    "pairing-germany.csv.gz": ("stored", "pairing_germany"),
    "pairing-germany-rho.csv.gz": ("stored", "copula_rho"),
    "germany-forecasts.csv.gz": ("stored", "germany_forecasts"),
    "comparison.csv": ("evidence", "checkpoint_scores"),
    "compare-daily.csv": ("evidence", "compare_daily"),
    "block-combinations.csv": ("evidence", "block_combinations"),
}

PROGRESS_TABLES = {
    "loss": "progress/loss.csv",
    "seven_day": "progress/seven-day.csv",
    "horizons": "horizon-scores.csv",
}
"""Each run's progress tables in the `finetune` set, by the progress
step's option, from the run's reports folder (workflow rules run_progress
and horizon_table). Stored as `progress/<run>/<file name>`; the BDBV check
is left out, as it holds Ebola rows."""

# The dates a German row is held out by, as the table guard reads them
# (`paperpipe.config.check_no_heldout`), and `origin` of
# `nowcastgermany`'s own tables: a development origin's targets may run
# past the split.
DATE_COLUMNS = ("origin_date", "nowcast_date", "origin")


def sets(config: Mapping[str, Any], results: str) -> dict[str, dict[str, str]]:
    """Return each set's files: its name in the bundle and where it is made.

    Paths are as the config gives them (`$ROOT` unexpanded in the raw
    config). The examples are the SI's picked seeds of the development
    round.
    """
    synthetic = config["synthetic"]
    evidence = config.get("evidence") or {}
    dev_sims = synthetic["simulations"]["development"]
    seeds = sorted(
        {str(sim).split("-")[0] for sim, _ in synthetic["examples"]["picks"]}
    )
    examples = {}
    for seed in seeds if dev_sims else []:
        examples[f"examples/counts-{seed}.csv.gz"] = (
            f"{dev_sims}/counts-{seed}.csv.gz"
        )
        examples[f"examples/settings-{seed}.csv"] = (
            f"{dev_sims}/settings-{seed}.csv"
        )
    rounds = config.get("rounds_dir")
    runs_dir = config.get("runs_dir")
    runs = [
        config.get("live_run"),
        *(config.get("comparisons") or {}).values(),
    ]
    # Each paper run's progress tables (workflow rules run_progress and
    # horizon_table).
    progress = {
        f"progress/{run}/{Path(table).name}": (
            f"{runs_dir}/{run}/reports/{table}"
        )
        for run in dict.fromkeys(r for r in runs if r)
        for table in PROGRESS_TABLES.values()
        if runs_dir
    }

    def file_of(name: str) -> str | None:
        entry = evidence.get(name) or {}
        return entry.get("file")

    found = {
        "synthetic": {
            "synthetic-dev.csv.gz": (
                synthetic["forecasts"]["development"] if dev_sims else None
            ),
            "pairing-synthetic.csv.gz": (
                f"{results}/evidence/pairing/pairing-synthetic.csv.gz"
            ),
        },
        "finetune": {
            **progress,
            "comparison.csv": file_of("checkpoint_scores"),
            "compare-daily.csv": file_of("compare_daily"),
            "block-combinations.csv": file_of("block_combinations"),
        },
        "germany": {
            "pairing-germany.csv.gz": (
                f"{results}/evidence/pairing/pairing-germany.csv.gz"
            ),
            "pairing-germany-rho.csv.gz": (
                f"{results}/evidence/pairing/pairing-germany-rho.csv.gz"
            ),
            # The German benchmark's development table, every model.
            "germany-forecasts.csv.gz": (
                f"{config['germany']['results']}/paper-forecasts.csv.gz"
            ),
            # Our models' rows of it as the fine-tuning workflow's units
            # made them, and the checkpoint they load, which Site results
            # reads (`nowcastgermany units`).
            "germany-units.csv.gz": (
                f"{config['germany']['results']}/forecasts.csv.gz"
            ),
            "germany-units.json": (
                f"{config['germany']['results']}/units.json"
            ),
        },
        "simulations": {
            "sim-diseases.csv": f"{rounds}/report/sim-diseases.csv"
            if rounds
            else None,
            "sim-summaries.csv": f"{results}/evidence/sim-summaries.csv",
            "sim-realism.csv": f"{results}/evidence/sim-realism.csv",
            **examples,
        },
    }
    return {
        name: {k: v for k, v in files.items() if v}
        for name, files in found.items()
    }


def content_hash(files: Mapping[str, Path | str]) -> str:
    """Return the hash of the files' names and contents."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(hashlib.sha256(Path(files[name]).read_bytes()).digest())
    return digest.hexdigest()


def tag(set_name: str, files: Mapping[str, Path | str]) -> str:
    """Return the set's release tag for these contents."""
    return f"{PREFIX}{set_name}-{content_hash(files)[:12]}"


def check_development(
    files: Mapping[str, Path | str],
    config: Mapping[str, Any],
    *,
    german: bool = False,
) -> None:
    """Fail if a table holds a held-out row.

    A held-out synthetic seed (in `seed` or a synthetic `location`), a
    German row on or after the held-out date, or an Ebola row. With
    `german` every row is German, as in the German pairing tables.
    """
    heldout = config["heldout"]
    seeds = {int(s) for s in heldout["synthetic_seeds"]}
    since = pd.Timestamp(heldout["germany_from"])
    for name, path in files.items():
        if not str(name).endswith((".csv", ".csv.gz")):
            continue
        frame = pd.read_csv(path)
        problems = []
        study = frame.get("study")
        if study is not None and (study == "ebola").any():
            problems.append("Ebola rows")
        if "seed" in frame and frame["seed"].isin(seeds).any():
            problems.append("held-out seeds")
        if "location" in frame:
            synthetic = frame["location"].astype(str).str.match(r"^\d+-\d+$")
            found = frame.loc[synthetic, "location"].map(seed_of)
            if found.isin(seeds).any():
                problems.append("held-out seeds")
        german_rows = pd.Series(german, index=frame.index)
        if study is not None:
            german_rows = german_rows | (study == "germany")
        for column in DATE_COLUMNS:
            if column in frame:
                late = pd.to_datetime(frame[column], errors="coerce") >= since
                if (late & german_rows).any():
                    problems.append(f"German {column} held out")
        if problems:
            raise ValueError(
                f"{name} holds rows that are held out: {', '.join(problems)}."
            )


def bundle(set_name: str, files: Mapping[str, Path | str], out: Path) -> Path:
    """Pack the files with a manifest into `out`/paper-inputs.tar.gz."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "set": set_name,
        "tag": tag(set_name, files),
        "files": sorted(files),
    }
    path = out / ASSET
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / "manifest.json"
        record.write_text(json.dumps(manifest, indent=2))
        with tarfile.open(path, "w:gz") as tar:
            tar.add(record, arcname="manifest.json")
            for name in sorted(files):
                tar.add(files[name], arcname=name)
    return path


def unpack(bundle_path: Path, folder: Path) -> Path:
    """Unpack a bundle into `folder` (replacing it); return the folder."""
    if folder.exists():
        for item in sorted(folder.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path) as tar:
        tar.extractall(folder, filter="data")
    return folder


def overlay(folders: Mapping[str, Path]) -> dict[str, Any]:
    """Return the config overlay pointing the paper at each unpacked set."""
    config: dict[str, Any] = {"stored": {}, "evidence": {}}
    for folder in folders.values():
        forecasts = folder / "synthetic-dev.csv.gz"
        if forecasts.is_file():
            config["synthetic"] = {"forecasts": {"development": str(forecasts)}}
        if (folder / "examples").is_dir():
            config["stored"]["examples"] = str(folder / "examples")
        if (folder / "progress").is_dir():
            config["stored"]["progress"] = str(folder / "progress")
        for name, (block, key) in STORED.items():
            path = folder / name
            if not path.is_file():
                continue
            if block == "evidence":
                config["evidence"][key] = {"file": str(path)}
            else:
                config["stored"][key] = str(path)
    return {k: v for k, v in config.items() if v}


def fetch(
    out: Path,
    pointer: Callable[[str], Mapping[str, Any]],
    download: Callable[[str, Path], Path],
    names: Sequence[str] = SETS,
) -> dict[str, Any]:
    """Unpack each set's current release under `out`; return the overlay.

    `pointer` returns the record of an endpoint's current version
    (`nowcastutils.pointers.read`), and raises `LookupError` for a set
    never promoted, which is left out with a warning.
    """
    folders = {}
    for name in names:
        endpoint = f"{PREFIX}{name}"
        try:
            release = str(pointer(endpoint)["tag"])
        except LookupError as err:
            print(f"::warning::{name}: {err} Its results stay pending.")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = download(release, Path(tmp))
            folders[name] = unpack(path, out / name)
        print(f"{name}: {release}")
    return overlay(folders)


# GitHub -------------------------------------------------------------------


def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY") or REPO


def gh_download(release: str, dest: Path) -> Path:
    """Download a release's bundle into `dest`."""
    subprocess.run(
        ["gh", "release", "download", release, "-R", _repo(), "-p", ASSET,
         "-D", str(dest), "--clobber"],
        check=True,
    )  # fmt: skip
    return dest / ASSET


def gh_publish(set_name: str, files: Mapping[str, Path], git_sha: str) -> str:
    """Release a set once per contents; return its tag."""
    release = tag(set_name, files)
    seen = subprocess.run(
        ["gh", "release", "view", release, "-R", _repo()],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if seen.returncode == 0:
        print(f"{release} is already published")
        return release
    with tempfile.TemporaryDirectory() as tmp:
        path = bundle(set_name, files, Path(tmp))
        notes = (
            f"The paper's host-made {set_name} outputs (development only), "
            f"from {git_sha}: {', '.join(sorted(files))}. The Site build "
            f"reads the release `{pointers.tag(PREFIX + set_name)}` names."
        )
        subprocess.run(
            ["gh", "release", "create", release, str(path), "-R", _repo(),
             "--prerelease", "--latest=false", "--title",
             f"Paper {set_name} outputs {release.rsplit('-', 1)[-1]}",
             "--notes", notes],
            check=True,
        )  # fmt: skip
    return release


def promote(
    released: Path, branches: Sequence[str], run: Any = subprocess.run
) -> dict[str, Any]:
    """Name a published set's release its current one; return the record.

    `released` is the record `publish` wrote (`set`, `tag`, `hash`).
    """
    record = json.loads(Path(released).read_text())
    name = record["set"]
    return pointers.promote(
        f"{PREFIX}{name}",
        record["tag"],
        hash=record["hash"],
        produced_by=f"the paper workflow (rule release_paper_{name})",
        branches=branches,
        repo=_repo(),
        title=f"Paper {name} outputs, current",
        needs=ASSET,
        run=run,
    )


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("set")
    pub.add_argument("--heldout", required=True)
    pub.add_argument("--file", action="append", default=[], required=True)
    pub.add_argument("--record", type=Path, required=True)
    up = sub.add_parser("promote")
    up.add_argument("--released", type=Path, required=True)
    up.add_argument("--record", type=Path, required=True)
    up.add_argument("--branches", nargs="+", default=[])
    get = sub.add_parser("fetch")
    get.add_argument("out", type=Path)
    get.add_argument("--overlay", type=Path, required=True)
    get.add_argument(
        "--set",
        action="append",
        choices=SETS,
        default=None,
        help="Fetch this set only (repeat for more); every set by default.",
    )
    return parser


def main(argv: Sequence[str]) -> int:
    """Publish or promote one set, or fetch sets; return the exit code."""
    args = parser().parse_args(argv)
    if args.command == "publish":
        config = {"heldout": json.loads(args.heldout)}
        files = dict(entry.split("=", 1) for entry in args.file)
        files = {name: Path(path) for name, path in files.items()}
        check_development(files, config, german=args.set == "germany")
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False,
        ).stdout.strip()  # fmt: skip
        release = gh_publish(args.set, files, sha)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(
            json.dumps(
                {"set": args.set, "tag": release, "hash": content_hash(files)}
            )
            + "\n"
        )
        return 0
    if args.command == "promote":
        record = promote(args.released, args.branches)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record) + "\n")
        return 0
    pointer = functools.partial(pointers.read, repo=_repo())
    config = fetch(args.out, pointer, gh_download, args.set or SETS)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.write_text(yaml.safe_dump(config, sort_keys=True))
    print(f"Overlay: {args.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
