"""The paper's host-made outputs, published as releases and read back.

The workflow's hosts make what the Site build cannot: the synthetic
study's forecasts, the fine-tuning runs' curves and checkpoint scores, and
the simulation summaries. Each set of them is published as a release,
`paper-<set>-<hash>`, whose hash is of the files' contents, so a set is
published once per version (rules `release_paper_<set>`, target
`paper_inputs`). Only development outputs are published: `publish` refuses
a held-out synthetic seed, a held-out German date or an Ebola row.

The Site build fetches each set's newest release and writes a config
overlay that points the paper workflow at its files (`fetch`), so they are
inputs there, never jobs. A set with no release is left out, and what reads
it stays pending.

    python -m paperpipe.stored publish SET --heldout JSON --file NAME=PATH...
    python -m paperpipe.stored fetch OUT_DIR --overlay OVERLAY
"""

from __future__ import annotations

import argparse
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

from paperpipe.config import seed_of

PREFIX = "paper-"
"""Every set's release tags start `paper-<set>-`."""

ASSET = "paper-inputs.tar.gz"
"""Each release's one asset."""

REPO = "seabbs/bdbv-hub-modelling"
"""The repository the releases are read from, unless GITHUB_REPOSITORY."""

# Each stored file and the config entry it stands in for.
STORED = {
    "pairing-synthetic.csv.gz": ("stored", "pairing_synthetic"),
    "validation-curve.csv": ("stored", "finetune_validation"),
    "sim-diseases.csv": ("stored", "sim_diseases"),
    "sim-summaries.csv": ("stored", "sim_summaries"),
    "sim-realism.csv": ("stored", "sim_realism"),
    "pairing-germany.csv.gz": ("stored", "pairing_germany"),
    "pairing-germany-rho.csv.gz": ("stored", "copula_rho"),
    "comparison.csv": ("evidence", "checkpoint_scores"),
    "compare-daily.csv": ("evidence", "compare_daily"),
    "block-combinations.csv": ("evidence", "block_combinations"),
}

DATE_COLUMNS = ("origin_date", "target_date", "nowcast_date", "as_of_date")


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
    final = config.get("final_run")

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
            "validation-curve.csv": f"{final}/validation-curve.csv"
            if final
            else None,
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
        for name, (block, key) in STORED.items():
            path = folder / name
            if not path.is_file():
                continue
            if block == "evidence":
                config["evidence"][key] = {"file": str(path)}
            else:
                config["stored"][key] = str(path)
    return {k: v for k, v in config.items() if v}


def newest(releases: Sequence[Mapping[str, Any]], set_name: str) -> str | None:
    """Return the newest release tag of a set, or None."""
    prefix = f"{PREFIX}{set_name}-"
    mine = [r for r in releases if str(r["tag_name"]).startswith(prefix)]
    if not mine:
        return None
    return max(mine, key=lambda r: r["created_at"])["tag_name"]


def fetch(
    out: Path,
    releases: Sequence[Mapping[str, Any]],
    download: Callable[[str, Path], Path],
    names: Sequence[str] = ("synthetic", "finetune", "simulations", "germany"),
) -> dict[str, Any]:
    """Unpack each set's newest release under `out`; return the overlay."""
    folders = {}
    for name in names:
        release = newest(releases, name)
        if release is None:
            print(
                f"{name}: no {PREFIX}{name}-* release; its results stay pending"
            )
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = download(release, Path(tmp))
            folders[name] = unpack(path, out / name)
        print(f"{name}: {release}")
    return overlay(folders)


# GitHub -------------------------------------------------------------------


def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY") or REPO


def gh_releases() -> list[dict[str, Any]]:
    """Return the repository's releases (tag and creation time)."""
    out = subprocess.run(
        [
            "gh", "api", "--paginate", f"repos/{_repo()}/releases",
            "-q", ".[] | {tag_name, created_at}",
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return [json.loads(line) for line in out.stdout.splitlines() if line]


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
            "reads the newest `paper-" + set_name + "-*` release."
        )
        subprocess.run(
            ["gh", "release", "create", release, str(path), "-R", _repo(),
             "--prerelease", "--latest=false", "--title",
             f"Paper {set_name} outputs {release.rsplit('-', 1)[-1]}",
             "--notes", notes],
            check=True,
        )  # fmt: skip
    return release


def main(argv: Sequence[str]) -> int:
    """Publish one set, or fetch every set; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("set")
    pub.add_argument("--heldout", required=True)
    pub.add_argument("--file", action="append", default=[], required=True)
    pub.add_argument("--record", type=Path, required=True)
    get = sub.add_parser("fetch")
    get.add_argument("out", type=Path)
    get.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args(argv)
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
            json.dumps({"set": args.set, "tag": release}) + "\n"
        )
        return 0
    config = fetch(args.out, gh_releases(), gh_download)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.write_text(yaml.safe_dump(config, sort_keys=True))
    print(f"Overlay: {args.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
