"""Attach every model's results to the round releases.

    results-release attach BUILD SITE

submit.yml's `release` job publishes `round-<reference date>` from the
commit the round was submitted from. This adds the Pages build's
results to each such release whose round's hub release has hub tables
in the build (BUILD/results/hub/<hub release>) and every stored fit at
it of this code (BUILD/results/fit-store.json): those hub tables, the
results tables, a zip of BUILD/data and the built site SITE as
site.zip. site.zip goes last, so a release with it is complete; one
without it is filled again by the next build.

Needs GH_TOKEN and GITHUB_SHA, git and gh. Standard library only, so the
Pages `release` job, which has no forecast environment, runs it as
`python3 -I <this file> attach BUILD SITE`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

SITE_URL = "https://samabbott.co.uk/bdbv-hub-modelling/"
"""Where the site is published."""

LEDGER = "submission-ledger"
"""The branch whose `state.json` lists the submitted rounds."""

NOTES = """
## Results

Every model's results at this round's hub release, {origin}, built from commit {sha}.

- `hub.zip`: each model's forecast in the hub's format, samples and quantiles for the country, provinces and health zones (`hub/{origin}/<model>-samples.csv.gz`, `hub/{origin}/<model>-quantiles.csv.gz`), and the scored hub comparison (`hub-comparison/`).
- `forecasts.csv.gz`: every model's forecasts at each real origin.
- `*-scores.csv`, `*-energy.csv`, `nowcast-*.csv`, `zone-*.csv`: the scores behind the Backtest, Nowcast and Comparison pages.
- `backtest-fits.csv`, `delay-summary.csv`, `fit-store.json`: the stored fits' diagnostics and learnt delays, and the fit behind each model and origin.
- `data.zip`: the tidy hub data the build used.
- `site.zip`: the rendered site; see `README.txt` inside to view it offline.

Site: {site}
"""  # noqa: E501
"""The release notes' results section, after the notes up to it."""


def rounds(results: Path, state: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield each round and hub release whose results the build holds.

    A round counts when the build has hub tables at its hub release
    (`as_of`) and every stored fit there is of this code.
    """
    status = json.loads((results / "fit-store.json").read_text())
    for ref, entry in state["rounds"].items():
        origin = entry.get("as_of")
        fits = [s for s in status if s["origin"] == origin]
        if (
            (results / "hub" / str(origin)).is_dir()
            and fits
            and all(s["current"] for s in fits)
        ):
            yield ref, origin


def _zip(out: Path, root: Path, names: Sequence[str]) -> None:
    """Zip the files under ROOT's NAMES into OUT, by their paths in ROOT."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            top = root / name
            files = [top] if top.is_file() else sorted(top.rglob("*"))
            for f in files:
                z.write(f, f.relative_to(root).as_posix())


def _gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=check
    )


def attach_round(
    build: Path, site: Path, repo: str, ref: str, origin: str
) -> bool:
    """Attach the results to round REF's release; return if attached."""
    tag = f"round-{ref}"
    view = _gh(
        "release", "view", tag, "-R", repo, "--json", "assets",
        "-q", '[.assets[].name] | join(" ")', check=False,
    )  # fmt: skip
    if view.returncode != 0 or "site.zip" in view.stdout.split():
        return False
    results = build / "results"
    files = Path(tempfile.mkdtemp(prefix="round."))
    for pattern in ("*.csv", "*.csv.gz", "*.json"):
        for f in sorted(results.glob(pattern)):
            shutil.copy(f, files / f.name)
    _zip(files / "hub.zip", results, [f"hub/{origin}", "hub-comparison"])
    data = build / "data"
    _zip(files / "data.zip", data, sorted(p.name for p in data.iterdir()))
    stage = Path(tempfile.mkdtemp(prefix="site."))
    shutil.copytree(site, stage / "bdbv-hub-modelling")
    (stage / "README.txt").write_text(
        "Run `python3 -m http.server` in this folder and open\n"
        "bdbv-hub-modelling/ at the address it prints.\n"
    )
    _zip(stage / "site.zip", stage, ["README.txt", "bdbv-hub-modelling"])
    uploads = [str(f) for f in sorted(files.iterdir())]
    _gh("release", "upload", tag, *uploads, "-R", repo, "--clobber")
    body = _gh(
        "release", "view", tag, "-R", repo, "--json", "body", "-q", ".body"
    )
    lines = body.stdout.splitlines()
    cut = lines.index("## Results") if "## Results" in lines else len(lines)
    kept = "\n".join(lines[:cut])
    sha = os.environ.get("GITHUB_SHA", "")
    notes = stage / "notes.md"
    notes.write_text(
        kept + "\n" + NOTES.format(origin=origin, sha=sha, site=SITE_URL)
    )
    _gh("release", "edit", tag, "-R", repo, "--notes-file", str(notes))
    _gh(
        "release",
        "upload",
        tag,
        str(stage / "site.zip"),
        "-R",
        repo,
        "--clobber",
    )
    print(f"Attached results to {tag}.")
    return True


def attach(build: Path, site: Path) -> int:
    """Attach the build's results to every round release they complete."""
    repo = os.environ.get("GITHUB_REPOSITORY") or "seabbs/bdbv-hub-modelling"
    fetched = subprocess.run(
        ["git", "fetch", "-q", "--depth", "1", "origin", LEDGER],
        capture_output=True,
    )
    if fetched.returncode != 0:
        print("No ledger: no round to attach results to.")
        return 0
    state = json.loads(
        subprocess.run(
            ["git", "show", "FETCH_HEAD:state.json"],
            capture_output=True,
            check=True,
        ).stdout
    )
    for ref, origin in rounds(build / "results", state):
        attach_round(build, site, repo, ref, origin)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run `attach BUILD SITE`; return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 3 and args[0] == "attach":
        return attach(Path(args[1]), Path(args[2]))
    print("usage: results-release attach BUILD SITE", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
