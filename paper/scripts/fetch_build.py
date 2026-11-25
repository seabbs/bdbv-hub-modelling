"""Download the paper's built inputs into paper/build/ (`task paper:preview`).

    python paper/scripts/fetch_build.py [--url URL] [--paper DIR]

The Site workflow publishes the `build/` folder it rendered the site's
paper from (figures, tables, results.yml and evidence.yml) as the asset
paper-build.tar.gz of the `paper-build` release. This replaces
`<paper>/build/` with it, so Quarto renders the paper without running its
workflow. The repository is public, so no token is needed. Standard
library only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/seabbs/bdbv-hub-modelling/releases/download/"
    "paper-build/paper-build.tar.gz"
)
"""The Site workflow's latest paper build."""


def fetch(url: str, paper: Path) -> Path:
    """Replace `paper`/build with the bundle at `url`; return the folder."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "paper-build.tar.gz"
        with urllib.request.urlopen(url) as src, bundle.open("wb") as out:
            shutil.copyfileobj(src, out)
        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(bundle) as tar:
            # Refuses absolute paths, links and anything outside `unpacked`.
            tar.extractall(unpacked, filter="data")
        build = paper / "build"
        if not (unpacked / "build").is_dir():
            raise SystemExit(f"{url} holds no build/ folder")
        shutil.rmtree(build, ignore_errors=True)
        shutil.move(unpacked / "build", build)
    return build


def main(argv: list[str]) -> int:
    """Fetch the build; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--paper", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    build = fetch(args.url, args.paper)
    print(f"The paper's inputs are in {build}, from {args.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
