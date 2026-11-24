# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""MkDocs hook: each subproject's citation, from its own CITATION.cff.

Every package (`packages/<name>/CITATION.cff`) and the paper
(`paper/CITATION.cff`) has its own citation, with its own authors and
abstract; the root `CITATION.cff` is the repository as a whole. Those
files are the one source: `docs/citation.md` lists them where it says
`MARKER`, and each README's Citation section is `readme_section(cff)`
(tests/test_citations.py checks they match).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- subproject citations -->"
"""Where `docs/citation.md` lists the subprojects' citations."""


def subprojects(root: Path = ROOT) -> list[Path]:
    """Return each subproject's CITATION.cff: the packages, then the paper."""
    return [
        *sorted(root.glob("packages/*/CITATION.cff")),
        root / "paper/CITATION.cff",
    ]


def bibtex(cff: Path) -> str:
    """Return a BibTeX entry for a CITATION.cff."""
    meta = yaml.safe_load(cff.read_text())
    name = cff.parent.name
    authors = " and ".join(
        f"{a['family-names']}, {a['given-names']}" for a in meta["authors"]
    )
    first = meta["authors"][0]["family-names"].lower()
    return (
        f"@software{{{first}_{name},\n"
        f"  author = {{{authors}}},\n"
        f"  title = {{{{{meta['title']}}}}},\n"
        "  year = {2026},\n"
        f"  url = {{{meta['repository-code']}}}\n"
        "}"
    )


def readme_section(cff: Path) -> str:
    """Return a README's Citation section for its CITATION.cff."""
    return (
        "## Citation\n\n"
        "Cite it with [CITATION.cff](CITATION.cff); the repository as a "
        f"whole has [its own]({_up(cff)}).\n\n"
        f"```bibtex\n{bibtex(cff)}\n```\n"
    )


def _up(cff: Path) -> str:
    depth = len(cff.parent.relative_to(ROOT).parts)
    return "/".join([".."] * depth) + "/CITATION.cff"


def listing(root: Path = ROOT) -> str:
    """Return the docs page's list of the subprojects' citations."""
    out = []
    for cff in subprojects(root):
        meta = yaml.safe_load(cff.read_text())
        where = cff.parent.relative_to(root).as_posix()
        link = f"https://github.com/seabbs/bdbv-hub-modelling/blob/main/{where}/CITATION.cff"
        out.append(
            f"### {cff.parent.name}\n\n{meta['abstract'].strip()}\n\n"
            f"[CITATION.cff]({link})\n\n```bibtex\n{bibtex(cff)}\n```\n"
        )
    return "\n".join(out)


def on_page_markdown(markdown: str, **_: Any) -> str:
    """List the subprojects' citations on the citation page."""
    if MARKER in markdown:
        return markdown.replace(MARKER, listing())
    return markdown
