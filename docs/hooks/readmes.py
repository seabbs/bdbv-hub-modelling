# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""MkDocs hook: a package's page is its README, one source.

A page that holds `<!-- readme: packages/<name> -->` shows that README,
its relative links pointed at the file on GitHub, so the README is not
copied into the docs and the site still builds with `--strict`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BLOB = "https://github.com/seabbs/bdbv-hub-modelling/blob/main"
MARKER = re.compile(r"<!-- readme: ([\w./-]+) -->")
LINK = re.compile(r"(\]\()(?!https?://|#|mailto:)([^)\s]+)(\))")


def render(folder: str, root: Path = ROOT) -> str:
    """Return `folder`'s README with its relative links made absolute."""
    text = (root / folder / "README.md").read_text()
    # The page has its own title: drop the README's first heading.
    text = re.sub(r"\A# [^\n]*\n+", "", text)

    def absolute(m: re.Match[str]) -> str:
        target = (Path(folder) / m.group(2)).as_posix()
        parts: list[str] = []
        for part in target.split("/"):
            if part == "..":
                parts.pop()
            elif part != ".":
                parts.append(part)
        return f"{m.group(1)}{BLOB}/{'/'.join(parts)}{m.group(3)}"

    return LINK.sub(absolute, text)


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Put each marked README in its page."""
    return MARKER.sub(lambda m: render(m.group(1)), markdown)
