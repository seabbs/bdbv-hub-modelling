# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""MkDocs hook: the package sites, found in packages/ rather than listed.

Every `packages/<name>/mkdocs.yml` is a package's own docs site, built by
the Package docs workflow and published at `<site>/<name>/`. This hook
adds a link to each under the Packages section of the navigation, and
fills the `<!-- package sites -->` marker of docs/packages/index.md with a
table of them, from each site's name and description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- package sites -->"
SECTION = "Packages"


def sites(root: Path = ROOT) -> list[dict[str, str]]:
    """Return each package site's name and description, by name."""
    found = []
    for config in sorted((root / "packages").glob("*/mkdocs.yml")):
        data = yaml.safe_load(config.read_text())
        found.append(
            {
                "name": config.parent.name,
                "description": str(data.get("site_description", "")).strip(),
            }
        )
    return found


def table(root: Path = ROOT) -> str:
    """Return the package sites as a Markdown table, as seen from /packages/."""
    rows = [
        "| | Package | What it holds | Docs |",
        "|---|---|---|---|",
    ]
    for site in sites(root):
        name = site["name"]
        logo = f"../assets/logos/{name}/logo-mark.svg"
        icon = (
            f'![]({logo}){{ width="40" height="40" }}'
            if (root / "docs" / logo.removeprefix("../")).is_file()
            else ""
        )
        rows.append(
            f"| {icon} | [**{name}**](../{name}/){{ #{name} }} "
            f"| {site['description']} "
            f"| [Docs site](../{name}/), [API](../{name}/reference/) |"
        )
    return "\n".join(rows)


def on_config(config: Any, **_: Any) -> Any:
    """Link each package site under the Packages section, after its index."""
    url = str(config["site_url"]).rstrip("/")
    links = [{s["name"]: f"{url}/{s['name']}/"} for s in sites()]
    for item in config["nav"]:
        if isinstance(item, dict) and SECTION in item:
            entries = item[SECTION]
            item[SECTION] = [entries[0], *links, *entries[1:]]
    return config


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Put the package sites' table in the page that marks it."""
    if MARKER not in markdown:
        return markdown
    return markdown.replace(MARKER, table())
