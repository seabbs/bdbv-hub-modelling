"""The Quick links on the home page and in the README, grouped by project."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://samabbott.co.uk/bdbv-hub-modelling/"
SITES = {s.parent.name for s in (ROOT / "packages").glob("*/mkdocs.yml")}
PROJECTS = [
    "Fine-tuned foundation",
    "Joint-model-informed renewal",
    "Hub submission",
]
GROUPS = [
    ("Shared", ["nowcastutils", "nowcasteval"]),
    ("Models", ["nowcastbaselines", "nowcastfoundation", "nowcastrenewal"]),
    ("Studies", ["bdbvhub", "nowcastbdbv", "nowcastgermany"]),
]
PACKAGES = [name for _, group in GROUPS for name in group]
ON_DISK = sorted(
    p.parent.name for p in (ROOT / "packages").glob("*/pyproject.toml")
)
HOME_PROJECT = re.compile(r"\*\*([\w\- ]+)\*\*\]\(([\w/]+index\.md)\)")
HOME_PACKAGE = re.compile(r"\| (\w*) \| \[.*?\*\*(\w+)\*\*\]\(([^)\s]+)\)")
README_LINK = re.compile(r'<a href="([^"]+)">.*?<b>([^<]+)</b></a>')
README_GROUP = re.compile(r'<td rowspan="\d+">(\w+)</td>')
WIDTH = re.compile(r'width="(\d+)"')


def _home() -> str:
    text = (ROOT / "docs" / "index.md").read_text()
    start = text.index("## Quick links")
    return text[start : text.index("\n## ", start + 1)]


def _readme() -> str:
    text = (ROOT / "README.md").read_text()
    start = text.index("## Quick links")
    return text[start : text.index("\n## ", start + 1)]


def _projects(where: str) -> list[str]:
    if where == "home":
        return [n for n, _ in HOME_PROJECT.findall(_home())]
    first = _readme().split("</table>")[0]
    return [n for _, n in README_LINK.findall(first)]


def _packages(where: str) -> list[tuple[str, str, str]]:
    """Return (group, package, link), the group on its first row only."""
    if where == "home":
        return HOME_PACKAGE.findall(_home())
    rows = _readme().split("</table>")[1].split("<tr>")[2:]
    out = []
    for row in rows:
        group = README_GROUP.search(row)
        url, name = README_LINK.findall(row)[0]
        out.append((group.group(1) if group else "", name, url))
    return out


def _target(name: str, home: bool) -> str:
    if name in SITES:
        return f"{name}/" if home else f"{SITE}{name}/"
    # A package without its own site has a page from its README.
    return f"packages/{name}.md" if home else f"{SITE}packages/{name}/"


def test_every_package_is_grouped_once() -> None:
    assert sorted(PACKAGES) == ON_DISK


@pytest.mark.parametrize("where", ["home", "readme"])
def test_the_projects_come_first_in_one_row(where: str) -> None:
    assert _projects(where) == PROJECTS


@pytest.mark.parametrize("where", ["home", "readme"])
def test_the_packages_are_grouped(where: str) -> None:
    rows = _packages(where)
    assert [n for _, n, _ in rows] == PACKAGES
    heads = [(g, n) for g, n, _ in rows if g]
    assert heads == [(g, names[0]) for g, names in GROUPS]


@pytest.mark.parametrize("where", ["home", "readme"])
def test_a_package_links_to_its_own_site_where_it_has_one(where: str) -> None:
    links = {n: u for _, n, u in _packages(where)}
    for name in PACKAGES:
        assert links[name] == _target(name, where == "home"), name


@pytest.mark.parametrize("where", ["home", "readme"])
def test_every_logo_exists_in_light_and_dark(where: str) -> None:
    text = _home() if where == "home" else _readme()
    marks = re.findall(r"assets/logos/([\w-]+)/logo-mark(-dark)?\.svg", text)
    slugs = {s for s, _ in marks}
    assert len(slugs) == len(PROJECTS) + len(PACKAGES)
    # The docs site serves its own copy of the marks.
    logos = ROOT / ("docs" if where == "home" else "") / "assets" / "logos"
    for slug in slugs:
        for f in ("logo-mark.svg", "logo-mark-dark.svg"):
            assert (logos / slug / f).is_file(), slug
        assert (slug, "") in marks, slug
        assert (slug, "-dark") in marks, slug


@pytest.mark.parametrize("where", ["home", "readme"])
def test_quick_links_use_compact_marks(where: str) -> None:
    text = _home() if where == "home" else _readme()
    widths = {int(w) for w in WIDTH.findall(text)}
    assert widths
    assert max(widths) <= 48


def test_the_readme_puts_the_badges_after_the_quick_links() -> None:
    text = (ROOT / "README.md").read_text()
    intro = text[: text.index("## Quick links")]
    assert "badge" not in intro
    status = text.index("## Status")
    assert text.index("## Quick links") < status
    table = text[status : text.index("\n## ", status + 1)]
    assert table.index("<details>") < table.index("| Component |")
