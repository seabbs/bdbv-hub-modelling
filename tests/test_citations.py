"""Each subproject has its own citation, from its CITATION.cff."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "docs" / "hooks"))

import citations  # noqa: E402

PACKAGES = sorted(p.parent.name for p in ROOT.glob("packages/*/pyproject.toml"))


def test_every_package_and_the_paper_has_a_citation():
    have = {cff.parent.name for cff in citations.subprojects()}
    assert have == {*PACKAGES, "paper"}
    for cff in citations.subprojects():
        meta = yaml.safe_load(cff.read_text())
        for key in ("authors", "abstract", "title"):
            assert meta[key], (cff, key)


@pytest.mark.parametrize(
    "cff", citations.subprojects(), ids=lambda p: p.parent.name
)
def test_each_readme_cites_from_its_cff(cff):
    # One source: the README's section is generated from the file.
    text = (cff.parent / "README.md").read_text()
    assert citations.readme_section(cff) in text


def test_the_foundation_work_names_charlotte_holt():
    for cff in (
        ROOT / "packages" / "nowcastfoundation" / "CITATION.cff",
        ROOT / "paper" / "CITATION.cff",
    ):
        names = [
            a["family-names"]
            for a in yaml.safe_load(cff.read_text())["authors"]
        ]
        assert "Holt" in names, cff


def test_the_root_points_to_each_subproject():
    root = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert "packages/<name>/CITATION.cff" in root["message"]
    readme = (ROOT / "README.md").read_text()
    for cff in citations.subprojects():
        assert cff.relative_to(ROOT).as_posix() in readme
    page = (ROOT / "docs" / "citation.md").read_text()
    assert citations.MARKER in page
    assert "docs/hooks/citations.py" in (ROOT / "mkdocs.yml").read_text()


def test_the_citation_page_lists_every_subproject():
    page = (ROOT / "docs" / "citation.md").read_text()
    got = citations.on_page_markdown(page)
    assert citations.MARKER not in got
    for cff in citations.subprojects():
        assert citations.bibtex(cff) in got
