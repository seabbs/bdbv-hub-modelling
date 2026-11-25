"""Short READMEs where they help, one source for what repeats."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "docs" / "hooks"))

import readmes  # noqa: E402

# The workspace packages; packages/bdbvsubmission has its own environment
# and lock, so it is not installed with `uv sync --package`.
PACKAGES = sorted(
    p.parent.name
    for p in ROOT.glob("packages/*/pyproject.toml")
    if p.parent.name != "bdbvsubmission"
)

# Every top-level folder, and the CI workflows.
FOLDERS = [
    "packages",
    "packages/bdbvsubmission",
    "workflow",
    "paper",
    "docs",
    "data",
    "config",
    "tests",
    "assets",
    ".github/workflows",
]


@pytest.mark.parametrize(
    "folder", [*FOLDERS, *(f"packages/{p}" for p in PACKAGES)]
)
def test_the_folder_has_a_readme(folder):
    assert (ROOT / folder / "README.md").is_file()


@pytest.mark.parametrize("package", PACKAGES)
def test_each_package_readme_says_how_to_install_use_and_test_it(package):
    text = (ROOT / "packages" / package / "README.md").read_text()
    for heading in ("## Install", "## Example", "## Tasks", "## Citation"):
        assert heading in text, heading
    assert f"uv sync --package {package}" in text
    assert f"task {package}:test" in text


def _table(path: Path) -> str:
    text = path.read_text()
    block = text.split("<!-- packages table -->")[1].split(
        "<!-- /packages table -->"
    )[0]
    return block.replace('"../assets/', '"assets/')


def test_the_packages_table_is_one_table():
    assert _table(ROOT / "packages" / "README.md") == _table(ROOT / "README.md")


def test_a_package_without_its_own_site_has_a_page_from_its_readme():
    own_site = {p.parent.name for p in ROOT.glob("packages/*/mkdocs.yml")}
    for package in sorted(set(PACKAGES) - own_site):
        page = (ROOT / "docs" / "packages" / f"{package}.md").read_text()
        assert f"<!-- readme: packages/{package} -->" in page
        rendered = readmes.on_page_markdown(page)
        # Every link is absolute, so --strict finds no missing file.
        relative = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", rendered)
        assert relative == [], (package, relative)


def test_the_docs_readme_is_not_a_site_page():
    # It would clash with docs/index.md under mkdocs --strict.
    import yaml

    config = yaml.safe_load(
        (ROOT / "mkdocs.yml").read_text().split("\nnav:")[0]
    )
    assert "README.md" in config["exclude_docs"].split()
