"""Each package's docs site follows the template and its examples run."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "hooks"))
SITES = sorted((ROOT / "packages").glob("*/mkdocs.yml"))
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
TEMPLATE = [
    {"Getting started": "index.md"},
    {"How to": "how-to.md"},
    {"Reference": "reference.md"},
    {"How it is used in the workflow": "workflow.md"},
]


def _config(site: Path) -> dict:
    return yaml.safe_load(site.read_text())


def _main_config() -> dict:
    """mkdocs.yml, its `!!python/name:` tags read as plain strings."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda _loader, suffix, _node: suffix
    )
    return yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=Loader)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_a_package_site_follows_the_template(site: Path) -> None:
    config = _config(site)
    assert config["site_name"] == site.parent.name
    assert config["nav"] == TEMPLATE
    for page in TEMPLATE:
        assert (site.parent / "docs" / next(iter(page.values()))).is_file()


def test_every_package_has_its_own_site() -> None:
    packages = {
        p.parent.name for p in (ROOT / "packages").glob("*/pyproject.toml")
    }
    assert packages == {s.parent.name for s in SITES}


def test_the_main_site_lists_the_package_sites_it_finds() -> None:
    # docs/hooks/packages.py reads packages/*/mkdocs.yml, so no list of
    # packages is kept by hand in mkdocs.yml or docs/packages/index.md.
    import packages as hook

    names = [site["name"] for site in hook.sites(ROOT)]
    assert names == sorted(s.parent.name for s in SITES)
    index = (ROOT / "docs" / "packages" / "index.md").read_text()
    table = hook.on_page_markdown(index)
    for name in names:
        assert f"](../{name}/)" in table
        assert name not in index
    nav = _main_config()["nav"]
    config = {"nav": nav, "site_url": "https://example.org/site/"}
    hook.on_config(config)
    (packages,) = [i["Packages"] for i in config["nav"] if "Packages" in i]
    links = [next(iter(i.values())) for i in packages if isinstance(i, dict)]
    for name in names:
        assert f"https://example.org/site/{name}/" in links
    assert not any(f"packages/{n}.md" in str(packages) for n in names)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_a_package_site_links_back_to_the_main_site(site: Path) -> None:
    import site_links

    base = yaml.safe_load((ROOT / "packages" / "mkdocs-base.yml").read_text())
    assert "../../docs/hooks/site_links.py" in base["hooks"]
    config = {
        "nav": list(_config(site)["nav"]),
        "extra": dict(base["extra"]),
    }
    site_links.on_config(config)
    home = base["extra"]["homepage"]
    assert {"Project site": home} in config["nav"]
    assert {"All packages": f"{home}packages/"} in config["nav"]


def test_the_site_build_only_assembles_the_package_sites() -> None:
    pages = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
    build = "\n".join(s.get("run", "") for s in pages["jobs"]["build"]["steps"])
    assert "packages/*/mkdocs.yml" not in build
    # The deploy keeps each package's folder, which its own job publishes.
    deploy = pages["jobs"]["deploy"]["steps"]
    (step,) = [s for s in deploy if "JamesIves" in s.get("uses", "")]
    assert "steps.keep.outputs" in step["with"]["clean-exclude"]


def test_each_package_site_is_built_when_it_changes() -> None:
    flow = yaml.safe_load(
        (ROOT / ".github/workflows/package-docs.yml").read_text()
    )
    on = flow[True]
    assert "packages/**" in on["push"]["paths"]
    assert on["push"]["branches"] == ["main"]
    text = (ROOT / ".github/workflows/package-docs.yml").read_text()
    assert "mkdocs build --strict" in text
    assert "HEAD:gh-pages" in text


@pytest.mark.slow
@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_package_site_examples_run(site: Path, tmp_path: Path) -> None:
    # Every page's blocks run in the nav's order and share their names, as
    # a reader runs them, in a new interpreter so that no other package's
    # imports stand in for the package's own.
    pages = [next(iter(p.values())) for p in TEMPLATE]
    code = "\n".join(
        block
        for page in pages
        for block in BLOCK.findall((site.parent / "docs" / page).read_text())
    )
    assert code, "a package site has no example"
    # The package's own source first: bdbvsubmission has its own
    # environment and is not installed in the root one.
    env = {**os.environ, "PYTHONPATH": str(site.parent / "src")}
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_the_main_site_does_not_cross_reference_package_api() -> None:
    # Each package's API is on its own site, so an autorefs link such as
    # `[x][nowcastrenewal.model]` from the main site has no target there.
    # Link to `../<package>/reference/#<object>` instead.
    names = "|".join(s.parent.name for s in SITES)
    autoref = re.compile(
        rf"\]\[`?(?:{names})(?:\.\w+)*`?\]"
        rf"|\[`?(?:{names})(?:\.\w+)*`?\]\[\]"
        rf"|^:::\s*(?:{names})\b",
        re.MULTILINE,
    )
    hits = [
        f"{page.relative_to(ROOT)}: {m.group(0)}"
        for page in sorted((ROOT / "docs").rglob("*"))
        if page.suffix in {".md", ".py", ".ipynb"} and "hooks" not in page.parts
        for m in autoref.finditer(page.read_text())
    ]
    assert not hits, hits
