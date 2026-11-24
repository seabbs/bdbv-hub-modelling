"""bdbvhub.reporting.pages: the shared shell of the interactive reports."""

from __future__ import annotations

from pathlib import Path

from bdbvhub.reporting import pages

ROOT = Path(__file__).resolve().parents[3]


def test_a_page_has_the_theme_the_docs_style_and_its_scripts() -> None:
    doc = pages.page("Hub rounds", ["<h1>x</h1>"], pages.EXPLORER_SCRIPTS, ROOT)
    assert doc.startswith("<!doctype html>")
    assert "<title>Hub rounds</title>" in doc
    assert "--md-default-fg-color" in doc
    assert "slate" in doc
    assert "<h1>x</h1>" in doc
    # The docs scripts are inlined, the explorer's last.
    explorer = (ROOT / "docs" / "javascripts" / "explorer.js").read_text()
    assert explorer in doc
    assert doc.index(explorer) > doc.index("<h1>x</h1>")


def test_the_score_scripts_include_the_explorers() -> None:
    assert set(pages.EXPLORER_SCRIPTS) <= set(pages.SCORE_SCRIPTS)
    assert pages.SCORE_SCRIPTS[-1] == "scoresummary.js"


def test_the_data_folder_is_beside_the_page_and_starts_empty(
    tmp_path: Path,
) -> None:
    out = tmp_path / "this-round.html"
    first = pages.data_dir(out)
    (first / "old.json").write_text("{}")
    again = pages.data_dir(out)
    assert again == tmp_path / "this-round-data"
    assert list(again.iterdir()) == []


def test_a_page_is_written_whole(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "page.html"
    assert pages.write_page(out, "<p>x</p>") == out
    assert out.read_text() == "<p>x</p>"
    assert not out.with_name("page.html.tmp").exists()
