"""Shared pieces of the interactive report pages.

The hub rounds report, the this-round page and the BDBV fine-tuning
report are standalone HTML pages built with the docs site's own
interactive charts (`docs/javascripts`): the forecast explorer
(`nowcasteval.explorer.explorer_data` and `explorer_html`) and the
score summary (`nowcasteval.scoresummary.summary_rows` and
`summary_html`). This module gives them one look: the docs theme's
colours in light and dark (`THEME`, `SCHEME`), the docs scripts and
stylesheet inlined (`page`), each page's data folder beside it
(`data_dir`), a table the reader can sort (`html_table`) and an atomic
write (`write_page`).

    parts = ["<h1>...</h1>", explorer_html(f"{rel}/x.json", "Forecasts")]
    write_page(out, page("Title", parts))
"""

from __future__ import annotations

import html
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

__all__ = [
    "EXPLORER_SCRIPTS",
    "SCHEME",
    "SCORE_SCRIPTS",
    "THEME",
    "data_dir",
    "html_table",
    "page",
    "write_page",
]

EXPLORER_SCRIPTS: Final = (
    "toggles.js",
    "zoom.js",
    "summarise.js",
    "explorer.js",
)
"""The docs scripts the forecast explorer needs, in load order."""

SCORE_SCRIPTS: Final = (
    "toggles.js",
    "zoom.js",
    "summarise.js",
    "energy.js",
    "tablesort.js",
    "tablefilter.js",
    "placemap.js",
    "explorer.js",
    "scorechart.js",
    "scoresummary.js",
)
"""The docs scripts the explorer and the score summary need, in order."""

THEME: Final = """
:root{--md-default-bg-color:#fff;--md-default-fg-color:rgba(0,0,0,.87);
--md-default-fg-color--light:rgba(0,0,0,.54);
--md-default-fg-color--lighter:rgba(0,0,0,.32);
--md-default-fg-color--lightest:rgba(0,0,0,.07);
--md-accent-fg-color:#526cfe;--md-typeset-a-color:#4051b5}
[data-md-color-scheme=slate]{--md-default-bg-color:#1e2129;
--md-default-fg-color:rgba(226,228,233,.82);
--md-default-fg-color--light:rgba(226,228,233,.56);
--md-default-fg-color--lighter:rgba(226,228,233,.32);
--md-default-fg-color--lightest:rgba(226,228,233,.12);
--md-accent-fg-color:#5e8bde;--md-typeset-a-color:#5e8bde}
body{background:var(--md-default-bg-color);color:var(--md-default-fg-color);
font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;
padding:16px}
a{color:var(--md-typeset-a-color)}
table{border-collapse:collapse;font-size:13px;margin:8px 0 16px}
td,th{border-bottom:1px solid var(--md-default-fg-color--lightest);
padding:3px 8px;text-align:right}
td:first-child,th:first-child{text-align:left}
.mut{color:var(--md-default-fg-color--light)}
details{margin:8px 0}
summary{cursor:pointer}
.table-wrap{overflow-x:auto}
"""
"""The docs theme's colours (mkdocs-material variables the docs scripts
and stylesheet read), light and `slate` dark, and the page layout."""

SCHEME: Final = """
(function () {
  var dark = window.matchMedia("(prefers-color-scheme: dark)");
  function set() {
    document.body.setAttribute("data-md-color-scheme",
      dark.matches ? "slate" : "default");
  }
  set();
  dark.addEventListener("change", set);
})();
"""
"""Follow the reader's light or dark preference, as the docs do."""


def page(
    title: str,
    parts: Sequence[str],
    scripts: Sequence[str] = SCORE_SCRIPTS,
    root: Path | str = ".",
) -> str:
    """Return a standalone report page.

    Parameters
    ----------
    title : str
        The page's title.
    parts : sequence of str
        The body's HTML, in order.
    scripts : sequence of str
        Docs scripts (`docs/javascripts`) to inline, in load order.
    root : pathlib.Path or str
        The repository root, holding `docs/`.

    Returns
    -------
    str
        The page, with `THEME`, the docs stylesheet and `scripts` inlined.
    """
    docs = Path(root) / "docs"
    js = "\n".join((docs / "javascripts" / f).read_text() for f in scripts)
    css = (docs / "stylesheets" / "extra.css").read_text()
    return (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{THEME}{css}</style>"
        f"<body data-md-color-scheme=default><script>{SCHEME}</script>"
        + "\n".join(parts)
        + f"<script>{js}</script></body></html>"
    )


def html_table(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    digits: int = 2,
) -> str:
    """Return a table the reader can sort by any column (`tablesort.js`).

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows.
    columns : mapping of str to str
        Each column shown and its heading, in order.
    digits : int
        Decimal places of floats; missing values show as `-`.

    Returns
    -------
    str
        A `table` inside a scrolling `tablesort` block.
    """
    head = "".join(f"<th>{html.escape(h)}</th>" for h in columns.values())
    rows = []
    for row in frame[list(columns)].itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                text = "-" if pd.isna(v) else f"{v:.{digits}f}"
            else:
                text = str(v)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='tablesort table-wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def data_dir(out: Path) -> Path:
    """Return a page's data folder, `<stem>-data` beside it, emptied."""
    data = out.with_name(out.stem + "-data")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    return data


def write_page(out: Path, doc: str) -> Path:
    """Write a page through a temporary file and return its path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(doc)
    tmp.rename(out)
    return out
