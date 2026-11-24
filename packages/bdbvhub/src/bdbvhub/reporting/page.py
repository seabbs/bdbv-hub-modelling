# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The shell of the standalone report pages: one style for all of them.

The hub rounds report, the this-round page and the BDBV foundation-model
check are single HTML files outside the docs site that use its
interactive charts (`docs/javascripts`) and stylesheet
(`docs/stylesheets/extra.css`). `standalone` wraps a page's parts in that
style, with a light and a dark colour scheme that follow the reader's
system setting, so every page looks and behaves the same.

- `THEME`: the docs theme's colour variables and the page layout.
- `SCHEME`: sets the light or dark scheme from the system setting.
- `SCRIPTS`: the docs scripts the charts need, in the docs' load order.
- `html_table`: a table the reader can sort (`tablesort.js`).
- `standalone`: the whole page.
- `write_page`: writes it atomically.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

__all__ = [
    "SCHEME",
    "SCRIPTS",
    "THEME",
    "html_table",
    "standalone",
    "write_page",
]

SCRIPTS: Final = (
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
"""The docs scripts the charts need, in the docs' load order."""

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
"""The docs theme's colour variables, light and dark, and the layout."""

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
"""Sets the light or dark scheme from the reader's system setting."""


def html_table(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    digits: int = 2,
) -> str:
    """Return a table the reader can sort by any column.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows.
    columns : mapping of str to str
        Each column shown and its heading, in order.
    digits : int
        Decimal places of floats; missing values show as a dash.

    Returns
    -------
    str
        A `table` inside a `tablesort` block.
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


def standalone(
    title: str,
    parts: Sequence[str],
    docs: Path | str = "docs",
    scripts: Sequence[str] = SCRIPTS,
) -> str:
    """Return a standalone page in the docs site's style.

    Parameters
    ----------
    title : str
        The page title.
    parts : sequence of str
        The body's HTML, in order.
    docs : pathlib.Path or str
        The repository's `docs` folder, for the stylesheet and scripts.
    scripts : sequence of str
        The docs scripts inlined, in load order.

    Returns
    -------
    str
        The page.
    """
    docs = Path(docs)
    css = (docs / "stylesheets" / "extra.css").read_text()
    js = "\n".join((docs / "javascripts" / f).read_text() for f in scripts)
    return (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{THEME}{css}</style>"
        f"<body data-md-color-scheme=default><script>{SCHEME}</script>"
        + "\n".join(parts)
        + f"<script>{js}</script></body></html>"
    )


def write_page(out: Path | str, page: str) -> Path:
    """Write a page through a temporary file, so a reader never sees half.

    Parameters
    ----------
    out : pathlib.Path or str
        The page.
    page : str
        Its HTML.

    Returns
    -------
    pathlib.Path
        `out`.
    """
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(page)
    tmp.rename(out)
    return out
