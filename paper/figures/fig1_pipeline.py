# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Draw Figure 1, a schematic of the fine-tuning pipeline.

Writes `fig1-pipeline.svg` next to this script, and `fig1-pipeline.pdf`
and, with `--png`, a 200 dpi `fig1-pipeline.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig1_pipeline.py [--png]

The panels are:

- A, the five layers of the simulations (`nowcastfoundation.simulator`),
  one icon each;
- B, the model inputs (`nowcastfoundation.forecaster.layout_block`):
  each place's target, lag series and weekly row in one block, the lag
  series reported so far entering as known-future inputs;
- C, a training block cut from a simulated release history
  (`nowcastfoundation.finetune.training_block`) and the LoRA adapters,
  and below them the attention biases (`nowcastfoundation.series_bias`)
  on a map of the places and in the group-attention grid;
- D, forecasting with the fine-tuned model
  (`nowcastfoundation.forecaster.chronos_sampler`).

Units are points at the final size, 6.5 inches wide, so font sizes are
the printed sizes. Panels are laid out in `WIDTH`, and `Svg.render`
adds a white margin of `MARGIN` on every side.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fig1-pipeline"
PAGE = 468.0
MARGIN = 7.0
WIDTH = PAGE - 2 * MARGIN

FONT = "Helvetica, Arial, sans-serif"
FG = "#1a1e27"
MUTED = "#4f5666"
LINE = "#b9c0cd"
PANEL = "#eceff4"
CELL = "#d9dee8"
ACCENT = "#2a56c6"
ACCENT_SOFT = "#dde6fa"
COV = "#9a5b07"
COV_SOFT = "#f5e5c8"
HATCH = "#a3aabb"
TRUTH = "#1b7f5f"
TRUTH_SOFT = "#d4eee3"
# The attention-bias colours: a place's own series, and a place and its
# parent (up) or its children (down).
SLOTS = ("#6f42c1", "#c0397a")
PARENT_SLOT = "#2f7d4f"
CHILD_SLOT = COV

# Body text, small notes and headings, in points.
BODY = 7.5
SMALL = 7.0
HEAD = 8.0
TITLE = 9.0


class Svg:
    """An SVG document built from primitives in points."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        """Append raw SVG."""
        self.parts.append(s)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        sw: float = 0.6,
        rx: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw a rectangle."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        s: str | list[str],
        size: float = BODY,
        fill: str = FG,
        anchor: str = "start",
        weight: str = "normal",
        lead: float = 1.25,
        italic: bool = False,
    ) -> None:
        """Write text, one line per item of a list."""
        lines = [s] if isinstance(s, str) else s
        style = ' font-style="italic"' if italic else ""
        spans = "".join(
            f'<tspan x="{x:.2f}" dy="{0 if i == 0 else size * lead:.2f}">'
            f"{_escape(line)}</tspan>"
            for i, line in enumerate(lines)
        )
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" xml:space="preserve"{style}>'
            f"{spans}</text>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = MUTED,
        sw: float = 0.8,
        dash: str | None = None,
        arrow: str | None = None,
    ) -> None:
        """Draw a line, with an arrowhead marker id at its end."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{arrow})"' if arrow else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'
        )

    def path(
        self,
        d: str,
        stroke: str = MUTED,
        sw: float = 0.8,
        fill: str = "none",
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        """Draw a path."""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" '
            f'fill="{fill}" opacity="{opacity}"{da}/>'
        )

    def poly(self, pts: list[tuple[float, float]], **kw: object) -> None:
        """Draw a polyline through points."""
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.path(d, **kw)  # type: ignore[arg-type]

    def label(self, x: float, y: float, letter: str, title: str) -> None:
        """Write a panel's letter and title."""
        self.text(x, y, letter, size=11, weight="bold")
        self.text(x + 12, y, title, size=TITLE, weight="bold")

    def render(self, height: float) -> str:
        """Return the document, with a white margin on every side."""
        markers = "".join(
            f'<marker id="{name}" viewBox="0 0 6 6" refX="5.5" refY="3" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 z" fill="{colour}"/></marker>'
            for name, colour in (
                ("ah", MUTED),
                ("aha", ACCENT),
                ("ahup", PARENT_SLOT),
                ("ahdown", CHILD_SLOT),
            )
        )
        defs = (
            "<defs>"
            '<pattern id="hatch" width="3.5" height="3.5" '
            'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="3.5" height="3.5" fill="{PANEL}"/>'
            f'<line x1="0" y1="0" x2="0" y2="3.5" stroke="{HATCH}" '
            'stroke-width="0.9"/></pattern>'
            f"{markers}</defs>"
        )
        total = height + 2 * MARGIN
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{PAGE}pt" height="{total}pt" '
            f'viewBox="0 0 {PAGE} {total}" font-family="{FONT}">'
            f"{defs}"
            f'<rect width="{PAGE}" height="{total}" fill="white"/>'
            f'<g transform="translate({MARGIN} {MARGIN})">'
            + "".join(self.parts)
            + "</g></svg>\n"
        )


def _escape(s: str) -> str:
    """Escape text for XML."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell(g: Svg, x: float, y: float, w: float, h: float, kind: str) -> None:
    """Draw one cell of a triangle or block, styled by what it holds.

    `kind` is `final` (a complete count), `lag` (a count reported so
    far), `future` (to forecast), `truth` (a final count the training
    loss scores), `none` (no release at that lag) or `unseen` (not yet
    reported).
    """
    if kind == "final":
        g.rect(x, y, w, h, fill=CELL, stroke="white", sw=0.5)
    elif kind == "lag":
        g.rect(x, y, w, h, fill=COV_SOFT, stroke=COV, sw=0.6)
    elif kind == "future":
        g.rect(
            x, y, w, h, fill=ACCENT_SOFT, stroke=ACCENT, sw=0.6, dash="2 1.2"
        )
    elif kind == "truth":
        g.rect(x, y, w, h, fill=TRUTH_SOFT, stroke=TRUTH, sw=0.8)
    elif kind == "none":
        g.rect(x, y, w, h, fill="url(#hatch)", stroke="white", sw=0.5)
    elif kind == "unseen":
        g.rect(x, y, w, h, fill="none", stroke=LINE, sw=0.5, dash="1.5 1.5")


def lag_rows(cols: int, latest: int, last_complete: int, lags: int):
    """Return the cell kinds of a target and its lag series by row.

    Steps are 1 to `cols`. The target is complete up to the last
    complete step and to forecast after it. Lag `d` is reported up to
    `latest - d`, so every lag series is solid from the left and the
    longer lags end earlier.
    """
    rows = [
        [
            "final" if c <= last_complete else "future"
            for c in range(1, cols + 1)
        ]
    ]
    for d in range(lags):
        rows.append(
            ["lag" if c <= latest - d else "unseen" for c in range(1, cols + 1)]
        )
    return rows


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, the five layers of the simulations."""
    g.label(0, oy + 9, "A", "Simulated outbreaks")
    names = [
        "Infections",
        "Observation",
        "Reporting delays",
        "Reporting system",
        "Places and strata",
    ]
    gap = 14.0
    w = (WIDTH - gap * (len(names) - 1)) / len(names)
    top = oy + 17
    height = 46.0
    for i, name in enumerate(names):
        x = i * (w + gap)
        g.rect(x, top, w, height, fill="white", stroke=LINE, sw=0.7, rx=3)
        g.rect(x, top, w, 30, fill=PANEL, stroke="none", rx=3)
        g.rect(x, top + 26, w, 4, fill=PANEL, stroke="none", rx=0)
        _glyph(g, i, x + w / 2, top + 14)
        g.text(
            x + w / 2,
            top + 40,
            name,
            size=BODY,
            weight="bold",
            anchor="middle",
        )
        if i < len(names) - 1:
            g.line(
                x + w + 1.5,
                top + 15,
                x + w + gap - 1.5,
                top + 15,
                arrow="ah",
            )
    return top + height


def _curve(
    cx: float, base: float, width: float, peak: float, skew: float = 0.0
) -> list[tuple[float, float]]:
    """Return points of a smooth epidemic curve over `width` points."""
    pts = []
    for k in range(31):
        u = k / 30
        centre = 0.5 + skew
        v = 2.718 ** (-(((u - centre) / 0.2) ** 2))
        pts.append((cx - width / 2 + u * width, base - peak * v))
    return pts


def _area(
    g: Svg, pts: list[tuple[float, float]], base: float, **kw: object
) -> None:
    """Fill the area under a curve down to `base`."""
    d = f"M{pts[0][0]:.2f},{base:.2f} " + " ".join(
        f"L{x:.2f},{y:.2f}" for x, y in pts
    )
    d += f" L{pts[-1][0]:.2f},{base:.2f} Z"
    g.path(d, **kw)  # type: ignore[arg-type]


def _node(
    g: Svg, x: float, y: float, r: float, text: str = "", fill: str = "white"
) -> None:
    """Draw a place as a circle, with its letter."""
    g.add(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" '
        f'stroke="{FG}" stroke-width="0.8"/>'
    )
    if text:
        g.text(x, y + 2.8, text, size=HEAD, anchor="middle", weight="bold")


def _glyph(g: Svg, i: int, cx: float, cy: float) -> None:
    """Draw the icon of layer `i` of panel A."""
    base = cy + 9
    if i == 0:
        # Infections: an epidemic curve.
        pts = _curve(cx, base, 46, 16, -0.05)
        _area(g, pts, base, stroke="none", fill=ACCENT, opacity=0.85)
        g.line(cx - 24, base, cx + 24, base, stroke=MUTED, sw=0.6)
    elif i == 1:
        # Observation: noisy counts about their expectation.
        pts = _curve(cx, base, 44, 15, 0.0)
        g.poly(pts, stroke=MUTED, sw=0.8)
        noise = [1.5, -2.5, 2, -1, 3, -3, 0.5, 2.5, -2, 1, -1.5]
        for k, e in enumerate(noise):
            x, y = pts[k * 3]
            g.add(
                f'<circle cx="{x:.2f}" cy="{min(y + e, base):.2f}" r="1.2" '
                f'fill="{ACCENT}"/>'
            )
        g.line(cx - 23, base, cx + 23, base, stroke=MUTED, sw=0.6)
    elif i == 2:
        # Reporting delays: a delay distribution.
        probs = [0.15, 0.55, 0.85, 0.7, 0.48, 0.3, 0.18, 0.1, 0.05]
        for k, p in enumerate(probs):
            g.rect(
                cx - 20 + k * 4.6, base - 17 * p, 3.4, 17 * p, fill=ACCENT, rx=0
            )
        g.line(cx - 22, base, cx + 22, base, stroke=MUTED, sw=0.6)
    elif i == 3:
        # The reporting system: reports by weekday, two days with none.
        for k, h in enumerate([12, 8, 0, 9, 15, 0, 5]):
            if h:
                g.rect(cx - 16 + k * 4.8, base - h, 3.4, h, fill=ACCENT, rx=0)
        g.line(cx - 18, base, cx + 18, base, stroke=MUTED, sw=0.6)
    else:
        # Places and strata: a country over two places.
        top, low = (cx, cy - 6), [(cx - 12, cy + 6), (cx + 12, cy + 6)]
        for x, y in low:
            g.line(top[0], top[1], x, y, stroke=MUTED, sw=0.7)
        for x, y in [top, *low]:
            g.add(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{ACCENT}"/>'
            )


def _block(
    g: Svg,
    bx: float,
    by: float,
    cols: int,
    bw: float,
    rh: float,
    kinds: list[list[str]],
) -> float:
    """Draw a block of series from cell kinds by row; return its bottom."""
    y = by
    for k, row in enumerate(kinds):
        if k == 1:
            y += 2
        for c in range(cols):
            cell(g, bx + c * bw, y, bw, rh, row[c])
        y += rh + 1
    return y - 1


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, every place's series in one block."""
    g.label(0, oy + 9, "B", "Model inputs")
    # Days 1 to 14, the latest release on day 11 and a maximum delay of
    # three days, so the last complete day is day 8.
    cols, latest, lags = 14, 11, 3
    last = latest - lags
    bw, rh = 21.0, 7.0
    bx, by = 66.0, oy + 30
    names = ["target", *(f"lag {d}" for d in range(lags)), "weekly"]
    colours = [ACCENT, *([COV] * lags), MUTED]
    tops: list[float] = []
    y = by
    for p, place in enumerate(("C", "A", "B")):
        tops.append(y)
        bottom = _block(
            g, bx, y, cols, bw, rh, lag_rows(cols, latest, last, lags)
        )
        # The weekly row: each week's total, complete or to forecast.
        wy = bottom + 3
        for k in range(cols // 7):
            kind = "final" if 7 * (k + 1) <= last else "future"
            cell(g, bx + 7 * k * bw, wy, 7 * bw, rh, kind)
        place_bottom = wy + rh
        g.text(
            6,
            (y + place_bottom) / 2 + 3,
            place,
            size=11,
            weight="bold",
            anchor="middle",
        )
        if p == 0:
            ys = [y + k * (rh + 1) + (2 if k else 0) for k in range(lags + 1)]
            ys.append(wy)
            for yy, name, colour in zip(ys, names, colours, strict=True):
                g.text(
                    bx - 4,
                    yy + rh - 1,
                    name,
                    size=SMALL - 0.5,
                    anchor="end",
                    fill=colour,
                )
            # The lags reported after the last complete day enter as
            # known-future inputs.
            x0 = bx + last * bw
            ly = [y + (d + 1) * (rh + 1) + 2 for d in range(lags)]
            pts = [(x0, ly[0]), (bx + latest * bw, ly[0])]
            for d in range(lags):
                xr = bx + (latest - d) * bw
                pts += [(xr, ly[d] + rh)]
                if d + 1 < lags:
                    pts += [(bx + (latest - d - 1) * bw, ly[d] + rh)]
            pts += [(x0, ly[-1] + rh), (x0, ly[0])]
            g.poly(pts, stroke=COV, sw=1.4)
            kx = bx + cols * bw + 8
            ky = ly[0] + rh / 2
            g.line(kx - 2, ky, bx + latest * bw + 2, ky, stroke=COV, sw=0.7)
            g.text(kx, ky + 2.5, "known future", size=SMALL, fill=COV)
        y = place_bottom + 7
    bottom = y - 7
    # Column headings.
    for lo, hi, name in [
        (1, last, "context"),
        (last + 1, latest, "nowcast"),
        (latest + 1, cols, "forecast"),
    ]:
        g.text(
            bx + (lo - 1 + hi) / 2 * bw,
            by - 5,
            name,
            size=SMALL,
            anchor="middle",
            fill=MUTED,
        )
    # The last complete day and the latest release.
    exx = bx + last * bw
    g.line(exx, by - 2, exx, bottom + 3, stroke=ACCENT, sw=1.1)
    g.text(
        exx,
        bottom + 12,
        "last complete",
        size=SMALL,
        weight="bold",
        fill=ACCENT,
        anchor="middle",
    )
    rx_ = bx + latest * bw
    g.line(rx_, by - 2, rx_, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    g.text(rx_, bottom + 12, "latest data", size=SMALL, anchor="middle")
    # The legend, beside the block.
    lx = bx + cols * bw + 8
    items = [
        ("final", "complete"),
        ("lag", "reported so far"),
        ("unseen", "not yet reported"),
        ("future", "to forecast"),
    ]
    for k, (kind, name) in enumerate(items):
        ly_ = tops[2] + 10 + 13 * k
        cell(g, lx, ly_ - 6.5, 9, 7.5, kind)
        g.text(lx + 13, ly_, name, size=SMALL)
    return bottom + 14


# The attention-bias relations (panel C, lower part), each with its colour.
RELATIONS = (
    ("same", "same place", SLOTS[0]),
    ("up", "up (child \u2192 parent)", PARENT_SLOT),
    ("down", "down (parent \u2192 child)", CHILD_SLOT),
    ("itself", "itself (no bias)", "white"),
    ("none", "unrelated (no bias)", CELL),
)


def _relation(row: tuple[str, int], col: tuple[str, int]) -> str:
    """Return the relation of a query row to a key column of the toy block.

    Places A and B are children of the country C. A row and itself have
    no bias, drawn apart from the same-place pairs around them.
    """
    (p, i), (q, j) = row, col
    if p == q:
        return "same" if i != j else "itself"
    if q == "C":
        return "up"
    if p == "C":
        return "down"
    return "none"


def _arc(
    g: Svg,
    a: tuple[float, float],
    b: tuple[float, float],
    bend: float,
    colour: str,
    arrow: str,
) -> None:
    """Draw a curved arrow from `a` to `b`, bent to its left by `bend`."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    qx, qy = mx + bend * dy / n, my - bend * dx / n
    g.add(
        f'<path d="M{a[0]:.2f},{a[1]:.2f} Q{qx:.2f},{qy:.2f} '
        f'{b[0]:.2f},{b[1]:.2f}" stroke="{colour}" stroke-width="1.0" '
        f'fill="none" marker-end="url(#{arrow})"/>'
    )


def _toward(
    a: tuple[float, float], b: tuple[float, float], r: float
) -> tuple[float, float]:
    """Return the point `r` from `a` toward `b`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    return (a[0] + r * dx / n, a[1] + r * dy / n)


def attention_biases(g: Svg, oy: float) -> float:
    """Draw the attention biases: a map of the places and the grid."""
    g.text(
        0,
        oy + 8,
        "Attention biases in group attention",
        size=HEAD,
        weight="bold",
    )
    top = oy + 16
    # The map: country C over places A and B, with the up and down
    # relations between them as arrows.
    r = 8.0
    c = (44.0, top + 14)
    kids = {"A": (18.0, top + 58), "B": (70.0, top + 58)}
    for k in kids.values():
        lo = _toward(k, c, r + 1.5)
        hi = _toward(c, k, r + 1.5)
        _arc(g, lo, hi, 4.5, PARENT_SLOT, "ahup")
        _arc(g, hi, lo, 4.5, CHILD_SLOT, "ahdown")
    for name, (x, y) in [("C", c), *kids.items()]:
        _node(g, x, y, r, name)
    # The grid: group attention at one step, rows attending to columns.
    places = ("C", "A", "B")
    n_series = 3
    rows = [(p, k) for p in places for k in range(n_series)]
    cs = 8.0
    gx, gy = 132.0, top + 9
    g.text(
        gx + cs * len(rows) / 2,
        top + 1,
        "at one step, rows attend to columns",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    colour = {key: c for key, _, c in RELATIONS}
    for a, row in enumerate(rows):
        for b, col in enumerate(rows):
            rel = _relation(row, col)
            g.rect(
                gx + b * cs + 0.4,
                gy + 9 + a * cs + 0.4,
                cs - 0.8,
                cs - 0.8,
                fill=colour[rel],
                rx=1,
                stroke=LINE if rel == "itself" else "none",
                sw=0.6,
            )
    for k, p in enumerate(places):
        mid = n_series * cs * (k + 0.5)
        g.text(gx + mid, gy + 6, p, size=SMALL, anchor="middle", weight="bold")
        g.text(
            gx - 4,
            gy + 9 + mid + 2.5,
            p,
            size=SMALL,
            anchor="end",
            weight="bold",
        )
    grid_bottom = gy + 9 + len(rows) * cs
    # The relations, one a line, beside the grid.
    lx = gx + len(rows) * cs + 22
    for k, (key, name, c_) in enumerate(RELATIONS):
        y = gy + 11 + k * 12
        edge = LINE if key == "itself" else "none"
        g.rect(lx, y - 7, 9, 7.5, fill=c_, rx=1, stroke=edge, sw=0.6)
        g.text(lx + 13, y, name, size=SMALL)
    g.text(
        lx + 128,
        gy + 11,
        [
            "one learned bias per",
            "relation, layer and head,",
            "added to the attention",
            "scores",
        ],
        size=SMALL,
        fill=MUTED,
    )
    return max(grid_bottom, top + 58 + r)


def panel_c(g: Svg, oy: float) -> float:
    """Draw panel C, training blocks, the LoRA adapters and the biases."""
    g.label(0, oy + 9, "C", "Training blocks and LoRA fine-tuning")
    top = oy + 22
    # The training block, cut at a random cut-off, laid out as in B.
    cw, rh = 12.0, 8.0
    n_ctx, lags, n_fc = 8, 3, 3
    cut = n_ctx + lags
    cols = cut + n_fc
    bx = 34.0
    by = top + 18
    rows = lag_rows(cols, cut, n_ctx, lags)
    rows[0] = ["final" if c < n_ctx else "truth" for c in range(cols)]
    bottom = by + (lags + 1) * (rh + 1) + 1
    # A stack of blocks behind, for many outbreaks and cut-offs.
    for k in (2, 1):
        g.rect(
            bx - 2 + 3 * k,
            by - 2 - 3 * k,
            cols * cw + 4,
            bottom - by + 4,
            fill="white",
            stroke=LINE,
            sw=0.6,
            rx=2,
        )
    _block(g, bx, by, cols, cw, rh, rows)
    g.text(
        bx - 4, by + rh - 1.5, "target", size=SMALL, anchor="end", fill=ACCENT
    )
    g.text(
        bx - 4,
        (by + rh + 2 + bottom) / 2 + 2.5,
        "lags",
        size=SMALL,
        anchor="end",
        fill=COV,
    )
    g.text(
        bx + n_ctx * cw / 2,
        by - 10,
        "context",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    g.text(
        bx + (n_ctx + cols) / 2 * cw,
        by - 10,
        "final counts",
        size=SMALL,
        anchor="middle",
        fill=TRUTH,
        weight="bold",
    )
    exx = bx + n_ctx * cw
    g.line(exx, by - 2, exx, bottom + 3, stroke=ACCENT, sw=1.1)
    g.text(
        exx,
        bottom + 12,
        "last complete",
        size=SMALL,
        weight="bold",
        fill=ACCENT,
        anchor="middle",
    )
    cutx = bx + cut * cw
    g.line(cutx, by - 2, cutx, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    g.text(cutx + 2, bottom + 12, "cut-off", size=SMALL, anchor="start")
    left_bottom = bottom + 14

    # LoRA: the frozen model with small trained adapters.
    mw, aw = 104.0, 24.0
    mx = WIDTH - mw - aw - 30
    y0 = top
    g.rect(mx, y0, mw, 13, fill=CELL, rx=2)
    g.text(
        mx + mw / 2, y0 + 9.3, "input embedding", size=SMALL, anchor="middle"
    )
    g.rect(
        mx - 4,
        y0 + 17,
        mw + aw + 12,
        41,
        fill="none",
        stroke=LINE,
        sw=0.7,
        dash="3 2",
        rx=3,
    )
    g.rect(mx, y0 + 21, mw, 17, fill=CELL, rx=2)
    g.text(
        mx + mw / 2,
        y0 + 28.6,
        "time and group attention",
        size=SMALL,
        anchor="middle",
    )
    g.text(
        mx + mw / 2,
        y0 + 35.6,
        "+ attention biases",
        size=SMALL,
        anchor="middle",
        fill=ACCENT,
        weight="bold",
    )
    g.rect(mx, y0 + 41, mw, 13, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 50.3, "feed-forward", size=SMALL, anchor="middle")
    g.text(mx + mw + aw + 11, y0 + 40, "\u00d7 12", size=SMALL, fill=MUTED)
    g.rect(mx, y0 + 62, mw, 13, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 71.3, "output layer", size=SMALL, anchor="middle")
    for yy, hh in ((y0 + 21, 17), (y0 + 62, 13)):
        g.rect(
            mx + mw + 4,
            yy,
            aw,
            hh,
            fill=ACCENT_SOFT,
            stroke=ACCENT,
            sw=0.9,
            rx=2,
        )
        g.text(
            mx + mw + 4 + aw / 2,
            yy + hh / 2 + 2.5,
            "B\u00b7A",
            size=SMALL,
            anchor="middle",
            fill=ACCENT,
            weight="bold",
        )
    lgy = y0 + 88
    g.rect(mx, lgy - 6.5, 9, 7.5, fill=CELL, rx=1)
    g.text(mx + 12, lgy, "frozen", size=SMALL)
    g.rect(
        mx + 50,
        lgy - 6.5,
        9,
        7.5,
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        sw=0.8,
        rx=1,
    )
    g.text(mx + 62, lgy, "trained", size=SMALL)
    # The arrow from the blocks to the model.
    g.line(bx + cols * cw + 14, y0 + 38, mx - 10, y0 + 38, arrow="ah")
    y = max(left_bottom, lgy + 2) + 8
    # The attention biases, set apart below a rule.
    g.line(0, y, WIDTH, y, stroke=LINE, sw=0.5)
    return attention_biases(g, y + 4)


def panel_d(g: Svg, oy: float) -> float:
    """Draw panel D, forecasting with the fine-tuned model."""
    g.label(0, oy + 9, "D", "Forecasting with the fine-tuned model")
    top = oy + 18
    h = 58.0
    gap = 18.0
    widths = [86.0, 86.0, 104.0]
    widths.append(WIDTH - sum(widths) - 3 * gap)
    xs = [0.0]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    labels = [
        "latest data",
        "Chronos-2",
        "quantiles \u2192 draws",
        "predictions",
    ]
    for k, (x, w, lab) in enumerate(zip(xs, widths, labels, strict=True)):
        g.rect(
            x,
            top,
            w,
            h,
            fill=PANEL if k == 1 else "white",
            stroke=LINE,
            sw=0.7,
            rx=3,
        )
        g.text(
            x + w / 2, top + 11, lab, size=SMALL, anchor="middle", weight="bold"
        )
        if k:
            g.line(x - gap + 2, top + h / 2, x - 2, top + h / 2, arrow="ah")
    # Stage 1: the latest data, laid out as in B.
    cw, rh = 7.0, 6.0
    rows = lag_rows(10, 8, 5, 3)
    _block(g, xs[0] + (widths[0] - 70) / 2, top + 20, 10, cw, rh, rows)
    # Stage 2: the model and its adapter.
    x2, w2 = xs[1], widths[1]
    g.rect(
        x2 + w2 / 2 - 34,
        top + 25,
        68,
        15,
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        sw=0.9,
        rx=2,
    )
    g.text(
        x2 + w2 / 2,
        top + 35.5,
        "+ adapter \u00b7 biases",
        size=SMALL,
        anchor="middle",
        fill=ACCENT,
        weight="bold",
    )
    # Stage 3: a quantile function read at a uniform level.
    x3, w3 = xs[2], widths[2]
    px, py, pw, ph = x3 + 24, top + 18, w3 - 36, 30.0
    g.line(px, py + ph, px + pw, py + ph, stroke=MUTED, sw=0.6)
    g.line(px, py, px, py + ph, stroke=MUTED, sw=0.6)
    g.text(px + pw + 2, py + ph + 2.5, "u", size=SMALL, fill=MUTED)
    g.text(px - 3, py + 6, "Q(u)", size=SMALL, anchor="end", fill=MUTED)
    levels = [
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
    ]
    pts = []
    for u in levels:
        q = 0.5 + 0.32 * ((u - 0.5) / 0.5) ** 3 + 0.16 * (u - 0.5)
        pts.append((px + u * pw, py + ph - (0.08 + 0.84 * q) * ph))
    g.poly(pts, stroke=ACCENT, sw=1.1)
    for x, y in pts:
        g.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.1" fill="{ACCENT}"/>')
    ux, uy = pts[levels.index(0.7)]
    g.line(ux, py + ph, ux, uy, stroke=FG, sw=0.7, dash="1.5 1.5")
    g.line(ux, uy, px, uy, stroke=FG, sw=0.7, dash="1.5 1.5", arrow="ah")
    # Stage 4: the nowcast up to the latest data and the forecast after.
    x4, w4 = xs[3], widths[3]
    fx, fy, fw, fh = x4 + 10, top + 24, w4 - 20, 28.0
    n, last, latest = 14, 6, 9
    step = fw / (n - 1)
    base = [12, 13, 15, 18, 20, 24, 27, 30, 32, 34, 35, 35, 34, 33]
    reported = [12, 13, 15, 18, 20, 23, 24, 20, 14, 8]

    def yv(v: float) -> float:
        return fy + fh - v / 42 * fh

    ex, lx = fx + last * step, fx + latest * step
    g.rect(ex, fy - 3, lx - ex, fh + 3, fill=PANEL, rx=0)
    g.text(
        (ex + lx) / 2,
        fy - 5,
        "nowcast",
        size=SMALL - 0.5,
        fill=MUTED,
        anchor="middle",
    )
    g.text(
        (lx + fx + fw) / 2,
        fy - 5,
        "forecast",
        size=SMALL - 0.5,
        fill=MUTED,
        anchor="middle",
    )
    for wdt, op in [(1.5, 0.18), (0.75, 0.32)]:
        up, lo = [], []
        for k in range(last, n):
            s = (k - last) * wdt
            up.append((fx + k * step, yv(base[k] + s)))
            lo.append((fx + k * step, yv(base[k] - s * 0.8)))
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in up)
        d += " L" + " L".join(f"{x:.2f},{y:.2f}" for x, y in lo[::-1])
        g.path(d + " Z", stroke="none", fill=ACCENT, opacity=op)
    g.poly(
        [(fx + k * step, yv(base[k])) for k in range(last, n)],
        stroke=ACCENT,
        sw=1.0,
    )
    g.poly(
        [(fx + k * step, yv(base[k])) for k in range(last + 1)],
        stroke=FG,
        sw=0.9,
    )
    for k in range(latest + 1):
        g.add(
            f'<circle cx="{fx + k * step:.2f}" cy="{yv(reported[k]):.2f}" '
            f'r="1.3" fill="{COV}"/>'
        )
    g.line(lx, fy - 3, lx, fy + fh, stroke=FG, sw=0.7, dash="2 1.5")
    g.line(fx, fy + fh, fx + fw, fy + fh, stroke=MUTED, sw=0.6)
    return top + h


def build() -> str:
    """Return Figure 1 as SVG."""
    g = Svg()
    y = panel_a(g, 0.0)
    y = panel_b(g, y + 12)
    y = panel_c(g, y + 8)
    y = panel_d(g, y + 10)
    return g.render(round(y))


def main() -> None:
    """Write the SVG, PDF and optionally PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also write PNG.")
    args = parser.parse_args()
    svg = OUT.with_suffix(".svg")
    svg.write_text(build())
    rsvg = shutil.which("rsvg-convert")
    if rsvg is None:
        msg = "rsvg-convert (librsvg) is needed to write the PDF and PNG."
        raise FileNotFoundError(msg)
    subprocess.run(
        [rsvg, "-f", "pdf", "-o", str(OUT.with_suffix(".pdf")), str(svg)],
        check=True,
    )
    if args.png:
        subprocess.run(
            [
                rsvg,
                "-d",
                "200",
                "-p",
                "200",
                "-o",
                str(OUT.with_suffix(".png")),
                str(svg),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
