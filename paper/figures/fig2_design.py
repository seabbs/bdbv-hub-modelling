# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Draw Figure 2, the study design.

Writes `fig2-design.svg` next to this script, and `fig2-design.pdf`
and, with `--png`, a 200 dpi `fig2-design.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig2_design.py [--png]

The panels are:

- A, what we fine-tune Chronos-2 on: base Chronos-2, LoRA, and one box
  per fine-tuned model with a small input block glyph, marked main text
  (filled dot) or Supporting Information (outlined dot);
- B, how the models are used: rows by model, columns by study (held-out
  simulations, the German nowcast hub and the Ebola forecasting hub),
  each cell marked main text or Supporting Information, starred where
  submitted to the Ebola hub, and empty where the model does not apply.

It reuses the drawing helpers and palette of Figure 1
(`fig1_pipeline.py`). Units are points at the final size, 6.5 inches
wide, so font sizes are the printed sizes.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

from fig1_pipeline import (
    ACCENT,
    ACCENT_SOFT,
    BODY,
    CELL,
    COV,
    COV_SOFT,
    FG,
    HEAD,
    LINE,
    MUTED,
    PANEL,
    PARENT_SLOT,
    SLOTS,
    SMALL,
    WIDTH,
    Svg,
)

OUT = Path(__file__).resolve().parent / "fig2-design"

# The role of a fine-tuned model or a cell: main text or SI.
MAIN = "main"
SI = "si"
ALL_BIASES = ("same", "parent")


def role_dot(g: Svg, cx: float, cy: float, role: str) -> None:
    """Draw a main text (filled) or SI (outlined) dot."""
    fill = ACCENT if role == MAIN else "white"
    g.add(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.1" fill="{fill}" '
        f'stroke="{ACCENT}" stroke-width="0.9"/>'
    )


def star(g: Svg, cx: float, cy: float, r: float = 3.6) -> None:
    """Draw a five-pointed star, the mark of an Ebola hub submission."""
    pts = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        a = math.pi * (k / 5 - 0.5)
        pts.append(f"{cx + rad * math.cos(a):.2f},{cy + rad * math.sin(a):.2f}")
    g.add(f'<polygon points="{" ".join(pts)}" fill="{COV}"/>')


def mini(
    g: Svg,
    x: float,
    y: float,
    lags: bool,
    biases: tuple[str, ...],
    cw: float = 4.2,
    rh: float = 3.0,
) -> None:
    """Draw a small input block: a target row and, optionally, lag rows.

    Cells are shaded as in Figure 1. The bars to the left mark the
    attention biases in Figure 1's colours, the same-place bias (`same`)
    and the parent and child biases (`parent`). The block starts at `x`
    plus room for the bars.
    """
    cols = 7
    rows = [["final" if c < 4 else "future" for c in range(cols)]]
    if lags:
        rows += [
            ["lag" if c < 5 - d else "none" for c in range(cols)]
            for d in range(3)
        ]
    style = {
        "final": (CELL, "white"),
        "future": (ACCENT_SOFT, ACCENT),
        "lag": (COV_SOFT, COV),
        "none": (PANEL, "white"),
    }
    bx = x + 7
    yy = y
    for k, row in enumerate(rows):
        if k == 1:
            yy += 1
        for c, kind in enumerate(row):
            fill, edge = style[kind]
            g.rect(
                bx + c * cw, yy, cw, rh, fill=fill, stroke=edge, sw=0.35, rx=0.5
            )
        yy += rh + 0.5
    # Bars at least 7 points tall, centred on the block.
    h = yy - y - 0.5
    bh = max(h, 7.0)
    bars = (("same", SLOTS[0]), ("parent", PARENT_SLOT))
    for k, (kind, colour) in enumerate(bars):
        if kind in biases:
            g.rect(x + 3 * k, y + (h - bh) / 2, 2.0, bh, fill=colour, rx=1.0)


def block_height(lags: bool, rh: float = 3.0) -> float:
    """Return the height of a `mini` block."""
    return rh + (3 * (rh + 0.5) + 1 if lags else 0)


# The fine-tuned models (panel A): name, data, role, lag series, biases.
SIMULATED = ["simulated"]
FINETUNES = (
    ("All biases", SIMULATED, MAIN, True, ALL_BIASES),
    ("No biases", SIMULATED, SI, True, ()),
    ("Same place only", SIMULATED, SI, True, ("same",)),
    ("Parent and child only", SIMULATED, SI, True, ("parent",)),
    ("No lags", SIMULATED, SI, False, ALL_BIASES),
    (
        "Real data",
        ["Germany to 9 Feb 2022", "Ebola hub to 8 Nov 2026"],
        SI,
        True,
        ALL_BIASES,
    ),
)


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, what we fine-tune Chronos-2 on."""
    g.label(0, oy + 9, "A", "What we fine-tune Chronos-2 on")
    top = oy + 24
    bh, gap = 32.0, 5.0
    nrow = 3
    bottom = top + nrow * bh + (nrow - 1) * gap
    mid = (top + bottom) / 2
    # Base Chronos-2, the start of every fine-tune.
    cw, ch = 64.0, 34.0
    g.rect(0, mid - ch / 2, cw, ch, fill=PANEL, stroke=LINE, sw=0.7, rx=3)
    g.text(
        cw / 2, mid - 1, "Chronos-2", size=HEAD, weight="bold", anchor="middle"
    )
    g.text(
        cw / 2, mid + 9, "pretrained", size=SMALL, fill=MUTED, anchor="middle"
    )
    # One LoRA arrow to a bracket over the fine-tuned models.
    gx = 112.0
    g.line(cw + 2, mid, gx - 8, mid, arrow="ah")
    g.text(
        (cw + gx - 6) / 2,
        mid - 4,
        "LoRA",
        size=SMALL,
        fill=MUTED,
        anchor="middle",
    )
    g.rect(
        gx - 6,
        top - 5,
        WIDTH - gx + 6,
        bottom - top + 10,
        fill="none",
        stroke=LINE,
        sw=0.6,
        rx=4,
        dash="2 1.5",
    )
    bw = (WIDTH - gx - 12) / 2
    for k, (name, data, role, lags, biases) in enumerate(FINETUNES):
        r, c = divmod(k, 2)
        x = gx + c * (bw + 6)
        y = top + r * (bh + gap)
        main = role == MAIN
        g.rect(
            x,
            y,
            bw,
            bh,
            fill="white",
            stroke=ACCENT if main else LINE,
            sw=1.0 if main else 0.6,
            rx=3,
        )
        h = block_height(lags)
        mini(g, x + 6, y + (bh - h) / 2, lags, biases)
        tx = x + 48
        lines = len(data)
        ty = y + bh / 2 - 4.2 * lines + 2.8
        g.text(tx, ty, name, size=BODY, weight="bold")
        g.text(tx, ty + 9, data, size=SMALL, fill=MUTED, lead=1.2)
        role_dot(g, x + bw - 7, y + 7, role)
    return bottom


# The studies, the columns of panel B: a name and a tag.
APPS = (
    ("Held-out simulations", ""),
    ("German nowcast hub", "max delay 40 d"),
    ("Ebola forecasting hub", "max delay 42 d"),
)

# The rows of panel B: a name, a glyph (lag series, biases) or None, and
# per study a role, a note and whether it was submitted to the Ebola hub,
# or None where the row does not apply.
USES = (
    (
        "Zero-shot, complete counts",
        (False, ()),
        [(MAIN, "", False)] * 3,
    ),
    (
        "Zero-shot, reporting triangle",
        (True, ()),
        [(MAIN, "", False)] * 3,
    ),
    (
        "Fine-tuned",
        (True, ALL_BIASES),
        [(MAIN, "", False), (MAIN, "", False), (MAIN, "", True)],
    ),
    (
        "Random walk nowcast",
        None,
        [(MAIN, "", False), (MAIN, "", False), (MAIN, "", True)],
    ),
    (
        "Nowcast → Chronos-2",
        None,
        [(SI, "", False)] * 3,
    ),
    (
        "Fine-tuning ablations",
        None,
        [(SI, "validation", False), (SI, "to 9 Feb 2022", False), None],
    ),
    (
        "Real-data fine-tune",
        None,
        [
            None,
            (SI, "from 10 Feb 2022", False),
            (SI, "after 8 Nov 2026", False),
        ],
    ),
    (
        "Renewal models",
        None,
        [None, None, (MAIN, "", True)],
    ),
)


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, how the models are used in each study."""
    g.label(0, oy + 9, "B", "How the models are used")
    cx0 = 150.0
    cw = (WIDTH - cx0) / len(APPS)
    top = oy + 18
    for k, (name, tag) in enumerate(APPS):
        x = cx0 + k * cw + 8
        g.text(x, top + 8, name, size=HEAD, weight="bold")
        if tag:
            g.text(x, top + 17, tag, size=SMALL, fill=MUTED)
    rh = 15.0
    y = top + 22
    g.line(0, y, WIDTH, y, stroke=FG, sw=0.6)
    for r, (name, glyph, cells) in enumerate(USES):
        if r % 2 == 0:
            g.rect(0, y, WIDTH, rh, fill=PANEL, rx=0)
        if glyph is not None:
            lags, biases = glyph
            h = block_height(lags, rh=2.4)
            mini(g, 4, y + (rh - h) / 2, lags, biases, cw=3.4, rh=2.4)
        g.text(40, y + rh / 2 + 2.6, name, size=BODY)
        for k, use in enumerate(cells):
            if use is None:
                continue
            role, note, submitted = use
            x = cx0 + k * cw + 12
            role_dot(g, x, y + rh / 2, role)
            if submitted:
                star(g, x + 10, y + rh / 2)
            if note:
                g.text(x + 7, y + rh / 2 + 2.4, note, size=SMALL, fill=MUTED)
        y += rh
        if r == 3:
            # The four models in every main results figure, above the line.
            g.line(0, y, WIDTH, y, stroke=MUTED, sw=0.5, dash="2 1.5")
    g.line(0, y, WIDTH, y, stroke=FG, sw=0.6)
    for k in range(len(APPS)):
        x = cx0 + k * cw
        g.line(x, top, x, y, stroke=LINE, sw=0.5)
    # One legend line.
    y += 12
    x = 4.0
    items = (
        ("main", "main text"),
        ("si", "Supporting Information"),
        ("star", "submitted to the Ebola hub"),
        ("same", "same-place bias"),
        ("parent", "parent and child biases"),
    )
    for kind, words in items:
        if kind in (MAIN, SI):
            role_dot(g, x, y - 2.5, kind)
        elif kind == "star":
            star(g, x, y - 2.5)
        else:
            colour = SLOTS[0] if kind == "same" else PARENT_SLOT
            g.rect(x - 1, y - 7, 2.0, 9, fill=colour, rx=1.0)
        g.text(x + 6, y, words, size=SMALL)
        x += 6 + 3.55 * len(words) + 14
    return y + 2


def build() -> str:
    """Return Figure 2 as SVG."""
    g = Svg()
    y = panel_a(g, 0.0)
    y = panel_b(g, y + 12)
    return g.render(round(y + 3))


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
