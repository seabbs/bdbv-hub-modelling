# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Draw Figure 2, the study design.

Writes `fig2-design.svg` next to this script, and `fig2-design.pdf`
and, with `--png`, a 200 dpi `fig2-design.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig2_design.py [--png]

The panels are:

- A, what we fine-tune Chronos-2 on: a tree from pretrained Chronos-2
  to zero-shot use and to LoRA fine-tunes on simulated or real data,
  with the main fine-tuned model's ablations as its leaves. Each
  fine-tuned model's node carries a small input block glyph;
- B, how the models are used: a web linking each model to the studies
  it is run in (held-out simulations, the German nowcast hub and the
  Ebola forecasting hub), starred where submitted to the Ebola hub.

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

ALL_BIASES = ("same", "parent")


def star(g: Svg, cx: float, cy: float, r: float = 3.8) -> None:
    """Draw a five-pointed star, the mark of an Ebola hub submission."""
    pts = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        a = math.pi * (k / 5 - 0.5)
        pts.append(f"{cx + rad * math.cos(a):.2f},{cy + rad * math.sin(a):.2f}")
    g.add(
        f'<polygon points="{" ".join(pts)}" fill="{COV}" '
        'stroke="white" stroke-width="0.5"/>'
    )


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


def link(
    g: Svg,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = LINE,
    sw: float = 0.8,
) -> None:
    """Draw a smooth horizontal S-curve from one node to another."""
    mx = (x1 + x2) / 2
    g.path(
        f"M{x1:.2f},{y1:.2f} C{mx:.2f},{y1:.2f} {mx:.2f},{y2:.2f} "
        f"{x2:.2f},{y2:.2f}",
        stroke=stroke,
        sw=sw,
    )


def node(
    g: Svg,
    x: float,
    y: float,
    w: float,
    h: float,
    name: str,
    sub: list[str] | None = None,
    glyph: tuple[bool, tuple[str, ...]] | None = None,
    main: bool = False,
) -> None:
    """Draw a node: a box with an optional glyph, a name and a note."""
    g.rect(
        x,
        y,
        w,
        h,
        fill="white",
        stroke=ACCENT if main else LINE,
        sw=1.1 if main else 0.6,
        rx=3,
    )
    tx = x + 6
    if glyph is not None:
        lags, biases = glyph
        bh = block_height(lags)
        mini(g, x + 5, y + (h - bh) / 2, lags, biases)
        tx = x + 47
    sub = sub or []
    n = 1 + len(sub)
    ty = y + h / 2 - 4.4 * (n - 1) + 2.6
    g.text(
        tx,
        ty,
        name,
        size=BODY,
        weight="bold",
        fill=ACCENT if main else FG,
    )
    if sub:
        g.text(tx, ty + 8.8, sub, size=SMALL, fill=MUTED, lead=1.25)


# Panel A: the ablations of the main fine-tuned model, name and glyph.
ABLATIONS = (
    ("No biases", (True, ())),
    ("Same place only", (True, ("same",))),
    ("Parent and child only", (True, ("parent",))),
    ("No lags", (False, ALL_BIASES)),
)


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, the tree of what we fine-tune Chronos-2 on."""
    g.label(0, oy + 9, "A", "What we fine-tune Chronos-2 on")
    top = oy + 22
    # Column positions: root, branch split, model, ablation leaves.
    lw, lh, lgap = 124.0, 18.0, 4.0
    lx = WIDTH - lw
    mw = 132.0
    mx = lx - 30 - mw
    # Zero-shot at the top, the simulated subtree, then real data.
    zy, zh = top, 16.0
    ay = zy + zh + 10
    ab_bottom = ay + len(ABLATIONS) * lh + (len(ABLATIONS) - 1) * lgap
    main_h = 26.0
    main_y = (ay + ab_bottom) / 2 - main_h / 2
    ry, rh = ab_bottom + 10, 32.0
    bottom = ry + rh
    # Root, centred between the simulated and real-data models.
    cw, ch = 66.0, 34.0
    cy = ((main_y + main_h / 2) + (ry + rh / 2)) / 2
    g.rect(0, cy - ch / 2, cw, ch, fill=PANEL, stroke=LINE, sw=0.7, rx=3)
    g.text(
        cw / 2, cy - 1, "Chronos-2", size=HEAD, weight="bold", anchor="middle"
    )
    g.text(
        cw / 2, cy + 9, "pretrained", size=SMALL, fill=MUTED, anchor="middle"
    )
    # Zero-shot, no fine-tuning: a small node.
    zw = 52.0
    link(g, cw, cy - 8, mx, zy + zh / 2, stroke=MUTED, sw=0.8)
    g.rect(mx, zy, zw, zh, fill="white", stroke=LINE, sw=0.6, rx=3)
    g.text(
        mx + zw / 2, zy + zh / 2 + 2.6, "Zero-shot", size=BODY, anchor="middle"
    )
    # LoRA trunk to a split by data.
    sx = 126.0
    g.line(cw, cy, sx, cy, stroke=MUTED, sw=0.8)
    g.text(
        (cw + sx) / 2,
        cy + 9,
        "LoRA fine-tune",
        size=SMALL,
        fill=MUTED,
        anchor="middle",
    )
    g.add(f'<circle cx="{sx:.2f}" cy="{cy:.2f}" r="1.6" fill="{MUTED}"/>')
    link(g, sx, cy, mx, main_y + main_h / 2, stroke=MUTED, sw=0.8)
    link(g, sx, cy, mx, ry + rh / 2, stroke=MUTED, sw=0.8)
    g.text(
        mx - 4,
        main_y + main_h / 2 - 3.5,
        "simulated",
        size=SMALL,
        fill=MUTED,
        italic=True,
        anchor="end",
    )
    g.text(
        mx - 4,
        ry + rh / 2 + 9,
        "real data",
        size=SMALL,
        fill=MUTED,
        italic=True,
        anchor="end",
    )
    # The main fine-tuned model and its ablations.
    for k, (name, glyph) in enumerate(ABLATIONS):
        y = ay + k * (lh + lgap)
        link(
            g,
            mx + mw,
            main_y + main_h / 2,
            lx,
            y + lh / 2,
            stroke=MUTED,
            sw=0.8,
        )
        node(g, lx, y, lw, lh, name, glyph=glyph)
    node(
        g,
        mx,
        main_y,
        mw,
        main_h,
        "Fine-tuned",
        ["all biases"],
        glyph=(True, ALL_BIASES),
        main=True,
    )
    node(
        g,
        mx,
        ry,
        mw,
        rh,
        "Real data",
        ["Germany to 9 Feb 2022", "Ebola hub to 8 Nov 2026"],
        glyph=(True, ALL_BIASES),
    )
    return bottom


# Panel B: the studies, name and maximum delay.
STUDIES = (
    ("Held-out simulations", ""),
    ("German nowcast hub", "max delay 40 d"),
    ("Ebola forecasting hub", "max delay 42 d"),
)
S, G, E = range(3)

# Panel B: the models, ordered to keep links from crossing. Each has a
# name, a glyph or None, and its links as (study, label, submitted).
ALL3 = ((S, "", False), (G, "", False), (E, "", False))
MODELS = (
    (
        "Fine-tuning ablations",
        None,
        ((S, "validation", False), (G, "development period", False)),
    ),
    ("Zero-shot, complete counts", (False, ()), ALL3),
    ("Zero-shot, reporting triangle", (True, ()), ALL3),
    ("Nowcast \u2192 Chronos-2", None, ALL3),
    (
        "Fine-tuned",
        (True, ALL_BIASES),
        ((S, "", False), (G, "", False), (E, "", True)),
    ),
    (
        "Random walk nowcast",
        None,
        ((S, "", False), (G, "", False), (E, "", True)),
    ),
    ("Renewal models", None, ((E, "", True),)),
    (
        "Real-data fine-tune",
        None,
        ((G, "test period", False), (E, "after 8 Nov 2026", False)),
    ),
)


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, the web of which models run in which studies."""
    g.label(0, oy + 9, "B", "How the models are used")
    top = oy + 22
    # Left: the models.
    mw, mh, mgap = 150.0, 16.0, 7.0
    ys = [top + k * (mh + mgap) for k in range(len(MODELS))]
    bottom = ys[-1] + mh
    # Right: the studies, spread over the same height.
    sw_, sx = 120.0, WIDTH - 120.0
    sgap = 14.0
    sh = (bottom - top - 2 * sgap) / 3
    sy = [top + k * (sh + sgap) for k in range(3)]
    # Ports on each study's left edge, in the order of their models.
    ins: dict[int, list[int]] = {s: [] for s in range(3)}
    for i, (_, _, uses) in enumerate(MODELS):
        for s, _, _ in uses:
            ins[s].append(i)
    port: dict[tuple[int, int], float] = {}
    for s, models in ins.items():
        n = len(models)
        pad = 7.0
        for k, i in enumerate(models):
            port[(i, s)] = sy[s] + pad + k * (sh - 2 * pad) / (n - 1)
    # Links, grey first and the fine-tuned model's on top.
    order = sorted(
        range(len(MODELS)), key=lambda i: MODELS[i][0] == "Fine-tuned"
    )
    labels = []
    stars = []
    for i in order:
        name, _, uses = MODELS[i]
        main = name == "Fine-tuned"
        y1 = ys[i] + mh / 2
        for s, label, submitted in uses:
            y2 = port[(i, s)]
            link(
                g,
                mw,
                y1,
                sx,
                y2,
                stroke=ACCENT if main else LINE,
                sw=1.1 if main else 0.8,
            )
            if label:
                labels.append((i, s, y1, y2, label))
            if submitted:
                stars.append((sx, y2))
    for i, (name, glyph, _) in enumerate(MODELS):
        node(g, 0, ys[i], mw, mh, name, glyph=glyph, main=name == "Fine-tuned")
    for s, (name, tag) in enumerate(STUDIES):
        g.rect(sx, sy[s], sw_, sh, fill=PANEL, stroke=LINE, sw=0.6, rx=3)
        cy = sy[s] + sh / 2
        if tag:
            g.text(sx + 10, cy - 1, name, size=HEAD, weight="bold")
            g.text(sx + 10, cy + 8.5, tag, size=SMALL, fill=MUTED)
        else:
            g.text(sx + 10, cy + 2.8, name, size=HEAD, weight="bold")
    # Link labels inside each study, beside the link's port.
    for _, _, _, y2, label in labels:
        g.text(
            sx + 5, y2 + 2.3, label, size=SMALL - 0.5, fill=MUTED, italic=True
        )
    for x, y in stars:
        star(g, x, y)
    # One legend line.
    y = bottom + 15
    x = 0.0
    items = (
        ("star", "submitted to the Ebola hub"),
        ("same", "same-place bias"),
        ("parent", "parent and child biases"),
    )
    for kind, words in items:
        if kind == "star":
            star(g, x + 3, y - 2.5)
        else:
            colour = SLOTS[0] if kind == "same" else PARENT_SLOT
            g.rect(x + 2, y - 7, 2.0, 9, fill=colour, rx=1.0)
        g.text(x + 9, y, words, size=SMALL)
        x += 9 + 3.55 * len(words) + 16
    return y + 2


def build() -> str:
    """Return Figure 2 as SVG."""
    g = Svg()
    y = panel_a(g, 0.0)
    y = panel_b(g, y + 14)
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
