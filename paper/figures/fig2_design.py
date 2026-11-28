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
- B, how the models are used: a flow from each model of panel A and
  our two baselines, through what it is given (complete counts, a
  nowcast as complete counts or the reporting triangle), to the studies
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


# Panel A: the ablations of the main fine-tuned model: name, note, glyph.
ABLATIONS = (
    ("No attention biases", "", (True, ())),
    ("Same place", "attention bias only", (True, ("same",))),
    ("Parent and child", "attention biases only", (True, ("parent",))),
    ("No delay information", "", (False, ALL_BIASES)),
)


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, the tree of what we fine-tune Chronos-2 on."""
    g.label(0, oy + 9, "A", "What we fine-tune Chronos-2 on")
    top = oy + 22
    # Column positions: root, branch split, model, ablation leaves.
    lw, lh, lgap = 124.0, 22.0, 3.0
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
    for k, (name, sub, glyph) in enumerate(ABLATIONS):
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
        node(g, lx, y, lw, lh, name, sub=[sub] if sub else None, glyph=glyph)
    node(
        g,
        mx,
        main_y,
        mw,
        main_h,
        "Fine-tuned",
        ["all attention biases"],
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


# Panel B: the models, named and glyphed as in panel A, then our two
# baselines. Each has a name, a note, a glyph or None and a height.
ZS, FT, AB, RD, RW, RN = range(6)
MODELS = (
    ("Zero-shot", [], None, 16.0),
    ("Fine-tuned", ["all attention biases"], (True, ALL_BIASES), 26.0),
    ("Ablations", ["of attention biases", "and delay information"], None, 26.0),
    (
        "Real data",
        ["Germany to 9 Feb 2022", "Ebola hub to 8 Nov 2026"],
        (True, ALL_BIASES),
        32.0,
    ),
    ("Random walk nowcast", [], None, 16.0),
    ("Renewal models", [], None, 16.0),
)
SUBMITTED = (FT, RW, RN)

# Panel B: one colour per model, its node's stripe and its links
# (colour-blind safe, after Okabe and Ito and Tol).
MODEL_COLOURS = (
    "#E69F00",
    ACCENT,
    "#AA3377",
    "#332288",
    "#D55E00",
    "#555555",
)

# Panel B: what each model is given, a name and a note.
CC, NC, RT = range(3)
INPUTS = (
    ("Complete counts", ["latest counts only"]),
    ("Nowcast", ["random walk nowcast", "as complete counts"]),
    ("Reporting triangle", ["counts and lag series"]),
)

# Panel B: the studies, a name and the maximum delay.
S, G, E = range(3)
STUDIES = (
    ("Held-out simulations", ""),
    ("German nowcast hub", "max delay 40 d"),
    ("Ebola forecasting hub", "max delay 42 d"),
)

# Panel B: the flows, as (model, input, studies, width). Each study is
# (study, note), the note naming the data scored where it differs.
THIN = 0.8
FLOWS = (
    (ZS, CC, ((S, ""), (G, ""), (E, ""))),
    (ZS, NC, ((S, ""), (G, ""), (E, ""))),
    (ZS, RT, ((S, ""), (G, ""), (E, ""))),
    (FT, NC, ((S, ""), (G, ""), (E, ""))),
    (FT, RT, ((S, ""), (G, ""), (E, ""))),
    (AB, CC, ((S, ""), (G, ""))),
    (AB, RT, ((S, "validation"), (G, "development period"))),
    (RD, RT, ((G, "test period"), (E, "after 8 Nov 2026"))),
    (RW, RT, ((S, ""), (G, ""), (E, ""))),
    (RN, RT, ((E, ""),)),
)


def flow_width(model: int, inp: int) -> float:
    """Return a link's width: thin for the no-delays ablation."""
    return THIN if (model, inp) == (AB, CC) else 1.5


def stack(
    links: list[tuple[float, float, object]], top: float, h: float
) -> dict[object, float]:
    """Centre a stack of links on an edge, in order of their sort key.

    Each link is (sort key, width, id); returns each id's centre.
    """
    gap = 0.4
    total = sum(w for _, w, _ in links) + gap * (len(links) - 1)
    y = top + (h - total) / 2
    out = {}
    for _, w, key in sorted(links, key=lambda t: t[0]):
        out[key] = y + w / 2
        y += w + gap
    return out


def spread(heights: list[float], top: float, bottom: float) -> list[float]:
    """Return the tops of boxes spread evenly from `top` to `bottom`."""
    gap = (bottom - top - sum(heights)) / (len(heights) - 1)
    ys, y = [], top
    for h in heights:
        ys.append(y)
        y += h + gap
    return ys


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, each model's flow through its input to each study."""
    g.label(0, oy + 9, "B", "How the models are used")
    top = oy + 22
    # Columns: models, inputs and studies.
    mw = 132.0
    iw, ix = 98.0, 192.0
    sw_ = 108.0
    sx = WIDTH - sw_
    mgap = 6.0
    mh = [m[3] for m in MODELS]
    my = spread(mh, top, top + sum(mh) + mgap * (len(mh) - 1))
    bottom = my[-1] + mh[-1]
    ih = [10.0 + 9.0 * (1 + len(sub)) for _, sub in INPUTS]
    iy = spread(ih, top + 4, bottom - 4)
    # Study notes, one line per noted flow.
    notes: dict[int, list[tuple[int, str]]] = {s: [] for s in range(3)}
    for model, _, uses in FLOWS:
        for s, note in uses:
            if note:
                notes[s].append((model, note))
    sh = [
        12.0 + 9.5 * (1 + bool(tag) + len(notes[s]))
        for s, (_, tag) in enumerate(STUDIES)
    ]
    sy = spread(sh, top + 8, bottom - 8)
    mc = [y + h / 2 for y, h in zip(my, mh, strict=True)]
    ic = [y + h / 2 for y, h in zip(iy, ih, strict=True)]
    sc = [y + h / 2 for y, h in zip(sy, sh, strict=True)]
    # Ports: each edge's links stacked in order of the far end's height.
    m_out: dict[int, list] = {i: [] for i in range(len(MODELS))}
    i_in: dict[int, list] = {i: [] for i in range(3)}
    i_out: dict[int, list] = {i: [] for i in range(3)}
    s_in: dict[int, list] = {s: [] for s in range(3)}
    for model, inp, uses in FLOWS:
        w = flow_width(model, inp)
        m_out[model].append(((ic[inp], inp), w, (model, inp)))
        i_in[inp].append(((mc[model], model), w, (model, inp)))
        for s, _ in uses:
            i_out[inp].append(((sc[s], model), w, (model, inp, s)))
            s_in[s].append(((ic[inp], model), w, (model, inp, s)))
    pm: dict = {}
    for i, links in m_out.items():
        pm.update(stack(links, my[i], mh[i]))
    pi_in: dict = {}
    pi_out: dict = {}
    for i in range(3):
        pi_in.update(stack(i_in[i], iy[i], ih[i]))
        pi_out.update(stack(i_out[i], iy[i], ih[i]))
    ps: dict = {}
    for s in range(3):
        ps.update(stack(s_in[s], sy[s], sh[s]))
    # Links in each model's colour, the fine-tuned model's on top.
    order = sorted(FLOWS, key=lambda f: f[0] == FT)
    for model, inp, uses in order:
        colour = MODEL_COLOURS[model]
        w = flow_width(model, inp)
        link(g, mw, pm[(model, inp)], ix, pi_in[(model, inp)], colour, w)
        for s, _ in uses:
            link(
                g,
                ix + iw,
                pi_out[(model, inp, s)],
                sx,
                ps[(model, inp, s)],
                colour,
                w,
            )
    # Model nodes, with their colour stripe and a star if submitted.
    for i, (name, sub, glyph, h) in enumerate(MODELS):
        node(g, 0, my[i], mw, h, name, sub=sub, glyph=glyph, main=i == FT)
        g.rect(mw - 3, my[i] + 2, 3, h - 4, fill=MODEL_COLOURS[i], rx=1)
        if i in SUBMITTED:
            star(g, mw - 10, my[i])
    for i, (name, sub) in enumerate(INPUTS):
        node(g, ix, iy[i], iw, ih[i], name, sub=sub)
    for s, (name, tag) in enumerate(STUDIES):
        g.rect(sx, sy[s], sw_, sh[s], fill=PANEL, stroke=LINE, sw=0.6, rx=3)
        y = sy[s] + 13
        g.text(sx + 7, y, name, size=HEAD, weight="bold")
        if tag:
            y += 9.5
            g.text(sx + 7, y, tag, size=SMALL, fill=MUTED)
        for model, note in notes[s]:
            y += 9.5
            g.line(sx + 7, y - 2.4, sx + 15, y - 2.4, MODEL_COLOURS[model], 1.5)
            g.text(sx + 18, y, note, size=SMALL, fill=MUTED, italic=True)
    # One legend line.
    y = bottom + 15
    x = 0.0
    items = (
        ("star", "submitted to the Ebola hub"),
        ("same", "same-place attention bias"),
        ("parent", "parent and child attention biases"),
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
