# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Draw Figure 2, the study design.

Writes `fig2-design.svg` next to this script, and `fig2-design.pdf`
and, with `--png`, a 200 dpi `fig2-design.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig2_design.py [--png]

The panels are:

- A, what we fine-tune Chronos-2 on, one row per fine-tuned model, each
  marked main text or Supporting Information, from base Chronos-2;
- B, how the models are used: models and their inputs by application
  (held-out simulations, the German nowcast hub and the Ebola
  forecasting hub), each cell marked main text or Supporting
  Information, and left empty where the model does not apply.

It reuses the drawing helpers and palette of Figure 1
(`fig1_pipeline.py`). Units are points at the final size, 6.5 inches
wide, so font sizes are the printed sizes.
"""

from __future__ import annotations

import argparse
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
    SLOTS,
    SMALL,
    WIDTH,
    Svg,
)

OUT = Path(__file__).resolve().parent / "fig2-design"

# The role of a fine-tuned model or a cell: main text or SI.
MAIN = "main"
SI = "si"


def badge(g: Svg, x: float, y: float, role: str) -> float:
    """Draw a Main (filled) or SI (outlined) badge; return its width."""
    w, h = 24.0, 10.0
    if role == MAIN:
        g.rect(x, y, w, h, fill=ACCENT, stroke=ACCENT, sw=0.8, rx=2)
        colour, name = "white", "Main"
    else:
        g.rect(x, y, w, h, fill="white", stroke=ACCENT, sw=0.8, rx=2)
        colour, name = ACCENT, "SI"
    g.text(
        x + w / 2,
        y + 7.3,
        name,
        size=SMALL - 0.5,
        anchor="middle",
        fill=colour,
        weight="bold",
    )
    return w


def _layout(g: Svg, x: float, y: float, lags: bool, biases: bool) -> None:
    """Draw a small input block: a target row and, optionally, lag rows.

    Cells are shaded as in Figure 1 (`cell`), drawn finer at this size.
    A bar to the left marks the place attention biases.
    """
    cols, cw, rh = 7, 4.6, 3.2
    rows = [["final" if c < 4 else "future" for c in range(cols)]]
    if lags:
        for d in range(3):
            rows.append(
                ["lag" if d <= c < cols - d else "none" for c in range(cols)]
            )
    style = {
        "final": (CELL, "white"),
        "future": (ACCENT_SOFT, ACCENT),
        "lag": (COV_SOFT, COV),
        "none": (PANEL, "white"),
    }
    yy = y
    for k, row in enumerate(rows):
        if k == 1:
            yy += 1
        for c, kind in enumerate(row):
            fill, edge = style[kind]
            g.rect(
                x + c * cw, yy, cw, rh, fill=fill, stroke=edge, sw=0.35, rx=0.6
            )
        yy += rh + 0.6
    if biases:
        g.rect(x - 4, y, 2.2, yy - y - 0.6, fill=SLOTS[0], rx=1.1)


# What each fine-tuned model is trained on (panel A): its source, a
# description, its name in panel B, the role, whether its blocks have lag
# series and whether it has the attention biases.
FINETUNES = (
    (
        "Simulated surveillance data",
        "reporting triangle, with attention biases",
        "Fine-tuned",
        MAIN,
        True,
        True,
    ),
    (
        "Simulated data, no attention biases",
        "reporting triangle only",
        "No biases",
        SI,
        True,
        False,
    ),
    (
        "Simulated data as complete counts",
        "no lag series, with attention biases",
        "No lags",
        SI,
        False,
        True,
    ),
    (
        "Real data",
        "German development half, Ebola hub releases",
        "Real data",
        SI,
        True,
        True,
    ),
)


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, what we fine-tune Chronos-2 on."""
    g.label(0, oy + 9, "A", "What we fine-tune Chronos-2 on")
    top = oy + 26
    rh, gap = 22.0, 4.0
    n = len(FINETUNES)
    bottom = top + n * rh + (n - 1) * gap
    # Base Chronos-2, the start of every fine-tune, also used zero-shot.
    bw = 82.0
    by = top + 8
    bh = bottom - top - 16
    g.rect(0, by, bw, bh, fill=PANEL, stroke=LINE, sw=0.7, rx=3)
    g.text(
        bw / 2,
        by + 18,
        "Chronos-2",
        size=HEAD,
        weight="bold",
        anchor="middle",
    )
    g.text(
        bw / 2,
        by + 29,
        ["pretrained", "weights"],
        size=SMALL,
        anchor="middle",
        fill=MUTED,
        lead=1.2,
    )
    g.text(
        bw / 2,
        by + bh - 26,
        ["used zero-shot,", "no fine-tuning"],
        size=SMALL,
        anchor="middle",
        fill=FG,
        lead=1.2,
    )
    badge(g, bw / 2 - 12, by + bh - 14, MAIN)
    # The LoRA fine-tunes fan out to the rows.
    rx = 116.0
    g.text(
        (bw + rx) / 2,
        top - 3,
        "LoRA fine-tune",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    mid = (top + bottom) / 2
    for k, (head, words, name, role, lags, biases) in enumerate(FINETUNES):
        y = top + k * (rh + gap)
        g.path(
            f"M{bw:.2f},{mid:.2f} C{bw + 16:.2f},{mid:.2f} "
            f"{rx - 22:.2f},{y + rh / 2:.2f} {rx - 7:.2f},{y + rh / 2:.2f}",
            stroke=ACCENT if role == MAIN else MUTED,
            sw=1.1 if role == MAIN else 0.7,
        )
        g.line(
            rx - 8,
            y + rh / 2,
            rx - 2,
            y + rh / 2,
            stroke=ACCENT if role == MAIN else MUTED,
            sw=1.1 if role == MAIN else 0.7,
            arrow="aha" if role == MAIN else "ah",
        )
        g.rect(
            rx,
            y,
            WIDTH - rx,
            rh,
            fill="white",
            stroke=ACCENT if role == MAIN else LINE,
            sw=0.9 if role == MAIN else 0.6,
            rx=3,
        )
        gh = 3.2 + (3 * 3.8 + 1 if lags else 0)
        _layout(g, rx + 10, y + (rh - gh) / 2, lags, biases)
        g.text(rx + 52, y + 9.5, head, size=BODY, weight="bold")
        g.text(rx + 52, y + 18, words, size=SMALL, fill=MUTED)
        g.text(WIDTH - 34, y + 13.8, name, size=BODY, anchor="end")
        badge(g, WIDTH - 29, y + (rh - 10) / 2, role)
    return bottom


# Applications, the columns of panel B: a name and a note.
APPS = (
    ("Held-out simulations", "test set"),
    ("German nowcast hub", "7-day sums, max delay 40 days"),
    ("Ebola forecasting hub", "weekly targets, max delay 42 days"),
)

# Models and inputs, the rows of panel B: the model, its input, and per
# application a role and a note, or None where the model does not apply.
USES = (
    (
        "Fine-tuned",
        "reporting triangle",
        [(MAIN, ""), (MAIN, ""), (MAIN, "submitted")],
    ),
    (
        "Zero-shot",
        "reporting triangle",
        [(MAIN, ""), (MAIN, ""), (MAIN, "")],
    ),
    (
        "Zero-shot",
        "complete counts: latest counts, no lags",
        [(MAIN, ""), (MAIN, ""), (MAIN, "")],
    ),
    (
        "Random walk nowcast",
        "baselinenowcast, extended to forecast",
        [(MAIN, ""), (MAIN, ""), (MAIN, "submitted")],
    ),
    (
        "Zero-shot and fine-tuned",
        "random walk point nowcast as complete counts",
        [(SI, ""), (SI, ""), (SI, "")],
    ),
    (
        "No biases, no lags",
        "fine-tuned without attention biases or lags",
        [(SI, "validation set"), (SI, "development half"), None],
    ),
    (
        "Real data",
        "fine-tuned on real data, reporting triangle",
        [None, (SI, "held-out half"), (SI, "later rounds")],
    ),
    (
        "Renewal models",
        "epinowcast on PyRenew; variants in SI",
        [None, None, (MAIN, "submitted")],
    ),
)


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, how the models are used."""
    g.label(0, oy + 9, "B", "How the models are used")
    cx0 = 180.0
    cw = (WIDTH - cx0) / len(APPS)
    top = oy + 20
    for k, (name, note) in enumerate(APPS):
        x = cx0 + k * cw
        g.text(x + 6, top + 8, name, size=HEAD, weight="bold")
        g.text(x + 6, top + 17, note, size=SMALL, fill=MUTED)
    g.text(0, top + 17, "Model and input", size=SMALL, fill=MUTED)
    rh = 20.0
    y = top + 22
    g.line(0, y, WIDTH, y, stroke=FG, sw=0.6)
    for r, (model, words, cells) in enumerate(USES):
        if r % 2 == 0:
            g.rect(0, y, WIDTH, rh, fill=PANEL, rx=0)
        g.text(4, y + 8.5, model, size=BODY, weight="bold")
        g.text(4, y + 16.5, words, size=SMALL, fill=MUTED)
        for k, use in enumerate(cells):
            if use is None:
                continue
            role, note = use
            x = cx0 + k * cw + 6
            badge(g, x, y + (rh - 10) / 2, role)
            if note:
                g.text(x + 28, y + rh / 2 + 2.5, note, size=SMALL, fill=MUTED)
        y += rh
        if r == 3:
            # The four models in every main results figure, above the line.
            g.line(0, y, WIDTH, y, stroke=MUTED, sw=0.5, dash="2 1.5")
    g.line(0, y, WIDTH, y, stroke=FG, sw=0.6)
    for k in range(1, len(APPS)):
        x = cx0 + k * cw
        g.line(x, top, x, y, stroke=LINE, sw=0.5)
    g.line(cx0, top, cx0, y, stroke=LINE, sw=0.5)
    # Legend and notes.
    y += 12
    x = 0.0
    for role, name in [(MAIN, "main text"), (SI, "Supporting Information")]:
        x += badge(g, x, y - 7.5, role) + 4
        g.text(x, y, name, size=SMALL)
        x += 3.6 * len(name) + 14
    g.rect(
        x,
        y - 7.5,
        24,
        10,
        fill="none",
        stroke=LINE,
        sw=0.5,
        dash="1.5 1.5",
    )
    g.text(x + 28, y, "empty: does not apply", size=SMALL)
    notes = [
        "One forward pass gives the daily counts and, from each place's"
        " aggregate row, the 7-day sums or weekly targets.",
        "Ebola: two passes per nowcast date, the active block and one block"
        " per province, to fit in GPU memory.",
        "Ebola only: draws paired across places by a copula for the hub's"
        " trajectories, with the alternatives in the SI.",
    ]
    y += 12
    for line in notes:
        g.rect(1, y - 4.5, 2.5, 2.5, fill=COV, rx=1.2)
        g.text(8, y, line, size=SMALL, fill=FG)
        y += 9.5
    return y - 6


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
