"""Static HTML report of our hub rounds against every hub team.

Reads the `compare-hub` output over the hub's own files (no model of
ours is run), each week scored once it is complete by the scorer's rule
(`nowcasteval.nowcast.complete_weeks`). For our last
rounds it shows which chain model went out under each hub id (the
ledger's `submissions.csv`), each round's fans against the latest data
(`nowcasteval.plots.plot_hub_round`), and WIS, WIS relative to the hub
baseline on the cells both forecast, and 50% and 90% coverage by model,
horizon and location type, with a summary up top.

    python workflow/scripts/hub_rounds_report.py OUT.html COMPARISON_DIR \
        DATA_DIR SUBMISSIONS_CSV HEAD_FILE N_ROUNDS
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]

from bdbvhub.data.tidy import load_target  # noqa: E402
from nowcasteval.plots import comparison_colours, plot_hub_round  # noqa: E402
from nowcasteval.scoring import relative_skill  # noqa: E402
from nowcastutils.hierarchy import is_zone  # noqa: E402

BASELINE = "BDBVHub-baseline"
"""The hub's baseline, which relative WIS divides by."""

OURS = {"seabbs-renewal": "renewalbvd", "seabbs-rwnowcast": "randomwalknowcast"}
"""Our hub ids and the model whose colour each takes."""

LOCATIONS = ("CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO")
"""The national and province panels of the fans."""

MATCH = ("reference_date", "truth_vintage", "location", "target_end_date")
"""What makes two models' scores the same cell."""


def location_type(loc: str) -> str:
    """Return `national`, `province` or `zone`."""
    if loc == "CD":
        return "national"
    return "zone" if is_zone(loc) else "province"


def last_rounds(quantiles: pd.DataFrame, n: int) -> list[str]:
    """Return the last `n` rounds any of our hub ids forecast."""
    ours = quantiles.loc[quantiles["model"].isin(list(OURS))]
    return sorted(set(ours["reference_date"].astype(str)))[-n:]


def score_table(scores: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Mean WIS, coverage and WIS relative to the hub baseline.

    Relative WIS is the model's mean WIS over the baseline's on the cells
    both forecast (`relative_skill`); missing where they share none.
    """
    if scores.empty:
        return pd.DataFrame()
    group = ["model", *by]
    out = (
        scores.groupby(group)
        .agg(
            n=("wis", "size"),
            wis=("wis", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_90=("coverage_90", "mean"),
        )
        .reset_index()
    )
    if BASELINE in set(scores["model"]):
        rel = relative_skill(
            scores, baseline=BASELINE, by=by, metrics=("wis",), match_on=MATCH
        )
        rel = rel.rename(columns={"n": "n_shared"})
        out = out.merge(rel, on=group, how="left")
    else:
        out = out.assign(n_shared=0, relative_wis=np.nan)
    ours = out["model"].isin(list(OURS))
    return out.assign(ours=ours).sort_values([*by, "relative_wis", "wis"])


def fmt(v: object, n: int = 2) -> str:
    """Return a number for a table cell."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{n}f}"


def html_table(tab: pd.DataFrame, by: Sequence[str] = ()) -> str:
    """Return a score table, our hub ids in bold."""
    if tab.empty:
        return "<p class=mut>Nothing scored yet.</p>"
    heads = [
        *by,
        "model",
        "n",
        "mean WIS",
        f"rel. WIS ({BASELINE})",
        "cells shared",
        "cover 50%",
        "cover 90%",
    ]
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for r in tab.itertuples():
        name = html.escape(str(r.model))
        name = f"<b>{name}</b>" if r.ours else name
        cells = [
            *[html.escape(str(getattr(r, b))) for b in by],
            name,
            fmt(r.n),
            fmt(r.wis, 1),
            fmt(r.relative_wis),
            str(int(r.n_shared)) if np.isfinite(r.n_shared) else "",
            fmt(r.coverage_50),
            fmt(r.coverage_90),
        ]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells))
    return "\n".join([*out, "</table>"])


def chain_table(table: pd.DataFrame, rounds: Sequence[str]) -> str:
    """Return which model went out under each hub id each round."""
    if table.empty:
        return "<p class=mut>The ledger has no submissions table.</p>"
    t = table.loc[table["reference_date"].astype(str).isin(list(rounds))]
    heads = [
        "round",
        "hub id",
        "model submitted",
        "chain start",
        "substituted",
        "ruled out",
        "gate",
        "source",
    ]
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for r in t.itertuples():
        cells = [
            r.reference_date,
            r.hub_id,
            r.model,
            r.chain_start,
            r.substituted,
            r.ruled_out,
            r.gate_status,
            r.source,
        ]
        out.append(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
        )
    return "\n".join([*out, "</table>"])


def summary_text(tab: pd.DataFrame, rounds: Sequence[str]) -> str:
    """Return the paragraph up top, from the overall score table."""
    parts = [
        f"We submitted {len(rounds)} round(s) in this window "
        f"({', '.join(rounds)})."
    ]
    label = "complete"
    if tab.empty:
        parts.append("No scores yet.")
    else:
        ranked = tab.dropna(subset=["relative_wis"])
        ranked = ranked.loc[ranked["n_shared"] > 0].sort_values("relative_wis")
        for hub_id in OURS:
            row = tab.loc[tab["model"] == hub_id]
            if row.empty:
                parts.append(f"{hub_id} has no {label} scores.")
                continue
            r = row.iloc[0]
            if hub_id in set(ranked["model"]):
                pos = list(ranked["model"]).index(hub_id) + 1
                rel = (
                    f"relative WIS {fmt(r.relative_wis)} against "
                    f"{BASELINE} on {int(r.n_shared)} shared cells "
                    f"({pos} of {len(ranked)} models with shared cells, "
                    "each on its own)"
                )
            else:
                rel = f"no cells shared with {BASELINE}"
            parts.append(
                f"On {label} truth, {hub_id} has mean WIS {fmt(r.wis, 1)} "
                f"over {int(r.n)} cells, {rel}, and covers "
                f"{fmt(r.coverage_50)} (50%) and {fmt(r.coverage_90)} (90%)."
            )
    return " ".join(parts)


def fan(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    ref: str,
    complete_after_weeks: int,
    mode: str,
) -> str:
    """Return one round's fans as an inline PNG."""
    import matplotlib.pyplot as plt

    q = quantiles.assign(
        source=np.where(quantiles["model"].isin(list(OURS)), "submitted", "hub")
    )
    teams = sorted(set(q.loc[q["source"] == "hub", "model"]))
    colours = comparison_colours(
        {}, {f"{k} (submitted)": v for k, v in OURS.items()}, teams
    )
    fig = plot_hub_round(
        q,
        target,
        ref,
        colours,
        locations=LOCATIONS,
        complete_after_weeks=complete_after_weeks,
        mode=mode,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return (
        f"<img class=fan-{mode} alt='Fans for the {ref} round' "
        f"src='data:image/png;base64,{data}'>"
    )


def by_type(scores: pd.DataFrame) -> pd.DataFrame:
    """Scores with `location_type`."""
    if scores.empty:
        return scores
    return scores.assign(location_type=scores["location"].map(location_type))


STYLE = """
:root{--bg:#fff;--ink:#1a1a1a;--mut:#666;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--ink:#e8e8e8;
--mut:#999;--line:#333}}
body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,
sans-serif;max-width:1200px;margin:0 auto;padding:16px}
.mut{color:var(--mut)}table{border-collapse:collapse;margin:8px 0 20px;
font-size:13px;display:block;overflow-x:auto}td,th{border-bottom:1px solid
var(--line);padding:3px 8px;text-align:right}td:first-child,th:first-child
{text-align:left}img{max-width:100%}.fan-dark{display:none}
@media (prefers-color-scheme:dark){.fan-light{display:none}
.fan-dark{display:block}}
"""


def build(
    out: Path,
    comparison: Path,
    data_dir: Path,
    submissions: Path,
    head: Path,
    n_rounds: int,
) -> Path:
    """Write the report to `out` and return it."""

    def read(d: Path, name: str) -> pd.DataFrame:
        f = d / name
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    comp = read(comparison, "scores.csv")
    quantiles = read(comparison, "quantiles.csv.gz")
    pending = read(comparison, "pending.csv")
    meta = json.loads((comparison / "summary.json").read_text())
    table = (
        pd.read_csv(submissions, dtype=str, keep_default_na=False)
        if submissions.exists()
        else pd.DataFrame()
    )
    target = load_target(data_dir)
    rounds = last_rounds(quantiles, n_rounds)

    def window(s: pd.DataFrame) -> pd.DataFrame:
        if s.empty:
            return s
        return by_type(s.loc[s["reference_date"].astype(str).isin(rounds)])

    comp = window(comp)
    overall = score_table(comp)
    teams = sorted(set(quantiles["model"]) - set(OURS))
    parts = [
        "<h1>Our hub rounds against the hub teams</h1>",
        "<p class=mut>"
        f"Truth: the hub's release of {html.escape(meta['truth_vintage'])} "
        f"(hub data commit {html.escape(head.read_text().strip()[:9])}). "
        f"Built {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC; rebuilt when "
        "the hub's data change.</p>",
        f"<p>{html.escape(summary_text(overall, rounds))}</p>",
        "<h2>What went out</h2>",
        chain_table(table, rounds),
        "<h2>How it is scored</h2>",
        "<ul>"
        "<li>Every file is scored as the hub has it, on the hub's cells "
        "(no model of ours is rerun).</li>"
        "<li>A week is scored once complete by the scorer's rule (its count "
        "has stopped growing between releases), on the latest release; "
        "later releases may still revise it.</li>"
        f"<li>Relative WIS is mean WIS over {BASELINE}'s on the cells both "
        "forecast (below 1 beats it), so each model's is on its own shared "
        "cells. Mean WIS is over each model's own cells, which differ "
        "(zones with no cases pull it down), so compare models by relative "
        "WIS, not mean WIS.</li>"
        "<li>Only past weeks (horizons below 0) are complete so far; the "
        "forecast weeks are scored as their data complete.</li>"
        "<li>Hub teams in these rounds: "
        + html.escape(", ".join(teams))
        + ". epiforecasts-* and mccabe-* have no files in the weekly "
        "rounds.</li></ul>",
    ]
    if not pending.empty:
        parts.append(
            "<p class=mut>Rounds with no complete week yet: "
            + html.escape(", ".join(pending["reference_date"].astype(str)))
            + ".</p>"
        )
    for label, s in (("Scores", comp),):
        parts.append(f"<h2>{label}: all rounds</h2>")
        parts.append(html_table(overall))
        parts.append(f"<h3>{label}: by location type</h3>")
        parts.append(
            html_table(score_table(s, ("location_type",)), ("location_type",))
        )
        parts.append(f"<h3>{label}: by horizon (weeks after the round)</h3>")
        parts.append(html_table(score_table(s, ("horizon",)), ("horizon",)))
        parts.append(f"<h3>{label}: by round</h3>")
        parts.append(
            html_table(score_table(s, ("reference_date",)), ("reference_date",))
        )
    parts.append("<h2>Forecasts against the data</h2>")
    parts.append(
        "<p class=mut>Each model's median (point), 50% (thick) and 90% "
        "(thin) intervals by week, ours on top, the hub teams below; the "
        "line is the latest release, shaded where still incomplete.</p>"
    )
    cut = int(meta["complete_after_weeks"])
    for ref in reversed(rounds):
        parts.append(f"<h3>Round {html.escape(ref)}</h3>")
        parts.append(fan(quantiles, target, ref, cut, "light"))
        parts.append(fan(quantiles, target, ref, cut, "dark"))
    doc = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Hub rounds</title><style>{STYLE}</style>" + "\n".join(parts)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(doc)
    tmp.rename(out)
    return out


def main(argv: Sequence[str]) -> None:
    """Build the report from the command line."""
    out, comp, data, subs, head, n = argv
    print(
        build(
            Path(out),
            Path(comp),
            Path(data),
            Path(subs),
            Path(head),
            int(n),
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
