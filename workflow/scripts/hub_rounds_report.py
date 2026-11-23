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
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def pairwise_relative_wis(
    scores: pd.DataFrame, baseline: str, min_shared: int = 1
) -> pd.DataFrame:
    """Pairwise relative WIS, as in the hubs (scoringutils).

    For each pair of models, the ratio of their mean WIS on the cells both
    forecast; a model's relative skill is the geometric mean of its ratios
    against every other model it shares cells with, divided by the
    baseline's. The spread is the 10% and 90% quantiles of its pairwise
    ratios, scaled the same way.
    """
    key = list(MATCH)
    wide = scores.pivot_table(index=key, columns="model", values="wis")
    models = list(wide.columns)
    ratios: dict[str, list[float]] = {m: [] for m in models}
    for a, b in itertools.permutations(models, 2):
        both = wide[[a, b]].dropna()
        if len(both) < min_shared:
            continue
        mb = both[b].mean()
        if mb > 0:
            ratios[a].append(both[a].mean() / mb)
    theta = {
        m: float(np.exp(np.mean(np.log(r)))) if r else np.nan
        for m, r in ratios.items()
    }
    base = theta.get(baseline, np.nan)
    if not np.isfinite(base):
        # The baseline has no cells here: scale to the geometric mean of
        # every model's skill instead (`scaled_to` says which).
        finite = [t for t in theta.values() if np.isfinite(t)]
        base = float(np.exp(np.mean(np.log(finite)))) if finite else np.nan
        scaled_to = "geometric mean of all models"
    else:
        scaled_to = baseline
    rows = []
    for m in models:
        r = np.array(ratios[m])
        cov = scores.loc[scores["model"] == m]
        rows.append(
            {
                "model": m,
                "relative_skill": theta[m] / base if base else np.nan,
                "low": (np.quantile(r, 0.1) / base) if len(r) else np.nan,
                "high": (np.quantile(r, 0.9) / base) if len(r) else np.nan,
                "pairs": len(r),
                "n": len(cov),
                "rounds": cov["reference_date"].nunique(),
                "locations": cov["location"].nunique(),
                "cell_share": len(cov) / len(wide),
                "scaled_to": scaled_to,
            }
        )
    return pd.DataFrame(rows).sort_values("relative_skill")


def ranking_figure(
    groups: Mapping[str, pd.DataFrame], ours: Sequence[str], mode: str = "light"
) -> Any:
    """One panel per group: relative skill with its spread, by model.

    `groups` maps a panel title to a `pairwise_relative_wis` table. Our
    hub ids are coloured; the marker size shows the share of cells.
    """
    import matplotlib.pyplot as plt

    ink = "#1a1a1a" if mode == "light" else "#e8e8e8"
    mut = "#888888"
    col = {"seabbs-renewal": "#d55e00", "seabbs-rwnowcast": "#0072b2"}
    n = max(len(groups), 1)
    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.6), squeeze=False, layout="constrained"
    )
    for ax, (title, tab) in zip(axes[0], groups.items(), strict=False):
        t = tab.dropna(subset=["relative_skill"]).sort_values(
            "relative_skill", ascending=False
        )
        y = np.arange(len(t))
        for yi, r in zip(y, t.itertuples(), strict=True):
            c = col.get(r.model, mut)
            ax.hlines(yi, r.low, r.high, color=c, lw=1.5, alpha=0.7)
            ax.plot(
                r.relative_skill, yi, "o", color=c,
                ms=4 + 10 * min(r.cell_share, 1.0),
            )  # fmt: skip
        ax.set_yticks(y, [str(m) for m in t["model"]], fontsize=8)
        for lab in ax.get_yticklabels():
            if lab.get_text() in ours:
                lab.set_color(col[lab.get_text()])
                lab.set_fontweight("bold")
        ax.axvline(1.0, color=mut, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9, color=ink)
        ax.set_xlabel("relative WIS (log scale)", color=ink, fontsize=8)
        ax.tick_params(colors=ink, labelsize=8)
        for s in ax.spines.values():
            s.set_color(mut)
        ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    return fig


def all_teams_fans(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    ref: str,
    locations: Sequence[str],
    ours: Sequence[str],
    mode: str = "light",
) -> Any:
    """One panel per location: every model's median and 50% band, one round."""
    import matplotlib.pyplot as plt

    ink = "#1a1a1a" if mode == "light" else "#e8e8e8"
    mut = "#999999"
    col = {"seabbs-renewal": "#d55e00", "seabbs-rwnowcast": "#0072b2"}
    q = quantiles.loc[quantiles["reference_date"].astype(str) == ref]
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    locs = [loc for loc in locations if loc in set(q["location"])]
    fig, axes = plt.subplots(
        1, max(len(locs), 1), figsize=(4.0 * max(len(locs), 1), 3.6),
        squeeze=False, layout="constrained",
    )  # fmt: skip
    r0 = pd.Timestamp(ref)
    for ax, loc in zip(axes[0], locs, strict=False):
        ql = q.loc[q["location"] == loc]
        for m, g in ql.groupby("model"):
            g = g.sort_values("target_end_date")
            x = pd.to_datetime(g["target_end_date"])
            c = col.get(m, mut)
            lw, a = (2.0, 0.25) if m in ours else (0.9, 0.08)
            ax.fill_between(x, g["q25"], g["q75"], color=c, alpha=a, lw=0)
            ax.plot(x, g["q50"], color=c, lw=lw, alpha=1 if m in ours else 0.7)
            ax.annotate(
                str(m),
                (x.iloc[-1], g["q50"].iloc[-1]),
                fontsize=6, color=c if m in ours else ink, alpha=0.8,
                xytext=(2, 0), textcoords="offset points", va="center",
            )  # fmt: skip
        d = real.loc[(real["vintage"] == latest) & (real["location"] == loc)]
        x = pd.to_datetime(d["target_end_date"])
        sel = x >= r0 - pd.Timedelta(weeks=8)
        ax.plot(x[sel], d["total"][sel], "o-", color=ink, ms=3, lw=1)
        ax.axvline(r0, color=mut, lw=0.8, ls=":")
        ax.set_title(loc, fontsize=9, color=ink)
        ax.tick_params(colors=ink, labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
        for s in ax.spines.values():
            s.set_color(mut)
        ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    return fig


def png(fig: Any, alt: str, mode: str) -> str:
    """Return a figure as an inline PNG for one colour scheme."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return (
        f"<img class=fan-{mode} alt='{html.escape(alt)}' "
        f"src='data:image/png;base64,{data}'>"
    )


def pairwise_table(tab: pd.DataFrame) -> str:
    """Return the pairwise relative WIS table, our hub ids in bold."""
    if tab.empty:
        return "<p class=mut>Nothing scored yet.</p>"
    heads = ["model", "relative WIS", "spread (10%-90%)", "pairs", "cells",
             "share of cells", "rounds", "locations"]  # fmt: skip
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for r in tab.itertuples():
        name = html.escape(str(r.model))
        name = f"<b>{name}</b>" if r.model in OURS else name
        cells = [name, fmt(r.relative_skill), f"{fmt(r.low)}-{fmt(r.high)}",
                 str(r.pairs), str(r.n), fmt(r.cell_share), str(r.rounds),
                 str(r.locations)]  # fmt: skip
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells))
    return "\n".join([*out, "</table>"])


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
    parts.append("<h2>Pairwise comparison</h2>")
    parts.append(
        "<p class=mut>As the hubs score teams (scoringutils): for each pair "
        "of models, the ratio of their mean WIS on the cells both "
        "forecast; a model's relative WIS is the geometric mean of its "
        f"ratios against every other model, divided by {BASELINE}'s "
        "(below 1 beats it). The spread is the 10% to 90% range of its "
        "pairwise ratios, scaled the same way. Teams with few cells "
        "(see cells) are noisy. Where the baseline has no cells, the "
        "level is scaled to the geometric mean of every model.</p>"
    )
    levels: dict[str, pd.DataFrame] = {}
    if not comp.empty:
        parts.append("<h3>All locations</h3>")
        parts.append(pairwise_table(pairwise_relative_wis(comp, BASELINE)))
        for lv in ("national", "province", "zone"):
            sub = comp.loc[comp["location_type"] == lv]
            if sub.empty:
                continue
            tab = pairwise_relative_wis(sub, BASELINE)
            levels[f"{lv} (scaled to {tab['scaled_to'].iloc[0]})"] = tab
            parts.append(f"<h3>{lv.capitalize()}</h3>")
            parts.append(pairwise_table(tab))
        weeks = {
            "nowcast weeks (horizon below 0)": comp["horizon"] < 0,
            "forecast weeks (horizon 0 on)": comp["horizon"] >= 0,
        }
        for name, sel in weeks.items():
            if sel.any():
                levels[name] = pairwise_relative_wis(comp.loc[sel], BASELINE)
            else:
                parts.append(f"<p class=mut>No {name} are complete yet.</p>")
        parts.append("<h3>Ranking</h3>")
        parts.append(
            "<p class=mut>Relative WIS (point; size: share of cells "
            "covered) and its spread, by level and by nowcast or forecast "
            "weeks, log scale; ours in colour.</p>"
        )
        for mode in ("light", "dark"):
            parts.append(
                png(ranking_figure(levels, list(OURS), mode), "Ranking", mode)
            )
    parts.append("<h2>Forecasts against the data</h2>")
    parts.append(
        "<p class=mut>Each model's median (point), 50% (thick) and 90% "
        "(thin) intervals by week, ours on top, the hub teams below; the "
        "line is the latest release, shaded where still incomplete.</p>"
    )
    if rounds:
        last = rounds[-1]
        parts.append(f"<h3>Every team together, round {html.escape(last)}</h3>")
        parts.append(
            "<p class=mut>Each model's median (line) and 50% interval "
            "(band), ours in colour; points are the latest release.</p>"
        )
        for mode in ("light", "dark"):
            fig = all_teams_fans(
                quantiles, target, last, LOCATIONS, list(OURS), mode
            )
            parts.append(png(fig, f"Every team, round {last}", mode))
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
