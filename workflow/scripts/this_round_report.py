"""Interactive HTML page of this week's round: every candidate's forecast.

No scores: the round has no truth yet. For the round the latest dry run
(or live run) of Submit on main targeted, it shows every candidate's
forecasts and nowcasts against the data as of the round, with the docs
site's forecast explorer (`docs/javascripts/explorer.js`: location by
level, model and view controls), the `wk R_t` quantiles of the renewal
candidates the same way, the gate's pick and why, each candidate's
checks, and the renewal fits' convergence.

Inputs: the round's hub tables of every candidate (`fit-store archive`),
the fit store (for the renewal fits' `rt.csv.gz` and `diagnostics.json`),
the hub data, and the run's `inputs`, `gate` and `baseline-gate`
artefacts. The page reads `<OUT stem>-data/` beside it.

    python workflow/scripts/this_round_report.py OUT.html CANDIDATES \
        STORE DATA_DIR RUN_DIR
"""

from __future__ import annotations

import base64
import datetime as dt
import gzip
import html
import io
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

# The page style and colour scheme of the hub rounds report, beside it
# (`python` puts the script's folder first on the path).
from hub_rounds_report import SCHEME, THEME

from bdbvhub import compare as hc
from bdbvhub.store.store import read_fit
from nowcastbdbv.data.tidy import load_target
from nowcasteval.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from nowcasteval.plots import comparison_colours, location_names

ROOT = Path(__file__).parents[2]

CANDIDATES = (
    "renewalbvd",
    "renewalsp",
    "renewal",
    "randomwalknowcast",
    "randomwalk",
    "fm_chronos_daily_ragged",
    "fm_chronos_daily_ragged_ft",
)
"""The models shown, in order: the renewal chain, the walks, Chronos-2
zero-shot and fine-tuned."""

RENEWAL = ("renewalbvd", "renewalsp", "renewal")
"""The candidates with a fit directory (`rt.csv.gz`, `diagnostics.json`)."""

RT_COLUMNS = {0.05: "q05", 0.25: "q25", 0.5: "q50", 0.75: "q75", 0.95: "q95"}
"""The `wk R_t` levels the explorer draws, as its quantile columns."""

SCRIPTS = ("toggles.js", "zoom.js", "summarise.js", "explorer.js")
"""The docs scripts the explorer needs, in the docs' load order."""


def newest_fit(store: Path, model: str, origin: str) -> Any | None:
    """Return the newest good stored fit of a model at an origin."""
    best = None
    for path in sorted((store / model / origin).glob("*.json.gz")):
        fit = read_fit(path)
        if fit.meta.get("status", "ok") != "ok":
            continue
        if best is None or fit.meta["fitted_at"] > best.meta["fitted_at"]:
            best = fit
    return best


def fit_file(fit: Any, name: str) -> bytes | None:
    """Return one of a stored fit's files, decoded."""
    data = fit.files.get(name)
    return None if data is None else base64.b64decode(data)


def rt_quantiles(store: Path, origin: str, ref: str) -> pd.DataFrame:
    """Return the renewal candidates' `wk R_t` as the explorer's quantiles."""
    rows = []
    for model in RENEWAL:
        fit = newest_fit(store, model, origin)
        raw = fit_file(fit, "rt.csv.gz") if fit else None
        if raw is None:
            continue
        rt = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
        rt = rt.loc[rt["output_type_id"].round(3).isin(list(RT_COLUMNS))]
        wide = rt.pivot_table(
            index=["location", "target_end_date"],
            columns="output_type_id",
            values="value",
        ).rename(columns=lambda q: RT_COLUMNS[round(q, 3)])
        rows.append(
            wide.reset_index().assign(
                model=model, source="real", reference_date=ref, ours=True
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def convergence(store: Path, origin: str) -> str:
    """Return the renewal fits' convergence as a small table."""
    heads = ["model", "fitted", "code", "worst R-hat", "lowest bulk ESS",
             "divergences", "sampling time"]  # fmt: skip
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for model in RENEWAL:
        fit = newest_fit(store, model, origin)
        raw = fit_file(fit, "diagnostics.json") if fit else None
        if raw is None:
            out.append(f"<tr><td>{model}</td><td colspan=6>no stored fit</td>")
            continue
        d = json.loads(raw)
        params = d.get("parameters", [])
        rhat = max((p["rhat"] for p in params), default=float("nan"))
        ess = min((p["ess_bulk"] for p in params), default=float("nan"))
        div = sum(c.get("divergences", 0) for c in d.get("chains", []))
        cells = [
            model,
            str(fit.meta["fitted_at"])[:16].replace("T", " "),
            str(fit.meta.get("commit", ""))[:9],
            f"{rhat:.3f}",
            f"{ess:.0f}",
            str(div),
            f"{d.get('sampling_seconds', 0) / 60:.0f} min",
        ]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells))
    return "\n".join([*out, "</table>"])


def gate_table(prov: dict[str, Any]) -> str:
    """Return the gate's attempts: each candidate's status and why."""
    heads = ["candidate", "status", "fit", "checks", "why"]
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for a in prov.get("attempts", []):
        why = "; ".join(a.get("reasons") or a.get("warnings") or [])
        cells = [a["id"], a["status"], a.get("fit", ""), a.get("checks", ""),
                 why[:300]]  # fmt: skip
        out.append(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
        )
    return "\n".join([*out, "</table>"])


def gate_section(run: Path) -> str:
    """Return what the gate would pick and why, from the run's artefacts."""
    parts = []
    for name, title in (("gate", "seabbs-renewal"),
                        ("baseline-gate", "seabbs-rwnowcast")):  # fmt: skip
        found = sorted((run / name).glob("**/provenance.json"))
        if not found:
            parts.append(
                f"<h3>{title}</h3><p class=mut>No gate result in the "
                "run yet.</p>"
            )
            continue
        prov = json.loads(found[0].read_text())
        pick = prov.get("selected") or "nothing (round skipped)"
        parts.append(
            f"<h3>{title}: {html.escape(str(pick))}</h3>"
            f"<p>{html.escape(str(prov.get('reason', '')))}</p>"
            + gate_table(prov)
        )
    return "\n".join(parts)


def team_quantiles(round_dir: Path, ref: str) -> pd.DataFrame:
    """Return every hub team's submission for the round, as quantiles.

    Read from the hub's files `this_round_inputs` fetched (`hub/`, openly
    licensed, and `hub-raw/`), as each team submitted them. Our own hub
    ids are left out: our candidates are shown instead.
    """
    roots = [round_dir / "hub", round_dir / "hub-raw"]
    if not any(r.exists() for r in roots):
        return pd.DataFrame()
    subs = hc.load_hub_submissions(*[r for r in roots if r.exists()])
    subs = subs.loc[
        (subs["reference_date"].astype(str) == ref)
        & ~subs["model"].astype(str).str.startswith("seabbs-")
    ]
    if subs.empty:
        return pd.DataFrame()
    return hc.forecast_quantiles(subs).assign(ours=False)


def build(
    out: Path, candidates: Path, store: Path, data_dir: Path, run: Path
) -> Path:
    """Write the page and its data folder; return the page."""
    ctx = json.loads((run / "inputs" / "inputs" / "context.json").read_text())
    origin, ref = str(ctx["as_of"]), str(ctx["reference_date"])
    meta = json.loads((run / "run.json").read_text())
    target = load_target(data_dir)
    real = target.loc[target["kind"] == "real"]
    asof = real.loc[real["vintage"].astype(str) == origin]
    names = location_names(real)
    table = pd.read_csv(candidates, dtype={"key": str})
    samples = table.loc[table["output_type"] == "sample"].assign(
        source="real",
        sample=lambda f: f["output_type_id"].astype(int),
    )
    q = hc.forecast_quantiles(samples).assign(ours=True)
    shown = [m for m in CANDIDATES if m in set(q["model"])]
    q = q.loc[q["model"].isin(shown)]
    q = q.iloc[q["model"].map(shown.index).argsort(kind="stable")]
    missing = [m for m in CANDIDATES if m not in shown]
    teams = team_quantiles(candidates.parent, ref)
    names_t = sorted(set(teams["model"])) if len(teams) else []
    colours = comparison_colours({m: m for m in shown}, {}, names_t)
    if len(teams):
        q = pd.concat([q, teams], ignore_index=True)

    data = out.with_name(out.stem + "-data")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    rel = data.name
    write_explorer(
        data / "cases.json",
        explorer_data(
            q,
            asof.assign(vintage=origin),
            names,
            0,
            colours,
            truth_vintage=origin,
            complete_through=origin,
            data_label=f"data as of {origin}",
        ),
    )
    rt = rt_quantiles(store, origin, ref)
    if len(rt):
        empty = asof.iloc[0:0].assign(vintage=origin)
        write_explorer(
            data / "rt.json",
            explorer_data(
                rt,
                empty,
                names,
                0,
                {m: colours[m] for m in rt["model"].unique()},
                truth_vintage=origin,
                complete_through=origin,
                unit="Rt",
                data_label="no data (Rt is not observed)",
            ),
        )
    js = "\n".join(
        (ROOT / "docs" / "javascripts" / f).read_text() for f in SCRIPTS
    )
    css = (ROOT / "docs" / "stylesheets" / "extra.css").read_text()
    note = (
        f"<p class=mut>Not shown, with no good stored fit at {origin}: "
        f"{html.escape(', '.join(missing))}.</p>"
        if missing
        else ""
    )
    parts = [
        f"<h1>This round: {html.escape(ref)}</h1>",
        f"<p class=mut>Data as of {html.escape(origin)}. Submit run "
        f"{html.escape(str(meta.get('run', '')))} on main "
        f"{html.escape(str(meta.get('sha', ''))[:9])} "
        f"({html.escape(str(meta.get('mode', '')))}). Built "
        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC, again when the "
        "run or the fits change. No scores: the round has no truth "
        "yet.</p>",
        "<h2>What the gate picks</h2>",
        gate_section(run),
        "<h2>Forecasts and nowcasts</h2>",
        "<p class=mut>Every candidate against the data as of the round, "
        "beside every other team's submission for this round so far, "
        "as each submitted it (read from the hub; toggle each in the "
        "key). Pick the location (grouped by national, province and "
        "zone), the models and the view.</p>",
        note,
        explorer_html(f"{rel}/cases.json", "Candidate forecasts"),
        "<h2>wk R_t</h2>",
        "<p class=mut>The renewal candidates' weekly effective "
        "reproduction number by week of infection, as quantiles.</p>",
        explorer_html(f"{rel}/rt.json", "Rt")
        if len(rt)
        else "<p>No stored fit has Rt yet.</p>",
        "<h2>Convergence</h2>",
        convergence(store, origin),
    ]
    doc = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>This round</title><style>{THEME}{css}</style>"
        f"<body data-md-color-scheme=default><script>{SCHEME}</script>"
        + "\n".join(parts)
        + f"<script>{js}</script></body></html>"
    )
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(doc)
    tmp.rename(out)
    return out


def main(argv: Sequence[str]) -> None:
    """Build the page from the command line."""
    out, cands, store, data, run = argv
    print(build(Path(out), Path(cands), Path(store), Path(data), Path(run)))


if __name__ == "__main__":
    main(sys.argv[1:])
