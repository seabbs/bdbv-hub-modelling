"""Interactive HTML report of our hub rounds against every hub team.

Reads the `compare-hub` output over the hub's own files (no model of ours
is run) and writes one page with the docs site's interactive charts
(`docs/javascripts`): the score summary (`scoresummary.js`), relative to
`seabbs-rwnowcast`, with its metric picker (WIS or WIS on the log scale,
coverage), and the forecast explorer (`explorer.js`) with its location,
round and model controls. A plain summary comes first and one small table
last: each model's pairwise relative WIS on the cells it shares with
every other model, scaled to `seabbs-rwnowcast`. Models `compare-hub`
reran on current code for each round (`--models`) are shown beside our
submissions as `<model> (current code)`.

The page reads its data from the folder `<OUT stem>-data/` beside it.

    python workflow/scripts/hub_rounds_report.py OUT.html COMPARISON_DIR \
        DATA_DIR SUBMISSIONS_CSV HEAD_FILE N_ROUNDS
"""

from __future__ import annotations

import datetime as dt
import html
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhub.reporting.page import standalone, write_page
from nowcastbdbv.data.tidy import load_target
from nowcasteval.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from nowcasteval.plots import (
    comparison_colours,
    comparison_label,
    location_names,
)
from nowcasteval.scorechart import write_score_chart
from nowcasteval.scoresummary import summary_html, summary_rows
from nowcasteval.scoring import pairwise_skill

ROOT = Path(__file__).parents[2]

OURS = {"seabbs-renewal": "renewalbvd", "seabbs-rwnowcast": "randomwalknowcast"}
"""Our hub ids and the model whose colour each takes."""

REFERENCE = "seabbs-rwnowcast"
"""The hub id every relative score divides by."""

MATCH = ("reference_date", "truth_vintage", "location", "target_end_date")
"""What makes two models' scores the same cell."""


def last_rounds(quantiles: pd.DataFrame, n: int) -> list[str]:
    """Return the last `n` rounds any of our hub ids forecast."""
    ours = quantiles.loc[quantiles["model"].isin(list(OURS))]
    return sorted(set(ours["reference_date"].astype(str)))[-n:]


def as_submitted(frame: pd.DataFrame) -> pd.DataFrame:
    """Our hub ids' rows as `submitted`, whichever copy they came from."""
    ours = frame["model"].isin(list(OURS))
    return frame.assign(source=np.where(ours, "submitted", frame["source"]))


def label(model: str, source: str) -> str:
    """Return a row's model name on the page.

    A hub team's id; `<hub id> (submitted)`; or `<model> (current code)`
    for a model rerun on current code for the round.
    """
    if source in ("hub", "submitted"):
        return comparison_label(model, source)
    return f"{model} (current code)"


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Each row's model named as the page shows it."""
    return frame.assign(
        model=[
            label(m, s)
            for m, s in zip(frame["model"], frame["source"], strict=True)
        ]
    )


def pairwise(scores: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Pairwise relative WIS, scaled to `reference` (the hubs' way).

    `nowcasteval.scoring.pairwise_skill` on the cells, with each model's
    cells, rounds and mean coverage.
    """
    skill = pairwise_skill(
        scores, reference, metrics=("wis",), match_on=MATCH
    ).rename(columns={"n": "cells"})
    cover = scores.groupby("model")[["coverage_50", "coverage_90"]].mean()
    tab = skill.merge(cover, left_on="model", right_index=True)
    cols = ["model", "relative_wis", "cells", "rounds"]
    return tab[[*cols, *cover.columns]].sort_values("relative_wis")


def fmt(v: float, n: int = 2) -> str:
    """Return a number for a table cell."""
    return "" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def small_table(tab: pd.DataFrame) -> str:
    """Return the one table: pairwise relative WIS and coverage by model."""
    heads = ["model", "relative WIS", "cells", "rounds", "50% cover",
             "90% cover"]  # fmt: skip
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for r in tab.itertuples():
        name = html.escape(str(r.model))
        if str(r.model).split(" ")[0] in OURS or "(current code)" in r.model:
            name = f"<b>{name}</b>"
        cells = [name, fmt(r.relative_wis), str(r.cells), str(r.rounds),
                 fmt(r.coverage_50), fmt(r.coverage_90)]  # fmt: skip
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells))
    return "\n".join([*out, "</table>"])


def summary_text(
    tab: pd.DataFrame, table: pd.DataFrame, rounds: Sequence[str]
) -> str:
    """Return the plain summary up top."""
    went = ""
    if not table.empty:
        t = table.loc[table["reference_date"].isin(rounds)]
        sub = t.loc[t["hub_id"] == "seabbs-renewal"]
        models = sorted(set(sub["model"]))
        swapped = int((sub["substituted"] == "true").sum())
        went = (
            " Under seabbs-renewal the chain submitted "
            f"{', '.join(models) or 'nothing'}"
            + (
                f", with {swapped} substituted round(s)"
                if swapped
                else ", never a fallback"
            )
            + "; seabbs-rwnowcast is the random walk on a nowcast."
        )
    parts = [f"We submitted {len(rounds)} rounds ({', '.join(rounds)}).{went}"]
    if not tab.empty:
        ranked = tab.dropna(subset=["relative_wis"]).reset_index(drop=True)
        mine = ranked.loc[ranked["model"].str.startswith("seabbs-renewal")]
        if len(mine):
            i = int(mine.index[0])
            r = ranked.iloc[i]
            better = ranked.loc[: i - 1, "model"].tolist() if i else []
            parts.append(
                "On the weeks scored so far, seabbs-renewal's WIS is "
                f"{fmt(r.relative_wis)} times seabbs-rwnowcast's, "
                f"{i + 1} of {len(ranked)} models"
                + (f" (behind {', '.join(better)})" if better else "")
                + f"; {fmt(r.coverage_50)} of truths fall in its 50% "
                f"intervals and {fmt(r.coverage_90)} in its 90%, most of "
                "them zone-weeks with no cases."
            )
        now = ranked.loc[ranked["model"] == "randomwalknowcast (current code)"]
        if len(now):
            parts.append(
                "randomwalknowcast on current code, rerun on each round's "
                f"data, scores {fmt(now.iloc[0].relative_wis)} against the "
                "seabbs-rwnowcast we submitted."
            )
        parts.append(
            "Teams that forecast few cells (the table at the end) have "
            "noisy scores. Only past weeks are complete so far, so these "
            "are nowcast scores."
        )
    return " ".join(parts)


def build(
    out: Path,
    comparison: Path,
    data_dir: Path,
    submissions: Path,
    head: Path,
    n_rounds: int,
) -> Path:
    """Write the page and its data folder; return the page."""
    meta = json.loads((comparison / "summary.json").read_text())
    quantiles = as_submitted(pd.read_csv(comparison / "quantiles.csv.gz"))
    scores = as_submitted(pd.read_csv(comparison / "scores.csv"))
    energy = as_submitted(pd.read_csv(comparison / "energy.csv"))
    table = (
        pd.read_csv(submissions, dtype=str, keep_default_na=False)
        if submissions.exists()
        else pd.DataFrame()
    )
    target = load_target(data_dir)
    names = location_names(target)
    rounds = last_rounds(quantiles, n_rounds)

    def window(f: pd.DataFrame) -> pd.DataFrame:
        return f.loc[f["reference_date"].astype(str).isin(rounds)]

    quantiles = window(quantiles)
    scores, energy = window(scores), window(energy)
    source = quantiles["source"]
    teams = sorted(set(quantiles.loc[source == "hub", "model"]))
    rerun = sorted(
        set(quantiles.loc[~source.isin(["hub", "submitted"]), "model"])
    )
    ours = [f"{h} (submitted)" for h in OURS]
    current = {label(m, "real"): m for m in rerun}
    colours = comparison_colours(
        current, {f"{h} (submitted)": m for h, m in OURS.items()}, teams
    )
    reference = f"{REFERENCE} (submitted)"
    models = [*ours, *current, *teams]

    data = out.with_name(out.stem + "-data")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    rel = data.name
    srcs = {}
    for key, kind in (("complete", "complete"), ("latest", "reported")):
        s = labelled(scores.loc[scores["target"] == kind])
        if s.empty or reference not in set(s["model"]):
            continue
        s = s.assign(
            kind="real", vintage=s["origin"], origin=s["reference_date"]
        )
        e = labelled(energy.loc[energy["target"] == kind])
        e = e.assign(origin=e["reference_date"])
        write_score_chart(
            data / f"scores-{key}.json",
            summary_rows(
                s,
                models,
                colours,
                names,
                baseline=reference,
                default_on=models,
                energy=e,
            ),
        )
        srcs[key] = f"{rel}/scores-{key}.json"
    write_explorer(
        data / "explorer.json",
        explorer_data(
            labelled(quantiles).assign(ours=quantiles["source"] != "hub"),
            target,
            names,
            int(meta["complete_after_weeks"]),
            colours,
            truth_vintage=meta["truth_vintage"],
            samples=None,
        ),
    )
    complete = labelled(scores.loc[scores["target"] == "complete"])
    tab = pairwise(complete, reference) if len(complete) else pd.DataFrame()

    commit = html.escape(head.read_text().strip()[:9])
    parts = [
        "<h1>Our hub rounds against the hub teams</h1>",
        "<p class=mut>Truth: the hub's release of "
        f"{html.escape(meta['truth_vintage'])} (hub data commit {commit}). "
        f"Built {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC, again "
        "whenever the hub's data change. The hub's files as submitted; no "
        "model of ours is rerun.</p>",
        f"<p>{html.escape(summary_text(tab, table, rounds))}</p>",
        "<h2>Scores</h2>",
        "<p class=mut>Relative scores divide by seabbs-rwnowcast on the "
        "cells both forecast, so seabbs-rwnowcast is 1 and below 1 is "
        "better. Pick the metric (WIS, shown as CRPS, on the natural or "
        "the log scale; coverage), the release (complete weeks or the "
        "latest release), the scale and the models.</p>",
        summary_html(srcs, "Scores by model", view="overall")
        if srcs
        else "<p>No round has a score yet.</p>",
        "<h3>By horizon</h3>",
        summary_html(srcs, "Scores by horizon", view="horizon") if srcs else "",
        "<h2>Forecasts</h2>",
        "<p class=mut>Pick a location, a round and the models; bands are "
        "the intervals, against the latest release.</p>",
        explorer_html(f"{rel}/explorer.json", "Forecast explorer"),
        "<h2>Pairwise relative WIS</h2>",
        "<p class=mut>As the hubs rank teams (scoringutils' pairwise "
        "comparisons): each pair's mean WIS ratio on the cells both "
        "forecast, a model's geometric mean over every model, itself "
        "included, scaled so seabbs-rwnowcast is 1. Complete weeks.</p>",
        small_table(tab) if len(tab) else "<p>Nothing scored yet.</p>",
    ]
    return write_page(out, standalone("Hub rounds", parts, ROOT / "docs"))


def main(argv: Sequence[str]) -> None:
    """Build the report from the command line."""
    out, comp, data, subs, head, n = argv
    print(
        build(Path(out), Path(comp), Path(data), Path(subs), Path(head), int(n))
    )


if __name__ == "__main__":
    main(sys.argv[1:])
