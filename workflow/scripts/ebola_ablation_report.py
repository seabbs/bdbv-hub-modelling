"""The Ebola ablation page (the workflow's `ablation_report`).

Reads each variant's scores (`ebola_ablation.py summary`), its releases
and the training run's block releases (`releases.csv`, written by the
block builder), and writes one page with the docs theme
(`bdbvhub.reporting.pages`). Every number on it is computed here.

    python workflow/scripts/ebola_ablation_report.py DIR... --out FILE \
        --checkpoint slots-<hash> [--releases releases.csv]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import html
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bdbvhub.reporting import pages

ZS, FT = "foundation_zeroshot", "foundation"
ZS_PW, FT_PW = "foundation_zeroshot_pseudoweeks", "foundation_pseudoweeks"
MODELS = (
    (ZS, "Zero-shot"),
    (FT, "Fine-tuned"),
    (ZS_PW, "Zero-shot, pseudo weeks"),
    (FT_PW, "Fine-tuned, pseudo weeks"),
)
"""The models the page compares, each against the reference."""

SIM_LABELS = {
    "daily": (
        "Control",
        "The held-out outbreaks as simulated, released daily from the "
        "first report (most training blocks).",
    ),
    "weekly": (
        "Weekly releases",
        "Released every 7 days over the 40 weeks to the last round.",
    ),
    "short": (
        "Weekly, 7 weeks before the first round",
        "Weekly from 49 days before the first round, no gap.",
    ),
    "fresh": (
        "Weekly from the first round",
        "Weekly from the first round only, so its first release holds "
        "every earlier onset.",
    ),
    "gap": (
        "Weekly with a 50-day gap",
        "Weekly over 40 weeks, but none in the 50 days before the first round.",
    ),
    "few": (
        "BDBV releases",
        "One release 50 days before the first round, then weekly from it, "
        "as BDBV's releases.",
    ),
    "few_daily": (
        "BDBV gap, then daily",
        "One release 50 days before the first round, then daily from it.",
    ),
    "late_start": (
        "Late-starting zones",
        "Daily releases. A third of zones report nothing before a day "
        "within 3 weeks of the first round, then all their earlier cases "
        "at once.",
    ),
    "small": (
        "Small counts",
        "Daily releases. Each outbreak thinned to a peak of 60 cases a week.",
    ),
    "weekday": ("Weekday cycle", "Daily releases. Weekend onsets halved."),
    "long_delays": (
        "Longer delays",
        "Daily releases. 30% of reports arrive 14 to 35 days later than "
        "simulated.",
    ),
    "stops": (
        "Zones stop reporting",
        "Daily releases. A third of zones stop reporting from a day near "
        "the first round.",
    ),
    "bdbv_like": (
        "Full Ebola-like",
        "BDBV releases, late-starting zones, small counts and longer "
        "delays together.",
    ),
}
"""Each simulated variant's label and definition, in the page's order."""

BDBV_LABELS = {
    "as_is": ("BDBV hub data", "The hub data as released."),
    "rebuilt": ("Rebuilt", "The hub data rewritten by the ablation's writer."),
    "no_zones": ("No zones", "No health-zone data."),
    "no_late_provinces": (
        "No late provinces",
        "Provinces and zones first reported after the first release left out.",
    ),
    "flat_weekdays": (
        "Flat weekdays",
        "Each epiweek's counts spread evenly over its days.",
    ),
    "cap_d": ("Delays capped at D", "No report more than D days after onset."),
    "daily_releases": (
        "Daily releases",
        "A release each day between releases, linear in each count.",
    ),
    "weekly_source": ("Weekly source", "No daily data: the weekly target."),
    "all": ("All changes", "Every change above together."),
    "all_but_releases": (
        "All but releases",
        "Every change above but daily releases.",
    ),
    "no_first_release": (
        "No first release",
        "The first release, 50 days before the next, left out.",
    ),
    "releases_cap": (
        "Daily releases, capped delays",
        "Daily releases and delays capped at D, without zones.",
    ),
    "releases_cap_late": (
        "Daily releases, capped delays, no late provinces",
        "As above, with late provinces left out.",
    ),
}
"""Each BDBV variant's label and definition."""

LABELS = {"sim": SIM_LABELS, "bdbv": BDBV_LABELS}


@dataclass(frozen=True)
class Scores:
    """One variant's scores (`ebola_ablation.py summary`).

    Attributes
    ----------
    overall : dict
        Each model's relative CRPS and relative log CRPS.
    weeks : dict
        Each model's relative CRPS by weeks back.
    medians : dict
        Each model and level's summed medians over the truth by weeks
        back.
    """

    overall: dict[str, tuple[float, float]]
    weeks: dict[str, dict[int, float]]
    medians: dict[tuple[str, str], dict[int, float]]

    def crps(self, model: str, log: bool = False) -> float | None:
        """Return a model's relative (log) CRPS, None if not scored."""
        got = self.overall.get(model)
        return None if got is None else got[int(log)]

    def median(
        self, model: str, weeks: Sequence[int], level: str = "national"
    ) -> list[float]:
        """Return a model's medians over the truth at `weeks` back."""
        row = self.medians.get((model, level), {})
        return [row[k] for k in weeks if k in row]


def load(folder: Path) -> Scores:
    """Return a variant folder's scores."""
    s = json.loads((folder / "summary.json").read_text())
    overall = {
        r["model"]: (r["relative_crps"], r["relative_log_crps"])
        for r in s["overall"]
    }
    weeks: dict[str, dict[int, float]] = {}
    for r in s["by_weeks_back"]:
        weeks.setdefault(r["model"], {})[int(r["weeks_back"])] = r[
            "relative_crps"
        ]
    medians: dict[tuple[str, str], dict[int, float]] = {}
    for r in s["calibration"]:
        medians.setdefault((r["model"], r["level"]), {})[
            int(r["weeks_back"])
        ] = r["median_over_truth"]
    return Scores(overall, weeks, medians)


def release_history(folder: Path, reference: str) -> tuple[list[int], int]:
    """Return the releases at each scored origin and the longest gap.

    The scored origins are those the reference model forecast
    (`release-runs.csv.gz`); the releases are the daily releases' cut-offs
    on or before each. The gap is the most days between two releases up
    to the last origin.
    """
    daily = pd.read_csv(folder / "data" / "daily.csv", usecols=["cutoff"])
    cuts = sorted(pd.to_datetime(daily["cutoff"].unique()))
    with gzip.open(
        folder / "forecasts" / reference / "release-runs.csv.gz", "rt"
    ) as f:
        runs = pd.read_csv(f)
    origins = sorted(
        pd.to_datetime(runs.loc[runs["status"] == "forecast", "origin"])
    )
    counts = [sum(c <= o for c in cuts) for o in origins]
    upto = [c for c in cuts if c <= origins[-1]]
    gaps = [(b - a).days for a, b in itertools.pairwise(upto)]
    return counts, max(gaps, default=0)


def training_releases(
    releases: pd.DataFrame, at_most: Sequence[int]
) -> pd.DataFrame:
    """Return the training run's daily-step blocks by releases at the cut.

    Training blocks only (not the held-out part), with a daily source,
    as BDBV's model runs on daily data. Rows: all, each `releases or
    fewer` in `at_most`, and those whose first release backfills (a late
    start). Columns `blocks` and `share`.
    """
    t = releases.loc[
        (releases["part"] != "held-out")
        & releases["source"].astype(str).str.startswith("daily")
    ]
    rows = {"All": len(t)}
    for k in at_most:
        rows[f"{k} releases or fewer"] = int((t["releases"] <= k).sum())
    rows["Late start (first release backfills)"] = int(
        (t["release_start_days"] > 0).sum()
    )
    out = pd.DataFrame({"blocks": pd.Series(rows)})
    out["share"] = out["blocks"] / max(len(t), 1)
    return out


# ------------------------------------------------------------------ page


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _span(xs: Sequence[float]) -> str:
    """Return the range of some ratios, as '1.2 to 1.7' or one value."""
    if not xs:
        return "-"
    lo, hi = min(xs), max(xs)
    return (
        f"{lo:.1f}" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f} to {hi:.1f}"
    )


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        "<div class=table-wrap><table><thead><tr>"
        + th
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _label(side: str, name: str) -> str:
    return LABELS[side].get(name, (name, ""))[0]


def _results(found: dict, side: str, models: list[str]) -> str:
    """Return one side's table: relative CRPS, ratios and medians."""
    head = ["Variant"]
    head += [lab for m, lab in MODELS if m in models]
    head += [
        "FT / ZS, CRPS",
        "FT / ZS, log CRPS",
        "FT median / truth, weeks back -1, 0, 1",
        "ZS median / truth",
    ]
    if FT_PW in models:
        head.append("FT pseudo weeks median / truth")
    rows = []
    for (s, name), sc in found.items():
        if s != side:
            continue
        row = [_label(side, name)]
        for m, _ in MODELS:
            if m in models:
                row.append(
                    f"{_f(sc.crps(m))} <span class=mut>/ "
                    f"{_f(sc.crps(m, True))}</span>"
                )
        ft, zs = sc.crps(FT), sc.crps(ZS)
        ftl, zsl = sc.crps(FT, True), sc.crps(ZS, True)
        ratio = ft / zs if ft and zs else None
        cell = _f(ratio)
        row.append(f"<b>{cell}</b>" if ratio and ratio > 1.5 else cell)
        row.append(_f(ftl / zsl if ftl and zsl else None))
        for m in (FT, ZS, FT_PW) if FT_PW in models else (FT, ZS):
            row.append(", ".join(_f(x) for x in sc.median(m, (-1, 0, 1))))
        rows.append(row)
    return _table(head, rows)


def _weeks(found: dict, models: list[str]) -> str:
    span = range(-1, 6)
    rows = []
    for (side, name), sc in found.items():
        for m, lab in MODELS:
            if m in models and m in sc.weeks:
                rows.append(
                    [_label(side, name), lab]
                    + [_f(sc.weeks[m].get(k)) for k in span]
                )
    return _table(["Variant", "Model", *map(str, span)], rows)


def _conclusion(found: dict) -> list[str]:
    """Return the conclusion's paragraphs, each from the scores it needs."""

    def sc(name: str, side: str = "sim") -> Scores | None:
        return found.get((side, name))

    out = []
    daily, few, bdbv = sc("daily"), sc("few"), sc("as_is", "bdbv")
    if daily and few:
        text = (
            f"With BDBV's release pattern (one release, a 50-day gap, then "
            f"weekly) the fine-tuned model's relative CRPS goes from "
            f"{_f(daily.crps(FT))} on the control to {_f(few.crps(FT))}, "
            f"while zero-shot goes from {_f(daily.crps(ZS))} to "
            f"{_f(few.crps(ZS))}. Its national medians run at "
            f"{_span(few.median(FT, (-1, 0, 1)))} times the truth at weeks "
            "back -1 to 1."
        )
        if bdbv:
            text += (
                f" On the BDBV hub data it scores {_f(bdbv.crps(FT))} "
                f"against zero-shot's {_f(bdbv.crps(ZS))}, with provincial "
                f"medians at "
                f"{_span(bdbv.median(FT, (-1, 0, 1), 'province'))} times "
                "the latest release."
            )
        out.append(text)
    parts = [
        (name, SIM_LABELS[name][0].lower())
        for name in ("weekly", "short", "fresh", "gap", "few_daily")
        if sc(name)
    ]
    if parts:
        listed = "; ".join(
            f"{lab} {_f(found[('sim', n)].crps(FT))} "
            f"(zero-shot {_f(found[('sim', n)].crps(ZS))})"
            for n, lab in parts
        )
        out.append(
            f"Split by release history, the fine-tuned model scores: {listed}."
        )
    others = [
        n
        for n in ("late_start", "small", "weekday", "long_delays", "stops")
        if sc(n)
    ]
    if others:
        listed = "; ".join(
            f"{SIM_LABELS[n][0].lower()} {_f(found[('sim', n)].crps(FT))} "
            f"(zero-shot {_f(found[('sim', n)].crps(ZS))})"
            for n in others
        )
        out.append(f"With daily releases and one other BDBV feature: {listed}.")
    sims = [v for (s, _), v in found.items() if s == "sim"]
    ahead = [
        v
        for v in sims
        if v.crps(FT, True) is not None
        and v.crps(ZS, True) is not None
        and v.crps(FT, True) < v.crps(ZS, True)
    ]
    if sims:
        out.append(
            f"On the log scale, which weighs outbreaks evenly, the "
            f"fine-tuned model is ahead of zero-shot in {len(ahead)} of "
            f"{len(sims)} simulated variants."
        )
    return out


def _pseudo(found: dict) -> list[str]:
    """Return the pseudo weeks section, if the pseudo-week models ran."""
    rows = [(k, v) for k, v in found.items() if v.crps(FT_PW) is not None]
    if not rows:
        return []
    listed = "; ".join(
        f"{_label(*k).lower()} {_f(v.crps(FT))} to {_f(v.crps(FT_PW))}"
        for k, v in rows
    )
    return [
        "<h2>Pseudo weeks</h2>",
        "<p>With pseudo weeks (<code>FoundationModel.pseudo_weeks</code>, "
        "<code>nowcastutils.releases.pseudo_releases</code>) the model adds "
        "a release every 7 days wherever the release history has a gap of "
        "two weeks or more, and before its first release.\n"
        "Each pseudo release is the latest release's daily counts times the "
        "share of the delay line's cases of that place and onset week "
        "reported by its cut-off.\n"
        "The counts in the gap are estimated from a delay model, not "
        "reported.\n"
        "The simulated variants carry a delay line of each province's cases "
        "by onset week, in days, as the hub's.</p>",
        f"<p>The fine-tuned model's relative CRPS without and with pseudo "
        f"weeks: {listed}.</p>",
    ]


def _release_counts(
    folders: list[Path], reference: str, releases: Path | None
) -> list[str]:
    """Return the release-count section: the variants and training."""
    rows = []
    most = 0
    for f in folders:
        side = f.parent.name
        if not (f / "forecasts" / reference / "release-runs.csv.gz").exists():
            continue
        if side == "sim" and f.name not in ("daily", "few", "fresh", "gap"):
            continue
        counts, gap = release_history(f, reference)
        most = max(most, max(counts))
        rows.append(
            [
                _label(side, f.name),
                ", ".join(map(str, counts))
                if len(counts) <= 8 and max(counts) <= 60
                else f"{min(counts)} to {max(counts)}",
                f"{gap} days" if gap > 7 else "none",
            ]
        )
    parts = [
        "<h2>Release counts</h2>",
        "<p>Releases on or before each scored round, and the longest gap "
        "between releases.</p>",
        _table(["Data", "Releases at the round", "Longest gap"], rows),
    ]
    if releases is not None:
        bdbv = [
            max(release_history(f, reference)[0])
            for f in folders
            if f.parent.name == "bdbv"
            and (f / "forecasts" / reference / "release-runs.csv.gz").exists()
        ]
        at_most = sorted({2, *bdbv, *([most] if most <= 20 else [])})
        t = training_releases(pd.read_csv(releases), at_most)
        parts += [
            "<p>The training run's blocks "
            f"(<code>{html.escape(releases.name)}</code>, written by the "
            "block builder with the grid and release layer), daily-step "
            "training blocks only, as BDBV's model runs on daily data.</p>",
            _table(
                ["Training blocks (daily step)", "Blocks", "Share"],
                [
                    [lab, f"{int(r.blocks):,}", f"{r.share:.1%}"]
                    for lab, r in t.iterrows()
                ],
            ),
        ]
    return parts


def report(
    folders: list[Path],
    out: Path,
    checkpoint: str,
    reference: str = "randomwalknowcast",
    releases: Path | None = None,
    root: Path | str = ".",
) -> Path:
    """Write the ablation page from the variants' folders.

    Parameters
    ----------
    folders : list of pathlib.Path
        Each variant's folder (`summary.json`, `data/`, `forecasts/`),
        under a `sim` or `bdbv` folder for its side.
    out : pathlib.Path
        The page.
    checkpoint : str
        The fine-tuned checkpoint the variants ran.
    reference : str
        The model the scores are relative to.
    releases : pathlib.Path, optional
        The training run's `releases.csv`; without it the page has no
        training block counts.
    root : pathlib.Path or str
        The repository root, holding `docs/` (`pages.page`).

    Returns
    -------
    pathlib.Path
        The page.
    """
    order = {n: i for i, n in enumerate([*SIM_LABELS, *BDBV_LABELS])}
    folders = sorted(
        folders,
        key=lambda f: (f.parent.name != "sim", order.get(f.name, 99), f.name),
    )
    found = {(f.parent.name, f.name): load(f) for f in folders}
    models = [
        m
        for m, _ in MODELS
        if any(v.crps(m) is not None for v in found.values())
    ]
    defs = [[_label(s, n), LABELS[s].get(n, ("", ""))[1]] for s, n in found]
    day = dt.date.today().isoformat()
    parts = [
        "<h1>Ebola (BDBV): the fine-tuned Chronos-2 and the release "
        "history</h1>",
        f"<p class=mut>Built {day}. Checkpoint chronos-"
        f"{html.escape(checkpoint)} against zero-shot Chronos-2, both in "
        f"the ragged layout. {html.escape(reference)} is the reference.</p>",
        "<h2>Conclusion</h2>",
        *(f"<p>{p}</p>" for p in _conclusion(found)),
        *_pseudo(found),
        "<h2>Variants</h2>",
        "<p>Six held-out simulated outbreaks become six provinces, each "
        "outbreak's leaves its zones.\n"
        "Each is shifted so its peak week ends on 3 October.\n"
        "Eight weekly rounds run from 3 weeks before to 4 weeks after it.\n"
        "A release 150 days after the last round is the truth.\n"
        "The simulated reports are changed one feature at a time, then "
        "written as hub data; every variant uses the same outbreaks and "
        "random seed.\n"
        "The BDBV variants change the hub data one feature at a time "
        "towards the simulations.</p>",
        _table(["Variant", "Definition"], defs),
        "<h2>Results</h2>",
        "<p>Each variant runs through the BDBV check unchanged "
        "(<code>bdbvhub ft-check forecast</code>, scored by "
        "<code>ftcheck_report.skill_tables</code>).\n"
        "The hub's 7-day target is scored at the country and provinces, on "
        "weeks the truth release holds in full.\n"
        f"Model cells are relative CRPS / relative log CRPS against "
        f"{html.escape(reference)}, pairwise as on the check pages.\n"
        "The ratio columns are fine-tuned over zero-shot, bold where "
        "fine-tuned is more than 1.5 times worse.\n"
        "Weeks back -1 is the week after the round, 0 the round's week, 1 "
        "and on earlier weeks still being reported.</p>",
        "<h3>Simulated variants</h3>",
        _results(found, "sim", models),
    ]
    if any(s == "bdbv" for s, _ in found):
        parts += [
            "<h3>BDBV hub data</h3>",
            "<p>Scored against the latest hub release.</p>",
            _results(found, "bdbv", models),
        ]
    parts += [
        "<details><summary>Relative CRPS by weeks back</summary>",
        _weeks(found, models),
        "</details>",
        *_release_counts(folders, reference, releases),
        "<h2>Limits</h2>",
        "<p>One set of six outbreaks and eight rounds per variant.\n"
        "Relative CRPS pools across outbreaks, so the largest dominate it; "
        "the log CRPS weighs them evenly.\n"
        "The draws, the longest delay and the block limit are the "
        "workflow's <code>ablation</code> settings, so BDBV scores differ "
        "from the published check.\n"
        "The variants change the held-out simulations' reports after "
        "simulation; they are not new simulator runs.</p>",
        "<p class=mut>Code: <code>workflow/scripts/ebola_ablation.py</code> "
        "and <code>ebola_ablation_report.py</code>, run by the workflow's "
        "<code>ablation</code> target "
        "(<code>workflow/rules/ablation.smk</code>).</p>",
    ]
    return pages.write_page(
        out, pages.page("Ebola ablation", parts, ("tablesort.js",), root)
    )


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", default="randomwalknowcast")
    ap.add_argument("--releases", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=Path())
    a = ap.parse_args()
    report(a.folders, a.out, a.checkpoint, a.reference, a.releases, a.root)


if __name__ == "__main__":
    main()
