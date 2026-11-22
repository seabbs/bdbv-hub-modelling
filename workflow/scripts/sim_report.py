"""A self-contained HTML report on the science of simulated rounds.

From the batches themselves (`sims/`), the per-outbreak summaries
(`grid_summaries.py outbreaks`), the German development half and the
paper's reference values (`paper/data/reference-values.csv`):

1. a gallery of example outbreaks across the grid's regimes, each with
   infections, R and effective R by stratum, expected onsets and reported
   cases, the counts as known at three cut-offs against the final counts,
   the reporting triangle and the delay distribution over time;
2. by layer, the drawn settings and the outcomes, each against the German
   national value (development half) and the reference values where one
   exists;
3. flags: attack rates above 1, redrawn draws and why, and the sim-realism
   audit's odd combinations.

    uv run python workflow/scripts/sim_report.py germany \
        <COVID-19_hospitalizations.csv> <split_date> <out.csv>
    uv run python workflow/scripts/sim_report.py render <out.html> <title> \
        <reference-values.csv> <germany.csv> \
        <name>=<sims_dir>:<outbreaks.csv> [...]
"""

from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import grid_summaries as gs

DEV_FIRST = "2021-04-06"
"""The first German reference date used, as the sim-realism audit."""

INK = "#1f2933"
MUTED = "#8a94a0"
DE_COLOUR = "#c2410c"
SIM_COLOUR = "#2a78d6"
REF_COLOUR = "#1baf7a"

# --- Germany ---------------------------------------------------------------


def german_series(wide: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Return each German series as changes by reference and report date.

    Changes after 80 days are put at 81 days, as `hub_triangle` does.
    """
    cols = [c for c in wide.columns if c.startswith("value_")]
    long = wide.melt(
        id_vars=["date", "location", "age_group"],
        value_vars=cols,
        var_name="delay",
        value_name="count",
    ).dropna(subset=["count"])
    long["delay"] = long["delay"].map({c: i for i, c in enumerate(cols)})
    ref = pd.to_datetime(long["date"])
    long = long.assign(
        reference_date=ref,
        report_date=ref + pd.to_timedelta(long["delay"], unit="D"),
    )
    return dict(tuple(long.groupby(["location", "age_group"])))


def germany(triangle: Path, split: str) -> pd.DataFrame:
    """Return the German development half's series statistics.

    Reference dates from `DEV_FIRST` to the day before `split`. Delay
    quantiles are within 80 days; `after_80_share` is the rest. One row
    per statistic: the national series, and the medians over states and
    over age groups.
    """
    wide = pd.read_csv(triangle)
    wide = wide.loc[(wide["date"] >= DEV_FIRST) & (wide["date"] < split)]
    rows = []
    for (loc, age), c in german_series(wide).items():
        within = c.loc[c["delay"] <= 80]
        s = gs.series_stats(c)
        s |= {
            k: v
            for k, v in gs.series_stats(within).items()
            if k.startswith("delay_q")
        }
        total = c["count"].sum()
        s["after_80_share"] = (
            float(c.loc[c["delay"] > 80, "count"].sum() / total)
            if total > 0
            else np.nan
        )
        if loc == "DE" and age == "00+":
            kind = "national"
        elif age == "00+":
            kind = "states"
        elif loc == "DE":
            kind = "ages"
        else:
            continue
        rows += [
            {"kind": kind, "statistic": k, "value": v} for k, v in s.items()
        ]
    long = pd.DataFrame(rows)
    return long.groupby(["kind", "statistic"], as_index=False)["value"].median()


# --- Reading the rounds ----------------------------------------------------


def read_sources(args: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return the outbreaks, their failed draws and each round's folder.

    The outbreaks are the summaries joined to their settings.
    """
    outs, fails, dirs = [], [], {}
    for arg in args:
        name, paths = arg.split("=", 1)
        sims, summary = paths.split(":", 1)
        sims = Path(sims)
        dirs[name] = sims
        summ = pd.read_csv(summary)
        settings = pd.concat(
            [pd.read_csv(p) for p in sorted(sims.glob("settings-*.csv"))],
            ignore_index=True,
        )
        settings = settings.loc[:, ~settings.columns.duplicated()]
        keep = [c for c in settings.columns if c not in summ.columns]
        d = summ.merge(
            settings[["outbreak", *keep]].rename(columns={"strata": "layout"}),
            on="outbreak",
            how="left",
        )
        if "strata" in settings and "layout" not in d:
            d = d.merge(
                settings[["outbreak", "strata"]].rename(
                    columns={"strata": "layout"}
                ),
                on="outbreak",
            )
        outs.append(d.assign(round=name))
        for p in sorted(sims.glob("failures-*.csv")):
            f = pd.read_csv(p, usecols=lambda c: c in FAILURE_COLUMNS)
            fails.append(f.assign(round=name))
    out = derived(pd.concat(outs, ignore_index=True).copy())
    failures = (
        pd.concat(fails, ignore_index=True)
        if fails
        else pd.DataFrame(columns=[*FAILURE_COLUMNS, "round"])
    )
    return out, failures, dirs


FAILURE_COLUMNS = ("outbreak", "try", "reason", "window", "path")


def _sum(x: object) -> float:
    if isinstance(x, str):
        return float(sum(float(p) for p in x.split(";") if p not in ("", "NA")))
    return float(x)


def derived(d: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the report reads that the summaries do not have."""
    d = d.copy()
    d["variants"] = d.get(
        "variant_at", pd.Series(np.nan, index=d.index)
    ).notna()
    d["population_total"] = d.get(
        "population", pd.Series(np.nan, index=d.index)
    ).map(_sum)
    days = d.get("report_weekdays", pd.Series("all", index=d.index)).astype(str)
    d["report_weekdays_n"] = days.map(
        lambda s: 7 if s in ("all", "nan") else len(s.split(";"))
    )
    d["sparse_reporting"] = d["report_weekdays_n"] <= 2
    d["max_delay_days"] = pd.to_numeric(d.get("max_delay"), errors="coerce")
    d["leaves"] = d.get("n_places", 1) * d.get("n_ages", 1)
    d["nesting_depth"] = d.get("layout", pd.Series("", index=d.index)).map(
        {"single": 0, "places": 1, "ages": 1, "places_ages": 2}
    )
    d["waves_per_year"] = d["n_peaks"] / (d["weeks"] / 52)
    return d


# --- Figures ---------------------------------------------------------------


def png(fig) -> str:
    """Return a figure as an <img> with the PNG inline."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="" src="data:image/png;base64,{data}">'


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)
    ax.title.set_fontsize(9)


def outbreak_figure(counts: pd.DataFrame, latent: pd.DataFrame):
    """Return the six panels of one example outbreak."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    latent = latent.assign(date=pd.to_datetime(latent["date"]))
    c = counts.assign(
        reference_date=pd.to_datetime(counts["reference_date"]),
        report_date=pd.to_datetime(counts["report_date"]),
    )
    strata = sorted(latent["location"].unique())
    many = len(strata) > 8
    colours = plt.get_cmap("tab10")
    inf = latent.pivot_table(
        index="date", columns="location", values="infections", aggfunc="sum"
    )
    ax = axes[0, 0]
    for i, s in enumerate(strata):
        ax.plot(
            inf.index,
            inf[s],
            lw=0.8 if many else 1.3,
            color=MUTED if many else colours(i % 10),
            label=None if many else s,
        )
    if many:
        ax.plot(inf.index, inf.sum(axis=1), color=INK, lw=1.5, label="all")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Infections by stratum ({len(strata)})")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax = axes[0, 1]
    for col, ls in (("r", "-"), ("reff", "--")):
        p = latent.pivot_table(
            index="date", columns="location", values=col, aggfunc="mean"
        )
        for i, s in enumerate(strata[:8]):
            if s in p:
                ax.plot(
                    p.index,
                    p[s],
                    ls=ls,
                    lw=1,
                    color=MUTED if many else colours(i % 10),
                )
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_ylim(0, min(4, max(1.5, float(latent["r"].max(skipna=True)) + 0.2)))
    ax.set_title("R (solid) and effective R (dashed)")
    ax = axes[0, 2]
    daily = latent.groupby("date")["onsets"].sum()
    final = c.groupby("reference_date")["count"].sum()
    ax.plot(daily.index, daily, color=MUTED, lw=1.2, label="expected onsets")
    ax.plot(final.index, final, color=SIM_COLOUR, lw=1, label="reported, final")
    ax.set_title("Expected onsets and reported cases by reference date")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 0]
    ax.plot(final.index, final, color=INK, lw=1.4, label="final")
    lo, hi = c["reference_date"].min(), c["reference_date"].max()
    for i, share in enumerate((0.4, 0.65, 0.9)):
        cut = lo + (hi - lo) * share
        known = c.loc[c["report_date"] <= cut]
        known = known.groupby("reference_date")["count"].sum()
        ax.plot(
            known.index,
            known,
            lw=1,
            color=plt.get_cmap("Blues")(0.45 + 0.2 * i),
            label=f"known at {cut:%d %b}",
        )
        ax.axvline(cut, color=MUTED, lw=0.5, ls=":")
    ax.set_title("Counts as known at three cut-offs")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 1]
    delay = (c["report_date"] - c["reference_date"]).dt.days
    tri = (
        c.assign(delay=delay)
        .pivot_table(
            index="delay",
            columns="reference_date",
            values="count",
            aggfunc="sum",
        )
        .fillna(0)
    )
    ax.imshow(
        np.log1p(tri.clip(lower=0).to_numpy()),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(0, tri.shape[1], tri.index.min() - 0.5, tri.index.max() + 0.5),
    )
    ax.set_xlabel("reference day", fontsize=7)
    ax.set_ylabel("delay (days)", fontsize=7)
    ax.set_title("Reporting triangle, log(1 + count)")
    ax = axes[1, 2]
    block = (c["reference_date"] - lo).dt.days // 28
    cmap = plt.get_cmap("Blues")
    nb = int(block.max()) + 1
    top = 0
    for b, part in c.assign(delay=delay).groupby(block):
        by = part.groupby("delay")["count"].sum()
        if by.sum() <= 0:
            continue
        cum = by.sort_index().cumsum() / by.sum()
        top = max(top, gs.delay_quantile(by, 0.99))
        ax.step(
            cum.index,
            cum,
            where="post",
            lw=1,
            color=cmap(0.3 + 0.7 * b / max(nb - 1, 1)),
        )
    ax.set_xlim(0, max(top, 7))
    ax.set_title("Delay CDF by 4-week reference block (darker later)")
    for a in axes.flat:
        _style(a)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def bands_of(
    refs: pd.DataFrame, quantity: str
) -> list[tuple[str, float, float]]:
    """Return (disease, low, high) of a quantity, in the summaries' units.

    One band per disease, from the lowest to the highest of its rows' low,
    central and high values. A `*` marks a disease with a median or a
    serial interval among its rows.
    """
    r = refs.loc[refs["quantity"] == quantity]
    out = []
    for disease, rows in r.groupby("disease", sort=False):
        vals, starred = [], False
        for _, row in rows.iterrows():
            v = pd.to_numeric(
                pd.Series([row.get(k) for k in ("low", "central", "high")]),
                errors="coerce",
            ).dropna()
            if not len(v):
                continue
            if quantity == "growth_rate":
                if row["unit"] == "days":
                    v = np.log(2) / v
                elif row["unit"] == "per week":
                    v = v / 7
            measure = str(row.get("measure", ""))
            starred |= (
                "median" in measure.lower() or "serial" in measure.lower()
            )
            vals += list(v)
        if vals:
            label = f"{disease}{'*' if starred else ''}"
            out.append((label, float(min(vals)), float(max(vals))))
    return out


def hist_panel(ax, values, title, de=None, bands=(), log=False, clip=None):
    """Draw a statistic's histogram, with Germany and the reference bands."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if clip is not None:
        v = v.clip(*clip)
    if log:
        v = v[v > 0]
        bins = (
            np.logspace(np.log10(v.min()), np.log10(v.max()), 40)
            if len(v)
            else 10
        )
        ax.set_xscale("log")
    else:
        bins = 40
    full = v
    if len(v):
        hidden = ""
        if not log:
            # The middle 99%, so a few extremes do not squash the rest.
            lo, hi = np.quantile(v, [0.005, 0.995])
            pad = 0.05 * (hi - lo) if hi > lo else 1
            ax.set_xlim(lo - pad, hi + pad)
            out = (v < lo) | (v > hi)
            if out.any():
                hidden = f" ({int(out.sum())} outside)"
            v = v[~out]
        ax.hist(v, bins=bins, color=SIM_COLOUR, alpha=0.75)
        title += hidden
    ymax = ax.get_ylim()[1]
    xmax = ax.get_xlim()[1]
    note = ""
    if de is not None and np.isfinite(de):
        pct = 100 * (full < de).mean() if len(full) else np.nan
        ax.axvline(de, color=DE_COLOUR, lw=1.6)
        note = f"  DE {de:.3g} (p{pct:.0f})"
    for i, (label, lo, hi) in enumerate(bands):
        y = ymax * (1.02 + 0.07 * i)
        ax.plot(
            [lo, hi] if hi > lo else [lo],
            [y, y] if hi > lo else [y],
            color=REF_COLOUR,
            lw=2.5,
            marker="|" if hi > lo else "o",
            ms=4,
        )
        ax.text(
            min(hi, xmax), y, f" {label}", fontsize=6, va="center", color=INK
        )
    if bands:
        ax.set_ylim(0, ymax * (1.1 + 0.07 * len(bands)))
    ax.set_title(title + note)
    _style(ax)


def bar_panel(ax, values, title):
    """Draw the shares of a categorical setting or outcome."""
    s = values.astype(str).value_counts(normalize=True).sort_index()
    ax.barh(s.index, s.to_numpy(), color=SIM_COLOUR)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    _style(ax)


# --- Sections --------------------------------------------------------------

REGIMES = [
    ("early window", lambda d: d["window"] == "early"),
    ("endemic window", lambda d: d["window"] == "endemic"),
    ("established window", lambda d: d["window"] == "established"),
    ("multi-wave", lambda d: d["pattern"] == "multi-wave"),
    ("oscillating path", lambda d: d["path"] == "oscillating"),
    ("sharp rise and fall", lambda d: d["path"] == "rise_fall_sharp"),
    ("controlled", lambda d: d["path"] == "controlled"),
    ("decline", lambda d: d["path"] == "decline"),
    ("variants", lambda d: d["variants"]),
    ("attack rate above 1", lambda d: d["attack_max"] > 1),
    ("backlog shock", lambda d: d.get("shocks") == "backlog"),
    ("outage shock", lambda d: d.get("shocks") == "outage"),
    (
        "slow delays (top 10% mean delay)",
        lambda d: d["delay_mean"] >= d["delay_mean"].quantile(0.9),
    ),
    (
        "longest delay window",
        lambda d: d["max_delay_days"] == d["max_delay_days"].max(),
    ),
    ("delay that walks", lambda d: d["delay_change"] == "walk"),
    ("step change in delay", lambda d: d["delay_change"] == "step"),
    ("large nested system", lambda d: d["leaves"] == d["leaves"].max()),
    (
        "places, independent",
        lambda d: (
            (d["layout"] == "places") & (d.get("strata_link") == "independent")
        ),
    ),
    (
        "ages, linked",
        lambda d: (d["layout"] == "ages") & (d.get("strata_link") == "linked"),
    ),
    ("weekly reporting", lambda d: d["report_days"] == "weekly"),
    ("fixed report days", lambda d: d["report_days"] == "fixed_days"),
    ("revisions", lambda d: d["revisions"] == "yes"),
    (
        "strong reference weekday effect",
        lambda d: (
            d["weekday_amplitude"] >= d["weekday_amplitude"].quantile(0.95)
        ),
    ),
    ("fewer than 100 cases", lambda d: d["reported"] < 100),
]


def pick_gallery(d: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """One outbreak per regime: the lowest id that matches, none twice."""
    seen, out = set(), []
    for label, rule in REGIMES:
        try:
            m = d.loc[rule(d).fillna(False).astype(bool)]
        except (KeyError, TypeError):
            continue
        m = m.loc[~m["outbreak"].isin(seen)].sort_values("outbreak")
        if len(m):
            o = m.iloc[0]
            seen.add(o["outbreak"])
            out.append((label, o))
    return out


RANDOM_SEED = 20261122
"""The random gallery's seed, so reruns show the same outbreaks."""


def pick_random(d: pd.DataFrame, k: int = 24) -> list[tuple[str, pd.Series]]:
    """`k` outbreaks drawn at random (seeded), not chosen by regime."""
    d = d.sort_values(["round", "outbreak"]).reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = sorted(rng.choice(len(d), size=min(k, len(d)), replace=False))
    return [(f"random {i + 1}", d.iloc[r]) for i, r in enumerate(rows)]


def _n_items(x: object) -> int:
    """Return how many values a settings cell holds (`;`-separated)."""
    if isinstance(x, str):
        return sum(p not in ("", "NA") for p in x.split(";"))
    return 0 if pd.isna(x) else 1


def _any_true(x: object) -> bool:
    """Return whether any of a settings cell's values is true."""
    return any(
        p.strip().upper() in ("TRUE", "1", "1.0") for p in str(x).split(";")
    )


SHARES = [
    ("R path", lambda d: d["path"]),
    ("window", lambda d: d["window"]),
    ("place R relation", lambda d: d["place_rt"]),
    ("strata link", lambda d: d["strata_link"]),
    ("spread", lambda d: d["spread"]),
    (
        "R shifts",
        lambda d: (
            d["shift_at"]
            .map(_n_items)
            .clip(upper=3)
            .map({0: "0", 1: "1", 2: "2", 3: "3+"})
        ),
    ),
    (
        "a reactive shift",
        lambda d: (
            d["shift_reactive"].map(_any_true).map({True: "yes", False: "no"})
        ),
    ),
    ("variants", lambda d: d["variants"].map({True: "yes", False: "no"})),
    ("reporting shocks", lambda d: d["shocks"]),
    (
        "R at its 95% over days",
        lambda d: pd.cut(
            d["r_q95"],
            [0, 1.5, 3, 6, np.inf],
            labels=["below 1.5", "1.5 to 3", "3 to 6", "above 6"],
        ).astype(str),
    ),
]
"""The round-wide shares: what each outbreak's R path is made of."""


def shares(d: pd.DataFrame) -> str:
    """Return the shares of R path kinds, shifts, shocks and relations."""
    rows = []
    for name, of in SHARES:
        try:
            x = of(d)
        except KeyError:
            continue
        counts = x.fillna("none").astype(str).value_counts()
        for level, n in counts.items():
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(level)}</td>"
                f"<td>{n}</td><td>{n / len(d):.0%}</td></tr>"
            )
            name = ""
    return (
        "<table><tr><th>setting</th><th>level</th><th>outbreaks</th>"
        "<th>share</th></tr>" + "".join(rows) + "</table>"
    )


def gallery(picks: list[tuple[str, pd.Series]], dirs: dict) -> str:
    """Return the figures of the outbreaks in `picks`."""
    parts = []
    cache: dict = {}
    for label, o in picks:
        seed = o["outbreak"].split("-")[0]
        sims = dirs[o["round"]]
        if (sims, seed) not in cache:
            counts = pd.read_csv(sims / f"counts-{seed}.csv.gz")
            counts = counts.loc[counts["outbreak"].str.endswith("-1")]
            counts = counts.assign(outbreak=counts["outbreak"].str[:-2])
            cache = {
                (sims, seed): (
                    dict(tuple(counts.groupby("outbreak"))),
                    dict(
                        tuple(
                            pd.read_csv(sims / f"latent-{seed}.csv.gz").groupby(
                                "outbreak"
                            )
                        )
                    ),
                )
            }
        counts, latent = cache[(sims, seed)]
        if o["outbreak"] not in counts or o["outbreak"] not in latent:
            continue
        fig = outbreak_figure(counts[o["outbreak"]], latent[o["outbreak"]])
        facts = (
            f"{o['window']} window, {o['path']} path, {o['pattern']}; "
            f"{o.get('layout')} ({int(o['leaves'])} strata, "
            f"{o.get('strata_link')}); {o['weeks']:.0f} weeks; "
            f"GT mean {o['gt_mean']:.1f} d; attack {o['attack_max']:.2f}; "
            f"reporting {o['report_days']}, "
            f"delay mean {o['delay_mean']:.1f} d, "
            f"max delay {o['max_delay_days']:.0f} d, "
            f"change {o['delay_change']}; "
            f"shocks {o.get('shocks')}; revisions {o['revisions']}; "
            f"{o['reported']:.0f} cases"
        )
        parts.append(
            f"<h3>{html.escape(label)}: {o['outbreak']} ({o['round']})</h3>"
            f"<p class=facts>{html.escape(facts)}</p>{png(fig)}"
        )
    return "\n".join(parts)


LAYERS = [
    (
        "Layer 1: infections",
        [
            (
                "gt_mean",
                "Generation time mean (days)",
                "generation_time",
                False,
            ),
            ("r_q95", "R, 95% over days", "R", False),
            ("r_q05", "R, 5% over days", None, False),
            (
                "max_growth_14d",
                "Fastest 14-day growth of onsets (/day)",
                "growth_rate",
                False,
            ),
            (
                "weekly_growth_max",
                "Largest weekly log growth, reported",
                None,
                False,
            ),
            ("weekly_growth_sd", "Weekly log growth sd, reported", None, False),
            ("attack_max", "Attack rate, largest stratum", None, True),
            ("waves_per_year", "Major peaks per year", None, False),
            ("peak_share", "Peak timing (share of series)", None, False),
        ],
        [
            "window",
            "path",
            "pattern",
            "gt_family",
            "variants",
            "imports",
            "mixing",
        ],
    ),
    (
        "Layer 2: observation",
        [
            (
                "incubation_mean",
                "Infection to reference, mean (days)",
                "incubation",
                False,
            ),
            ("ascertainment", "Ascertainment", None, False),
            ("mean_daily", "Mean daily count, reported", None, True),
            ("zero_share", "Share of zero days", None, False),
        ],
        ["observation", "obs_strata", "observation_strata"],
    ),
    (
        "Layer 3: reporting delays",
        [
            (
                "delay_mean",
                "Drawn reporting delay mean (days)",
                "reporting_delay",
                False,
            ),
            ("delay_q50", "Delay median (days)", None, False),
            ("delay_q90", "Delay 90% (days)", None, False),
            ("delay_q99", "Delay 99% (days)", None, False),
            ("day0_share", "Share reported on day 0", None, False),
            ("mean_delay", "Mean delay, reported (days)", None, False),
        ],
        [
            "delay_change",
            "max_delay_days",
            "delay_family",
            "delay_dow",
            "delay_places",
        ],
    ),
    (
        "Layer 4: reporting system",
        [
            ("weekday_amplitude", "Reference weekday amplitude", None, True),
            (
                "negative_share",
                "Share of updates that are negative",
                None,
                False,
            ),
        ],
        [
            "report_days",
            "report_weekdays_n",
            "reference_dow",
            "report_dow",
            "revisions",
            "shocks",
        ],
    ),
    (
        "Layer 5: strata and aggregation",
        [
            ("leaves", "Strata (leaves)", None, False),
            ("nesting_depth", "Nesting depth", None, False),
        ],
        ["layout", "strata_link", "reporting_strata", "spread", "place_rt"],
    ),
]


def layers(d: pd.DataFrame, de: pd.DataFrame, refs: pd.DataFrame) -> str:
    """Return the settings and outcomes by layer."""
    national = de.loc[de["kind"] == "national"].set_index("statistic")["value"]
    parts = []
    for title, numeric, factors in LAYERS:
        n = len(numeric)
        cols = 3
        rows = -(-n // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.1 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (col, label, ref, log) in zip(axes, numeric, strict=False):
            if col not in d:
                ax.set_visible(False)
                continue
            bands = bands_of(refs, ref) if ref else ()
            hist_panel(
                ax, d[col], label, de=national.get(col), bands=bands, log=log
            )
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        present = [f for f in factors if f in d]
        fig2, axes2 = plt.subplots(
            -(-len(present) // 4), 4, figsize=(14, 2.4 * -(-len(present) // 4))
        )
        axes2 = np.atleast_1d(axes2).ravel()
        for ax, f in zip(axes2, present, strict=False):
            bar_panel(ax, d[f], f)
        for ax in axes2[len(present) :]:
            ax.set_visible(False)
        fig2.tight_layout()
        parts.append(f"<h3>{title}</h3>{png(fig)}{png(fig2)}")
    return "\n".join(parts)


def _fmt(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.3g}"


def germany_table(de: pd.DataFrame, d: pd.DataFrame) -> str:
    """Return the table of Germany's statistics against the simulations'."""
    wide = de.pivot(index="statistic", columns="kind", values="value")
    rows = []
    for stat, r in wide.iterrows():
        v = (
            pd.to_numeric(d[stat], errors="coerce").dropna()
            if stat in d
            else pd.Series(dtype=float)
        )
        q = np.quantile(v, [0.05, 0.5, 0.95]) if len(v) else [np.nan] * 3
        pct = 100 * (v < r.get("national")).mean() if len(v) else np.nan
        rows.append(
            f"<tr><td>{stat}</td>"
            + "".join(
                f"<td>{_fmt(r.get(k, np.nan))}</td>"
                for k in ("national", "states", "ages")
            )
            + f"<td>{' / '.join(_fmt(x) for x in q)}</td>"
            + f"<td>{'—' if np.isnan(pct) else f'p{pct:.0f}'}</td></tr>"
        )
    return (
        "<table><tr><th>Statistic</th><th>DE national</th>"
        "<th>DE states, median</th><th>DE ages, median</th>"
        "<th>Sims 5 / 50 / 95%</th><th>DE national's percentile</th></tr>"
        + "".join(rows)
        + "</table>"
    )


ODD = [
    (
        "Reporting on 1 or 2 days a week and a reference-day weekday effect",
        lambda d: d["sparse_reporting"] & (d["reference_dow"] == "yes"),
    ),
    (
        "Infections above 50% of the population (largest stratum)",
        lambda d: d["attack_max"] > 0.5,
    ),
    (
        "Shocks in an endemic window",
        lambda d: (d["shocks"] != "none") & (d["window"] == "endemic"),
    ),
    (
        "Revisions and reporting on 1 or 2 days a week",
        lambda d: (d["revisions"] == "yes") & d["sparse_reporting"],
    ),
    (
        "Early window and an oscillating or multi-wave path",
        lambda d: (
            (d["window"] == "early")
            & d["path"].isin(["oscillating", "multi_wave"])
        ),
    ),
    (
        "Fewer than 100 cases in the whole outbreak",
        lambda d: d["reported"] < 100,
    ),
    ("More than 1 million cases", lambda d: d["reported"] > 1e6),
    (
        "Maximum delay longer than half the series",
        lambda d: d["max_delay_days"] > d["weeks"] * 7 / 2,
    ),
    (
        "Population under 50,000 and ascertainment above 0.5",
        lambda d: (d["population_total"] < 5e4) & (d["ascertainment"] > 0.5),
    ),
    ("Mean delay above 21 days", lambda d: d["delay_mean"] > 21),
    (
        "Mean delay above 14 days and series under 12 weeks",
        lambda d: (d["delay_mean"] > 14) & (d["weeks"] < 12),
    ),
    (
        "Delay change on a series under 10 weeks",
        lambda d: (d["delay_change"] != "static") & (d["weeks"] < 10),
    ),
    (
        "Variant on a series under 16 weeks",
        lambda d: d["variants"] & (d["weeks"] < 16),
    ),
    ("R above 10 (95% over days)", lambda d: d["r_q95"] > 10),
    ("Effective R above 3 (95% over days)", lambda d: d["reff_q95"] > 3),
]


def _ids(d: pd.DataFrame, k: int = 5) -> str:
    return ", ".join(sorted(d["outbreak"])[:k])


def flags(d: pd.DataFrame, failures: pd.DataFrame) -> str:
    """Return the flags: attack rates above 1, redraws, odd combinations."""
    out = []
    high = d.loc[d["attack_max"] > 1]
    novar = high.loc[~high["variants"]]
    tab = pd.crosstab(
        high["window"],
        high["variants"].map({True: "variants", False: "no variants"}),
    )
    out.append(
        "<h3>Attack rate above 1</h3>"
        f"<p>{len(high)} of {len(d)} outbreaks "
        f"({len(high) / len(d):.1%}) infect "
        "more than their population in at least one stratum. "
        "Infections count every infection in the series, from the latent "
        "step's start, over the stratum's population. "
        "Immune-escape variants return immune people to the susceptible pool, "
        "so an outbreak with variants can pass 1 by reinfection. "
        f"Without variants it should not: {len(novar)} do "
        f"({_ids(novar, 10) or 'none'}).</p>" + tab.to_html(border=0)
    )
    draws = len(d) + len(failures)
    reasons = (
        failures["reason"]
        .astype(str)
        .str.replace(r":.*", ":", regex=True)
        .value_counts()
    )
    rows = "".join(
        f"<tr><td>{html.escape(r)}</td><td>{n}</td>"
        f"<td>{n / max(draws, 1):.1%}</td></tr>"
        for r, n in reasons.items()
    )
    tries = (
        d["tries"].value_counts().sort_index() if "tries" in d else pd.Series()
    )
    designs = (
        d["designs"].value_counts().sort_index()
        if "designs" in d
        else pd.Series()
    )
    out.append(
        "<h3>Redrawn draws</h3>"
        f"<p>{len(failures)} draws were redrawn, of {draws} drawn. "
        "Rounds simulated before the redraw fix also redrew draws with fewer "
        "than 10 cases; since it, only failed or empty draws are redrawn.</p>"
        "<table><tr><th>Reason</th><th>Draws</th><th>Share of draws</th></tr>"
        f"{rows}</table>"
        f"<p>Tries of the kept draw: {tries.to_dict()}. "
        f"Designs: {designs.to_dict()}.</p>"
    )
    rows = []
    for label, rule in ODD:
        try:
            m = d.loc[rule(d).fillna(False).astype(bool)]
        except KeyError:
            continue
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{len(m)}</td>"
            f"<td>{len(m) / len(d):.1%}</td><td>{_ids(m)}</td></tr>"
        )
    out.append(
        "<h3>Odd combinations</h3>"
        "<table><tr><th>Combination</th><th>n</th><th>Share</th>"
        "<th>First outbreaks</th></tr>" + "".join(rows) + "</table>"
    )
    return "\n".join(out)


CSS = """
:root{--ink:#1f2933;--muted:#5f6b78;--bg:#ffffff;--rule:#d9dee3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--ink:#e6e9ec;--muted:#a3adb8;--bg:#15191d;--rule:#2c333a}
img{background:#fff}}
body{font:15px/1.45 system-ui,sans-serif;color:var(--ink);background:var(--bg);
max-width:1200px;margin:2em auto;padding:0 16px}
img{max-width:100%;height:auto;display:block;margin:.4em 0 1.2em}
table{border-collapse:collapse;font-size:13px;margin:.6em 0 1.2em}
td,th{border-bottom:1px solid var(--rule);padding:3px 8px;text-align:left}
.facts{color:var(--muted);font-size:13px;margin:0}
h2{margin-top:2em;border-bottom:1px solid var(--rule)}
"""


def render(
    out: Path, title: str, refs: Path, de_csv: Path, sources: list[str]
) -> None:
    """Write the report of the rounds in `sources` to `out`."""
    d, failures, dirs = read_sources(sources)
    de = pd.read_csv(de_csv)
    ref = pd.read_csv(refs)
    rounds = ", ".join(dirs)
    body = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p>Rounds {html.escape(rounds)}: {len(d)} outbreaks, "
        f"{len(failures)} redrawn draws. "
        "From the simulated batches (first draw, strata summed). "
        "Germany is the hub's triangle for the development half "
        f"(reference dates {DEV_FIRST} to the split), never the held-out "
        "half or BDBV. Red lines are Germany's national value with its "
        "percentile among the simulations; green bars span the paper's "
        "reference values by disease (`paper/data/reference-values.csv`; "
        "* a disease with a median or serial interval among its rows; R "
        "rows mix R0 and Rt). "
        "Nesting is places crossed with ages (depth 2 at most) until the "
        "nested strata land; aggregates (exact, unallocated, late parents) "
        "are made by the trainer, not the simulator, so are not here.</p>",
        "<h2>1. Example outbreaks</h2>",
        "<h3>Shares over the round</h3>",
        shares(d),
        "<h3>At random</h3>",
        f"<p>{min(24, len(d))} outbreaks drawn at random (seed "
        f"{RANDOM_SEED}), as a typical round looks.</p>",
        gallery(pick_random(d), dirs),
        "<h3>By regime</h3>",
        "<p>One outbreak per regime, the lowest id that matches.</p>",
        gallery(pick_gallery(d), dirs),
        "<h2>2. Settings and outcomes by layer</h2>",
        germany_table(de, d),
        layers(d, de, ref),
        "<h2>3. Flags</h2>",
        flags(d, failures),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><html lang=en><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style><body>"
        + "\n".join(body)
        + "</body></html>"
    )
    print(f"{len(d)} outbreaks -> {out}")


def main(argv: list[str]) -> None:
    """Run `germany` or `render` from the command line."""
    if argv[0] == "germany":
        germany(Path(argv[1]), argv[2]).to_csv(argv[3], index=False)
        print(f"Germany -> {argv[3]}")
    elif argv[0] == "render":
        render(Path(argv[1]), argv[2], Path(argv[3]), Path(argv[4]), argv[5:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
