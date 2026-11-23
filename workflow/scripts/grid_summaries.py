"""Per-outbreak summaries of a simulated round, and their comparison.

Reads a round's `sims/` folder (`counts-<seed>.csv.gz`,
`latent-<seed>.csv.gz`, `settings-<seed>.csv`) and writes one row per
outbreak (first draw, strata summed):

- design: window, path, trajectory, delay change, GT family, strata;
- drawn values: GT mean and CV, incubation mean, ascertainment, delay
  mean (strata averaged where per stratum);
- size: infections, expected onsets, reported cases, attack rate, peak
  weekly reported cases;
- dynamics, on the 7-day centred mean of expected onsets: fastest growth
  over 14 days, half-peak to peak and peak to half-peak of each major
  peak (prominence at least 30% of the maximum, 28 days apart), number of
  major peaks and a pattern;
- R and effective R, infection weighted over strata: 5% and 95% over days;
- delays, from the reported counts: mean delay and share reported within
  7 days of the reference date;
- the series statistics the German development half is compared on
  (`series_stats`): delay median, 90% and 99%, the share reported on the
  reference date, the reference-weekday amplitude, the share of
  non-zero updates that are negative, mean daily count, share of zero
  days, and the sd and maximum of weekly log growth;
- the peak's place in the series (`peak_share`, 0 the first day, 1 the
  last) and the largest stratum attack rate;
- phase at the training cut-offs: the share of weekly cut-offs, from two
  weeks after the first report to `SIM_DELAY_WEEKS` + 5 weeks before the
  last (as `finetune.chronos.outbreak_blocks` places them), where expected
  onsets grow (14-day growth over 0.01 a day), fall (under -0.01), are
  flat, or are low (under 1 a day).

    uv run python workflow/scripts/grid_summaries.py outbreaks <sims> <out.csv>
    uv run python workflow/scripts/grid_summaries.py compare <out.csv> \
        <name>=<outbreaks.csv> [<name>=<outbreaks.csv> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from nowcastfoundation.finetune.chronos import SIM_DELAY_WEEKS

GROWING = 0.01
"""Daily 14-day log growth above which a cut-off is in a rising phase."""

LOW = 1.0
"""Expected onsets a day under which a cut-off is in a low phase."""

SETTINGS = [
    "window",
    "path",
    "trajectory",
    "delay_change",
    "gt_family",
    "incubation_family",
    "report_days",
    "report_dow",
    "reference_dow",
    "delay_dow",
    "revisions",
    "observation_strata",
    "reporting_strata",
    "n_places",
    "n_ages",
    "weeks",
]
VALUES = [
    "gt_mean",
    "gt_cv",
    "incubation_mean",
    "incubation_cv",
    "ascertainment",
    "delay_mean",
]


def _mean(x: object) -> float:
    """Return a value, or the mean of a `;`-separated per-stratum value."""
    if isinstance(x, str):
        parts = [float(p) for p in x.split(";") if p not in ("", "NA")]
        return float(np.mean(parts)) if parts else np.nan
    return float(x)


def dynamics(onsets: pd.Series) -> dict:
    """Return the dynamics summaries of a daily expected-onset series."""
    m = onsets.rolling(7, center=True, min_periods=7).mean().dropna()
    out: dict = {"n_peaks": 0, "pattern": "low"}
    if len(m) < 21 or m.max() < LOW:
        return out
    lg = np.log(m.clip(lower=0.5))
    g14 = (lg - lg.shift(14)) / 14
    out["max_growth_14d"] = float(g14.max())
    out["min_growth_14d"] = float(g14.min())
    v = m.to_numpy()
    peaks, _ = find_peaks(v, prominence=0.3 * v.max(), distance=28)
    rises, falls = [], []
    for p in peaks:
        left = np.where(v[:p] <= v[p] / 2)[0]
        right = np.where(v[p:] <= v[p] / 2)[0]
        if len(left):
            rises.append(int(p - left[-1]))
        if len(right):
            falls.append(int(right[0]))
    out["half_to_peak"] = float(np.median(rises)) if rises else np.nan
    out["peak_to_half"] = float(np.median(falls)) if falls else np.nan
    out["n_peaks"] = len(peaks)
    top = int(np.argmax(v))
    if len(peaks) >= 2:
        out["pattern"] = "multi-wave"
    elif len(peaks) == 1:
        out["pattern"] = "wave"
    elif top >= len(v) - 7:
        out["pattern"] = "rising"
    elif top <= 7:
        out["pattern"] = "falling"
    else:
        out["pattern"] = "flat"
    return out


def phases(onsets: pd.Series, first: pd.Timestamp, last: pd.Timestamp) -> dict:
    """Return the shares of weekly cut-offs in each phase."""
    m = onsets.rolling(7, center=True, min_periods=1).mean()
    lg = np.log(m.clip(lower=0.5))
    g14 = (lg - lg.shift(14)) / 14
    start = first + pd.Timedelta(weeks=2)
    end = last - pd.Timedelta(weeks=SIM_DELAY_WEEKS + 5)
    cuts = pd.date_range(start, end, freq="7D")
    cuts = cuts[cuts.isin(m.index)]
    if not len(cuts):
        return {}
    level, growth = m.loc[cuts], g14.loc[cuts].fillna(0)
    low = level < LOW
    up = ~low & (growth > GROWING)
    down = ~low & (growth < -GROWING)
    return {
        "cutoffs": len(cuts),
        "phase_low": float(low.mean()),
        "phase_rising": float(up.mean()),
        "phase_falling": float(down.mean()),
        "phase_flat": float((~low & ~up & ~down).mean()),
    }


def _max(x: object) -> float:
    """Return a value, or the largest of a `;`-separated per-stratum value."""
    if isinstance(x, str):
        parts = [float(p) for p in x.split(";") if p not in ("", "NA")]
        return float(np.max(parts)) if parts else np.nan
    return float(x)


def delay_quantile(by_delay: pd.Series, q: float) -> float:
    """Return the first delay by which a share `q` of the net count is in."""
    total = by_delay.sum()
    if total <= 0:
        return np.nan
    cum = by_delay.sort_index().cumsum() / total
    return float(cum.index[np.argmax(cum.to_numpy() >= q - 1e-9)])


def incubation_pmf(mean: float, cv: float, family: str) -> np.ndarray:
    """Return the infection-to-reference delay's daily pmf, to its 99.9%."""
    if family == "lognormal":
        sdlog = np.sqrt(np.log1p(cv**2))
        d = stats.lognorm(sdlog, scale=mean * np.exp(-(sdlog**2) / 2))
    else:
        d = stats.gamma(1 / cv**2, scale=mean * cv**2)
    edges = np.arange(int(np.ceil(d.ppf(0.999))) + 2)
    pmf = np.diff(d.cdf(edges))
    return pmf / pmf.sum()


def total_delay_stats(settings: pd.Series, by_delay: pd.Series) -> dict:
    """Return the infection-to-report delay's 50%, 90% and 99% quantiles.

    The strata's infection-to-reference pmfs, averaged, convolved with the
    reporting delay's net counts by delay (negative ones dropped). Rounds
    drawn before the CV and family were default to a gamma with CV 0.5.
    """
    means = str(settings.get("incubation_mean", "")).split(";")
    means = [float(m) for m in means if m not in ("", "NA", "nan")]
    cv = settings.get("incubation_cv", 0.5)
    cv = 0.5 if pd.isna(cv) else float(cv)
    family = settings.get("incubation_family", "gamma")
    family = "gamma" if pd.isna(family) else str(family)
    report = by_delay.clip(lower=0).sort_index()
    if not means or report.sum() <= 0:
        return {}
    pmfs = [incubation_pmf(m, cv, family) for m in means]
    inc = np.zeros(max(map(len, pmfs)))
    for p in pmfs:
        inc[: len(p)] += p / len(pmfs)
    rep = np.zeros(int(report.index.max()) + 1)
    rep[report.index.to_numpy(dtype=int)] = report.to_numpy(dtype=float)
    total = pd.Series(np.convolve(inc, rep / rep.sum()))
    return {f"total_q{q}": delay_quantile(total, q / 100) for q in (50, 90, 99)}


def series_stats(counts: pd.DataFrame) -> dict:
    """Return the statistics one series is compared with Germany on.

    `counts` has `reference_date`, `report_date` (dates) and `count`, the
    change reported on that report date. As the sim-realism audit: delays
    pooled over reference dates, the weekday amplitude the largest over
    the smallest weekday mean of the final count over its centred 7-day
    mean, and weekly growth over weeks with at least 10 cases.
    """
    delay = (counts["report_date"] - counts["reference_date"]).dt.days
    by_delay = counts["count"].groupby(delay).sum()
    total = by_delay.sum()
    out: dict = {}
    if total <= 0:
        return out
    for q in (50, 90, 99):
        out[f"delay_q{q}"] = delay_quantile(by_delay, q / 100)
    out["day0_share"] = float(by_delay.get(0, 0) / total)
    nonzero = counts["count"] != 0
    out["negative_share"] = float(
        (counts.loc[nonzero, "count"] < 0).mean() if nonzero.any() else 0
    )
    final = counts.groupby("reference_date")["count"].sum()
    final = final.asfreq("D", fill_value=0)
    out["mean_daily"] = float(final.mean())
    out["zero_share"] = float((final <= 0).mean())
    m = final.rolling(7, center=True, min_periods=7).mean()
    ratio = (final / m)[m > 0]
    if len(ratio) >= 14:
        by_day = ratio.groupby(ratio.index.day_of_week).mean()
        if by_day.min() > 0:
            out["weekday_amplitude"] = float(by_day.max() / by_day.min())
    weekly = final.resample("7D").sum()
    weekly = weekly.iloc[1:-1] if len(weekly) > 2 else weekly.iloc[:0]
    lg = np.log(weekly.where(weekly >= 10))
    growth = (lg - lg.shift(1)).dropna()
    if len(growth) >= 2:
        out["weekly_growth_sd"] = float(growth.std())
        out["weekly_growth_max"] = float(growth.max())
    return out


def weighted(latent: pd.DataFrame, col: str) -> pd.Series:
    """Return a latent column averaged over strata by infections, daily."""
    d = latent.dropna(subset=[col])
    w = d["infections"].where(d["infections"] > 0, 0) + 1e-12
    num = (d[col] * w).groupby(d["date"]).sum()
    return num / w.groupby(d["date"]).sum()


def outbreak_row(
    settings: pd.Series, latent: pd.DataFrame, counts: pd.DataFrame
) -> dict:
    """Return one outbreak's summaries."""
    row: dict = {"outbreak": settings["outbreak"]}
    row |= {k: settings.get(k) for k in SETTINGS}
    row |= {k: _mean(settings.get(k, np.nan)) for k in VALUES}
    row["strata"] = latent["location"].nunique()
    latent = latent.assign(date=pd.to_datetime(latent["date"]))
    daily = latent.groupby("date")[["infections", "onsets"]].sum()
    daily = daily.asfreq("D", fill_value=0)
    row["infections"] = float(daily["infections"].sum())
    row["onsets"] = float(daily["onsets"].sum())
    row["attack"] = _mean(settings.get("attack", np.nan))
    row["attack_max"] = _max(settings.get("attack", np.nan))
    row |= dynamics(daily["onsets"])
    if daily["onsets"].max() > 0 and len(daily) > 1:
        top = int(np.argmax(daily["onsets"].to_numpy()))
        row["peak_share"] = top / (len(daily) - 1)
    for col in ("r", "reff"):
        s = weighted(latent, col)
        if len(s):
            row[f"{col}_q05"], row[f"{col}_q95"] = np.quantile(s, [0.05, 0.95])
    c = counts.assign(
        reference_date=pd.to_datetime(counts["reference_date"]),
        report_date=pd.to_datetime(counts["report_date"]),
    )
    total = c["count"].sum()
    row["reported"] = float(total)
    if total > 0:
        delay = (c["report_date"] - c["reference_date"]).dt.days
        row["mean_delay"] = float((delay * c["count"]).sum() / total)
        row["reported_7d"] = float(c.loc[delay <= 7, "count"].sum() / total)
        weekly = c.groupby(pd.Grouper(key="reference_date", freq="7D"))["count"]
        row["peak_weekly"] = float(weekly.sum().max())
        row |= series_stats(c)
        by_delay = c["count"].groupby(delay).sum()
        row |= total_delay_stats(settings, by_delay)
        row |= phases(
            daily["onsets"], c["report_date"].min(), c["report_date"].max()
        )
    return row


def outbreaks(sims: Path) -> pd.DataFrame:
    """Return the summaries of every outbreak in a round's `sims/`."""
    rows = []
    for path in sorted(sims.glob("settings-*.csv")):
        seed = path.stem.split("-", 1)[1]
        settings = pd.read_csv(path)
        settings = settings.loc[:, ~settings.columns.duplicated()]
        latent = pd.read_csv(sims / f"latent-{seed}.csv.gz")
        counts = pd.read_csv(
            sims / f"counts-{seed}.csv.gz",
            usecols=["outbreak", "reference_date", "report_date", "count"],
        )
        counts = counts.loc[counts["outbreak"].str.endswith("-1")]
        counts = counts.assign(outbreak=counts["outbreak"].str[:-2])
        lat = dict(tuple(latent.groupby("outbreak")))
        cnt = dict(tuple(counts.groupby("outbreak")))
        for _, s in settings.iterrows():
            o = s["outbreak"]
            if o not in lat:
                continue
            rows.append(outbreak_row(s, lat[o], cnt.get(o, counts.iloc[:0])))
    return pd.DataFrame(rows)


QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
SHARES = [
    "window",
    "path",
    "pattern",
    "delay_change",
    "gt_family",
    "report_days",
]


def compare(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return quantiles of each numeric summary and shares of each factor.

    One row per (statistic, level) and one column block per source, so the
    sources' distributions read side by side.
    """
    rows = []
    numeric = [
        c
        for c in next(iter(sources.values())).columns
        if c not in ("outbreak", *SETTINGS, "pattern")
    ]
    for name, d in sources.items():
        for c in numeric:
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            if not len(v):
                continue
            q = np.quantile(v, QUANTILES)
            rows.append(
                {
                    "source": name,
                    "statistic": c,
                    "level": "",
                    "n": len(v),
                    "mean": v.mean(),
                    **{
                        f"q{int(p * 100):02d}": x
                        for p, x in zip(QUANTILES, q, strict=True)
                    },
                }
            )
        for c in SHARES:
            if c not in d:
                continue
            for level, share in d[c].value_counts(normalize=True).items():
                rows.append(
                    {
                        "source": name,
                        "statistic": f"share {c}",
                        "level": level,
                        "n": len(d),
                        "mean": share,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    """Run `outbreaks` or `compare` from the command line."""
    if argv[0] == "outbreaks":
        out = outbreaks(Path(argv[1]))
        out.to_csv(argv[2], index=False)
        print(f"{len(out)} outbreaks -> {argv[2]}")
    elif argv[0] == "compare":
        sources = {
            k: pd.read_csv(v) for k, v in (a.split("=", 1) for a in argv[2:])
        }
        compare(sources).to_csv(argv[1], index=False, float_format="%.4g")
        print(f"comparison -> {argv[1]}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
