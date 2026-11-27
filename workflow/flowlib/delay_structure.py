"""The structure of reporting delays in counts by reference and report date.

One set of summaries for simulated outbreaks and for real data alike
(the German triangle, the BDBV releases), so the simulations' delays can
be read against the data's: how the delay distribution moves over time
(by reference week and by release week), its weekday effects, how it
differs between strata (a parent and its children), the release
cadence with the wait it adds, and each stratum's delays week by week:
how strata's delays move together, how they follow the counts, bursts
of over-reporting withdrawn later and the delay's noise split into a
part common to strata, a part fixed by stratum and the rest.

Input rows have `stratum`, `ref` and `rep` (dates) and `count`, the
count of reference date `ref` first reported (or revised) on `rep`.
Counts are netted by delay before the delay summaries, so a revision
moves cases between delays and a net negative is dropped; the
first-report ratios, the shares by a delay and the withdrawn share use
the counts as reported, revisions included. Delay summaries use only
reference dates at least `horizon` days (the 95% delay) before the last
report, and release summaries only reports at least `horizon` days after
the first reference date, so the series' ends do not shorten or lengthen
the delays.

`structure` returns the summaries as plain JSON, `scalars` its single
numbers and `curves` its fixed-length curves, for comparing many series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

HORIZON_Q = 0.95
"""The delay quantile beyond which reference dates count as incomplete."""

MIN_COUNT = 10
"""The fewest cases a week, weekday or stratum needs for its summary."""

TIME_BINS = 10
"""Equal parts of the complete reference span for `delay_time`."""

GAP_BINS = 8
"""Release gaps of 1 to 7 days, then 8 or more."""

DELAY_POINTS = (0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 42)
"""The delays (days) the delay distribution's cumulative share is kept at."""

MAX_STRATA = 40
"""The most strata `structure` lists, the largest first."""

MAX_LINES = 16
"""The most strata `by_stratum` keeps week by week, the largest first."""

SHARE_DAYS = (1, 7)
"""The delays (days) each stratum's weekly share reported by is kept at."""

STRATUM_KEYS = ("by_stratum", "coupling", "bursts", "noise", "levels")
"""The keys of `structure` from `_by_stratum`."""

MIN_WEEKS = 4
"""The fewest weeks a correlation or slope between series needs."""


def _r(v: float | None, digits: int = 3) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), digits)


def _by_delay(rows: pd.DataFrame) -> pd.Series:
    """Return net counts by delay, net negatives dropped."""
    return rows.groupby("delay")["count"].sum().clip(lower=0).sort_index()


def _quantile(by_delay: pd.Series, q: float) -> int | None:
    total = by_delay.sum()
    if total <= 0:
        return None
    cum = by_delay.cumsum() / total
    return int(by_delay.index[np.searchsorted(cum.to_numpy(), q - 1e-9)])


def _summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Return the count, mean and 10, 50 and 90% delays of `rows`."""
    by = _by_delay(rows)
    n = float(by.sum())
    if n < MIN_COUNT:
        return {"n": n, "mean": None, "q10": None, "q50": None, "q90": None}
    return {
        "n": n,
        "mean": _r(float((by.index * by).sum()) / n),
        **{f"q{int(q * 100)}": _quantile(by, q) for q in (0.1, 0.5, 0.9)},
    }


def _cdf(rows: pd.DataFrame) -> list[float | None]:
    """Return the share reported by each of `DELAY_POINTS`."""
    by = _by_delay(rows)
    total = by.sum()
    if total <= 0:
        return [None] * len(DELAY_POINTS)
    return [_r(by[by.index <= d].sum() / total, 4) for d in DELAY_POINTS]


def _weekly(rows: pd.DataFrame, key: str, first: pd.Timestamp) -> dict:
    """Return each week's delay summary, weeks keyed on `key`'s date."""
    week = (rows[key] - first).dt.days // 7
    out: dict[str, list] = {
        k: [] for k in ("week", "n", "mean", "q10", "q50", "q90")
    }
    if rows.empty:
        return out
    for w in range(int(week.min()), int(week.max()) + 1):
        s = _summary(rows[week == w])
        out["week"].append((first + pd.Timedelta(weeks=w)).strftime("%Y-%m-%d"))
        for k in ("n", "mean", "q10", "q50", "q90"):
            out[k].append(s[k])
    return out


def _weekday_mean(rows: pd.DataFrame) -> list[float | None]:
    out = []
    for k in range(7):
        out.append(_summary(rows[rows["ref"].dt.day_of_week == k])["mean"])
    return out


def _profile(rows: pd.DataFrame, key: str) -> list[float]:
    """Return counts by weekday of `key` over their mean, Monday first."""
    by = rows.groupby(rows[key].dt.day_of_week)["count"].sum()
    by = by.reindex(range(7), fill_value=0).to_numpy(dtype=float)
    mean = by.mean()
    return [_r(v / mean) if mean > 0 else 0.0 for v in by]


def sim_parent(stratum: str) -> str | None:
    """Return a simulated stratum's parent: a place's, for a place's age."""
    head, sep, _ = str(stratum).partition("-")
    return head if sep else None


def _strata(
    rows: pd.DataFrame, parent: Callable[[str], str | None] | None
) -> list[dict[str, Any]]:
    total = {"name": "all", "level": "total", **_summary(rows)}
    names = (
        rows.groupby("stratum")["count"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    if len(names) < 2:
        return [total]
    names = list(names.index[:MAX_STRATA])
    of = {s: (parent(s) if parent else None) for s in names}
    out = [total]
    parents = list(dict.fromkeys(p for p in of.values() if p))
    for p in parents:
        kids = [s for s in names if of[s] == p]
        out.append(
            {
                "name": p,
                "level": "parent",
                **_summary(rows[rows["stratum"].map(of) == p]),
            }
        )
        out += [
            {
                "name": s,
                "level": "child",
                **_summary(rows[rows["stratum"] == s]),
            }
            for s in kids
        ]
    out += [
        {"name": s, "level": "child", **_summary(rows[rows["stratum"] == s])}
        for s in names
        if of[s] is None
    ]
    return out


def _stratum_weeks(rows: pd.DataFrame, first: pd.Timestamp) -> pd.DataFrame:
    """Return counts by stratum and week of reference date.

    `final`: the count as last reported. `net` and `dsum`: the count
    netted by delay and its delays' sum, for the mean delay. `by<d>`:
    the count reported by each of `SHARE_DAYS`, revisions included.
    `first`: the count at each reference date's first report.
    `withdrawn`: the size of the negative updates.
    """
    rows = rows.assign(week=(rows["ref"] - first).dt.days // 7)
    keys = ["stratum", "week"]
    net = (
        rows.groupby([*keys, "delay"])["count"]
        .sum()
        .clip(lower=0)
        .reset_index()
    )
    net = (
        net.assign(dsum=net["delay"] * net["count"])
        .groupby(keys)[["count", "dsum"]]
        .sum()
        .rename(columns={"count": "net"})
    )
    seen = rows[rows["count"] != 0]
    at = seen.groupby(["stratum", "ref"])["rep"].transform("min")
    by = rows.groupby(keys)["count"]
    cols = {
        "final": by.sum(),
        **{
            f"by{d}": rows[rows["delay"] <= d].groupby(keys)["count"].sum()
            for d in SHARE_DAYS
        },
        "first": seen[seen["rep"] == at].groupby(keys)["count"].sum(),
        "withdrawn": (-rows["count"].clip(upper=0))
        .groupby([rows["stratum"], rows["week"]])
        .sum()
        .rename_axis(keys),
    }
    return pd.DataFrame(cols).join(net).fillna(0.0).astype(float)


def _week_series(
    g: dict[str, np.ndarray], days: np.ndarray
) -> dict[str, np.ndarray]:
    """Return weekly series from `_stratum_weeks`' columns as arrays.

    `mean`: the mean delay (netted); `share<d>`: the share of the final
    count reported by each of `SHARE_DAYS`; `first_ratio`: the count at
    the first report over the final; `log_count`: the log of the final
    count a day (over the week's `days` reference dates, so a part week
    at the end is not low). Weeks with fewer than `MIN_COUNT` final
    cases are NaN.
    """
    final = g["final"]
    ok = final >= MIN_COUNT
    safe = np.where(ok, final, 1.0)
    net = np.where(g["net"] > 0, g["net"], 1.0)
    return {
        "mean": np.where(ok & (g["net"] > 0), g["dsum"] / net, np.nan),
        **{
            f"share{d}": np.where(ok, g[f"by{d}"] / safe, np.nan)
            for d in SHARE_DAYS
        },
        "first_ratio": np.where(ok, g["first"] / safe, np.nan),
        "log_count": np.where(ok, np.log(safe / days), np.nan),
    }


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Return the slope of `y` on `x` and their correlation.

    Over the points where both are known; None with fewer than
    `MIN_WEEKS` points or no variation.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_WEEKS:
        return None, None
    xd, yd = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    sxx, syy = float((xd * xd).sum()), float((yd * yd).sum())
    if sxx <= 1e-12 or syy <= 1e-12:
        return None, None
    sxy = float((xd * yd).sum())
    return sxy / sxx, sxy / np.sqrt(sxx * syy)


def _pooled(x: np.ndarray, y: np.ndarray) -> tuple[float | None, ...]:
    """Return `_fit` within strata: each stratum's series less its mean."""
    xs, ys = [], []
    for xi, yi in zip(x, y, strict=True):
        ok = np.isfinite(xi) & np.isfinite(yi)
        if ok.sum() >= MIN_WEEKS:
            xs.append(xi[ok] - xi[ok].mean())
            ys.append(yi[ok] - yi[ok].mean())
    if not xs:
        return None, None
    return _fit(np.concatenate(xs), np.concatenate(ys))


def _corr(mean: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Return each stratum's mean correlation with the others and all's.

    Pearson correlations of the strata's weekly mean delays, pair by
    pair over the weeks both have (at least `MIN_WEEKS`); NaN for a
    stratum with none.
    """
    k = len(mean)
    c: np.ndarray = (
        pd.DataFrame(mean.T).corr(min_periods=MIN_WEEKS).to_numpy(copy=True)
    )
    c[np.eye(k, dtype=bool)] = np.nan
    with np.errstate(invalid="ignore"):
        each = np.array(
            [np.nanmean(r) if np.isfinite(r).any() else np.nan for r in c]
        )
    pairs = c[np.triu_indices(k, 1)]
    pairs = pairs[np.isfinite(pairs)]
    return each, float(pairs.mean()) if len(pairs) else None


def _decompose(y: np.ndarray) -> dict[str, float | None] | None:
    """Return the variation of a strata-by-week matrix by its parts.

    An additive fit over the known cells: a time part common to strata
    (by week), a static stratum part and the stratum-by-week rest (which
    holds the weekly counts' sampling noise too). Each part's sd over
    the cells and its share of their sum of variances. With one stratum
    only the time part's sd. None with fewer than `MIN_WEEKS` cells.
    """
    ok = np.isfinite(y)
    if ok.sum() < MIN_WEEKS:
        return None
    strata = int((ok.sum(1) > 0).sum())
    if strata < 2:
        v = y[ok]
        return {
            "time_sd": _r(float(v.std())),
            "strata_sd": None,
            "residual_sd": None,
            "time_share": None,
            "strata_share": None,
            "residual_share": None,
        }
    z = np.where(ok, y, 0.0)
    n_s, n_w = ok.sum(1), ok.sum(0)
    mu = float(z.sum() / ok.sum())
    a, b = np.zeros(y.shape[0]), np.zeros(y.shape[1])
    for _ in range(200):
        r = np.where(ok, z - mu - a[:, None], 0.0)
        b = np.divide(r.sum(0), n_w, out=np.zeros_like(b), where=n_w > 0)
        r = np.where(ok, z - mu - b[None, :], 0.0)
        new = np.divide(r.sum(1), n_s, out=np.zeros_like(a), where=n_s > 0)
        done = np.abs(new - a).max() < 1e-10
        a = new
        if done:
            break
    parts = {
        "time": np.broadcast_to(b[None, :], y.shape)[ok],
        "strata": np.broadcast_to(a[:, None], y.shape)[ok],
        "residual": (y - mu - a[:, None] - b[None, :])[ok],
    }
    var = {k: float(v.var()) for k, v in parts.items()}
    total = sum(var.values())
    return {
        **{f"{k}_sd": _r(np.sqrt(v)) for k, v in var.items()},
        **{
            f"{k}_share": _r(v / total) if total > 0 else None
            for k, v in var.items()
        },
    }


def _sd(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)]
    return float(v.std()) if len(v) >= MIN_WEEKS else None


def _rl(v: np.ndarray, digits: int = 3) -> list[float | None]:
    return [_r(x, digits) for x in v]


def _levels(
    final: np.ndarray, days: np.ndarray, log_count: np.ndarray
) -> dict[str, Any]:
    """Return how the strata's weekly final counts move, for ascertainment.

    Ascertainment is not told apart from infections by the counts alone,
    so these show the counts' own movement. `rel`: each stratum's count
    a day by week over its mean (the total's in `total_rel`). `corr`:
    the mean pairwise correlation of the strata's log daily counts by
    week. `share_sd`: each stratum's sd of its log share of the strata's
    weekly count (weeks with `MIN_COUNT` cases, at least `MIN_WEEKS`),
    and `share_sd_all` their mean weighted by count; a stratum whose
    ascertainment moves on its own moves its share.
    """
    daily = np.clip(final, 0, None) / days
    mean = daily.mean(1, keepdims=True)
    rel = np.where(mean > 0, daily / np.where(mean > 0, mean, 1.0), np.nan)
    tot = daily.sum(0)
    total_rel = tot / tot.mean() if tot.mean() > 0 else tot * np.nan
    _, corr = _corr(log_count)
    ok = np.isfinite(log_count) & (tot > 0)[None, :]
    share = np.where(
        ok,
        np.log(np.where(ok, daily, 1.0) / np.where(tot > 0, tot, 1.0)[None, :]),
        np.nan,
    )
    sds = np.array([_sd(r) if len(daily) > 1 else None for r in share],
                   dtype=float)  # fmt: skip
    w = np.clip(final, 0, None).sum(1)
    use = np.isfinite(sds) & (w > 0)
    return {
        "rel": rel,
        "total_rel": total_rel,
        "corr": corr,
        "share_sd": sds,
        "share_sd_all": float((sds[use] * w[use]).sum() / w[use].sum())
        if use.any()
        else None,
    }


def _by_stratum(
    complete: pd.DataFrame, first: pd.Timestamp
) -> dict[str, Any] | None:
    """Return each stratum's delays week by week, and what they show.

    Keys: `by_stratum` (`week`, the weeks' first reference dates;
    `total`, `_week_series` of all strata summed; `strata`, the largest
    `MAX_LINES` strata, each with its `name`, `n` (final count),
    `withdrawn` (the share of its final count ever withdrawn), `corr`
    (its weekly mean delay's mean correlation with the other strata's),
    `share7_sd` (the sd of its weekly share by day 7), `counts_slope`
    and `counts_corr` (its weekly mean delay's slope on and correlation
    with its log daily count by week), `share1_corr` (the share by day
    1's correlation with the log count) and its `_week_series`; `corr`,
    the mean pairwise correlation of the strata's weekly mean delays,
    and `corr_min`, the lowest stratum's mean correlation); `coupling`
    (`slope`, `corr` and `share1_corr` pooled within strata);
    `bursts` (`max_ratio`, the largest weekly first-report-to-final
    ratio of any stratum; `withdrawn`, the share of the final count
    ever withdrawn; `withdrawn_max`, the largest stratum's share); and
    `noise` (`_decompose` of the strata's weekly log(1 + mean delay));
    and `levels` (`_levels`: `corr` and `share_sd`, also each line's
    `count_share_sd` and `rel_count`, and the total's `rel_count`).
    Strata are the `MAX_STRATA` largest.
    """
    if complete.empty:
        return None
    t = _stratum_weeks(complete, first)
    weeks = range(int(t.index.get_level_values("week").max()) + 1)
    sizes = (
        t.groupby(level="stratum")["final"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    names = list(sizes.index[:MAX_STRATA])
    g = {
        str(c): t[c]
        .unstack("week")
        .reindex(index=names, columns=weeks)
        .fillna(0.0)
        .to_numpy(dtype=float)
        for c in t.columns
    }
    tot = t.groupby(level="week").sum().reindex(weeks, fill_value=0.0)
    span = (complete["ref"].max() - first).days + 1
    days = np.array([min(7, span - 7 * w) for w in weeks], dtype=float)
    total = _week_series(
        {str(c): tot[c].to_numpy(dtype=float) for c in t.columns}, days
    )
    ser = _week_series(g, days)
    each, corr = _corr(ser["mean"])
    final = g["final"].sum(1)
    kept = final > 0
    withdrawn = g["withdrawn"].sum(1) / np.where(kept, final, 1.0)
    lv = _levels(g["final"], days, ser["log_count"])
    lines = []
    for i, name in enumerate(names[:MAX_LINES]):
        slope, cc = _fit(ser["log_count"][i], ser["mean"][i])
        _, c1 = _fit(ser["log_count"][i], ser["share1"][i])
        lines.append(
            {
                "name": str(name),
                "n": _r(final[i], 1),
                "withdrawn": _r(withdrawn[i], 4) if kept[i] else None,
                "corr": _r(each[i]),
                "share7_sd": _r(_sd(ser["share7"][i])),
                "counts_slope": _r(slope),
                "counts_corr": _r(cc),
                "share1_corr": _r(c1),
                "count_share_sd": _r(lv["share_sd"][i]),
                "rel_count": _rl(lv["rel"][i]),
                **{k: _rl(v[i]) for k, v in ser.items()},
            }
        )
    slope, cc = _pooled(ser["log_count"], ser["mean"])
    _, c1 = _pooled(ser["log_count"], ser["share1"])
    ratios = ser["first_ratio"][np.isfinite(ser["first_ratio"])]
    big = final >= MIN_COUNT
    all_final = float(final[kept].sum())
    eachs = each[np.isfinite(each)]
    return {
        "by_stratum": {
            "week": [
                (first + pd.Timedelta(weeks=w)).strftime("%Y-%m-%d")
                for w in weeks
            ],
            "total": {
                **{k: _rl(v) for k, v in total.items()},
                "rel_count": _rl(lv["total_rel"]),
            },
            "strata": lines,
            "corr": _r(corr),
            "corr_min": _r(float(eachs.min())) if len(eachs) else None,
        },
        "coupling": {
            "slope": _r(slope),
            "corr": _r(cc),
            "share1_corr": _r(c1),
        },
        "bursts": {
            "max_ratio": _r(float(ratios.max())) if len(ratios) else None,
            "withdrawn": _r(float(g["withdrawn"].sum()) / all_final, 4)
            if all_final > 0
            else None,
            "withdrawn_max": _r(float(withdrawn[big].max()), 4)
            if big.any()
            else None,
        },
        "noise": _decompose(np.log1p(ser["mean"])),
        "levels": {
            "corr": _r(lv["corr"]),
            "share_sd": _r(lv["share_sd_all"]),
        },
    }


def _releases(rows: pd.DataFrame, complete: pd.DataFrame) -> dict[str, Any]:
    """Return the release cadence and the wait it adds.

    A release is a report date with any count. `wait` is the
    count-weighted days from each complete reference date to the first
    release on or after it.
    """
    by_rep = rows.groupby("rep")["count"].apply(lambda c: (c != 0).any())
    dates = np.sort(by_rep.index[by_rep.to_numpy()].to_numpy())
    gaps = np.diff(dates) / np.timedelta64(1, "D")
    hist = np.zeros(GAP_BINS)
    for g in gaps.astype(int):
        hist[min(max(g, 1), GAP_BINS) - 1] += 1
    span = (dates[-1] - dates[0]) / np.timedelta64(1, "D") + 1
    days = pd.DatetimeIndex(dates).day_of_week
    by_ref = complete.groupby("ref")["count"].sum().clip(lower=0)
    nxt = np.searchsorted(dates, by_ref.index.to_numpy())
    ok = nxt < len(dates)
    wait = (dates[nxt[ok]] - by_ref.index.to_numpy()[ok]) / np.timedelta64(
        1, "D"
    )
    weights = by_ref.to_numpy()[ok]
    return {
        "n": len(dates),
        "per_week": _r(7 * len(dates) / span),
        "gaps": [_r(v / len(gaps)) for v in hist] if len(gaps) else None,
        "weekdays": [
            _r(float((days == k).mean())) if len(days) else 0.0
            for k in range(7)
        ],
        "wait": _r(float((wait * weights).sum() / weights.sum()))
        if weights.sum() > 0
        else None,
    }


def structure(
    rows: pd.DataFrame,
    parent: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    """Return the delay structure of counts by reference and report date.

    `parent` gives a stratum's parent (None for one with none); strata
    with no parent are children of the total. None if there are no
    counts.

    Keys: `horizon` (the 95% delay, days), `first`, `last_ref` and
    `last_rep`; `overall` (count, mean and 10, 50 and 90% delays of the
    complete reference dates) and `cdf` (`_cdf` of the same); `by_ref`
    and `by_rel` (the same by week of reference date, and by week of
    report over the complete releases); `time` (the mean in each of
    `TIME_BINS` equal parts of the complete reference span); `dow`
    (`ref_delay`, each reference weekday's mean delay less the mean of
    the seven; `ref_share` and `rep_share`, counts by weekday of
    reference and of report over their mean; `release_days`, the share
    of releases on each weekday);
    `strata` (the total, then each parent with its children);
    `releases` (`_releases`); and `by_stratum`, `coupling`, `bursts`
    and `noise` (`_by_stratum`, each None without complete reference
    dates).
    """
    if rows.empty or float(rows["count"].clip(lower=0).sum()) <= 0:
        return None
    rows = rows.assign(
        ref=pd.to_datetime(rows["ref"]), rep=pd.to_datetime(rows["rep"])
    )
    rows = rows.assign(delay=(rows["rep"] - rows["ref"]).dt.days)
    first, last_ref, last_rep = (
        rows["ref"].min(),
        rows["ref"].max(),
        rows["rep"].max(),
    )
    horizon = _quantile(_by_delay(rows), HORIZON_Q) or 0
    complete = rows[rows["ref"] <= last_rep - pd.Timedelta(days=horizon)]
    released = rows[rows["rep"] >= first + pd.Timedelta(days=horizon)]
    time: list[float | None] = []
    if not complete.empty:
        span = (complete["ref"].max() - first).days + 1
        part = ((complete["ref"] - first).dt.days * TIME_BINS) // span
        time = [_summary(complete[part == b])["mean"] for b in range(TIME_BINS)]
    wd = _weekday_mean(complete)
    known = [v for v in wd if v is not None]
    centre = float(np.mean(known)) if known else None
    rel = _releases(rows, complete)
    return {
        "horizon": int(horizon),
        "first": first.strftime("%Y-%m-%d"),
        "last_ref": last_ref.strftime("%Y-%m-%d"),
        "last_rep": last_rep.strftime("%Y-%m-%d"),
        "overall": _summary(complete),
        "cdf": _cdf(complete),
        "by_ref": _weekly(complete, "ref", first),
        "by_rel": _weekly(released, "rep", first),
        "time": time,
        "dow": {
            "ref_delay": [None if v is None else _r(v - centre) for v in wd],
            "ref_share": _profile(rows, "ref"),
            "rep_share": _profile(rows, "rep"),
            "release_days": rel["weekdays"],
        },
        "strata": _strata(complete, parent),
        "releases": rel,
        **(_by_stratum(complete, first) or dict.fromkeys(STRATUM_KEYS)),
    }


def _wsd(values: list[float], weights: list[float]) -> float:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    m = (v * w).sum() / w.sum()
    return float(np.sqrt((w * (v - m) ** 2).sum() / w.sum()))


def scalars(s: dict[str, Any]) -> dict[str, float | None]:
    """Return a structure's single numbers.

    `delay_drift`: the count-weighted sd of the weekly mean delays by
    reference week over the overall mean. `delay_trend`: the mean delay
    of the last third of the complete reference span less the first
    third's (days). `dow_delay`: the range of the reference weekdays'
    mean delays (days). `release_weekdays`: the weekdays with at least
    5% of releases. `wait`: the mean days to the next release, and
    `wait_share` its share of the mean delay. `strata_spread`: the
    count-weighted sd of the children's mean delays (days), None with
    fewer than two. `strata_corr` and `strata_corr_min`: the mean
    pairwise correlation of the strata's weekly mean delays and the
    lowest stratum's mean correlation. `share7_sd`: the largest
    stratum's sd of its weekly share by day 7. `counts_slope`,
    `counts_corr` and `share1_counts_corr`: the weekly mean delay's
    slope on and correlation with the log weekly count, and the share
    by day 1's correlation, within strata. `burst_ratio`: the largest
    weekly first-report-to-final ratio of any stratum. `withdrawn_share`
    and `withdrawn_max`: the share of the final count ever withdrawn,
    all strata and the largest stratum's. `noise_time`, `noise_strata`
    and `noise_residual`: the shares of the weekly log mean delay's
    variation common to strata, fixed by stratum and the rest.
    `strata_count_corr` and `count_share_sd`: the mean pairwise
    correlation of the strata's log weekly counts and the count-weighted
    sd of their log shares of the weekly count (`_levels`).
    """
    ref = s["by_ref"]
    pairs = [
        (m, n)
        for m, n in zip(ref["mean"], ref["n"], strict=True)
        if m is not None
    ]
    mean = s["overall"]["mean"]
    drift = (
        _wsd(*zip(*pairs, strict=True)) / mean
        if len(pairs) > 1 and mean
        else None
    )
    time = [v for v in s["time"] if v is not None]
    third = max(1, TIME_BINS // 3)
    early = [v for v in s["time"][:third] if v is not None]
    late = [v for v in s["time"][-third:] if v is not None]
    trend = np.mean(late) - np.mean(early) if early and late else None
    wd = [v for v in s["dow"]["ref_delay"] if v is not None]
    kids = [
        x
        for x in s["strata"]
        if x["level"] == "child" and x["mean"] is not None
    ]
    wait = s["releases"]["wait"]
    return {
        "delay_drift": _r(drift),
        "delay_trend": _r(trend) if time else None,
        "dow_delay": _r(max(wd) - min(wd)) if len(wd) > 1 else None,
        "release_weekdays": sum(v >= 0.05 for v in s["dow"]["release_days"]),
        "wait": wait,
        "wait_share": _r(min(wait / mean, 1.0))
        if wait is not None and mean
        else (0.0 if wait == 0 else None),
        "strata_spread": _r(
            _wsd([x["mean"] for x in kids], [x["n"] for x in kids])
        )
        if len(kids) > 1
        else None,
        **_stratum_scalars(s),
    }


def _stratum_scalars(s: dict[str, Any]) -> dict[str, float | None]:
    by = s.get("by_stratum") or {}
    cp = s.get("coupling") or {}
    bu = s.get("bursts") or {}
    no = s.get("noise") or {}
    lv = s.get("levels") or {}
    sds = [x["share7_sd"] for x in by.get("strata", [])]
    sds = [v for v in sds if v is not None]
    return {
        "strata_corr": by.get("corr"),
        "strata_corr_min": by.get("corr_min"),
        "share7_sd": max(sds) if sds else None,
        "counts_slope": cp.get("slope"),
        "counts_corr": cp.get("corr"),
        "share1_counts_corr": cp.get("share1_corr"),
        "burst_ratio": bu.get("max_ratio"),
        "withdrawn_share": bu.get("withdrawn"),
        "withdrawn_max": bu.get("withdrawn_max"),
        "noise_time": no.get("time_share"),
        "noise_strata": no.get("strata_share"),
        "noise_residual": no.get("residual_share"),
        "strata_count_corr": lv.get("corr"),
        "count_share_sd": lv.get("share_sd"),
    }


NOISE_SETTINGS = (
    ("time", "delay_jitter", ("yes",), "delay_jitter_sd"),
    ("strata", "delay_strata_noise", ("strata", "both"), "delay_strata_sd"),
    (
        "strata_time",
        "delay_strata_noise",
        ("strata_time", "both"),
        "delay_strata_jitter_sd",
    ),
)
"""Each delay noise layer: its name, the factor that turns it on, the
factor's levels that do and the setting with its drawn sd."""


ASCERTAINMENT_SETTINGS = (
    "ascertainment_change",
    "ascertainment_change_strata",
    "ascertainment_change_switch",
    "ascertainment_counts",
    "ascertainment_noise",
    "asc_own_change",
    "asc_rho",
    "asc_switch_at",
    "asc_counts_sign",
    "asc_counts_slope",
    "asc_counts_sd",
    "asc_noise_sd",
)
"""The settings that make a simulation's ascertainment, as drawn."""


def drawn_ascertainment(settings: Mapping[str, Any]) -> dict[str, str]:
    """Return a simulation's drawn ascertainment structure.

    Each of `ASCERTAINMENT_SETTINGS` the settings hold (not empty), as
    written: factor levels, and values per stratum `;`-joined.
    """
    out = {}
    for k in ASCERTAINMENT_SETTINGS:
        v = settings.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if str(v).strip() not in ("", "nan", "NA"):
            out[k] = str(v).strip()
    return out


def drawn_noise(settings: Mapping[str, Any]) -> dict[str, float | None]:
    """Return a simulation's drawn delay noise sds, by layer.

    From its settings: `time` (`delay_jitter_sd`), `strata`
    (`delay_strata_sd`) and `strata_time` (`delay_strata_jitter_sd`),
    each None where its factor leaves the layer off or the setting is
    missing (0 where the factor is there and off).
    """
    out: dict[str, float | None] = {}
    for name, factor, on, col in NOISE_SETTINGS:
        level = str(settings.get(factor, "")).strip()
        try:
            v = float(settings.get(col, ""))
        except (TypeError, ValueError):
            v = float("nan")
        if level and level not in on:
            out[name] = 0.0
        else:
            out[name] = _r(v) if np.isfinite(v) else None
    return out


def curves(s: dict[str, Any]) -> dict[str, list[float | None]]:
    """Return a structure's fixed-length curves.

    `cdf`: the share reported by each of `DELAY_POINTS`. `delay_time`:
    each part's mean delay over the overall mean. `ref_delay` and
    `release_days`: as in `structure`'s `dow`; `refdow` and `repdow` its
    `ref_share` and `rep_share`. `gaps`:
    the share of release gaps of 1 to 7 and 8 or more days.
    """
    mean = s["overall"]["mean"]
    return {
        "cdf": s["cdf"],
        "delay_time": [
            _r(v / mean) if v is not None and mean else None for v in s["time"]
        ]
        or [None] * TIME_BINS,
        "ref_delay": s["dow"]["ref_delay"],
        "refdow": s["dow"]["ref_share"],
        "repdow": s["dow"]["rep_share"],
        "release_days": s["dow"]["release_days"],
        "gaps": s["releases"]["gaps"] or [None] * GAP_BINS,
    }
