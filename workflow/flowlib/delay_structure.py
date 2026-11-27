"""The structure of reporting delays in counts by reference and report date.

One set of summaries for simulated outbreaks and for real data alike
(the German triangle, the BDBV releases), so the simulations' delays can
be read against the data's: how the delay distribution moves over time
(by reference week and by release week), its weekday effects, how it
differs between strata (a parent and its children), and the release
cadence with the wait it adds.

Input rows have `stratum`, `ref` and `rep` (dates) and `count`, the
count of reference date `ref` first reported (or revised) on `rep`.
Counts are netted by delay before any summary, so a revision moves cases
between delays and a net negative is dropped. Delay summaries use only
reference dates at least `horizon` days (the 95% delay) before the last
report, and release summaries only reports at least `horizon` days after
the first reference date, so the series' ends do not shorten or lengthen
the delays.

`structure` returns the summaries as plain JSON, `scalars` its single
numbers and `curves` its fixed-length curves, for comparing many series.
"""

from __future__ import annotations

from collections.abc import Callable
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

MAX_STRATA = 40
"""The most strata `structure` lists, the largest first."""


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
    names = rows.groupby("stratum")["count"].sum().sort_values(ascending=False)
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
    complete reference dates); `by_ref` and `by_rel` (the same by week
    of reference date, and by week of report over the complete
    releases); `time` (the mean in each of `TIME_BINS` equal parts of
    the complete reference span); `dow` (`ref_delay`, each reference
    weekday's mean delay less the mean of the seven; `ref_share` and
    `rep_share`, counts by weekday of reference and of report over their
    mean; `release_days`, the share of releases on each weekday);
    `strata` (the total, then each parent with its children); and
    `releases` (`_releases`).
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
    fewer than two.
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
    }


def curves(s: dict[str, Any]) -> dict[str, list[float | None]]:
    """Return a structure's fixed-length curves.

    `delay_time`: each part's mean delay over the overall mean.
    `ref_delay` and `release_days`: as in `structure`'s `dow`. `gaps`:
    the share of release gaps of 1 to 7 and 8 or more days.
    """
    mean = s["overall"]["mean"]
    return {
        "delay_time": [
            _r(v / mean) if v is not None and mean else None for v in s["time"]
        ]
        or [None] * TIME_BINS,
        "ref_delay": s["dow"]["ref_delay"],
        "release_days": s["dow"]["release_days"],
        "gaps": s["releases"]["gaps"] or [None] * GAP_BINS,
    }
