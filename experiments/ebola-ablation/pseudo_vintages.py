"""Pseudo vintages of the BDBV hub data from its reference/report pairs.

What the data would have shown at a past cut-off, counted from the pairs
of onset (reference) and report the hub publishes, with no delay model:

- The delay line (`weekly-reporting-delays-by-province.csv`, one file
  per release from 9 October): one row per case with an observed onset,
  its province, onset week and days from onset to report. A case's
  report day is its onset day plus the delay, and its onset day is
  known only to the week, so whether it was reported by a cut-off is
  certain unless its 7 possible report days straddle the cut-off
  (`observed_vintage`: `lower` counts the certain cases, `upper` adds
  the straddling ones, `even` spreads each straddling case evenly over
  its 7 onset days).
- The missing-onset counts (`daily-missing-symptom-by-report-date.csv`):
  cases with no onset date, by report day and place, exact
  (`missing_vintage`).
- The daily releases (`daily-confirmed-case.csv`): counts by onset day
  at each real release, the vintages themselves (`check`).

`python experiments/ebola-ablation/pseudo_vintages.py DATA --out FILE`
writes the sources, the pseudo vintages by cut-off and their check against
the real releases as JSON for the ablation page
(`ebola_ablation_report.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DELAYS = "weekly-reporting-delays-by-province.csv"
MISSING = "daily-missing-symptom-by-report-date.csv"
DAILY = "daily-confirmed-case.csv"


def provinces(locations: pd.DataFrame) -> dict[str, str]:
    """Return each province's code by its name (zones share some names)."""
    p = locations.loc[locations["location_type"] == "province"]
    return dict(zip(p["location_name"], p["location"], strict=True))


def observed_vintage(
    line: pd.DataFrame, cutoff: str, locations: pd.DataFrame
) -> pd.DataFrame:
    """Return the delay line's cases reported by `cutoff`, bounded.

    Parameters
    ----------
    line : pandas.DataFrame
        The delay line as published: `location_name`, `week_ending` (the
        onset week's Saturday) and `days_sym_to_report`.
    cutoff : str
        The cut-off, an ISO date; a case is reported by it when its
        report day is on or before it.
    locations : pandas.DataFrame
        The hub's locations (`location`, `location_name`,
        `location_type`).

    Returns
    -------
    pandas.DataFrame
        By province code and onset week: `lower` (cases reported by the
        cut-off whatever their onset day), `upper` (those that may have
        been) and `even` (each straddling case counted as the share of
        its 7 onset days that report by the cut-off).
    """
    cut = pd.Timestamp(cutoff)
    week = pd.to_datetime(line["week_ending"])
    delay = pd.to_timedelta(line["days_sym_to_report"], unit="D")
    first = week - pd.Timedelta(days=6) + delay
    last = week + delay
    share = ((cut - first).dt.days + 1).clip(0, 7) / 7
    frame = pd.DataFrame(
        {
            "location": line["location_name"].map(provinces(locations)),
            "onset_week": week.dt.strftime("%Y-%m-%d"),
            "lower": (last <= cut).astype(int),
            "upper": (first <= cut).astype(int),
            "even": share,
        }
    )
    return frame.groupby(["location", "onset_week"])[
        ["lower", "upper", "even"]
    ].sum()


def missing_vintage(missing: pd.DataFrame, cutoff: str) -> pd.Series:
    """Return the missing-onset cases reported by `cutoff`, by province.

    Parameters
    ----------
    missing : pandas.DataFrame
        One release's missing-onset counts: `report_date`, `location`,
        `location_type` and `n_miss_sym_onset`.
    cutoff : str
        The cut-off, an ISO date.

    Returns
    -------
    pandas.Series
        Cases by province code.
    """
    m = missing.loc[
        (missing["location_type"] == "province")
        & (missing["report_date"] <= cutoff)
    ]
    return m.groupby("location")["n_miss_sym_onset"].sum()


def cutoffs(releases: list[str], start: str) -> list[str]:
    """Return weekly cut-offs from `start` to the last release.

    Each release's date is one too.
    """
    days = pd.date_range(start, max(releases), freq="7D").strftime("%Y-%m-%d")
    return sorted({*days, *releases})


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a release's observed-onset counts by province and onset week."""
    d = daily.loc[(daily["location_type"] == "province") & ~daily["imputed"]]
    day = pd.to_datetime(d["target_end_date"])
    week = day + pd.to_timedelta((5 - day.dt.day_of_week) % 7, unit="D")
    return (
        d.assign(onset_week=week.dt.strftime("%Y-%m-%d"))
        .groupby(["location", "onset_week"])["observation"]
        .sum()
    )


def _cells(pseudo: pd.DataFrame, real: pd.Series) -> dict[str, float]:
    """Return how a pseudo vintage's cells sit against a real release's."""
    j = pseudo.join(real.rename("real"), how="outer").fillna(0)
    j = j.loc[(j["upper"] + j["real"]) > 0]
    exact = j["lower"] == j["upper"]
    return {
        "cells": len(j),
        "inside": float(
            ((j["real"] >= j["lower"]) & (j["real"] <= j["upper"])).mean()
        ),
        "below": float((j["real"] < j["lower"]).mean()),
        "above": float((j["real"] > j["upper"]).mean()),
        "exact_cells": int(exact.sum()),
        "exact_equal": int((exact & (j["lower"] == j["real"])).sum()),
    }


def build(data: Path, start: str = "2026-07-04") -> dict:
    """Return the sources, pseudo vintages and their check as a dict.

    Parameters
    ----------
    data : pathlib.Path
        The tidy hub data (`bdbvhub data`), with the raw files in `raw/`.
    start : str
        The first weekly cut-off.

    Returns
    -------
    dict
        `sources` (each file, its pairs and coverage), `vintages` (by
        cut-off: national totals of the pseudo vintage, and the real
        release's where one is that day) and `check` (each real release
        against the pseudo vintage from the latest files, and against
        its own delay file).
    """
    raw = data / "raw"
    locations = pd.read_csv(raw / "auxiliary-data" / "locations.csv")
    daily = pd.read_csv(raw / "auxiliary-data" / DAILY)
    daily["imputed"] = daily["imputed"].astype(str).str.upper() == "TRUE"
    missing = pd.read_csv(raw / "auxiliary-data" / MISSING)
    lines = {
        p.stem: pd.read_csv(p) for p in sorted((raw / "delays").glob("*.csv"))
    }
    releases = sorted(daily["as_of"].unique())
    latest_line = lines[max(lines)]
    latest_missing = missing.loc[missing["as_of"] == missing["as_of"].max()]
    sources = [
        {
            "file": f"auxiliary-data/{DAILY}",
            "holds": "Cases by onset day and release, observed and imputed "
            "onsets apart",
            "places": "country, provinces, zones",
            "reference": "onset day",
            "report": "the release (no report date)",
            "releases": f"{releases[0]} to {releases[-1]} ({len(releases)})",
        },
        {
            "file": f"auxiliary-data/{DELAYS} (archived per release)",
            "holds": "One row per case with an observed onset: onset week "
            "and days from onset to report",
            "places": "provinces",
            "reference": "onset week",
            "report": "onset day plus the delay: known to within 7 days",
            "releases": f"{min(lines)} to {max(lines)} ({len(lines)})",
        },
        {
            "file": f"auxiliary-data/{MISSING}",
            "holds": "Cases with no onset date (onsets the hub imputes), by "
            "report day",
            "places": "country, provinces, zones",
            "reference": "none",
            "report": "report day, exact",
            "releases": f"{missing['as_of'].min()} to "
            f"{missing['as_of'].max()} ({missing['as_of'].nunique()})",
        },
    ]
    vintages = []
    for cut in cutoffs(releases, start):
        obs = observed_vintage(latest_line, cut, locations).sum()
        row = {
            "cutoff": cut,
            "lower": int(obs["lower"]),
            "even": float(obs["even"]),
            "upper": int(obs["upper"]),
            "missing": int(missing_vintage(latest_missing, cut).sum()),
        }
        if cut in releases:
            r = daily.loc[
                (daily["as_of"] == cut) & (daily["location_type"] == "country")
            ]
            row["real_observed"] = int(
                r.loc[~r["imputed"], "observation"].sum()
            )
            row["real_imputed"] = int(r.loc[r["imputed"], "observation"].sum())
        vintages.append(row)
    check = []
    for rel in releases:
        real = _weekly(daily.loc[daily["as_of"] == rel])
        row = {
            "release": rel,
            **_cells(observed_vintage(latest_line, rel, locations), real),
        }
        if rel in lines:
            own = observed_vintage(lines[rel], rel, locations)
            joined = (
                own["upper"]
                .rename("own")
                .to_frame()
                .join(real.rename("real"), how="outer")
                .fillna(0)
            )
            row["own_equal"] = float((joined["own"] == joined["real"]).mean())
        check.append(row)
    negative = int((latest_line["days_sym_to_report"] < 0).sum())
    return {
        "latest_delay_line": max(lines),
        "latest_missing": str(latest_missing["as_of"].iloc[0]),
        "negative_delays": negative,
        "sources": sources,
        "vintages": vintages,
        "check": check,
    }


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2026-07-04")
    a = ap.parse_args()
    out = build(a.data, a.start)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
