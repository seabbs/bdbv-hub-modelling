"""German real-data training blocks, for the real-data ablation.

The German hub triangle cut into training blocks as the German tasks lay
it out, from reports before the first scored German date only
(`window`), so the real-data run is never scored on what it trained on.
Run as `python -m nowcastgermany.finetune_blocks TRIANGLE OUT
--first-date D --truth-delay N`, the window's two settings from
`config/germany.toml` (rule real_training_blocks).
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from nowcastfoundation.finetune import AHEAD_WEEKS, training_block
from nowcastfoundation.finetune import chronos as ft
from nowcastgermany import data as de
from nowcastutils.base import NATIONAL
from nowcastutils.dates import epiweek_end

SOURCE = "daily+rolling7"
"""The blocks' source: daily counts with each place's 7-day rolling sums,
as the German models lay out the daily releases (`forecaster.PERIODS`)."""

LAGS = (14, 21, 28, 35, 40, 49)
"""Release lags $D$ in days, taken in turn over the cuts: the ragged
models' default (28 days), the German truth delay (40 days) and the
simulations' range around them."""

PLACES = ("national", "states", "ages")
"""The blocks' places: Germany alone, or Germany with its states or its
age groups, as the German tasks group them."""

FIRST_CUT = "2021-04-20"
"""The first nowcast date cut, two weeks after the triangle's first
Meldedatum; cuts with under two steps of context are skipped."""


def window(first_date: str, truth_delay: int) -> tuple[str, str]:
    """Return the last report date used and the last date with truth.

    Reports up to the day before `first_date`, the first scored German
    nowcast date; a count is truth once `truth_delay` days have passed,
    as the benchmark takes its truth.
    """
    as_of = dt.date.fromisoformat(first_date) - dt.timedelta(days=1)
    truth = as_of - dt.timedelta(days=truth_delay)
    return as_of.isoformat(), truth.isoformat()


def cut_dates(first: str, truth_until: str) -> list[str]:
    """Return every nowcast date whose horizon ends by `truth_until`."""
    last = dt.date.fromisoformat(truth_until) - dt.timedelta(weeks=AHEAD_WEEKS)
    day, out = dt.date.fromisoformat(first), []
    while day <= last:
        out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return out


def german_counts(wide: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Return the German hub triangle's counts as reported by `as_of`.

    Each change of `value_0d` to `value_80d` is reported on its Meldedatum
    plus its delay; the change after 80 days is left out. States (all
    ages) and Germany's age groups are kept, named as the German tasks
    name them (`nowcastgermany.data.place_code`); Germany is their sum.

    Returns
    -------
    pandas.DataFrame
        `location`, `reference_date`, `report_date` and `count`, as
        `finetune.training_block` reads.
    """
    long = de.hub_changes(wide)
    long = long.loc[long["delay"] <= 80]
    states = (long["age_group"] == de.ALL_AGES) & (
        long["location"] != de.GERMANY
    )
    ages = (long["location"] == de.GERMANY) & (long["age_group"] != de.ALL_AGES)
    long = long.loc[states | ages]
    report = pd.to_datetime(long["date"]) + pd.to_timedelta(long["delay"], "D")
    out = pd.DataFrame(
        {
            "location": [
                de.place_code(loc, age)
                for loc, age in zip(
                    long["location"], long["age_group"], strict=True
                )
            ],
            "reference_date": long["date"].astype(str),
            "report_date": report.dt.date.astype(str),
            "count": long["change"].astype("int64"),
        }
    )
    out = out.loc[out["report_date"] <= as_of]
    return out.sort_values(
        ["location", "reference_date", "report_date"]
    ).reset_index(drop=True)


def _moved(dates: pd.Series, days: int) -> pd.Series:
    """Return ISO dates moved `days` days later."""
    return (pd.to_datetime(dates) + pd.Timedelta(days=days)).dt.date.astype(str)


def german_blocks(
    counts: pd.DataFrame,
    cuts: list[str],
    truth_until: str,
    first: int = 0,
) -> list[dict[str, Any]]:
    """Return the training blocks of the German data at nowcast dates.

    Each cut is laid out as the German tasks lay it out: dates move so the
    cut is a Saturday, every report date up to it is a release, and
    Germany is the country (`parent` exact). Each of `PLACES` gives a
    block by `finetune.training_block`, at a lag taken in turn from
    `LAGS` (from cut `first` on), padded as the simulated blocks are.

    Raises
    ------
    ValueError
        If a cut's horizon ends after `truth_until`.
    """
    places = sorted(counts["location"].unique())
    states = [p for p in places if "-A" not in p]
    ages = [p for p in places if "-A" in p]
    leaves = {"national": states, "states": states, "ages": ages}
    out = []
    for i, cut in enumerate(cuts, start=first):
        end = dt.date.fromisoformat(cut) + dt.timedelta(weeks=AHEAD_WEEKS)
        if end.isoformat() > truth_until:
            raise ValueError(
                f"cut {cut} needs truth to {end}, past the truth date "
                f"{truth_until}"
            )
        shift = (epiweek_end(cut) - dt.date.fromisoformat(cut)).days
        moved = counts.assign(
            reference_date=_moved(counts["reference_date"], shift),
            report_date=_moved(counts["report_date"], shift),
        )
        last = (dt.date.fromisoformat(cut) + dt.timedelta(shift)).isoformat()
        cutoffs = sorted(d for d in moved["report_date"].unique() if d <= last)
        for j, which in enumerate(PLACES):
            view = (NATIONAL,)
            if which != "national":
                view = (NATIONAL, *leaves[which])
            own = moved.loc[moved["location"].isin(leaves[which])]
            block = training_block(
                own,
                SOURCE,
                view,
                cutoffs,
                max_lag=LAGS[(i + j) % len(LAGS)],
            )
            if block is None or block["start"] < 2:
                continue
            padded = ft._pad(
                block,
                ft.CONTEXT,
                ft.LOSS_HORIZON_DAYS,
                ft.DAYS_PER_STEP[SOURCE],
            )
            levels = ft.target_levels(view)
            padded["levels"] = tuple(levels[k] for k in block["target_places"])
            padded["places"] = view
            padded["target_places"] = list(block["target_places"])
            padded["parent"] = "exact"
            padded["mix"] = "germany"
            padded["source"] = SOURCE
            padded["cut"] = cut
            out.append(padded)
    return out


def _cut(args: tuple[pd.DataFrame, str, int, str]) -> list[dict[str, Any]]:
    counts, cut, i, truth_until = args
    return german_blocks(counts, [cut], truth_until, first=i)


def make_blocks(
    triangle: Path,
    out: Path,
    first_date: str,
    truth_delay: int,
    workers: int = 4,
) -> int:
    """Cut the German blocks of `window` and pickle them in cut order."""
    as_of, truth_until = window(first_date, truth_delay)
    counts = german_counts(pd.read_csv(triangle), as_of)
    cuts = cut_dates(FIRST_CUT, truth_until)
    jobs = [(counts, c, i, truth_until) for i, c in enumerate(cuts)]
    with ProcessPoolExecutor(workers) as pool:
        blocks = [b for part in pool.map(_cut, jobs) for b in part]
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    print(
        f"{len(blocks)} blocks of {len(cuts)} cuts {cuts[0]} to {cuts[-1]}, "
        f"reports to {as_of}, truth to {truth_until}: {out}",
        flush=True,
    )
    return len(blocks)


def main() -> None:
    """Cut the German blocks of the window the options give."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triangle", type=Path)
    parser.add_argument("out", type=Path)
    # The first scored German nowcast date and the benchmark's truth
    # delay (`config/germany.toml` `nowcast.first_date`, `truth.delay`).
    parser.add_argument("--first-date", required=True)
    parser.add_argument("--truth-delay", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    make_blocks(
        args.triangle,
        args.out,
        args.first_date,
        args.truth_delay,
        args.workers,
    )


if __name__ == "__main__":
    main()
