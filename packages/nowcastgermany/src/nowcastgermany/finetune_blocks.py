"""German real-data training blocks, for the real-data runs.

Two kinds of run read them (workflow/config.yaml `runs`):

- a run with `real_training: germany` mixes them into its simulations
  (no committed run does): the German hub triangle
  cut into training blocks as the German tasks lay it out, from reports
  before the first scored German date only (`window`). Run as `python -m
  nowcastgermany.finetune_blocks TRIANGLE OUT --first-date D
  --truth-delay N`, the window's two settings from
  `packages/nowcastgermany/config/germany.toml` (rule
  real_training_blocks).
- `realdata` trains on real data alone (`foundation_real`): each nowcast
  date's German tasks, as the benchmark builds them
  (`data.nowcast_tasks`), laid out as the fine-tuned model forecasts them
  (`finetune.realtasks`), from the data as available on the last day of
  the development half (`realdata_window`, the table `[realdata_germany]`
  of packages/nowcastfoundation/config/finetune.toml). Run as `python -m
  nowcastgermany.finetune_blocks TRIANGLE OUT --realdata` (rule
  realdata_blocks_germany).
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastfoundation.finetune import AHEAD_WEEKS, training_block
from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import realtasks as rt
from nowcastgermany import data as de
from nowcastutils.base import NATIONAL

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


def german_blocks(
    counts: pd.DataFrame,
    cuts: list[str],
    truth_until: str,
    first: int = 0,
) -> list[dict[str, Any]]:
    """Return the training blocks of the German data at nowcast dates.

    Each cut is laid out as the German tasks lay it out, on its real
    dates: every report date up to it is a release, and Germany is the
    country (`parent` exact). Each of `PLACES` gives a
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
        cutoffs = sorted(d for d in counts["report_date"].unique() if d <= cut)
        for j, which in enumerate(PLACES):
            view = (NATIONAL,)
            if which != "national":
                view = (NATIONAL, *leaves[which])
            own = counts.loc[counts["location"].isin(leaves[which])]
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


FINETUNE_CONFIG = ft.ROOT / "packages/nowcastfoundation/config/finetune.toml"
"""The fine-tuning settings, with the real-data window `[realdata_germany]`."""

GERMANY_CONFIG = ft.ROOT / "packages/nowcastgermany/config/germany.toml"
"""The benchmark's settings: its tasks' horizons (`forecast_days`)."""

MAX_DELAY = 80
"""The longest delay kept, in days: the hub's change after 80 days holds
revisions made up to August 2022, so it is left out."""


@dataclass(frozen=True)
class RealdataWindow:
    """The German data the real-data checkpoint trains on.

    Attributes
    ----------
    as_of : str
        The last report date used: the last day of the development half.
    truth_delay : int
        Days a count must have had by `as_of` to be truth.
    first_cut : str
        The first nowcast date cut.
    held_out_dates : int
        The latest nowcast dates held out to monitor the loss.
    """

    as_of: str
    truth_delay: int
    first_cut: str
    held_out_dates: int

    @property
    def truth_until(self) -> str:
        """The last Meldedatum complete for `truth_delay` days by `as_of`."""
        day = dt.date.fromisoformat(self.as_of)
        return (day - dt.timedelta(days=self.truth_delay)).isoformat()

    def cuts(self) -> list[str]:
        """Every nowcast date from `first_cut` with its truth by then."""
        return cut_dates(self.first_cut, self.truth_until)


def realdata_window(path: Path | str = FINETUNE_CONFIG) -> RealdataWindow:
    """Return the real-data checkpoint's German window.

    Read from the table `[realdata_germany]` of the fine-tuning settings.
    """
    raw = tomllib.loads(Path(path).read_text())["realdata_germany"]
    return RealdataWindow(
        as_of=str(raw["as_of"]),
        truth_delay=int(raw["truth_delay"]),
        first_cut=str(raw["first_cut"]),
        held_out_dates=int(raw["held_out_dates"]),
    )


def task_horizons(path: Path | str = GERMANY_CONFIG) -> tuple[int, ...]:
    """Return the horizons the checkpoint units forecast each date at.

    `forecast_days` ahead down to the hub's last nowcast horizon, as
    `nowcastgermany.evaluation.load_germany_config` reads them.
    """
    days = int(
        tomllib.loads(Path(path).read_text())["nowcast"]["forecast_days"]
    )
    return tuple(range(days, min(de.HORIZONS) - 1, -1))


def realdata_triangle(path: Path | str, as_of: str) -> pd.DataFrame:
    """Return the hub triangle as available on `as_of`.

    The counts by Meldedatum and report date (`data.hub_triangle`), those
    reported by `as_of` and at most `MAX_DELAY` days after their date.
    """
    tri = de.within_delay(de.hub_triangle(path), MAX_DELAY)
    return tri.loc[tri["report_date"] <= as_of].reset_index(drop=True)


def realdata_blocks(
    triangle: pd.DataFrame,
    window: RealdataWindow,
    cuts: list[str] | None = None,
    block_limit: int = rt.BLOCK_LIMIT,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Return the real-data checkpoint's German blocks, in cut order.

    Each nowcast date's tasks (`data.nowcast_tasks`: Germany with its
    states, and Germany with its age groups), laid out as the fine-tuned
    model forecasts them (`finetune.realtasks.task_blocks`), with the
    counts as available on `window.as_of` as the truth, to
    `window.truth_until`.
    """
    horizons = task_horizons()
    model = rt.real_model(block_limit)
    truths = [
        rt.truth_of(model, task, window.truth_until)
        for task in de.nowcast_tasks(triangle, window.as_of, 1, seed, horizons)
    ]
    rng = np.random.default_rng(seed)
    out = []
    for cut in cuts if cuts is not None else window.cuts():
        tasks = de.nowcast_tasks(triangle, cut, 1, seed, horizons)
        for task, truth in zip(tasks, truths, strict=True):
            out += rt.task_blocks(
                model, task, truth, rng, data="germany", mix="germany", cut=cut
            )
    return out


def make_realdata_blocks(
    path: Path,
    out: Path,
    window: RealdataWindow,
    block_limit: int = rt.BLOCK_LIMIT,
) -> int:
    """Cut the real-data checkpoint's German blocks and pickle them."""
    triangle = realdata_triangle(path, window.as_of)
    blocks = realdata_blocks(triangle, window, block_limit=block_limit)
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    cuts = window.cuts()
    print(
        f"{len(blocks)} blocks of {len(cuts)} nowcast dates {cuts[0]} to "
        f"{cuts[-1]}, reports to {window.as_of}, truth to "
        f"{window.truth_until}: {out}",
        flush=True,
    )
    return len(blocks)


def main() -> None:
    """Cut the German blocks of the window the options give."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triangle", type=Path)
    parser.add_argument("out", type=Path)
    # The real-data checkpoint's window (`realdata_window`), or the mixed
    # run's: the first scored German nowcast date and the benchmark's
    # truth delay (`packages/nowcastgermany/config/germany.toml`
    # `nowcast.first_date`, `truth.delay`).
    parser.add_argument("--realdata", action="store_true")
    parser.add_argument("--block-limit", type=int, default=rt.BLOCK_LIMIT)
    parser.add_argument("--first-date")
    parser.add_argument("--truth-delay", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.realdata:
        make_realdata_blocks(
            args.triangle, args.out, realdata_window(), args.block_limit
        )
        return
    if args.first_date is None or args.truth_delay is None:
        parser.error("give --realdata, or --first-date and --truth-delay")
    make_blocks(
        args.triangle,
        args.out,
        args.first_date,
        args.truth_delay,
        args.workers,
    )


if __name__ == "__main__":
    main()
