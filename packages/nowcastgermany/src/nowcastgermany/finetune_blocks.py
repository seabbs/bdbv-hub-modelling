"""German real-data training blocks, for the real-data runs.

The German hub triangle cut into training blocks as the German tasks lay
it out. Two runs read them (workflow/config.yaml `runs`):

- `realdata` mixes them into Run A's simulations. Its blocks use reports
  before the first scored German date only (`window`), so the run is
  never scored on what it trained on. Run as `python -m
  nowcastgermany.finetune_blocks TRIANGLE OUT --first-date D
  --truth-delay N`, the window's two settings from
  `packages/nowcastgermany/config/germany.toml` (rule
  real_training_blocks).
- `realonly` trains on real data alone (`foundation_real`). Its blocks
  use the data as available on the last day of the development half,
  from the daily and the weekly releases (`realdata_window`, the table
  `[realdata_germany]` of packages/nowcastfoundation/config/finetune.toml).
  Run as `python -m nowcastgermany.finetune_blocks TRIANGLE OUT
  --realdata` (rule realonly_blocks_germany).
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

import pandas as pd

from nowcastfoundation.finetune import AHEAD_WEEKS, training_block
from nowcastfoundation.finetune import chronos as ft
from nowcastgermany import data as de
from nowcastutils.base import NATIONAL

SOURCE = "daily+rolling7"
"""The blocks' source: daily counts with each place's 7-day rolling sums,
as the German models lay out the daily releases (`forecaster.PERIODS`)."""

LAGS = (14, 21, 28, 35, 40, 49)
"""Release lags $D$ in days, taken in turn over the cuts: the ragged
models' default (28 days), the German truth delay (40 days) and the
simulations' range around them."""

WEEKLY_LAGS = (2, 3, 4, 5, 6, 7)
"""Release lags $D$ in weeks of the weekly blocks, taken in turn over the
cuts: the ragged models' default (4 weeks) and the range around it."""

PLACES = ("national", "states", "ages")
"""The blocks' places: Germany alone, or Germany with its states or its
age groups, as the German tasks group them."""

REAL_PLACES = ("states", "ages")
"""The real-only checkpoint's blocks' places: each German task whole,
every level in one block as the fine-tuned model forecasts it (Germany
with its states, or with its age groups)."""

FIRST_CUT = "2021-04-20"
"""The first nowcast date cut, two weeks after the triangle's first
Meldedatum; cuts with under two steps of context are skipped."""

FINETUNE_CONFIG = ft.ROOT / "packages/nowcastfoundation/config/finetune.toml"
"""The fine-tuning settings, with the real-data window `[realdata_germany]`."""


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
    sources : tuple of str
        The blocks' sources: `SOURCE` (the daily releases) and `weekly`.
    held_out_dates : int
        The latest nowcast dates held out to monitor the loss.
    """

    as_of: str
    truth_delay: int
    first_cut: str
    sources: tuple[str, ...]
    held_out_dates: int

    @property
    def places(self) -> tuple[str, ...]:
        """The blocks' places, each German task whole (`REAL_PLACES`)."""
        return REAL_PLACES

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
        sources=tuple(raw["sources"]),
        held_out_dates=int(raw["held_out_dates"]),
    )


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
    sources: tuple[str, ...] = (SOURCE,),
    places: tuple[str, ...] = PLACES,
) -> list[dict[str, Any]]:
    """Return the training blocks of the German data at nowcast dates.

    Each cut is laid out as the German tasks lay it out, on its real
    dates, and Germany is the country (`parent` exact). Each source and
    each of `places` (`PLACES` by default) gives a block by
    `finetune.training_block`, at a lag taken in turn (from cut `first`
    on), padded as the simulated blocks are. `SOURCE` takes every report date up to the cut as a release, at
    a lag from `LAGS`; `weekly` takes a release a week, on the cut's
    weekday, at a lag from `WEEKLY_LAGS`.

    Raises
    ------
    ValueError
        If a cut's horizon ends after `truth_until`.
    """
    every = sorted(counts["location"].unique())
    states = [p for p in every if "-A" not in p]
    ages = [p for p in every if "-A" in p]
    leaves = {"national": states, "states": states, "ages": ages}
    out = []
    for i, cut in enumerate(cuts, start=first):
        end = dt.date.fromisoformat(cut) + dt.timedelta(weeks=AHEAD_WEEKS)
        if end.isoformat() > truth_until:
            raise ValueError(
                f"cut {cut} needs truth to {end}, past the truth date "
                f"{truth_until}"
            )
        daily = sorted(d for d in counts["report_date"].unique() if d <= cut)
        for source in sources:
            weekly = source == "weekly"
            lags = WEEKLY_LAGS if weekly else LAGS
            # Weekly releases fall on the cut's weekday, back from the cut.
            cutoffs = daily[::-7][::-1] if weekly else daily
            for j, which in enumerate(places):
                view = (NATIONAL,)
                if which != "national":
                    view = (NATIONAL, *leaves[which])
                own = counts.loc[counts["location"].isin(leaves[which])]
                lag = lags[(i + j) % len(lags)]
                block = training_block(own, source, view, cutoffs, max_lag=lag)
                if block is None or block["start"] < 2:
                    continue
                padded = ft._pad(
                    block,
                    ft.CONTEXT,
                    ft.LOSS_HORIZON_DAYS,
                    ft.DAYS_PER_STEP[source],
                )
                levels = ft.target_levels(view)
                padded["levels"] = tuple(
                    levels[k] for k in block["target_places"]
                )
                padded["places"] = view
                padded["target_places"] = list(block["target_places"])
                padded["parent"] = "exact"
                padded["mix"] = "germany"
                padded["data"] = "germany"
                padded["source"] = source
                padded["cut"] = cut
                padded["lag"] = lag
                padded["truth_end"] = end.isoformat()
                out.append(padded)
    return out


def _cut(
    args: tuple[pd.DataFrame, str, int, str, tuple[str, ...], tuple[str, ...]],
) -> list[dict[str, Any]]:
    counts, cut, i, truth_until, sources, places = args
    return german_blocks(
        counts, [cut], truth_until, first=i, sources=sources, places=places
    )


def _cut_all(
    counts: pd.DataFrame,
    cuts: list[str],
    truth_until: str,
    sources: tuple[str, ...],
    workers: int,
    places: tuple[str, ...] = PLACES,
) -> list[dict[str, Any]]:
    """Cut every cut's blocks over `workers` processes, in cut order."""
    jobs = [
        (counts, c, i, truth_until, sources, places) for i, c in enumerate(cuts)
    ]
    with ProcessPoolExecutor(workers) as pool:
        return [b for part in pool.map(_cut, jobs) for b in part]


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
    blocks = _cut_all(counts, cuts, truth_until, (SOURCE,), workers)
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    print(
        f"{len(blocks)} blocks of {len(cuts)} cuts {cuts[0]} to {cuts[-1]}, "
        f"reports to {as_of}, truth to {truth_until}: {out}",
        flush=True,
    )
    return len(blocks)


def make_realdata_blocks(
    triangle: Path,
    out: Path,
    real: RealdataWindow,
    workers: int = 4,
) -> int:
    """Cut the real-data checkpoint's German blocks and pickle them.

    The triangle as available on the window's `as_of`, cut at each of its
    nowcast dates from each of its sources, in cut order.
    """
    counts = german_counts(pd.read_csv(triangle), real.as_of)
    cuts = real.cuts()
    blocks = _cut_all(
        counts, cuts, real.truth_until, real.sources, workers, real.places
    )
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    print(
        f"{len(blocks)} blocks of {len(cuts)} cuts {cuts[0]} to {cuts[-1]} "
        f"from {', '.join(real.sources)}, reports to {real.as_of}, "
        f"truth to {real.truth_until}: {out}",
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
    parser.add_argument("--first-date")
    parser.add_argument("--truth-delay", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.realdata:
        make_realdata_blocks(
            args.triangle, args.out, realdata_window(), args.workers
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
