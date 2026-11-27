"""BDBV real-data training blocks, for the real-only checkpoint.

The public BDBV data up to the release the real-data checkpoint trains
on (`realdata_cut`, the table `[realdata_bdbv]` of
packages/nowcastfoundation/config/finetune.toml), cut into training
blocks in the main fine-tuned model's layout, as it forecasts the hub:
the reporting triangle with its lag series, every level in one block
and, for daily releases, each place's 7-day rows. Daily blocks are cut
at each daily release from the health zones' releases (`daily_blocks`):
the country, its provinces and their zones in one block, each total
summed from the zones, cut into the simulated blocks' views
(`finetune.chronos.system_views`) where over the block limit. Weekly
blocks are cut at each Saturday from the hub's public per-case delay
table (`weekly_blocks`): the country with its provinces. Each is cut at
every release lag. The truth is the latest release's counts, masked
after `truth_until`, so every count trained on is at least the BDBV
truth delay old and no target the SI scores the model on is trained on.

    python -m nowcastbdbv.finetune_blocks DATA OUT

DATA is the hub's tidy data (`bdbvhub data`: `daily_zones.csv.gz`,
`delays.csv.gz`). The workflow's rule realonly_blocks_bdbv runs it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastfoundation.finetune import AHEAD_WEEKS, training_block
from nowcastfoundation.finetune import chronos as ft
from nowcastutils.base import NATIONAL
from nowcastutils.dates import epiweek_end

FINETUNE_CONFIG = ft.ROOT / "packages/nowcastfoundation/config/finetune.toml"
"""The fine-tuning settings, with the BDBV cut `[realdata_bdbv]`."""

DAILY = "daily+rolling7"
"""The daily blocks' source: daily counts with each place's 7-day rolling
sums, the fine-tuned model's layout of daily releases
(`forecaster.PERIODS`)."""

BLOCK_LIMIT = 1000
"""Most series in a block: the workflow's `finetune.block_limit`, over
which a whole system is cut into views as the simulated blocks are."""

LAGS = {DAILY: (14, 21, 28, 35, 40, 49), "weekly": (2, 3, 4, 5, 6, 7)}
"""Release lags $D$ of each source, in its steps, every one cut at each
cut: the German blocks' lags (`nowcastgermany.finetune_blocks`)."""


@dataclass(frozen=True)
class RealdataCut:
    """The BDBV data the real-data checkpoint trains on.

    Attributes
    ----------
    release : str
        The last release trained on.
    cutoff : str
        Its cut-off.
    truth_until : str
        The last onset day or week end a truth count may hold.
    """

    release: str
    cutoff: str
    truth_until: str


def realdata_cut(path: Path | str = FINETUNE_CONFIG) -> RealdataCut:
    """Return the real-data checkpoint's BDBV cut (`[realdata_bdbv]`).

    Raises
    ------
    ValueError
        If `truth_until` is not the cut-off less `truth_delay` days.
    """
    raw = tomllib.loads(Path(path).read_text())["realdata_bdbv"]
    cutoff = dt.date.fromisoformat(raw["cutoff"])
    truth = (cutoff - dt.timedelta(days=int(raw["truth_delay"]))).isoformat()
    if truth != raw["truth_until"]:
        raise ValueError(
            f"truth_until {raw['truth_until']} is not the cut-off less "
            f"{raw['truth_delay']} days, {truth}"
        )
    return RealdataCut(str(raw["release"]), str(raw["cutoff"]), truth)


def load_inputs(
    data: Path | str, cut: RealdataCut
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the zones' daily releases and the delay table up to the cut.

    Every daily release of the health zones up to `cut.release`, and the
    per-case delay table of that release.
    """
    data = Path(data)
    daily = pd.read_csv(data / "daily_zones.csv.gz", dtype=str)
    daily[["observed", "imputed"]] = daily[["observed", "imputed"]].astype(
        "int64"
    )
    daily = daily.loc[daily["vintage"] <= cut.release].reset_index(drop=True)
    delays = pd.read_csv(data / "delays.csv.gz", dtype={"delay": "int64"})
    delays = delays.loc[delays["vintage"] == cut.release].reset_index(drop=True)
    if delays.empty:
        raise ValueError(f"no delay table of the release {cut.release}")
    return daily, delays


def release_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Return daily releases as counts reported at each release's cut-off.

    `rows` are the hub's daily releases (`vintage`, `location`,
    `onset_date`, `observed`, `imputed`, `cutoff`) of the leaves. Each
    onset day's count at a cut-off, observed plus imputed (0 where a
    release has no row), less its count at the release before, is
    reported on that cut-off, so `finetune.releases` gives each release
    back.
    """
    value = rows.assign(value=rows["observed"] + rows["imputed"])
    wide = value.pivot_table(
        index=["location", "onset_date"],
        columns="cutoff",
        values="value",
        aggfunc="sum",
        fill_value=0,
    ).sort_index(axis=1)
    change = wide.diff(axis=1).fillna(wide)
    stacked = pd.Series(change.stack(), name="count")  # noqa: PD013
    long = stacked.reset_index()
    long = long.loc[long["count"] != 0]
    return pd.DataFrame(
        {
            "location": long["location"],
            "reference_date": long["onset_date"],
            "report_date": long["cutoff"],
            "count": long["count"].astype("int64"),
        }
    ).reset_index(drop=True)


def delay_counts(line_list: pd.DataFrame) -> pd.DataFrame:
    """Return a per-case delay table's counts by onset and report date.

    Each case of `line_list` (`location`, `onset_week`, the week's
    Saturday, and `delay`, days from onset to report) is reported on its
    onset week's end plus its delay.
    """
    onset = pd.to_datetime(line_list["onset_week"])
    report = onset + pd.to_timedelta(line_list["delay"].clip(lower=0), "D")
    return (
        pd.DataFrame(
            {
                "location": line_list["location"].to_numpy(),
                "reference_date": onset.dt.date.astype(str).to_numpy(),
                "report_date": report.dt.date.astype(str).to_numpy(),
                "count": 1,
            }
        )
        .groupby(["location", "reference_date", "report_date"], as_index=False)
        .sum()
    )


def block_times(start: int, horizon: int, source: str, cut: str) -> list[str]:
    """Return the step of each column of a block cut at `cut`.

    `training_block` lays its steps out to `AHEAD_WEEKS` weeks after the
    cut-off's week, a day or a week (`source`) apart.
    """
    end = epiweek_end(cut) + dt.timedelta(weeks=AHEAD_WEEKS)
    width = 7 if source == "weekly" else 1
    n = start + horizon
    return [
        (end - dt.timedelta(days=width * (n - 1 - k))).isoformat()
        for k in range(n)
    ]


def _short(place: str) -> str:
    """Return a place's name below the country (`CD-IT-01` is `IT-01`)."""
    return place.removeprefix(f"{NATIONAL}-")


def _hub(place: str) -> str:
    """Return a place's hub name (`IT-01` is `CD-IT-01`)."""
    return place if place == NATIONAL else f"{NATIONAL}-{place}"


def whole_system(leaves: list[str]) -> tuple[str, ...]:
    """Return every level of the leaves' system, as `chronos.whole_system`.

    The country, each group's total (a zone's province) and the leaves,
    named below the country, so `finetune.releases` sums each total.
    """
    short = sorted(_short(p) for p in leaves)
    groups = sorted({p.split("-")[0] for p in short if "-" in p})
    return (NATIONAL, *groups, *short)


def _finish(
    block: dict[str, Any],
    source: str,
    view: tuple[str, ...],
    kind: str,
    cutoffs: list[str],
    lag: int,
    truth_until: str,
) -> dict[str, Any] | None:
    """Mask, pad and label one block; None if it has no truth left.

    The targets' steps after `truth_until` are set missing, which
    Chronos-2's loss leaves out.
    """
    start, n = block["start"], block["n_targets"]
    times = block_times(start, block["horizon"], source, cutoffs[-1])
    kept = [i for i in range(start, len(times)) if times[i] <= truth_until]
    if not kept:
        return None
    late = [i for i in range(len(times)) if times[i] > truth_until]
    block["context"][:n, late] = np.nan
    padded = ft._pad(
        block, ft.CONTEXT, ft.LOSS_HORIZON_DAYS, ft.DAYS_PER_STEP[source]
    )
    # The step of each padded column, "" where padded.
    keep = min(start, ft.CONTEXT)
    after = padded["context"].shape[1] - ft.CONTEXT - block["horizon"]
    padded["times"] = (
        [""] * (ft.CONTEXT - keep)
        + times[start - keep :]
        + [""] * max(0, after)
    )[: padded["context"].shape[1]]
    levels = ft.target_levels(view)
    padded["levels"] = tuple(levels[k] for k in block["target_places"])
    # Hub names, so each place's parent is its nearest prefix
    # (`series_bias.parents_of`): a zone's province, a province's country.
    padded["places"] = tuple(_hub(p) for p in view)
    padded["target_places"] = list(block["target_places"])
    padded["parent"] = "exact"
    padded["mix"] = "bdbv"
    padded["data"] = "bdbv"
    padded["source"] = source
    padded["view"] = kind
    padded["cut"] = cutoffs[-1]
    padded["lag"] = lag
    padded["truth_end"] = times[max(kept)]
    return padded


def system_blocks(
    counts: pd.DataFrame,
    source: str,
    cutoffs: list[str],
    lag: int,
    truth_until: str,
    limit: int = BLOCK_LIMIT,
) -> list[dict[str, Any]]:
    """Return the blocks of the whole system at a cut, as the main run's.

    Every level in one block (`whole_system`), or, over `limit` series,
    the views of `chronos.system_views` that each fit, as the simulated
    blocks are cut (`chronos.outbreak_blocks`).
    """
    upto = counts.loc[counts["report_date"] <= cutoffs[-1]]
    if upto.empty:
        return []
    short = upto.assign(location=upto["location"].map(_short))
    places = whole_system(list(upto["location"].unique()))
    totals = any("-" in p for p in places)

    def cut(view: tuple[str, ...]) -> dict[str, Any] | None:
        block = training_block(
            short, source, view, cutoffs, max_lag=lag, totals=totals
        )
        return None if block is None or block["start"] < 2 else block

    whole = cut(places)
    if whole is None:
        return []
    views: list[tuple[str, tuple[str, ...], dict[str, Any] | None]] = [
        ("whole", places, whole)
    ]
    series = int(whole["context"].shape[0])
    if series > limit:
        cap = max(1, limit * len(places) // series)
        while True:
            views = [
                (kind, view, cut(view))
                for kind, view, _ in ft.system_views(places, cap)
            ]
            fits = all(
                b is None or b["context"].shape[0] <= limit for _, _, b in views
            )
            if fits or cap == 1:
                break
            cap -= 1
    out = []
    for kind, view, block in views:
        if block is None or block["context"].shape[0] > limit:
            continue
        done = _finish(block, source, view, kind, cutoffs, lag, truth_until)
        if done is not None:
            out.append(done)
    return out


def daily_blocks(
    rows: pd.DataFrame,
    truth_until: str,
    cuts: list[str] | None = None,
    limit: int = BLOCK_LIMIT,
) -> list[dict[str, Any]]:
    """Return daily blocks, one system per release and lag.

    Each release's cut-off is a cut, with the zones' releases up to it as
    the model sees them and the latest release's counts as the truth,
    masked after `truth_until`. The provinces and the country are the
    zones' totals.
    """
    counts = release_counts(rows)
    cutoffs = sorted(rows["cutoff"].unique())
    out = []
    for cut in cuts or cutoffs:
        upto = [c for c in cutoffs if c <= cut]
        for lag in LAGS[DAILY]:
            out += system_blocks(counts, DAILY, upto, lag, truth_until, limit)
    return out


def weekly_blocks(
    line_list: pd.DataFrame,
    last_cutoff: str,
    truth_until: str,
    limit: int = BLOCK_LIMIT,
) -> list[dict[str, Any]]:
    """Return weekly blocks from the public per-case delay table.

    The counts are rebuilt from the delay table (`delay_counts`), with no
    report after `last_cutoff`. Each Saturday from the first onset week
    to `last_cutoff` is a cut, with weekly releases on Saturdays, the
    country with its provinces, at every lag, and the truth masked after
    `truth_until`.
    """
    from nowcastfoundation.finetune import weekly_cutoffs

    counts = delay_counts(line_list)
    counts = counts.loc[counts["report_date"] <= last_cutoff]
    first = str(counts["reference_date"].min())
    saturdays = weekly_cutoffs(first, last_cutoff, 5)
    out = []
    for i in range(len(saturdays)):
        for lag in LAGS["weekly"]:
            out += system_blocks(
                counts, "weekly", saturdays[: i + 1], lag, truth_until, limit
            )
    return out


def make_blocks(
    data: Path, out: Path, cut: RealdataCut, limit: int = BLOCK_LIMIT
) -> int:
    """Cut the daily and weekly blocks and pickle them, daily first."""
    daily, delays = load_inputs(data, cut)
    blocks = daily_blocks(daily, cut.truth_until, limit=limit)
    n_daily = len(blocks)
    blocks += weekly_blocks(delays, cut.cutoff, cut.truth_until, limit)
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    print(
        f"{len(blocks)} BDBV blocks ({n_daily} daily) to {out}, release "
        f"{cut.release}, truth to {cut.truth_until}",
        flush=True,
    )
    return len(blocks)


def main() -> None:
    """Cut the BDBV blocks of the configured cut."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="The tidy hub data.")
    parser.add_argument("out", type=Path)
    # Most series in a block (the run's `finetune.block_limit`).
    parser.add_argument("--block-limit", type=int, default=BLOCK_LIMIT)
    args = parser.parse_args()
    make_blocks(args.data, args.out, realdata_cut(), args.block_limit)


if __name__ == "__main__":
    main()
