"""BDBV real-data training blocks, for the real-only checkpoint.

Every public BDBV release up to the one the real-data checkpoint trains
on (`realdata_cut`, the table `[realdata_bdbv]` of
packages/nowcastfoundation/config/finetune.toml) is one slice: its task,
built as the Ebola forecasts build it (`backtest.vintage_task`, with the
health zones), laid out as the fine-tuned model forecasts it
(`finetune.realtasks`): the daily releases with each place's 7-day rows,
the joint block of the country, its provinces and the active zones, and a
block per province of the province and its zones, each cut into views
over the block limit. The truth is the cut's release's counts, masked
after `truth_until`, so every count trained on is at least the BDBV truth
delay old and no target the SI scores the model on is trained on.

    python -m nowcastbdbv.finetune_blocks DATA OUT

DATA is the hub's tidy data (`bdbvhub data`). The workflow's rule
realonly_blocks_bdbv runs it.
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

from nowcastbdbv.backtest import vintage_task
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcastbdbv.data.zones import ZoneData, hub_places, load_zone_data
from nowcasteval.backtest import BacktestConfig, load_config
from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import realtasks as rt

FINETUNE_CONFIG = ft.ROOT / "packages/nowcastfoundation/config/finetune.toml"
"""The fine-tuning settings, with the BDBV cut `[realdata_bdbv]`."""

BACKTEST_CONFIG = ft.ROOT / "packages/nowcastbdbv/config/backtest.toml"
"""The Ebola forecasts' settings: each task's weeks."""

BLOCK_LIMIT = rt.BLOCK_LIMIT
"""Most series in a block, over which the model cuts it into views."""


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
        The last onset day a truth count may hold.
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


@dataclass(frozen=True)
class Inputs:
    """The hub's tidy tables the Ebola tasks are built from."""

    target: pd.DataFrame
    delays: pd.DataFrame
    daily: pd.DataFrame
    zones: ZoneData
    places: pd.DataFrame
    config: BacktestConfig


def load_inputs(data: Path | str) -> Inputs:
    """Return the hub's tidy tables in `data` and the forecasts' settings."""
    return Inputs(
        load_target(data),
        load_delays(data),
        load_daily(data),
        load_zone_data(data),
        hub_places(data),
        load_config(BACKTEST_CONFIG),
    )


def releases(inputs: Inputs, cut: RealdataCut) -> list[str]:
    """Return every real release up to the cut, oldest first."""
    real = inputs.target.loc[inputs.target["kind"] == "real", "vintage"]
    return sorted({str(v) for v in real if str(v) <= cut.release})


def _task(inputs: Inputs, vintage: str) -> Any:
    task, _ = vintage_task(
        inputs.target,
        inputs.delays,
        inputs.config,
        vintage,
        inputs.daily,
        zones=inputs.zones,
        places=inputs.places,
    )
    return task


def realdata_blocks(
    inputs: Inputs,
    cut: RealdataCut,
    block_limit: int = BLOCK_LIMIT,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Return the blocks of every release up to the cut, oldest first.

    Each release's task is laid out once as the fine-tuned model forecasts
    it, with the cut's release's counts to `cut.truth_until` as the truth.
    """
    model = rt.real_model(block_limit)
    truth = rt.truth_of(model, _task(inputs, cut.release), cut.truth_until)
    rng = np.random.default_rng(seed)
    out = []
    for vintage in releases(inputs, cut):
        task = _task(inputs, vintage)
        out += rt.task_blocks(
            model, task, truth, rng, data="bdbv", mix="bdbv", cut=vintage
        )
    return out


def make_blocks(
    data: Path, out: Path, cut: RealdataCut, limit: int = BLOCK_LIMIT
) -> int:
    """Cut the BDBV blocks and pickle them, oldest release first."""
    inputs = load_inputs(data)
    blocks = realdata_blocks(inputs, cut, limit)
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    print(
        f"{len(blocks)} BDBV blocks of {len(releases(inputs, cut))} releases "
        f"to {cut.release}, truth to {cut.truth_until}: {out}",
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
