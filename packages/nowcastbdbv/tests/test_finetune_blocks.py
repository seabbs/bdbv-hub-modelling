"""The BDBV real-data training blocks of the real-only checkpoint."""

import tomllib

import numpy as np
import pandas as pd
import pytest
from bdbv_cases import FIXTURE_DATA, ROOT

from nowcastbdbv import finetune_blocks as fb
from nowcastfoundation.finetune.stream import BLOCK_KEYS


def _config() -> dict:
    path = ROOT / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    return tomllib.loads(path.read_text())["realdata_bdbv"]


def test_the_cut_is_the_configs_release_and_truth_date():
    cut = fb.realdata_cut()
    raw = _config()
    assert cut.release == raw["release"] == "2026-11-08"
    assert cut.cutoff == raw["cutoff"]
    assert cut.truth_until == raw["truth_until"] == "2026-10-01"


def test_inputs_hold_no_release_after_the_cut():
    cut = fb.realdata_cut()
    daily, delays = fb.load_inputs(FIXTURE_DATA, cut)
    assert daily["vintage"].max() <= cut.release
    assert daily["cutoff"].max() <= cut.cutoff
    assert set(delays["vintage"]) == {cut.release}


def test_release_counts_give_each_release_back():
    # The daily releases are the health zones', which sum to the
    # provinces' and so to the country's.
    daily, _ = fb.load_inputs(FIXTURE_DATA, fb.realdata_cut())
    assert (daily["location"].str.count("-") == 2).all()
    counts = fb.release_counts(daily)
    for cutoff, rows in daily.groupby("cutoff"):
        want = (rows["observed"] + rows["imputed"]).sum()
        got = counts.loc[counts["report_date"] <= cutoff, "count"].sum()
        assert got == want


def _late(block: dict, truth_until: str) -> np.ndarray:
    """The targets' values at steps after the truth date."""
    late = [i for i, t in enumerate(block["times"]) if t > truth_until]
    return block["context"][: block["n_targets"], late]


def _blocks(source: str) -> list[dict]:
    cut = fb.realdata_cut()
    daily, delays = fb.load_inputs(FIXTURE_DATA, cut)
    if source == "daily+rolling7":
        return fb.daily_blocks(daily, cut.truth_until)
    return fb.weekly_blocks(delays, cut.cutoff, cut.truth_until)


@pytest.mark.slow
def test_daily_blocks_are_the_forecasting_models_layout():
    # As the fine-tuned model forecasts the hub: the daily releases with
    # each place's 7-day rows, every level in one block (the country, its
    # provinces and their zones), cut into the simulated blocks' views
    # where a whole system is over the block limit.
    blocks = _blocks("daily+rolling7")
    assert blocks
    for b in blocks:
        assert b["source"] == "daily+rolling7"
        assert len(b["places"]) > 1
        assert b["context"].shape[0] <= fb.BLOCK_LIMIT
        # Each target place has its daily row and its 7-day row.
        assert b["n_targets"] == 2 * len(set(b["target_places"]))
    levels = {lv for b in blocks for lv in b["levels"]}
    assert levels == {"national", "intermediate", "leaf"}
    whole = [b for b in blocks if b["view"] == "whole"]
    assert whole
    for b in whole:
        zones = [p for p in b["places"] if p.count("-") == 2]
        provinces = {z.rsplit("-", 1)[0] for z in zones}
        assert set(b["places"]) == {"CD", *provinces, *zones}


@pytest.mark.slow
def test_weekly_blocks_hold_the_country_with_its_provinces():
    blocks = _blocks("weekly")
    assert blocks
    for b in blocks:
        assert b["places"][0] == "CD"
        assert len(b["places"]) > 1
        assert all(p.count("-") == 1 for p in b["places"][1:])


@pytest.mark.slow
@pytest.mark.parametrize("source", ["daily+rolling7", "weekly"])
def test_no_bdbv_truth_after_the_truth_date(source):
    cut = fb.realdata_cut()
    blocks = _blocks(source)
    assert blocks
    for b in blocks:
        assert set(BLOCK_KEYS) <= set(b)
        assert (b["data"], b["mix"], b["source"]) == ("bdbv", "bdbv", source)
        # A province's view of its zones is under the province.
        assert b["places"][0] == "CD" or b["view"] == "subtree"
        assert b["cut"] <= cut.cutoff
        assert b["truth_end"] <= cut.truth_until
        assert len(b["times"]) == b["context"].shape[1]
        assert np.isnan(_late(b, cut.truth_until)).all()
        # Steps up to the truth date keep their counts.
        kept = [t for t in b["times"] if t and t <= cut.truth_until]
        assert kept


def test_the_times_end_at_the_horizon(monkeypatch):
    # A block's steps run to four weeks after its cut-off's week, as
    # `training_block` lays them out.
    times = fb.block_times(3, 2, "daily", "2026-10-07")
    assert times == [
        "2026-11-03",
        "2026-11-04",
        "2026-11-05",
        "2026-11-06",
        "2026-11-07",
    ]
    weeks = fb.block_times(2, 1, "weekly", "2026-10-07")
    assert weeks == ["2026-10-24", "2026-10-31", "2026-11-07"]
    assert pd.Timestamp(weeks[-1]).day_name() == "Saturday"
