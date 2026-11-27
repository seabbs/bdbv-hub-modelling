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
    daily, _ = fb.load_inputs(FIXTURE_DATA, fb.realdata_cut())
    provinces = daily.loc[daily["location"].str.count("-") == 1]
    counts = fb.release_counts(provinces)
    for cutoff, rows in provinces.groupby("cutoff"):
        want = (rows["observed"] + rows["imputed"]).sum()
        got = counts.loc[counts["report_date"] <= cutoff, "count"].sum()
        assert got == want


def _late(block: dict, truth_until: str) -> np.ndarray:
    """The targets' values at steps after the truth date."""
    late = [i for i, t in enumerate(block["times"]) if t > truth_until]
    return block["context"][: block["n_targets"], late]


@pytest.mark.slow
@pytest.mark.parametrize("source", ["daily", "weekly"])
def test_no_bdbv_truth_after_the_truth_date(source):
    cut = fb.realdata_cut()
    daily, delays = fb.load_inputs(FIXTURE_DATA, cut)
    if source == "daily":
        blocks = fb.daily_blocks(daily, cut.truth_until)
    else:
        blocks = fb.weekly_blocks(delays, cut.cutoff, cut.truth_until)
    assert blocks
    for b in blocks:
        assert set(BLOCK_KEYS) <= set(b)
        assert (b["data"], b["mix"], b["source"]) == ("bdbv", "bdbv", source)
        assert b["places"][0] == "CD"
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
