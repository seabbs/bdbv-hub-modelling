"""The BDBV real-data training blocks of the real-only checkpoint."""

import tomllib

import numpy as np
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


def test_every_release_to_the_cut_is_a_slice():
    # Every public release up to the cut, each cut once.
    inputs = fb.load_inputs(FIXTURE_DATA)
    releases = fb.releases(inputs, fb.realdata_cut())
    assert releases == sorted(set(releases))
    assert releases[-1] == "2026-11-08"
    assert all(r <= "2026-11-08" for r in releases)


@pytest.fixture(scope="module")
def blocks():
    cut = fb.realdata_cut()
    return fb.realdata_blocks(fb.load_inputs(FIXTURE_DATA), cut)


def _late(block: dict, truth_until: str) -> np.ndarray:
    """The targets' values at steps after the truth date."""
    late = [i for i, t in enumerate(block["times"]) if t > truth_until]
    return block["context"][: block["n_targets"], late]


@pytest.mark.slow
def test_no_bdbv_truth_after_the_truth_date(blocks):
    cut = fb.realdata_cut()
    assert blocks
    for b in blocks:
        assert set(BLOCK_KEYS) <= set(b)
        assert (b["data"], b["mix"]) == ("bdbv", "bdbv")
        assert b["cut"] <= cut.release
        assert b["truth_end"] <= cut.truth_until
        assert len(b["times"]) == b["context"].shape[1]
        assert np.isnan(_late(b, cut.truth_until)).all()


@pytest.mark.slow
def test_blocks_are_laid_out_as_the_ebola_forecast(blocks):
    # The forecasting path's layout: daily with 7-day rows, the joint block
    # of the country, provinces and active zones, and a block per province
    # of the province and its zones; no weekly-only blocks.
    assert {b["source"] for b in blocks} == {"daily+rolling7"}
    joint = [b for b in blocks if b["view"] == "joint"]
    provinces = [b for b in blocks if b["view"] == "province"]
    assert joint
    assert provinces
    for b in joint:
        assert b["places"][0] == "CD"
        assert any(p.count("-") == 1 for p in b["places"])
    for b in provinces:
        assert b["places"][0].count("-") == 1
        assert all(p.startswith(b["places"][0]) for p in b["places"])
    for b in blocks:
        assert b["n_targets"] == 2 * len(b["places"])
        assert b["context"].shape[0] <= fb.BLOCK_LIMIT
