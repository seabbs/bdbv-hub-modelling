"""The release layer: late release starts and cadences, with no resimulation.

Lead 10:35 (Sam): raggedness and weekly releases are drawn per block at
fine-tune time from the simulations already made. A block's releases
before its drawn start day are dropped, so its first kept release
backfills the onsets before it, its lag series start there and its counts
keep the whole history, as the BDBV hub's.
"""

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation.finetune import chronos as ft

DAYS = [d.date().isoformat() for d in pd.date_range("2026-01-03", "2026-05-01")]


def _counts():
    rows = [
        (place, onset, onset + pd.Timedelta(days=d), 5)
        for place in ("P01", "P02")
        for onset in pd.date_range("2026-01-03", periods=160)
        for d in (0, 2, 6)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    for col in ("reference_date", "report_date"):
        counts[col] = counts[col].dt.date.astype(str)
    return counts


def _blocks(layer, seed=0):
    old = ft.RELEASE_LAYER
    ft.RELEASE_LAYER = layer
    try:
        return ft.outbreak_blocks(
            _counts(), ["P01", "P02"], np.random.default_rng(seed), 1, 3
        )
    finally:
        ft.RELEASE_LAYER = old


START_ALL = {
    "start": {
        "share": 1.0,
        "delay_days": {"dist": "uniform", "lower": 40, "upper": 41},
    }
}


def test_a_late_start_drops_the_releases_before_it_and_keeps_the_cut():
    rng = np.random.default_rng(0)
    kept, delay = ft.release_start(DAYS, DAYS[0], START_ALL["start"], rng)
    assert delay == 40
    assert kept[0] == "2026-02-12"
    assert kept == [d for d in DAYS if d >= "2026-02-12"]
    assert kept[-1] == DAYS[-1]


def test_a_start_past_the_cut_leaves_the_cut_s_release_alone():
    spec = {
        "share": 1.0,
        "delay_days": {"dist": "uniform", "lower": 400, "upper": 401},
    }
    kept, delay = ft.release_start(
        DAYS, DAYS[0], spec, np.random.default_rng(0)
    )
    assert kept == DAYS[-1:]
    assert delay == 400


def test_most_blocks_start_on_time_at_the_drawn_share():
    spec = {
        "share": 0.3,
        "delay_days": {"dist": "uniform", "lower": 1, "upper": 182},
    }
    rng = np.random.default_rng(1)
    late = [
        ft.release_start(DAYS, DAYS[0], spec, rng)[1] > 0 for _ in range(4000)
    ]
    assert abs(np.mean(late) - 0.3) < 0.03


def test_with_no_late_starts_the_blocks_are_as_without_the_layer():
    # The layer draws from its own generator: every other draw, and every
    # later block's, is as before.
    off = _blocks(None)
    none_late = _blocks({"start": {**START_ALL["start"], "share": 0.0}})
    assert len(off) == len(none_late)
    for a, b in zip(off, none_late, strict=True):
        assert a["cadence"] == b["cadence"]
        np.testing.assert_array_equal(a["context"], b["context"])
        assert b["release_start_days"] == 0


def _filled(block):
    rows = block["context"][block["n_targets"] :]
    return np.isfinite(rows).sum()


def _history(block):
    """Return the first target's context, from its first onset."""
    row = block["context"][0, : ft.CONTEXT]
    return row[np.isfinite(row)]


def test_a_late_block_keeps_its_counts_and_loses_its_early_lag_cells():
    off = {(b["source"], b["cut"]): b for b in _blocks(None)}
    late = _blocks(START_ALL)
    assert late
    for b in late:
        a = off[(b["source"], b["cut"])]
        assert b["release_start_days"] == 40
        # A block cut before its start day has the cut's release alone.
        assert b["first_release"] >= "2026-02-12" or b["releases"] == 1
        # The counts row holds the whole history known at the cut, from
        # the first onset. Compared over the context both share: the lag
        # rule can give the late block another lag count, so its context
        # ends at another step, after which the row is the truth (#889).
        ha, hb = _history(a), _history(b)
        n = min(len(ha), len(hb))
        assert n > 0
        np.testing.assert_array_equal(ha[:n], hb[:n])
        if b["n_covariates"]:
            assert _filled(b) < _filled(a)


def test_the_layer_s_cadences_replace_the_block_design_s():
    layer = {"cadences": {"daily": {"weekly": 1.0}}}
    blocks = _blocks(layer)
    daily = [b for b in blocks if ft.DAYS_PER_STEP[b["source"]] == 1]
    assert daily
    assert {b["cadence"] for b in daily} == {"weekly"}


@pytest.mark.parametrize(
    "bad",
    [
        {
            "start": {
                "share": 1.5,
                "delay_days": {"dist": "uniform", "lower": 1, "upper": 2},
            }
        },
        {"start": {"share": 0.2, "delay_days": {"dist": "gamma"}}},
        {"cadences": {"daily": {"fortnightly": 1.0}}},
        {"cadences": {"daily": {"daily": 0.5}}},
        {"cadences": {"hourly": {"daily": 1.0}}},
        {"thin": 0.15},
    ],
)
def test_a_layer_that_cannot_be_drawn_is_refused(bad):
    with pytest.raises(ValueError, match="release layer"):
        ft.check_release_layer(bad)


def test_the_proposed_layer_file_is_valid():
    layer = ft.read_release_layer(
        ft.ROOT / "config" / "finetune" / "release-layer.yaml"
    )
    ft.check_release_layer(layer)
    # Daily-step blocks are 60% of all blocks; a quarter of them released
    # weekly is 15% of all blocks (lead 10:35).
    daily_share = sum(
        s for k, s in ft.SOURCE_SHARES.items() if ft.DAYS_PER_STEP[k] == 1
    )
    assert abs(daily_share * layer["cadences"]["daily"]["weekly"] - 0.15) < 0.01


def test_each_block_s_releases_are_recorded_under_a_layer(tmp_path):
    from nowcastfoundation.finetune import stream as fs

    assert fs.release_settings(None) == {}
    assert fs.release_settings(START_ALL) == {"release_layer": START_ALL}
    pairs = [("1-0001-1", b) for b in _blocks(START_ALL)]
    old = ft.RELEASE_LAYER
    try:
        ft.RELEASE_LAYER = None
        assert fs.record_releases(tmp_path, pairs, "held-out") == ""
        assert not (tmp_path / fs.RELEASES_FILE).exists()
        ft.RELEASE_LAYER = START_ALL
        line = fs.record_releases(tmp_path, pairs, "held-out")
        # A resumed run records the same blocks once.
        fs.record_releases(tmp_path, pairs, "held-out")
    finally:
        ft.RELEASE_LAYER = old
    assert "100% of" in line
    table = pd.read_csv(tmp_path / fs.RELEASES_FILE)
    assert len(table) == len(pairs)
    assert set(table["release_start_days"]) == {40}
