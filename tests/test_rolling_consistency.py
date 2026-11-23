"""The held-out check that a block's rolling-sum row matches its daily rows."""

import numpy as np
import pytest

from nowcastfoundation.finetune import evaluate as fe

H = 28
PAST = 14


def rolling_block(daily: np.ndarray) -> dict:
    """A one-place daily+rolling7 block over `daily`, the last `H` future."""
    sums = np.convolve(daily, np.ones(7))[: len(daily)]
    return {
        "context": np.stack([daily, sums]).astype(np.float32),
        "n_targets": 2,
        "prediction_length": H,
        "source": "daily+rolling7",
    }


def weekly_pattern(n: int) -> np.ndarray:
    """Daily counts with a day-of-week cycle around a slow rise."""
    t = np.arange(n)
    return (100 + t) * (1 + 0.5 * np.sin(2 * np.pi * t / 7))


def test_a_rolling_row_that_sums_its_daily_row_is_consistent():
    daily = weekly_pattern(PAST + H)
    block = rolling_block(daily)
    median = block["context"][:, -H:]
    got = fe.rolling_consistency(block, median)
    assert got["gap"] == pytest.approx(0, abs=1e-3 * got["scale"])
    # The sums of a weekly cycle are smooth, so the row is too.
    assert got["rough"] == pytest.approx(got["rough_daily_sum"], rel=1e-3)
    assert got["rough"] < 0.1 * np.abs(np.diff(daily[-H:], 2)).sum()


def test_a_rolling_row_that_copies_the_daily_cycle_is_flagged():
    daily = weekly_pattern(PAST + H)
    block = rolling_block(daily)
    median = block["context"][:, -H:].copy()
    # Leakage: the rolling row moves with the daily row's weekday swings.
    median[1] = 7 * daily[-H:]
    got = fe.rolling_consistency(block, median)
    assert got["gap"] > 0.05 * got["scale"]
    assert got["rough"] > 5 * got["rough_daily_sum"]
    assert got["rough_truth"] == pytest.approx(got["rough_daily_sum"], rel=1e-3)


def test_blocks_without_rolling_rows_are_skipped():
    block = {**rolling_block(weekly_pattern(PAST + H)), "source": "daily"}
    assert fe.rolling_consistency(block, block["context"][:, -H:]) is None


def test_the_check_sums_over_blocks_by_source():
    daily = weekly_pattern(PAST + H)
    block = rolling_block(daily)
    leaky = block["context"][:, -H:].copy()
    leaky[1] = 7 * daily[-H:]
    parts = [
        fe.rolling_consistency(block, block["context"][:, -H:]),
        fe.rolling_consistency(block, leaky),
    ]
    got = fe.summarise_rolling([("daily+rolling7", p) for p in parts])
    assert got["daily+rolling7_n"] == 2 * H
    assert 0 < got["daily+rolling7_gap"] < 1
    assert got["daily+rolling7_rough"] > got["daily+rolling7_rough_daily_sum"]
