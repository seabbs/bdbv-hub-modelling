"""The 7-day check: a block's 7-day row against its daily rows' sums.

Per checkpoint on held-out sims, (i) coherence, the
7-day row's forecast against the 7-day sums of the same block's daily
forecast (median ratio, and how often the sums fall in the row's 90%
interval), and (ii) weekday leak, the roughness of the 7-day median
(mean absolute second difference of its log) against the truth's.
"""

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


def _check(block, median, width=0.1):
    return fe.seven_day_parts(
        block, median, median * (1 - width), median * (1 + width)
    )


def test_a_7_day_row_that_sums_its_daily_row_is_coherent_and_smooth():
    block = rolling_block(weekly_pattern(PAST + H))
    got = fe.summarise_seven_day([_check(block, block["context"][:, -H:])])
    assert got["ratio"] == pytest.approx(1, abs=1e-3)
    assert got["inside90"] == 1
    assert got["rough_ratio"] == pytest.approx(1, abs=0.05)
    assert got["n"] == H


def test_a_7_day_row_that_copies_the_daily_cycle_is_flagged():
    daily = weekly_pattern(PAST + H)
    block = rolling_block(daily)
    median = block["context"][:, -H:].copy()
    # Leakage: the 7-day row moves with the daily row's weekday swings.
    median[1] = 7 * daily[-H:]
    got = fe.summarise_seven_day([_check(block, median, 0.02)])
    assert got["rough_ratio"] > 3
    assert got["inside90"] < 0.5


def test_blocks_without_7_day_rows_are_skipped():
    block = {**rolling_block(weekly_pattern(PAST + H)), "source": "daily"}
    assert _check(block, block["context"][:, -H:]) is None
    assert fe.summarise_seven_day([None]) == {}


def test_steps_past_the_truth_are_left_out():
    block = rolling_block(weekly_pattern(PAST + H))
    block["context"][:, -3:] = np.nan
    median = np.nan_to_num(block["context"][:, -H:], nan=1.0)
    got = fe.summarise_seven_day([_check(block, median)])
    assert got["n"] == H - 3
