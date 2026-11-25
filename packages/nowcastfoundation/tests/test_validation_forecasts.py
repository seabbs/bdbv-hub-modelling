"""Validation forecasts: a sample of held-out blocks' forecasts to view.

Each checkpoint's (and base Chronos-2's) forecasts of a fixed sample of
the run's held-out blocks, with the blocks' recent history and truth,
for the dashboard to set against the truth as it does German nowcasts.
"""

import numpy as np

from nowcastfoundation.finetune import evaluate as fe

H = 8


def _block(rows=3, levels=("national", "leaf", "leaf")):
    ctx = np.arange(rows * 20, dtype=np.float32).reshape(rows, 20)
    # A gap three steps before the horizon.
    ctx[:, 20 - H - 3] = np.nan
    return {
        "context": ctx,
        "n_targets": rows,
        "prediction_length": H,
        "nowcast": 3,
        "n_covariates": 0,
        "levels": levels[:rows],
        "places": ("DE", "P01", "P02"),
        "target_places": [0, 1, 2][:rows],
        "source": "daily",
        "cut_date": 120,
        "mix": "all_levels",
    }


def test_the_sample_is_spread_over_the_blocks_and_fixed():
    got = fe.validation_sample(100, 20)
    assert len(got) == 20
    assert got[0] == 0
    assert got == sorted(set(got))
    assert got == fe.validation_sample(100, 20)
    assert fe.validation_sample(5, 20) == [0, 1, 2, 3, 4]


def test_a_record_holds_history_truth_and_the_seven_levels():
    block = _block()
    q = np.tile(np.arange(7, dtype=float)[None, :, None], (3, 1, H))
    target = np.full((3, H), 5.0)
    target[:, -2:] = np.nan
    got = fe.forecast_record(7, block, q, target, tail=6)
    assert got["block"] == 7
    assert got["source"] == "daily"
    assert got["cut"] == 120
    assert got["mix"] == "all_levels"
    assert got["nowcast"] == 3
    assert [r["name"] for r in got["rows"]] == ["DE", "P01", "P02"]
    assert [r["level"] for r in got["rows"]] == ["national", "leaf", "leaf"]
    row = got["rows"][0]
    # The last 6 context steps before the horizon, a gap as None.
    assert len(row["history"]) == 6
    assert row["history"][3] is None
    assert row["truth"] == [5.0] * (H - 2) + [None, None]
    assert len(row["q"]) == 7
    assert row["q"][3] == [3.0] * H


def test_a_large_block_keeps_its_national_row_first_and_a_few_others():
    rows = fe.VALIDATION_ROWS + 5
    block = _block(rows=3)
    block["n_targets"] = rows
    block["context"] = np.ones((rows, 20), dtype=np.float32)
    block["levels"] = (*("leaf",) * (rows - 1), "national")
    block["target_places"] = list(range(rows))
    block["places"] = (*(f"P{i}" for i in range(rows - 1)), "DE")
    q = np.ones((rows, 7, H))
    got = fe.forecast_record(0, block, q, np.ones((rows, H)), tail=4)
    assert len(got["rows"]) == fe.VALIDATION_ROWS
    assert got["rows"][0]["name"] == "DE"


def test_the_levels_are_the_page_s_seven():
    seven = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    assert tuple(fe.VALIDATION_LEVELS) == seven
