"""The realised combinations of block layout and options (SI table)."""

import pandas as pd

from nowcastfoundation.finetune import combinations


def _block(source, mix, parent="exact"):
    return {"source": source, "mix": mix, "parent": parent}


def test_each_block_counts_once_under_its_combination():
    seen = {}
    combinations.count(
        seen,
        [
            _block("daily", "all_levels"),
            _block("daily", "all_levels"),
            _block("daily+weekly", "national"),
            _block("weekly", "leaves_national", "late_parent"),
        ],
    )
    combinations.count(seen, [_block("daily", "all_levels")])
    assert seen["combinations"] == {
        "daily|all_levels|exact": 3,
        "daily+weekly|national|exact": 1,
        "weekly|leaves_national|late_parent": 1,
    }


def test_shares_are_of_every_block_largest_first():
    got = combinations.shares(
        {
            "weekly|national|exact": 1,
            "daily|all_levels|exact": 3,
        }
    )
    assert list(got.columns) == [
        "layout",
        "levels",
        "parents",
        "blocks",
        "share",
    ]
    assert got["layout"].tolist() == ["daily", "weekly"]
    assert got["share"].tolist() == [0.75, 0.25]
    pd.testing.assert_series_equal(
        got["blocks"], pd.Series([3, 1], name="blocks")
    )
