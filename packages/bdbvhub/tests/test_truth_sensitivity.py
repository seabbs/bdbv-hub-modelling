"""How much the truth-final constants move what is scored."""

from __future__ import annotations

import pandas as pd
import pytest

from nowcasteval.nowcast import (
    complete_weeks,
    reporting_delay,
    truth_sensitivity,
)


def test_each_level_gives_its_truth_delay(
    target: pd.DataFrame, daily: pd.DataFrame
) -> None:
    got = truth_sensitivity(target, daily, levels=(0.9, 0.98), weeks=(4,))
    real = target.loc[target["kind"] == "real"]
    delays = got["delays"].set_index("level")["truth_delay"]
    for level in (0.9, 0.98):
        assert delays[level] == reporting_delay(real, level)


def test_complete_weeks_are_checked_against_the_latest_release(
    target: pd.DataFrame, daily: pd.DataFrame
) -> None:
    # Each earlier release's complete weeks against the latest counts:
    # the share later revised by more than 2% and by more than 10%.
    got = truth_sensitivity(
        target, daily, levels=(0.98,), weeks=(4, 6), growths=(0.02,)
    )
    rows = got["complete"].set_index(["weeks", "growth"])
    assert list(rows.columns) == [
        "releases",
        "complete",
        "revised_2pct",
        "revised_10pct",
    ]
    for key in ((4, 0.02), (6, 0.02)):
        r = rows.loc[key]
        assert 0 <= r["revised_10pct"] <= r["revised_2pct"] <= 1
        assert r["complete"] > 0
    # Older weeks only are complete under the longer age.
    assert rows.loc[(6, 0.02), "complete"] <= rows.loc[(4, 0.02), "complete"]
    real = target.loc[target["kind"] == "real"]
    releases = sorted(real["vintage"].astype(str).unique())
    one = complete_weeks(target, daily, releases[-2], weeks=4, growth=0.02)
    assert one["complete"].any()
    assert rows.loc[(4, 0.02), "releases"] == len(releases) - 1


def test_an_unreachable_level_has_no_delay(
    target: pd.DataFrame, daily: pd.DataFrame
) -> None:
    got = truth_sensitivity(target, daily, levels=(1.5,), weeks=(4,))
    assert got["delays"]["truth_delay"].isna().all()
    with pytest.raises(ValueError, match="level"):
        truth_sensitivity(target, daily, levels=(0.0,), weeks=(4,))
