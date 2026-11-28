"""Real-data blocks laid out as the fine-tuned model forecasts."""

import numpy as np

from nowcastfoundation.finetune import realtasks as rt


def test_the_lag_jitter_is_the_simulated_blocks():
    assert rt.lag_jitter() == (0.1, 0.05)


def test_a_share_of_blocks_scale_their_lag_count():
    rng = np.random.default_rng(3)
    scales = np.array([rt.lag_scale(rng, (0.1, 0.05)) for _ in range(4000)])
    jittered = scales != 1.0
    assert 0.07 < jittered.mean() < 0.13
    assert abs(scales[jittered].mean() - 1.0) < 0.02
    assert 0.03 < scales[jittered].std() < 0.07


def test_levels_name_the_country_totals_and_leaves():
    places = ("CD", "CD-IT", "CD-IT-01", "CD-NK")
    assert rt.levels_of(places) == ("national", "intermediate", "leaf", "leaf")


def test_truth_rows_are_missing_where_the_truth_has_none():
    truth = rt.Truth(
        ("CD", "CD-IT"),
        ["2026-09-01", "2026-09-02"],
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        "2026-09-30",
    )
    got = truth.rows(("CD-IT", "CD-NK"), ["2026-08-31", "2026-09-02"])
    assert np.isnan(got[0, 0])
    assert got[0, 1] == 4.0
    assert np.isnan(got[1]).all()
