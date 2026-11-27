"""Short over-reporting bursts (`reporting.revisions.OverReportingBursts`):
the first report several times the final count, withdrawn within days."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from test_reporting_simulate import AGES, DESIGN, ONSETS, PLACES, VALS

from nowcastrenewal.reporting.effects import ReferenceFrame
from nowcastrenewal.reporting.revisions import OverReportingBursts, revise
from nowcastrenewal.reporting.simulate import report_simulation

G, T, D = 2, 30, 6
FIRST = "2026-01-05"


def _frame() -> ReferenceFrame:
    return ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(T),
        place=np.arange(G),
        age=np.zeros(G, dtype=int),
    )


def _counts() -> np.ndarray:
    y = np.zeros((G, T, D))
    y[..., 0], y[..., 1], y[..., 3] = 200, 300, 500
    return y


def _bursts(**kw) -> OverReportingBursts:
    args = {
        "strata": [1],
        "at": [10 / T],
        "days": [3],
        "mult": [4.0],
        "keep": [2],
        **kw,
    }
    return OverReportingBursts("b", _frame(), **args)


def _run(bursts, counts, log_expected=None, seed=1) -> np.ndarray:
    if log_expected is None:
        log_expected = np.zeros(counts.shape)
    with handlers.seed(rng_seed=seed):
        return np.asarray(bursts(counts=counts, log_expected=log_expected))


def test_burst_first_reports_are_several_times_the_final_count() -> None:
    y = _counts()
    extra = _run(_bursts(), y)
    reported = y + extra
    final = reported.sum(axis=-1)
    # The final counts are unchanged.
    np.testing.assert_allclose(final, y.sum(axis=-1))
    for t in (10, 11, 12):
        first = reported[0, t, 0]
        assert first / final[0, t] == pytest.approx(4.0, rel=0.1)
        # Withdrawn on the first report day at least 2 days later.
        assert extra[0, t, 2] == -extra[0, t, 0]
        assert np.cumsum(reported[0, t])[3] == pytest.approx(final[0, t])
    # Other days and strata are untouched.
    assert np.all(extra[1] == 0)
    outside = np.setdiff1d(np.arange(T), [10, 11, 12])
    assert np.all(extra[0, outside] == 0)


def test_withdrawals_wait_for_a_report_day() -> None:
    y = _counts()
    log_expected = np.zeros(y.shape)
    log_expected[..., 2] = -np.inf
    log_expected[..., 4] = -np.inf
    extra = _run(_bursts(keep=[2]), y, log_expected)
    assert extra[0, 10, 3] < 0
    extra = _run(_bursts(keep=[4]), y, log_expected)
    assert extra[0, 10, 5] < 0
    assert np.all(extra[0, 10, 1:5] == 0)


def test_withdrawals_past_the_last_day_fall_on_it() -> None:
    # A burst on the last days is withdrawn by the last report day, so the
    # counts as of it are the final ones.
    y = _counts()
    extra = _run(_bursts(at=[(T - 3) / T], keep=[5]), y)
    reported = y + extra
    for t in range(T - 3, T):
        cells = np.arange(D) <= T - 1 - t
        assert reported[0, t, cells].sum() == pytest.approx(
            y[0, t, cells].sum()
        )
    assert extra[0, T - 3, 0] > 0
    assert extra[0, T - 3, 2] == -extra[0, T - 3, 0]


def test_bursts_are_reproducible() -> None:
    y = _counts()
    a = _run(_bursts(), y, seed=4)
    np.testing.assert_array_equal(a, _run(_bursts(), y, seed=4))
    assert not np.array_equal(a, _run(_bursts(), y, seed=5))


def test_bursts_ride_through_the_revisions() -> None:
    y = _counts()
    extra = _run(_bursts(), y)
    zeros = np.zeros_like(y)
    dense = np.asarray(
        revise(y, zeros, zeros, np.ones_like(y), [1.0, 1.0], bursts=extra)
    )
    # Day 10's report on day 10 holds the burst; day 12's its withdrawal.
    assert dense[0, 10, 10] == y[0, 10, 0] + extra[0, 10, 0]
    assert dense[0, 10, 12] == extra[0, 10, 2] < 0
    inside = np.arange(T)[:, None] + np.arange(D)[None, :] <= T - 1
    np.testing.assert_allclose(dense.sum(axis=-1), (y * inside).sum(axis=-1))


@pytest.mark.parametrize("revisions", ["no", "yes"])
def test_a_simulation_with_a_burst_keeps_its_final_counts(
    revisions: str,
) -> None:
    design = {**DESIGN, "revisions": revisions}
    values = {
        **VALS,
        "burst_strata": [2],
        "burst_at": [0.3],
        "burst_days": [4],
        "burst_mult": [5.0],
        "burst_keep": [3],
    }
    plain, _ = report_simulation(
        design, values, ONSETS, PLACES, AGES, seed=2, draws=1
    )
    burst, _ = report_simulation(
        {**design, "bursts": "yes"}, values, ONSETS, PLACES, AGES, seed=2,
        draws=1,
    )  # fmt: skip
    key = ["location", "reference_date"]
    a = plain.groupby(key)["count"].sum()
    b = burst.groupby(key)["count"].sum()
    pd.testing.assert_series_equal(a, b.reindex(a.index).fillna(0).astype(int))
    assert (burst["count"] < 0).any()
    # The first report of a burst day is several times its final count.
    p02 = burst[burst["location"] == "P02"]
    first = p02.sort_values("report_date").groupby("reference_date").first()
    ratio = first["count"] / p02.groupby("reference_date")["count"].sum()
    assert ratio.max() > 2
