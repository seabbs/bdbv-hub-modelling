"""The Schaake shuffle: draws reordered to the ranks of past windows."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from nowcastutils.trajectories import couple, fit_coupling, history_windows

LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
SPREAD = np.array([-2.0, -1.3, -0.7, 0.0, 0.7, 1.3, 2.0])


def _quantiles(series=2, steps=5):
    base = np.arange(1, series * steps + 1, dtype=float).reshape(series, steps)
    return base[..., None] * 10 + SPREAD


def _windows(n, series=2, steps=5, seed=0):
    """Windows whose cells move together: one level per window."""
    rng = np.random.default_rng(seed)
    level = rng.standard_normal((n, 1, 1))
    return level + 0.1 * rng.standard_normal((n, series, steps))


def test_each_cell_keeps_its_marginal():
    q = _quantiles()
    fit = {"method": "schaake", "windows": _windows(300)}
    draws = couple(q, LEVELS, "schaake", fit, n_samples=3000, seed=1)
    assert draws.shape == (3000, 2, 5)
    for i, j in [(0, 0), (1, 4)]:
        got = np.quantile(draws[:, i, j], [0.25, 0.5, 0.75])
        assert got == pytest.approx(q[i, j, 2:5], abs=0.15)


def test_draws_take_the_ranks_of_the_windows():
    q = _quantiles()
    windows = _windows(400)
    fit = {"method": "schaake", "windows": windows}
    draws = couple(q, LEVELS, "schaake", fit, n_samples=400, seed=2)
    # As many windows as samples: each sample is one window, so any two
    # cells' draws have the rank correlation of the windows' cells.
    for a, b in [((0, 0), (1, 3)), ((0, 1), (0, 4))]:
        got = spearmanr(draws[:, a[0], a[1]], draws[:, b[0], b[1]])
        want = spearmanr(windows[:, a[0], a[1]], windows[:, b[0], b[1]])
        # Up to ties in the flat tails beyond the end quantiles.
        assert got.statistic == pytest.approx(want.statistic, abs=0.01)
    # The windows move together, so the draws do too.
    assert spearmanr(draws[:, 0, 0], draws[:, 1, 4]).statistic > 0.9


def test_fewer_windows_than_samples_are_reused():
    q = _quantiles()
    fit = {"method": "schaake", "windows": _windows(20)}
    draws = couple(q, LEVELS, "schaake", fit, n_samples=1000, seed=3)
    assert draws.shape == (1000, 2, 5)
    assert spearmanr(draws[:, 0, 0], draws[:, 1, 4]).statistic > 0.8
    again = couple(q, LEVELS, "schaake", fit, n_samples=1000, seed=3)
    assert np.array_equal(draws, again)


def test_ties_in_the_windows_are_broken_at_random():
    # Zero counts tie; the tied draws still cover their whole marginal.
    q = _quantiles(series=1, steps=1)
    fit = {"method": "schaake", "windows": np.zeros((500, 1, 1))}
    draws = couple(q, LEVELS, "schaake", fit, n_samples=500, seed=4)
    assert np.median(draws) == pytest.approx(q[0, 0, 3], abs=0.2)
    assert draws.std() > 0.5


@pytest.mark.parametrize(
    "fit",
    [
        None,
        {"method": "schaake"},
        {"method": "schaake", "windows": np.zeros((5, 3, 5))},
        {"method": "schaake", "windows": np.full((5, 2, 5), np.nan)},
    ],
)
def test_a_schaake_fit_needs_windows_of_the_forecast_s_shape(fit):
    with pytest.raises(ValueError, match="windows"):
        couple(_quantiles(), LEVELS, "schaake", fit, n_samples=10, seed=1)


def test_history_windows_are_the_last_complete_stretches():
    values = np.arange(2 * 10, dtype=float).reshape(2, 10)
    got = history_windows(values, steps=3, n_windows=4)
    assert got.shape == (4, 2, 3)
    # Newest first, each one step earlier than the last.
    np.testing.assert_array_equal(got[0], values[:, 7:10])
    np.testing.assert_array_equal(got[3], values[:, 4:7])
    # No more windows than the history holds; NaN stretches are skipped.
    values[0, 8] = np.nan
    got = history_windows(values, steps=3, n_windows=100)
    # 8 stretches, of which the 2 ending after the gap hold it.
    assert got.shape == (6, 2, 3)
    assert np.isfinite(got).all()


def test_the_schaake_fit_is_its_windows_not_scores():
    with pytest.raises(ValueError, match="history_windows"):
        fit_coupling(np.zeros((3, 2, 4)), "schaake")
