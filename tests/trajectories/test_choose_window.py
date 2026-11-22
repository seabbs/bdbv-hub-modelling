"""The coupling's look-back window, chosen online by energy score."""

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from nowcastutils.trajectories import choose_window, energy_score

LEVELS = np.linspace(0.05, 0.95, 19)


def test_the_energy_score_of_a_point_forecast_is_its_distance():
    draws = np.full((50, 2, 3), 1.0)
    observed = np.zeros((2, 3))
    assert energy_score(draws, observed) == pytest.approx(np.sqrt(6))


def test_the_energy_score_reads_only_the_observed_cells():
    draws = np.full((50, 2, 3), 1.0)
    observed = np.full((2, 3), np.nan)
    observed[0, 0] = 0.0
    assert energy_score(draws, observed) == pytest.approx(1.0)
    assert np.isnan(energy_score(draws, np.full((2, 3), np.nan)))


def test_coupled_draws_score_better_on_coupled_truth():
    rng = np.random.default_rng(0)
    truth_level = rng.standard_normal()
    observed = np.full((1, 10), truth_level)
    together = np.repeat(rng.standard_normal((2000, 1, 1)), 10, axis=2)
    apart = rng.standard_normal((2000, 1, 10))
    assert energy_score(together, observed) < energy_score(apart, observed)


def _origins(rhos, series=3, steps=8, seed=1):
    """Standard normal quantiles, and truths with each origin's AR(1) rho."""
    rng = np.random.default_rng(seed)
    n = len(rhos)
    q = np.broadcast_to(ndtri(LEVELS), (n, series, steps, len(LEVELS)))
    z = np.empty((n, series, steps))
    for k, rho in enumerate(rhos):
        e = rng.standard_normal((series, steps))
        z[k, :, 0] = e[:, 0]
        for t in range(1, steps):
            z[k, :, t] = rho * z[k, :, t - 1] + np.sqrt(1 - rho**2) * e[:, t]
    return np.array(q), z


def test_a_recent_change_in_coupling_favours_the_short_window():
    # Long ago steps were independent; lately they move together.
    q, observed = _origins([0.0] * 40 + [0.95] * 12)
    scores = ndtri(np.clip(ndtr(observed), 1e-6, 1 - 1e-6))
    got = choose_window(
        scores, q, LEVELS, observed, candidates=(4, 40), n_eval=6, seed=2
    )
    assert got["window"] == 4
    assert got["energy"][4] < got["energy"][40]
    assert got["n_eval"] == 6


def test_the_choice_is_reproducible_and_checked():
    q, observed = _origins([0.5] * 12)
    kw = {"candidates": (2, 4), "n_eval": 3, "seed": 3}
    a = choose_window(observed, q, LEVELS, observed, **kw)
    b = choose_window(observed, q, LEVELS, observed, **kw)
    assert a == b
    with pytest.raises(ValueError, match="candidates"):
        choose_window(observed, q, LEVELS, observed, candidates=(), n_eval=3)
    with pytest.raises(ValueError, match="n_eval"):
        choose_window(observed, q, LEVELS, observed, candidates=(2,), n_eval=12)


def test_the_energy_score_estimates_the_scoring_package_s():
    # The same score as nowcasteval's fair estimator (scoringrules), from
    # draw pairs rather than all pairs: equal on average, noisier.
    from nowcasteval.scoring import energy_sample

    rng = np.random.default_rng(5)
    observed = rng.standard_normal(6)
    draws = rng.standard_normal((4000, 6)) + 0.3
    fair = energy_sample(observed, draws)
    assert energy_score(draws, observed) == pytest.approx(fair, rel=0.02)
