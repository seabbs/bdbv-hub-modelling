"""The coupling and its look-back window, chosen online by energy score."""

import numpy as np
import pytest
from scipy.special import ndtri

from nowcastutils.trajectories import (
    CANDIDATES,
    choose_coupling,
    energy_score,
)

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


def test_the_energy_score_estimates_the_scoring_package_s():
    # The same score as nowcasteval's fair estimator (scoringrules), from
    # draw pairs rather than all pairs: equal on average, noisier.
    from nowcasteval.scoring import energy_sample

    rng = np.random.default_rng(5)
    observed = rng.standard_normal(6)
    draws = rng.standard_normal((4000, 6)) + 0.3
    fair = energy_sample(observed, draws)
    assert energy_score(draws, observed) == pytest.approx(fair, rel=0.02)


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


GAUSSIAN = (("gaussian", 4), ("gaussian", 40))


def test_a_recent_change_in_coupling_favours_the_short_window():
    # Long ago steps were independent; lately they move together.
    q, observed = _origins([0.0] * 40 + [0.95] * 12)
    got = choose_coupling(
        q, LEVELS, observed, None, None, 200, seed=2, candidates=GAUSSIAN
    )
    assert (got["method"], got["window"]) == ("gaussian", 4)
    assert got["energy"]["gaussian-4"] < got["energy"]["gaussian-40"]
    assert got["chosen_by"] == "energy"
    # Fitted at the forecast from the latest 4 origins, which move together.
    assert got["rho"] > 0.5
    assert got["n_origins"] == 4


def test_steady_coupling_favours_the_long_window():
    q, observed = _origins([0.6] * 52, seed=4)
    got = choose_coupling(
        q, LEVELS, observed, None, None, 200, seed=2, candidates=GAUSSIAN
    )
    assert got["window"] == 40


def test_coupled_truth_is_coupled_and_independent_truth_is_not():
    both = (*GAUSSIAN, ("independent", None))
    q, coupled = _origins([0.9] * 30, seed=5)
    got = choose_coupling(
        q, LEVELS, coupled, None, None, 200, seed=1, candidates=both
    )
    assert got["method"] == "gaussian"
    assert got["energy"]["gaussian-40"] < got["energy"]["independent"]
    q, apart = _origins([0.0] * 30, seed=6)
    got = choose_coupling(
        q, LEVELS, apart, None, None, 200, seed=1, candidates=GAUSSIAN
    )
    # Shrunk to independence: no correlation the data do not show.
    assert abs(got["rho"]) < 0.1


def _mirrored(n, steps=6, seed=0):
    """Two series that mirror each other, and history that does too."""
    rng = np.random.default_rng(seed)
    q = np.broadcast_to(ndtri(LEVELS), (n, 2, steps, len(LEVELS)))
    level = rng.standard_normal((n, 1, steps))
    observed = np.concatenate([level, -level], axis=1)
    past = rng.standard_normal((1, 200))
    history = np.concatenate([past, -past])
    starts = np.full(n, history.shape[1])
    return np.array(q), observed, history, starts


def test_the_schaake_shuffle_wins_where_history_shows_the_dependence():
    # The exchangeable gaussian copula cannot mirror two series (s >= 0);
    # windows of a history that mirrors them can.
    q, observed, history, starts = _mirrored(12)
    got = choose_coupling(q, LEVELS, observed, history, starts, 200, seed=3)
    assert got["method"] == "schaake"
    assert got["n_windows"] == 200
    assert got["windows"].shape == (200, 2, 6)
    assert got["energy"]["schaake"] < min(
        v for k, v in got["energy"].items() if k != "schaake"
    )


def test_with_no_origin_to_compare_on_schaake_then_gaussian():
    q, observed, history, starts = _mirrored(2)
    got = choose_coupling(q, LEVELS, observed, history, starts, 50, seed=3)
    assert (got["method"], got["chosen_by"], got["n_eval"]) == (
        "schaake",
        "fallback",
        0,
    )
    got = choose_coupling(q, LEVELS, observed, None, None, 50, seed=3)
    assert (got["method"], got["chosen_by"]) == ("gaussian", "fallback")
    assert got["window"] == max(w for m, w in CANDIDATES if m == "gaussian")


def test_unmatured_cells_are_left_out_of_the_comparison():
    q, observed = _origins([0.9] * 12, seed=7)
    cut = observed.copy()
    cut[-3:, :, 4:] = np.nan
    cut[-1] = np.nan
    a = choose_coupling(
        q, LEVELS, cut, None, None, 100, seed=1, candidates=GAUSSIAN
    )
    assert a["n_eval"] == 12 - 2 - 1
    assert np.isfinite(list(a["energy"].values())).all()


def test_the_choice_is_reproducible_and_checked():
    q, observed = _origins([0.5] * 12)
    a = choose_coupling(q, LEVELS, observed, None, None, 100, seed=3)
    b = choose_coupling(q, LEVELS, observed, None, None, 100, seed=3)
    assert a == b
    with pytest.raises(ValueError, match="candidates"):
        choose_coupling(
            q, LEVELS, observed, None, None, 100, seed=3, candidates=()
        )
    with pytest.raises(ValueError, match="starts"):
        choose_coupling(q, LEVELS, observed, np.zeros((3, 9)), None, 100, 3)
