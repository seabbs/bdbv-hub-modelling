"""The delay linked to the stratum's expected counts by reference date
(`reporting.structure.CountCoupling`), of either sign."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers
from test_delay_change_strata import _design, _frame, _values

from nowcastrenewal.reporting.effects import ReferenceEffects
from nowcastrenewal.reporting.structure import CountCoupling

MAX_DELAY, N_DAYS, G = 8, 70, 3


def _log_lambda() -> np.ndarray:
    # A wave in every stratum, peaking at different sizes.
    t = np.arange(N_DAYS)
    wave = np.exp(-(((t - 35) / 12) ** 2))
    return np.log(np.array([5, 50, 500])[:, None] * wave + 0.5)


def _mu(sign: list[int], level: str = "shared", seed: int = 1, **kw: str):
    v: dict[str, Any] = _values(
        delay_counts_sign=sign, delay_counts_slope=0.2, delay_counts_sd=0.3
    )
    design = _design(delay_change="static", delay_counts=level, **kw)
    effects = ReferenceEffects("ref", _frame(), design, v, MAX_DELAY)
    with handlers.seed(rng_seed=seed), handlers.trace() as tr:
        _, hazard = effects(log_lambda=_log_lambda())
    loc = tr["ref::location"]["value"] if "ref::location" in tr else None
    return loc, np.asarray(hazard), tr


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("sign", [1, -1])
def test_delay_follows_the_counts_with_its_sign(sign: int) -> None:
    mu, _, _ = _mu([sign] * G)
    x = np.log1p(np.exp(_log_lambda()))
    for g in range(G):
        assert sign * _corr(np.asarray(mu)[g], x[g]) > 0.99


def test_the_term_is_the_slope_times_centred_log_counts() -> None:
    frame = _frame()
    with handlers.seed(rng_seed=0):
        c = np.asarray(
            CountCoupling("c", frame, [0.2, -0.1, 0.0])(
                log_lambda=_log_lambda()
            )
        )
    x = np.log1p(np.exp(_log_lambda()))
    want = np.array([0.2, -0.1, 0.0])[:, None] * (
        x - x.mean(axis=1, keepdims=True)
    )
    np.testing.assert_allclose(c, want, rtol=1e-5, atol=1e-6)
    # Centred within each stratum: no shift between strata's levels.
    np.testing.assert_allclose(c.mean(axis=1), 0, atol=1e-6)


def test_strata_can_take_opposite_signs() -> None:
    mu, _, _ = _mu([1, -1, 1], level="strata")
    x = np.log1p(np.exp(_log_lambda()))
    mu = np.asarray(mu)
    assert _corr(mu[0], x[0]) > 0.99
    assert _corr(mu[1], x[1]) < -0.99


def test_shared_slopes_are_the_same_and_strata_slopes_differ() -> None:
    _, _, tr = _mu([1] * G, level="shared")
    assert "ref::counts::strata_slope" not in tr
    _, _, tr = _mu([1] * G, level="strata")
    assert tr["ref::counts::strata_slope"]["value"].shape == (G,)
    assert "ref::counts::strata_jitter_slope" not in tr
    _, _, tr = _mu([1] * G, level="strata_time")
    assert tr["ref::counts::strata_jitter_slope"]["value"].shape[0] == G


def test_a_non_parametric_delay_takes_the_opposite_hazard() -> None:
    # Longer delays are lower hazards.
    _, hazard, _ = _mu([1] * G, delay_form="non_parametric")
    x = np.log1p(np.exp(_log_lambda()))
    assert _corr(hazard[2, :, 0], x[2]) < -0.99


def test_reproducible_and_needs_the_counts() -> None:
    a, _, _ = _mu([1] * G, level="strata_time", seed=3)
    b, _, _ = _mu([1] * G, level="strata_time", seed=3)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with (
        pytest.raises(ValueError, match="expected counts"),
        handlers.seed(rng_seed=0),
    ):
        CountCoupling("c", _frame(), 0.1)()
