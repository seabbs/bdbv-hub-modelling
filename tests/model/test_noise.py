"""The shared observation noise (`components.noise`) and the reporting
step's observed counts (`reporting.observation`).

Distributional checks: the draws' moments against each family's, and the
NB1 dispersion's scaling against the simulator engine's formula.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from pyrenew.deterministic import DeterministicVariable
from pyrenew.observation import NegativeBinomialNoise, PoissonNoise

from nowcastrenewal.components.noise import (
    FAMILIES,
    NegativeBinomial1Noise,
    count_noise,
)
from nowcastrenewal.reporting.delays import report_delay_pmf
from nowcastrenewal.reporting.observation import ObservedReports, nb1_scale

N = 200_000


def _draws(noise, mean: np.ndarray, seed: int = 0) -> np.ndarray:
    pred = jnp.broadcast_to(jnp.asarray(mean), (N, *np.shape(mean)))
    with handlers.seed(rng_seed=seed):
        return np.asarray(noise.sample("y", pred))


def test_count_noise_is_pyrenews_poisson_and_negative_binomial() -> None:
    assert isinstance(count_noise("poisson"), PoissonNoise)
    assert isinstance(count_noise("negbin", 2.0), NegativeBinomialNoise)
    assert isinstance(count_noise("negbin1d", 2.0), NegativeBinomial1Noise)
    assert set(FAMILIES) == {"poisson", "negbin", "negbin1d"}
    with pytest.raises(ValueError, match="family"):
        count_noise("binomial")


@pytest.mark.parametrize("family", ["poisson", "negbin", "negbin1d"])
def test_the_draws_have_each_familys_moments(family: str) -> None:
    mean = np.array([0.7, 4.0, 30.0])
    phi = 1.5
    noise = count_noise(family, phi)
    y = _draws(noise, mean)
    var = {
        "poisson": mean,
        "negbin": mean + mean**2 / phi,
        # NB1: NB2 with concentration mean * phi, variance mean (1 + 1/phi).
        "negbin1d": mean * (1 + 1 / phi),
    }[family]
    np.testing.assert_allclose(y.mean(axis=0), mean, rtol=0.02)
    np.testing.assert_allclose(y.var(axis=0), var, rtol=0.03)


def test_nb1_concentration_scales_with_the_mean() -> None:
    noise = NegativeBinomial1Noise(DeterministicVariable("phi", 0.5))
    with handlers.seed(rng_seed=1), handlers.trace() as tr:
        noise.sample("y", jnp.array([2.0, 20.0]))
    d = tr["y"]["fn"]
    np.testing.assert_allclose(d.concentration, [1.0, 10.0])


def test_nb1_scale_is_the_engines() -> None:
    # sqrt_phi_obs = obs_sqrt_phi * sqrt(lambda_hat * sum_p2) for NB1,
    # lambda_hat the median expected onset among those of at least 1
    # (at least 1), sum_p2 the delay pmf's sum of squares.
    onsets = np.array([[0.2, 3.0, 5.0], [9.0, 0.5, 1.0]])
    values = {"delay_median": 3.0, "delay_sdlog": 0.7}
    pmf = report_delay_pmf("lognormal", values, 10)
    lam, s2, scale = nb1_scale(onsets, "lognormal", values, 10)
    assert lam == pytest.approx(4.0)
    assert s2 == pytest.approx(float((pmf**2).sum()))
    assert scale == pytest.approx(np.sqrt(4.0 * s2))
    # No onset reaches 1: lambda_hat is 1.
    assert nb1_scale(onsets / 100, "lognormal", values, 10)[0] == 1.0


def test_observed_reports_draw_only_reported_cells() -> None:
    log_expected = jnp.log(jnp.array([[[3.0, 2.0, 0.0], [5.0, 1.0, 4.0]]]))
    obs = ObservedReports(
        "reports",
        family="negbin1d",
        sqrt_phi=np.array([0.5]),
        onsets=np.full((2, 1), 3.0),
        delay_family="lognormal",
        values={"delay_median": 3.0, "delay_sdlog": 0.7},
        max_delay=3,
    )
    with (
        jax.enable_x64(True),
        handlers.seed(rng_seed=2),
        handlers.trace() as tr,
    ):
        counts = np.asarray(obs(log_expected=log_expected))
    assert counts.shape == (1, 2, 3)
    # A cell with no expected reports (a day with no report) has none.
    assert counts[0, 0, 2] == 0
    assert "reports::counts" in tr
    assert np.all(counts >= 0)


def test_observed_reports_can_be_conditioned() -> None:
    log_expected = jnp.log(jnp.array([[[3.0, 2.0], [5.0, 1.0]]]))
    obs = ObservedReports(
        "reports",
        family="poisson",
        sqrt_phi=np.array([1.0]),
        onsets=np.full((2, 1), 3.0),
        delay_family="lognormal",
        values={"delay_median": 3.0, "delay_sdlog": 0.7},
        max_delay=2,
    )
    data = np.array([[[2, 1], [6, 0]]])
    with handlers.trace() as tr:
        obs(log_expected=log_expected, counts=data)
    site = tr["reports::counts"]
    assert site["is_observed"]
    np.testing.assert_array_equal(site["value"], data)
