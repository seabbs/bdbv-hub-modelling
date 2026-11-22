"""The shared hazards and reference events (`components.hazards`,
`components.reference`).

The simulator's reference events are also checked live against its R
engine in `test_latent_components.py`; the open hazards against
epinowcast's `hazard_to_log_prob` in `test_reporting_triangle.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nowcastrenewal.components.hazards import log_hazard_probs
from nowcastrenewal.components.reference import delayed_events

from .reference_kernels import hazard_to_pmf_reference


def test_closed_hazards_are_a_pmf_and_geometric_when_constant() -> None:
    pmf = np.exp(
        log_hazard_probs(jnp.full((3, 20), jnp.log(0.2 / 0.8)), close=True)
    )
    assert pmf.shape == (3, 21)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(pmf[0, :5], 0.2 * 0.8 ** np.arange(5), rtol=1e-5)


def test_closed_hazards_match_the_cumsum_reference() -> None:
    x = jnp.asarray(np.random.default_rng(5).normal(-2, 1, (4, 20)))
    np.testing.assert_allclose(
        np.exp(log_hazard_probs(x, close=True)),
        hazard_to_pmf_reference(x),
        rtol=1e-5,
        atol=1e-7,
    )


def test_open_hazards_use_the_last_as_given() -> None:
    # As in epinowcast, a finite last hazard leaves reports beyond the
    # window, and a last hazard of 1 closes it.
    with jax.enable_x64(True):
        p = np.exp(log_hazard_probs(jnp.zeros((1, 4))))
        shut = jnp.concatenate(
            [jnp.zeros((1, 3)), jnp.full((1, 1), jnp.inf)], 1
        )
        q = np.exp(log_hazard_probs(shut))
    np.testing.assert_allclose(p, [[0.5, 0.25, 0.125, 0.0625]])
    np.testing.assert_allclose(q, [[0.5, 0.25, 0.125, 0.125]])
    assert np.all(np.isfinite(q))


def test_hazards_have_gradients() -> None:
    g = jax.grad(lambda x: jnp.exp(log_hazard_probs(x, close=True))[3])(
        jnp.zeros(6)
    )
    assert bool(jnp.all(jnp.isfinite(g)))


def test_delayed_events_are_convolved_infections_with_their_history() -> None:
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.0, (30, 3))
    pmf = rng.dirichlet(np.ones(6))
    asc = np.array([0.5, 1.0, 0.2])
    before = np.array([0.0, 2.0, 5.0])
    with jax.enable_x64(True):
        out = np.asarray(delayed_events(x, pmf, asc, before))
    for j in range(3):
        padded = np.concatenate([np.full(len(pmf) - 1, before[j]), x[:, j]])
        want = asc[j] * np.convolve(padded, pmf, mode="valid")
        np.testing.assert_allclose(out[:, j], want, rtol=1e-12)
    # One pmf per place gives the same as a shared one.
    with jax.enable_x64(True):
        each = np.asarray(delayed_events(x, [pmf] * 3, asc, before))
    np.testing.assert_allclose(each, out, rtol=1e-12)


def test_delayed_events_need_one_pmf_per_place() -> None:
    with pytest.raises(ValueError, match="one delay per place"):
        delayed_events(np.ones((5, 2)), [np.ones(2) / 2])
