"""The shared mixing component: kernels and mixing matrices.

The simulator's place mixing is checked on closed-form cases in
`test_latent_components.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

from nowcastrenewal.components.mixing import (
    Mixing,
    distance_kernel,
    great_circle_km,
    mixing_matrix,
    normalise_kernel,
)

RNG = np.random.default_rng(578)
K_WEIGHTS = RNG.uniform(0.1, 1, (4, 4)) * (1 - np.eye(4))
F = np.array([0.4, 0.3, 0.2, 0.1])


def test_great_circle_distances() -> None:
    # A degree of latitude is about 111 km; the matrix is symmetric.
    d = great_circle_km([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    assert d[0, 1] == pytest.approx(111.19, abs=0.01)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0)
    with pytest.raises(ValueError, match="centre"):
        great_circle_km([0.0, np.nan], [0.0, 1.0])


def test_kernels_are_normalised_per_origin_or_receiver() -> None:
    np.testing.assert_allclose(
        normalise_kernel(K_WEIGHTS, "origin").sum(axis=0), 1.0
    )
    np.testing.assert_allclose(
        normalise_kernel(K_WEIGHTS, "receiver").sum(axis=1), 1.0
    )
    # A place with no weight stays at 0.
    quiet = K_WEIGHTS.copy()
    quiet[1, :] = 0
    assert np.all(normalise_kernel(quiet, "receiver")[1] == 0)
    with pytest.raises(ValueError, match="per"):
        normalise_kernel(K_WEIGHTS, "both")


def test_a_distance_kernel_weights_receivers_and_decays() -> None:
    d = great_circle_km([0.0, 1.0, 3.0], [0.0, 0.0, 0.0])
    pop = np.array([1.0, 2.0, 5.0])
    k = distance_kernel(d, 300.0, "origin", weights=pop)
    want = pop[:, None] * np.exp(-d / 300.0) * (1 - np.eye(3))
    np.testing.assert_allclose(k, want / want.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(np.diag(k), 0.0)


def test_sent_mixing_keeps_the_total_and_is_the_importation_form() -> None:
    K = normalise_kernel(K_WEIGHTS, "origin")
    eps = np.array([0.1, 0.05, 0.2, 0.0])
    with jax.enable_x64(True):
        W = np.asarray(mixing_matrix(K, eps, "sent"))
        W_pc = np.asarray(mixing_matrix(K, eps, "sent", fractions=F))
    # Each origin loses eps of its force and sends it over the kernel.
    want = np.diag(1 - eps) + K * eps[None, :]
    np.testing.assert_allclose(W, want, rtol=1e-12)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, rtol=1e-12)
    # Per resident: diag(1 - eps) + diag(1 / f) K diag(eps f).
    want_pc = np.diag(1 - eps) + K * (eps * F)[None, :] / F[:, None]
    np.testing.assert_allclose(W_pc, want_pc, rtol=1e-12)


def test_received_mixing_averages_each_place_with_its_sources() -> None:
    K = normalise_kernel(K_WEIGHTS, "receiver")
    eps = np.array([0.0, 0.1, 0.3])
    W = np.asarray(mixing_matrix(K, eps[:, None], "received"))
    assert W.shape == (3, 4, 4)
    for t, e in enumerate(eps):
        want = np.eye(4) + e * (K - np.diag(K.sum(axis=1)))
        np.testing.assert_allclose(W[t], want, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(W.sum(axis=-1), 1.0, rtol=1e-6)


def test_the_forms_agree_for_a_kernel_normalised_both_ways() -> None:
    K = normalise_kernel(1 - np.eye(4), "receiver")
    np.testing.assert_allclose(
        mixing_matrix(K, 0.2, "sent"), mixing_matrix(K, 0.2, "received")
    )


def test_no_share_is_no_mixing() -> None:
    K = normalise_kernel(K_WEIGHTS, "origin")
    for outflow in ("sent", "received"):
        np.testing.assert_allclose(
            mixing_matrix(K, 0.0, outflow, fractions=F), np.eye(4)
        )
    with pytest.raises(ValueError, match="outflow"):
        mixing_matrix(K, 0.1, "kept")


def test_the_component_samples_its_share() -> None:
    K = normalise_kernel(K_WEIGHTS, "origin")
    mix = Mixing(
        "mixing",
        DeterministicVariable("kernel", K),
        DistributionalVariable("importation", dist.Beta(1, 100)),
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        W = mix(fractions=jnp.asarray(F))
    eps = float(tr["importation"]["value"])
    with jax.enable_x64(True):
        want = mixing_matrix(K, eps, "sent", fractions=F)
    np.testing.assert_allclose(W, want, rtol=1e-5)
    assert repr(mix).startswith("Mixing(name='mixing'")


def test_the_component_checks_its_parts() -> None:
    kernel = DeterministicVariable("kernel", np.eye(2))
    with pytest.raises(TypeError, match="share_rv"):
        Mixing("mixing", kernel, 0.1)
    with pytest.raises(TypeError, match="kernel_rv"):
        Mixing("mixing", np.eye(2), DeterministicVariable("e", 0.1))
    with pytest.raises(ValueError, match="outflow"):
        Mixing("mixing", kernel, DeterministicVariable("e", 0.1), "kept")
