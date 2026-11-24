"""The shared renewal step (`components.renewal`).

The renewal without depletion is also checked live against epinowcast
in `test_latent_epinowcast.py`; the fitted models' use against the NumPy
reference in `test_components.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyrenew.latent.infection_functions import (
    compute_infections_from_rt,
    logistic_susceptibility_adjustment,
)

from nowcastrenewal.components.mixing import mixing_matrix, normalise_kernel
from nowcastrenewal.components.renewal import (
    DEPLETIONS,
    ExponentialDepletion,
    LinearDepletion,
    NoDepletion,
    Variants,
    run_renewal,
)

GT = np.array([0.2, 0.4, 0.3, 0.1])


def test_without_depletion_or_mixing_it_is_pyrenews_renewal() -> None:
    rng = np.random.default_rng(1)
    start = rng.uniform(1, 5, (len(GT), 1))
    rt = rng.uniform(0.8, 1.6, (40, 1))
    with jax.enable_x64(True):
        run = run_renewal(start, GT, rt, 1.0, depletion=NoDepletion())
        want = compute_infections_from_rt(
            jnp.asarray(start[:, 0]), jnp.asarray(rt[:, 0]), jnp.flip(GT)
        )
    np.testing.assert_allclose(run.infections[len(GT) :, 0], want, rtol=1e-12)
    np.testing.assert_allclose(run.reff, rt, rtol=1e-12)


def test_exponential_depletion_is_pyrenews_logistic_adjustment() -> None:
    # S (1 - e^{-y / N}) is PyRenew's adjustment, written with expm1.
    with jax.enable_x64(True):
        raw = jnp.array([65.0, 2.4, 800.0])
        s = jnp.array([9000.0, 4000.0, 15000.0])
        pop = jnp.array([1e4, 5e3, 2e4])
        new = ExponentialDepletion()(raw, s, pop, jnp.zeros(3))
        want = logistic_susceptibility_adjustment(raw, s / pop, pop)
    np.testing.assert_allclose(new, want, rtol=1e-9)


def test_exponential_depletion_is_linear_while_few_are_infected() -> None:
    with jax.enable_x64(True):
        raw, im = jnp.array([3.0]), jnp.array([0.5])
        s, pop = jnp.array([1e6]), jnp.array([1e6])
        lin = LinearDepletion()(raw, s, pop, im)
        exp = ExponentialDepletion()(raw, s, pop, im)
    np.testing.assert_allclose(exp, lin, rtol=1e-5)


def test_depletion_stops_at_the_susceptibles() -> None:
    with jax.enable_x64(True):
        raw, im = jnp.array([5e5]), jnp.array([20.0])
        s, pop = jnp.array([10.0]), jnp.array([1e3])
        for depletion in (LinearDepletion(), ExponentialDepletion()):
            assert float(depletion(raw, s, pop, im)[0]) == 10.0
    assert set(DEPLETIONS) == {"none", "linear", "exponential"}


def _per_resident() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    k = 3
    f = np.array([0.5, 0.3, 0.2])
    K = normalise_kernel(rng.uniform(0, 1, (k, k)) * (1 - np.eye(k)), "origin")
    with jax.enable_x64(True):
        W = np.asarray(mixing_matrix(K, 0.04, "sent", f))
    seeding = np.zeros((60, k))
    seeding[10:17, 2] = 1e-4
    return {
        # More starting days than the generation time: all count towards
        # depletion.
        "start": rng.uniform(1e-4, 1e-3, (len(GT) + 3, k)),
        "rt": rng.uniform(1.0, 2.5, (60, k)),
        "W": W,
        "seeding": seeding,
    }


def _loop(c: dict[str, np.ndarray], pool: float) -> tuple[np.ndarray, ...]:
    # The step written out: y = R (W F) + seeding, x = S (1 - e^{-y/nu}).
    hist = c["start"][-len(GT) :].copy()
    s = pool - c["start"].sum(axis=0)
    xs, reffs, imps = [], [], []
    off = c["W"] * (1 - np.eye(len(s)))
    for r, sd in zip(c["rt"], c["seeding"], strict=True):
        force = np.flip(GT) @ hist
        y = r * (c["W"] @ force) + sd
        x = s * -np.expm1(-y / pool)
        s = s - x
        xs.append(x)
        reffs.append(r * s / pool)
        imps.append(x * (r * (off @ force)) / y)
        hist = np.vstack([hist[1:], x])
    return np.array(xs), np.array(reffs), np.array(imps)


def test_the_fitted_form_runs_per_resident_with_seeding_and_mixing() -> None:
    c = _per_resident()
    with jax.enable_x64(True):
        run = run_renewal(
            c["start"],
            GT,
            c["rt"],
            0.05,
            mixing=c["W"],
            seeding=c["seeding"],
            depletion=ExponentialDepletion(),
        )
    x, reff, imported = _loop(c, 0.05)
    n0 = len(c["start"])
    np.testing.assert_allclose(run.infections[n0:], x, rtol=1e-12)
    np.testing.assert_allclose(run.reff, reff, rtol=1e-12)
    np.testing.assert_allclose(run.imported, imported, rtol=1e-12)
    # Infections never exceed the share at risk.
    assert np.all(np.asarray(run.infections).sum(axis=0) <= 0.05 + 1e-12)


def test_no_mixing_imports_nothing() -> None:
    c = _per_resident()
    with jax.enable_x64(True):
        run = run_renewal(c["start"], GT, c["rt"], 0.05)
    np.testing.assert_allclose(run.imported, 0.0)


def test_the_step_has_gradients_in_float32() -> None:
    c = _per_resident()

    def total(rt: jax.Array) -> jax.Array:
        run = run_renewal(
            c["start"].astype(np.float32),
            GT.astype(np.float32),
            rt,
            np.float32(0.05),
            mixing=c["W"].astype(np.float32),
            depletion=ExponentialDepletion(),
        )
        return run.infections.sum()

    g = jax.grad(total)(jnp.asarray(c["rt"], dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_a_variant_raises_r_and_returns_immune_people_once() -> None:
    v = Variants(at=[10.0], escape=[0.5], multiplier=[2.0], days=[4.0])
    with jax.enable_x64(True):
        m = np.asarray(v.multiplier_on(jnp.arange(1.0, 31.0)))
    assert m[0] == pytest.approx(1.0, abs=1e-4)
    assert m[-1] == pytest.approx(2.0, abs=1e-4)
    assert m[9] == pytest.approx(1.5)


def test_a_ceiling_bounds_r_smoothly_after_shifts_and_variants() -> None:
    # Variants multiply R; with a ceiling of 4, R stays below it, values
    # near 1 are kept and the ceiling bends R smoothly (no kink).
    start = np.full((len(GT), 1), 2.0)
    rt = np.linspace(0.5, 12.0, 60)[:, None]
    variants = Variants(at=[1.0], escape=[0.0], multiplier=[1.0], days=[1.0])
    with jax.enable_x64(True):
        free = run_renewal(start, GT, rt, 1e9, variants=variants)
        capped = run_renewal(
            start, GT, rt, 1e9, variants=variants, r_ceiling=4.0
        )
    r, rc = np.asarray(free.r)[:, 0], np.asarray(capped.r)[:, 0]
    assert rc.max() < 4.0
    near = r < 1.5
    np.testing.assert_allclose(rc[near], r[near], rtol=1e-3)
    assert (np.diff(rc) > 0).all()
    # No ceiling: R as it was.
    with jax.enable_x64(True):
        same = run_renewal(
            start, GT, rt, 1e9, variants=variants, r_ceiling=None
        )
    np.testing.assert_allclose(np.asarray(same.r), np.asarray(free.r))
