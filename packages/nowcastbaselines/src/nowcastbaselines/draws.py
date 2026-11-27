# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Draws of padded arrays that are those of the unpadded ones.

The walks and the nowcast draws pad their arrays to a few sizes
(`nowcastutils.shapes.bucket`), so JAX compiles a program once for many
dates. JAX draws a normal, uniform or gamma element from its flat index
alone, so the first elements of a longer flat array are those of the
shorter one. Its exact Poisson sampler is not: its rejection loop runs
until every element is accepted and keeps the latest accepted draw, so
an element's draw depends on how long the others take. `poisson` is
that sampler (`jax.random.poisson`'s `method="exact"`) with the padding
accepted from the start, so it gives the unpadded array's draws.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

__all__ = ["DRAW_STEPS", "poisson"]

DRAW_STEPS = 32
"""Padded sizes for each doubling of the nowcast draws' length
(`nowcastutils.shapes.bucket`). Their millions of draws are much of the
run time, so their padding is at most a thirty-second of their length."""


def poisson(key: Array, lam: Array, real: Array) -> Array:
    """Draw `jax.random.poisson(key, lam[real])` within a padded array.

    Parameters
    ----------
    key : jax.Array
        Random key.
    lam : jax.Array
        Rates, flat, with the unpadded array's first and 0 after.
    real : jax.Array
        True where an element is the unpadded array's, the shape of
        `lam`.

    Returns
    -------
    jax.Array
        int32 draws, those of `jax.random.poisson` on the unpadded rates
        where `real`, and 0 elsewhere.
    """
    lam = lam.astype(np.float32)
    knuth = jnp.isnan(lam) | (lam < 10)
    lam_knuth = lax.select(knuth, lam, jnp.zeros_like(lam))
    lam_rejection = lax.select(knuth, jnp.full_like(lam, 1e5), lam)
    result = lax.select(
        knuth,
        _knuth(key, lam_knuth),
        _rejection(key, lam_rejection, real),
    )
    return lax.select(lam == 0, jnp.zeros_like(result), result)


_MAX_ITERS = np.int32(np.iinfo(np.int32).max)


def _knuth(key: Array, lam: Array) -> Array:
    """Knuth's algorithm, as JAX's. An element of rate 0 takes no step."""

    def body(carry: Any) -> Any:
        i, k, rng, log_prod = carry
        rng, subkey = jax.random.split(rng)
        k = lax.select(log_prod > -lam, k + 1, k)
        u = jax.random.uniform(subkey, lam.shape, np.float32)
        return i + 1, k, rng, log_prod + jnp.log(u)

    def cond(carry: Any) -> Any:
        i, log_prod = carry[0], carry[3]
        return (log_prod > -lam).any() & (i < _MAX_ITERS)

    k = jnp.zeros(lam.shape, np.int32)
    log_rate = jnp.zeros(lam.shape, np.float32)
    out = lax.while_loop(cond, body, (0, k, key, log_rate))[1]
    return (out - 1).astype(np.int32)


def _rejection(key: Array, lam: Array, real: Array) -> Array:
    """Hormann's transformed rejection, as JAX's, with padding accepted."""
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)

    def body(carry: Any) -> Any:
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = jax.random.split(key, 3)
        u = jax.random.uniform(subkey_0, lam.shape, lam.dtype) - 0.5
        v = jax.random.uniform(subkey_1, lam.shape, lam.dtype)
        u_shifted = 0.5 - abs(u)
        k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
        t = -lam + k * log_lam - lax.lgamma(k + 1)
        accept1 = (u_shifted >= 0.07) & (v <= v_r)
        reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
        accept2 = s <= t
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(accept, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond(carry: Any) -> Any:
        i, _, accepted, _ = carry
        return (~accepted).any() & (i < _MAX_ITERS)

    k_init = jnp.full(lam.shape, -1, lam.dtype)
    out = lax.while_loop(cond, body, (0, k_init, ~real, key))[1]
    return out.astype(np.int32)
