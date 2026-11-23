# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Discrete-time hazards of report to probabilities.

$$p_d = h_d \prod_{j < d} (1 - h_j),$$

on the log scale, for the fitted models' delays (`model.hazard_delay`)
and the reporting triangle (`reporting.triangle`). With `close=True` a
last hazard of 1 is appended, so the probabilities sum to 1 over one more
delay; otherwise the last hazard is used as given, so a finite one leaves
some reports beyond the window (epinowcast's `hazard_to_log_prob`).

The running sum of $\log(1 - h)$ is a product with a triangular matrix,
faster than `cumsum` on CPU forward and backward. PyRenew has no hazard
parametrisation of a delay.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["log_hazard_probs"]


def log_hazard_probs(logit_hazard: Array, close: bool = False) -> Array:
    r"""Return $\log p_d$ from logit hazards on the last axis.

    Parameters
    ----------
    logit_hazard : Array
        Logit hazards for delays 0, ..., D - 1, shape `(..., D)`. Only the
        last may be $+\infty$ (a hazard of 1).
    close : bool
        Append a hazard of 1 at delay D, so the result has D + 1 entries
        summing to 1.

    Returns
    -------
    Array
        Shape `(..., D)`, or `(..., D + 1)` with `close`.
    """
    k = logit_hazard.shape[-1]
    log_h = jax.nn.log_sigmoid(logit_hazard)
    log_1mh = jax.nn.log_sigmoid(-logit_hazard)
    if close:
        steps, n_out = log_1mh, k + 1
        zero = jnp.zeros((*log_h.shape[:-1], 1), log_h.dtype)
        log_h = jnp.concatenate([log_h, zero], axis=-1)
    else:
        # The last hazard's log(1 - h) is never needed, and may be -inf.
        steps, n_out = log_1mh[..., :-1], k
    # Column d sums the terms before d: the log survival before each delay.
    upper = jnp.triu(jnp.ones((steps.shape[-1], n_out), log_h.dtype), 1)
    return log_h + steps @ upper
