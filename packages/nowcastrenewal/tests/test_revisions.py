"""Revisions of reported counts (`reporting.revisions`).

No external tool computes these revisions, so `revise` is checked on a
hand-worked case and `Revisions` on what revisions must leave alone.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from nowcastrenewal.reporting.revisions import Revisions, revise

G, T, D = 3, 30, 8
VALUES = {
    "revise_over": [0.05, 0.2, 0.1],
    "revise_move": [0.1, 0.3, 0.05],
    "revise_keep": [3.0, 14.0, 6.6],
}


def _counts() -> np.ndarray:
    rng = np.random.default_rng(7)
    y = rng.poisson(4.0, (G, T, D)).astype(float)
    t = np.arange(T)[:, None] + np.arange(D)[None, :]
    y[:, t > T - 1] = 0
    # The first reference day and the last report day have reports, so the
    # engine's first and last dates are the triangle's.
    y[:, 0, 0] = 3
    y[:, T - D, D - 1] = 2
    return y


COUNTS = _counts()


def test_revise_moves_errs_and_corrects_by_hand() -> None:
    # Two cells of one stratum, ten days. Cell (t=2, d=1), report day 3:
    # 5 cases, 2 in error, 1 moved a day earlier, corrected 3 days later
    # (day 6). Cell (t=8, d=1), report day 9, the last: 4 cases, 1 in
    # error, 2 moved 3 days later; the correction and the move are
    # clipped to day 9, so they cancel there. Exact.
    counts = np.zeros((1, 10, 3))
    errors, moved, shift = (np.zeros_like(counts) for _ in range(3))
    counts[0, 2, 1], errors[0, 2, 1], moved[0, 2, 1], shift[0, 2, 1] = (
        5,
        2,
        1,
        -1,
    )
    counts[0, 8, 1], errors[0, 8, 1], moved[0, 8, 1], shift[0, 8, 1] = (
        4,
        1,
        2,
        3,
    )
    out = np.asarray(revise(counts, errors, moved, shift, [3.0]))
    want = np.zeros((1, 10, 10))
    want[0, 2, 3] = 5 + 2 - 1
    want[0, 2, 6] = -2 + 1
    want[0, 1, 3] = 1
    want[0, 1, 6] = -1
    want[0, 8, 9] = 4
    np.testing.assert_array_equal(out, want)


def test_revisions_leave_the_final_counts() -> None:
    rev = Revisions("revisions", 0.3, 0.3, 5.0)
    with handlers.seed(rng_seed=1):
        out = np.asarray(rev(counts=COUNTS))
    # By the last report day each reference day has its unrevised total,
    # and cumulative counts by report day never go below zero.
    np.testing.assert_allclose(out.sum(axis=2), COUNTS.sum(axis=2))
    assert np.all(np.cumsum(out, axis=2) >= -1e-9)


def test_no_revisions_is_the_triangle_by_report_day() -> None:
    zero = jnp.zeros_like(COUNTS)
    out = np.asarray(
        revise(COUNTS, zero, zero, jnp.ones_like(COUNTS), [1.0] * G)
    )
    want = np.zeros((G, T, T))
    for dd in range(D):
        tt = np.arange(T - dd)
        want[:, tt, tt + dd] = COUNTS[:, tt, dd]
    np.testing.assert_allclose(out, want)
