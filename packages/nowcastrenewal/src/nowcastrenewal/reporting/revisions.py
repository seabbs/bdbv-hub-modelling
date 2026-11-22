# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Revisions of a reporting triangle's counts.

Each reported count $y$ of stratum $g$, reference day $t$ and report day
$r$ may be revised (the simulator R engine's `revise_counts`):

- cases reported in error, Poisson($o_g y$), are in the releases from $r$
  and removed $k_g$ days later;
- cases filed under a nearby reference day, Binomial($y$, $m_g$), moved
  $\pm(1 + \text{Poisson}(1))$ days (never after $r$, never before the
  first day), are moved back $k_g$ days later.

A correction past the last report day falls on it, so the counts as of
the last day are the unrevised ones. The revised counts are increments by
reference day and report day, `(G, T, T)`: they leave the delay window,
so they are not a `(G, T, D)` triangle.

The draws are numpyro sites, `"{name}::errors"`, `"{name}::moved"`,
`"{name}::later"` (direction, 0 earlier, 1 later) and `"{name}::extra"`.
PyRenew has no revision process, so this is our own `RandomVariable`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

__all__ = ["Revisions", "revise"]


def revise(
    counts: ArrayLike,
    errors: ArrayLike,
    moved: ArrayLike,
    shift: ArrayLike,
    keep: ArrayLike,
) -> Array:
    """Return the revised counts by reference and report day.

    Parameters
    ----------
    counts : array_like
        Reported counts, `(G, T, D)`; cells reported after the last day
        (`t + d >= T`) are dropped.
    errors, moved : array_like
        Each cell's cases in error and cases moved, `(G, T, D)`.
    shift : array_like
        Each cell's move in days, nonzero, `(G, T, D)`.
    keep : array_like
        Days until a correction, by stratum, `(G,)` (rounded, at least 1).

    Returns
    -------
    Array
        Increments by stratum, reference day and report day, `(G, T, T)`.
    """
    y = jnp.asarray(counts)
    G, T, D = y.shape
    g = jnp.broadcast_to(jnp.arange(G)[:, None, None], (G, T, D))
    t = jnp.broadcast_to(jnp.arange(T)[None, :, None], (G, T, D))
    r = t + jnp.arange(D)[None, None, :]
    inside = r <= T - 1
    k = jnp.maximum(1, jnp.round(jnp.asarray(keep))).astype(int)
    later = jnp.minimum(r + k[:, None, None], T - 1)
    to = jnp.clip(t + jnp.asarray(shift).astype(int), 0, r)
    rr = jnp.minimum(r, T - 1)

    def on(x: ArrayLike) -> Array:
        return jnp.where(inside, jnp.asarray(x), 0)

    err, mv = on(errors), on(moved)
    out = jnp.zeros((G, T, T), dtype=jnp.result_type(y, err, float))
    for ref, rep, x in (
        (t, rr, on(y)),
        (t, rr, err),
        (t, later, -err),
        (t, rr, -mv),
        (to, rr, mv),
        (t, later, mv),
        (to, later, -mv),
    ):
        out = out.at[g, ref, rep].add(x)
    return out


class Revisions(RandomVariable):
    """Revisions of a triangle's counts (module docstring).

    Parameters
    ----------
    name : str
        Name of the component.
    over, move, keep : array_like
        By stratum: the error rate $o$, the share moved $m$ and the days
        until a correction $k$ (`revise_over`, `revise_move`,
        `revise_keep`).
    """

    def __init__(
        self, name: str, over: ArrayLike, move: ArrayLike, keep: ArrayLike
    ) -> None:
        super().__init__(name=name)
        self.over = np.atleast_1d(np.asarray(over, dtype=float))
        self.move = np.atleast_1d(np.asarray(move, dtype=float))
        self.keep = np.atleast_1d(np.asarray(keep, dtype=float))

    def sample(self, counts: ArrayLike, **kwargs: object) -> Array:
        """Draw the revisions and return the revised counts.

        Parameters
        ----------
        counts : array_like
            Reported counts, `(G, T, D)`.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Increments by stratum, reference day and report day,
            `(G, T, T)`; recorded as `"{name}::counts"`.
        """
        y = jnp.asarray(counts)
        G = y.shape[0]
        over = jnp.broadcast_to(jnp.asarray(self.over), (G,))[:, None, None]
        move = jnp.broadcast_to(jnp.asarray(self.move), (G,))[:, None, None]
        n = y.astype(int)
        errors = numpyro.sample(f"{self.name}::errors", dist.Poisson(over * y))
        moved = numpyro.sample(
            f"{self.name}::moved", dist.Binomial(total_count=n, probs=move)
        )
        direction = numpyro.sample(
            f"{self.name}::later", dist.Bernoulli(probs=0.5 * jnp.ones_like(y))
        )
        extra = numpyro.sample(
            f"{self.name}::extra", dist.Poisson(jnp.ones_like(y))
        )
        shift = (2 * direction - 1) * (1 + extra)
        keep = jnp.broadcast_to(jnp.asarray(self.keep), (G,))
        out = revise(y, errors, moved, shift, keep)
        numpyro.deterministic(f"{self.name}::counts", out)
        return out
