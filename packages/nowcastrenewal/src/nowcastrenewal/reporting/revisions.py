# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Revisions of a reporting triangle's counts.

Each reported count $y$ of stratum $g$, reference day $t$ and report day
$r$ may be revised:

- cases reported in error, Poisson($o_g y$), are in the releases from $r$
  and removed $k_g$ days later;
- cases filed under a nearby reference day, Binomial($y$, $m_g$), moved
  $\pm(1 + \text{Poisson}(1))$ days (never after $r$, never before the
  first day), are moved back $k_g$ days later.

A correction past the last report day falls on it, so the counts as of
the last day are the unrevised ones.

Over-reporting bursts (`OverReportingBursts`) are short and large: for a
few reference days in one stratum, the first report is $m$ times the
day's final count, the excess, Poisson($m Y_{g,t} - y_{g,t,d^*}$) with
$Y$ the final count and $d^*$ the first delay with a report, withdrawn on
the first report day at least $k$ days later (or the last day). Bursts
add to the counts before the revisions (`revise`'s `bursts`).

The revised counts are increments by
reference day and report day, `(G, T, T)`: they leave the delay window,
so they are not a `(G, T, D)` triangle.

The draws are numpyro sites, `"{name}::errors"`, `"{name}::moved"`,
`"{name}::later"` (direction, 0 earlier, 1 later) and `"{name}::extra"`.
PyRenew has no revision process, so this is our own `RandomVariable`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

__all__ = ["OverReportingBursts", "Revisions", "revise"]


def revise(
    counts: ArrayLike,
    errors: ArrayLike,
    moved: ArrayLike,
    shift: ArrayLike,
    keep: ArrayLike,
    n_days: int | None = None,
    bursts: ArrayLike | None = None,
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
    n_days : int, optional
        The triangle's days, the first `n_days` of `T`; all of them if
        None. Later days pad the arrays to a shared shape and hold none.
    bursts : array_like, optional
        Increments added to the counts by reference day and delay,
        `(G, T, D)`, not revised (`OverReportingBursts`).

    Returns
    -------
    Array
        Increments by stratum, reference day and report day, `(G, T, T)`.
    """
    y = jnp.asarray(counts)
    G, T, D = y.shape
    last = (T if n_days is None else int(n_days)) - 1
    g = jnp.broadcast_to(jnp.arange(G)[:, None, None], (G, T, D))
    t = jnp.broadcast_to(jnp.arange(T)[None, :, None], (G, T, D))
    r = t + jnp.arange(D)[None, None, :]
    inside = r <= last
    k = jnp.maximum(1, jnp.round(jnp.asarray(keep))).astype(int)
    later = jnp.minimum(r + k[:, None, None], last)
    to = jnp.clip(t + jnp.asarray(shift).astype(int), 0, r)
    rr = jnp.minimum(r, last)

    def on(x: ArrayLike) -> Array:
        return jnp.where(inside, jnp.asarray(x), 0)

    err, mv = on(errors), on(moved)
    out = jnp.zeros((G, T, T), dtype=jnp.result_type(y, err, float))
    if bursts is not None:
        out = out.at[g, t, rr].add(on(bursts))
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
    n_days : int, optional
        The triangle's days (`revise`).
    """

    def __init__(
        self,
        name: str,
        over: ArrayLike,
        move: ArrayLike,
        keep: ArrayLike,
        n_days: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.n_days = n_days
        self.over = np.atleast_1d(np.asarray(over, dtype=float))
        self.move = np.atleast_1d(np.asarray(move, dtype=float))
        self.keep = np.atleast_1d(np.asarray(keep, dtype=float))

    def sample(
        self,
        counts: ArrayLike,
        bursts: ArrayLike | None = None,
        **kwargs: object,
    ) -> Array:
        """Draw the revisions and return the revised counts.

        Parameters
        ----------
        counts : array_like
            Reported counts, `(G, T, D)`.
        bursts : array_like, optional
            Increments added unrevised, `(G, T, D)` (`revise`).
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
        out = revise(y, errors, moved, shift, keep, self.n_days, bursts)
        numpyro.deterministic(f"{self.name}::counts", out)
        return out


class OverReportingBursts(RandomVariable):
    """Short over-reporting bursts, withdrawn within days (module docstring).

    Parameters
    ----------
    name : str
        Name of the component; its draws are `"{name}::errors"`, by
        stratum and reference day.
    frame : object
        The triangle's frame (`effects.ReferenceFrame`).
    strata : sequence of int
        Each burst's stratum, from 1.
    at : sequence of float
        Each burst's first reference day, a share of the observed days.
    days : sequence of int
        Each burst's reference days.
    mult : sequence of float
        Each burst's first report over the final count, $m$.
    keep : sequence of int
        Each burst's days until the excess is withdrawn, $k$.
    """

    def __init__(
        self,
        name: str,
        frame: Any,
        strata: Any,
        at: Any,
        days: Any,
        mult: Any,
        keep: Any,
    ) -> None:
        super().__init__(name=name)
        G, T = len(frame.place), len(frame.dates)
        self.last = int(frame.observed) - 1
        m = np.zeros((G, T))
        k = np.ones((G, T), dtype=int)
        for s, a, n, x, w in zip(
            np.atleast_1d(strata),
            np.atleast_1d(at),
            np.atleast_1d(days),
            np.atleast_1d(mult),
            np.atleast_1d(keep),
            strict=True,
        ):
            start = int(np.round(float(a) * frame.observed))
            window = np.arange(start, min(start + int(n), frame.observed))
            m[int(s) - 1, window] = np.maximum(m[int(s) - 1, window], x)
            k[int(s) - 1, window] = max(1, int(w))
        self.mult = m
        self.keep = k

    def sample(
        self, counts: ArrayLike, log_expected: ArrayLike, **kwargs: object
    ) -> Array:
        """Draw the bursts and return them as increments.

        Parameters
        ----------
        counts : array_like
            Reported counts, `(G, T, D)`.
        log_expected : array_like
            Log expected reports, `(G, T, D)`; a delay with none (minus
            infinity, as off a report day) takes no withdrawal.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Increments by stratum, reference day and delay, `(G, T, D)`:
            the excess at the first report and its withdrawal.
        """
        y = jnp.asarray(counts)
        G, T, D = y.shape
        reported = y > 0
        first = jnp.argmax(reported, axis=-1)
        total = y.sum(axis=-1)
        head = jnp.take_along_axis(y, first[..., None], axis=-1)[..., 0]
        d = jnp.arange(D)[None, None, :]
        keep = jnp.asarray(self.keep)
        report = jnp.isfinite(jnp.asarray(log_expected))
        later = report & (d >= (first + keep)[..., None])
        withdraw = jnp.argmax(later, axis=-1)
        t = jnp.arange(T)[None, :]
        withdraw = jnp.minimum(withdraw, self.last - t)
        mult = jnp.asarray(self.mult)
        on = (
            (mult > 0)
            & reported.any(axis=-1)
            & later.any(axis=-1)
            & (withdraw > first)
        )
        rate = jnp.where(on, jnp.maximum(mult * total - head, 0.0), 0.0)
        errors = numpyro.sample(f"{self.name}::errors", dist.Poisson(rate))
        g = jnp.broadcast_to(jnp.arange(G)[:, None], (G, T))
        tt = jnp.broadcast_to(t, (G, T))
        out = jnp.zeros((G, T, D), dtype=jnp.result_type(y, float))
        out = out.at[g, tt, first].add(errors)
        return out.at[g, tt, withdraw].add(-errors)
