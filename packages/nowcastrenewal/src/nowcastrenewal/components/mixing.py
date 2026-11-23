# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Mixing of transmission between places or strata.

A kernel $K$ has rows that receive: $K_{pq}$ is the weight of origin $q$
in what $p$ takes in, and its diagonal is 0. With shares $\varepsilon_q$
of each origin's force that move (per origin, and per day where they
change), the mixing matrix in counts is

$$W_{pq} = \delta_{pq}\,(1 - \ell_p) + K_{pq}\,\varepsilon_q,$$

applied to the generation-time-weighted infections $F$ as $W F$. What a
place keeps, $1 - \ell_p$, is set by `outflow`:

- `"sent"`: $\ell_p = \varepsilon_p \sum_r K_{rp}$, so each origin loses
  what it sends and the total force is kept. With a kernel normalised per
  origin (`per="origin"`) this is $\ell_p = \varepsilon_p$: the fitted
  models' importation.
- `"exported"`: $\ell_p = \varepsilon_p$, so each origin loses its
  whole share wherever it goes. With a block of a kernel normalised per
  origin over more receivers than the block lists, what goes to the
  unlisted receivers leaves: the zone models' importation, whose silent
  zones receive from the fitted zones after the fit. On a kernel whose
  columns sum to 1 it is `"sent"`.
- `"received"`: $\ell_p = \sum_q K_{pq} \varepsilon_q$, so each place's
  force is a weighted average of its own and its sources'. With a kernel
  normalised per receiver (`per="receiver"`) and one share a day this is
  $(1 - \varepsilon_t) F_p + \varepsilon_t \sum_q K_{pq} F_q$: the
  simulator's mixing.

The two agree when the kernel is normalised both ways (the simulator's
symmetric kind). Per resident, with population fractions $f$, the matrix
is $\operatorname{diag}(1 / f)\,W \operatorname{diag}(f)$.

PyRenew's `MobilitySubpopulationInfections` (Roy et al. 2025) mixes where
people are present, with a fixed column-stochastic mobility matrix and
depletion inside the step; it has no share that moves, no share that
changes over time and no kernel from distances. Hence this component.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

__all__ = [
    "OUTFLOWS",
    "Mixing",
    "distance_kernel",
    "great_circle_km",
    "mixing_matrix",
    "normalise_kernel",
]

OUTFLOWS = ("sent", "exported", "received")
"""What each place gives up (see the module docstring)."""

_EARTH_KM = 6371.0


def great_circle_km(latitude: ArrayLike, longitude: ArrayLike) -> np.ndarray:
    """Return the great-circle distances in km between points.

    Parameters
    ----------
    latitude, longitude : array_like
        Each point's latitude and longitude in degrees.

    Returns
    -------
    numpy.ndarray
        Symmetric, zero diagonal.

    Raises
    ------
    ValueError
        If a coordinate is missing.
    """
    lat = np.radians(np.asarray(latitude, dtype=float))
    lon = np.radians(np.asarray(longitude, dtype=float))
    if np.isnan(lat).any() or np.isnan(lon).any():
        raise ValueError("Missing place centre.")
    a = (
        np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    )
    return np.asarray(2 * _EARTH_KM * np.arcsin(np.sqrt(a)), dtype=float)


def normalise_kernel(weights: ArrayLike, per: str) -> np.ndarray:
    """Return the kernel normalised per origin or per receiver.

    Parameters
    ----------
    weights : array_like
        Non-negative weights, rows receive.
    per : {"origin", "receiver"}
        `"origin"`: each column (what an origin sends) sums to 1;
        `"receiver"`: each row (what a place takes in) sums to 1. A row or
        column with no weight stays 0.

    Returns
    -------
    numpy.ndarray
        The kernel.

    Raises
    ------
    ValueError
        If `per` is neither.
    """
    k = np.array(weights, dtype=float)
    if per not in ("origin", "receiver"):
        raise ValueError("per must be 'origin' or 'receiver'")
    axis = 0 if per == "origin" else 1
    total = k.sum(axis=axis, keepdims=True)
    return np.asarray(np.where(total > 0, k / np.where(total > 0, total, 1), 0))


def distance_kernel(
    distances: ArrayLike,
    decay: float,
    per: str,
    weights: ArrayLike | None = None,
) -> np.ndarray:
    r"""Return a kernel decaying with distance, rows receive.

    $K_{pq} \propto w_p e^{-d_{pq} / \text{decay}}$ for $p \ne q$, with $w$
    the receivers' weights (for example their populations), normalised
    by `normalise_kernel`.

    Parameters
    ----------
    distances : array_like
        Distances between places, symmetric.
    decay : float
        The decay length, in the distances' units.
    per : {"origin", "receiver"}
        How it is normalised (`normalise_kernel`).
    weights : array_like, optional
        Each receiver's weight; equal by default.

    Returns
    -------
    numpy.ndarray
        The kernel, zero diagonal.

    Raises
    ------
    ValueError
        If the kernel is not finite.
    """
    d = np.asarray(distances, dtype=float)
    w = np.ones(d.shape[0]) if weights is None else np.asarray(weights, float)
    k = w[:, None] * np.exp(-d / decay)
    np.fill_diagonal(k, 0.0)
    k = normalise_kernel(k, per)
    if not np.all(np.isfinite(k)):
        raise ValueError("The kernel is not finite.")
    return k


def mixing_matrix(
    kernel: ArrayLike,
    share: ArrayLike,
    outflow: str = "sent",
    fractions: ArrayLike | None = None,
) -> Array:
    r"""Return the mixing matrix, in counts or per resident.

    Parameters
    ----------
    kernel : array_like
        $K$, rows receive, `(k, k)`.
    share : array_like
        $\varepsilon$ by origin on the last axis, broadcast to `(..., k)`:
        a scalar, one per origin `(k,)`, or one per day `(n, 1)`.
    outflow : {"sent", "exported", "received"}
        What each place gives up (module docstring).
    fractions : array_like, optional
        Population fractions `(k,)`: the matrix is then per resident.

    Returns
    -------
    Array
        $W$, shape `(..., k, k)` with the leading shape of `share`.

    Raises
    ------
    ValueError
        If `outflow` is not one of `OUTFLOWS`.
    """
    if outflow not in OUTFLOWS:
        raise ValueError(f"outflow must be one of {OUTFLOWS}")
    K = jnp.asarray(kernel)
    k = K.shape[0]
    share_ = jnp.asarray(share)
    eps = jnp.broadcast_to(
        share_, (*share_.shape[:-1], k) if share_.ndim else (k,)
    )
    moved = K * eps[..., None, :]
    if outflow == "sent":
        leave = eps * K.sum(axis=0)
    elif outflow == "exported":
        leave = eps
    else:
        leave = moved.sum(axis=-1)
    W = moved + jnp.eye(k) * (1 - leave)[..., None, :]
    if fractions is None:
        return W
    f = jnp.asarray(fractions)
    return W * f[None, :] / f[:, None]


class Mixing(RandomVariable):
    r"""The mixing matrix from a kernel and its share (`mixing_matrix`).

    Parameters
    ----------
    name : str
        Name of the component.
    kernel_rv : RandomVariable
        Returns $K$, rows receive. A fixed kernel is a PyRenew
        `DeterministicVariable`.
    share_rv : RandomVariable
        Returns $\varepsilon$ (`mixing_matrix`).
    outflow : {"sent", "exported", "received"}
        What each place gives up.

    Raises
    ------
    TypeError
        If `kernel_rv` or `share_rv` is not a `RandomVariable`.
    ValueError
        If `outflow` is not one of `OUTFLOWS`.
    """

    def __init__(
        self,
        name: str,
        kernel_rv: RandomVariable,
        share_rv: RandomVariable,
        outflow: str = "sent",
    ) -> None:
        for label, rv in (("kernel_rv", kernel_rv), ("share_rv", share_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        if outflow not in OUTFLOWS:
            raise ValueError(f"outflow must be one of {OUTFLOWS}")
        super().__init__(name=name)
        self.kernel_rv = kernel_rv
        self.share_rv = share_rv
        self.outflow = outflow

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Mixing(name={self.name!r}, kernel_rv={self.kernel_rv!r}, "
            f"share_rv={self.share_rv!r}, outflow={self.outflow!r})"
        )

    def sample(
        self, fractions: ArrayLike | None = None, **kwargs: object
    ) -> Array:
        """Sample the mixing matrix.

        Parameters
        ----------
        fractions : array_like, optional
            Population fractions, for the matrix per resident.
        **kwargs
            Passed to `kernel_rv` and `share_rv` (for example the days).

        Returns
        -------
        Array
            $W$ (`mixing_matrix`).
        """
        return mixing_matrix(
            self.kernel_rv(**kwargs),
            self.share_rv(**kwargs),
            self.outflow,
            fractions,
        )
