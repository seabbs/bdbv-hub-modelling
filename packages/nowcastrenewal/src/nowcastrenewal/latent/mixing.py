# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The simulator's kernel between strata and its share over time.

`PlaceMixing` builds the kernel $K$ between strata (rows receive, each
age mixing with the same age elsewhere, normalised per receiver) and the
share $\varepsilon_t$ of each day, from the grid's mixing kinds. The
mixing matrix is the shared `components.mixing.mixing_matrix` with
`outflow="received"`: each stratum's force is
$(1 - \varepsilon_t) F_p + \varepsilon_t \sum_q K_{pq} F_q$.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

from ..components.mixing import distance_kernel, normalise_kernel

__all__ = ["KINDS", "PlaceMixing"]

KINDS = ("none", "symmetric", "asymmetric", "distance", "time_varying")
"""Mixing kinds: none; every place with every other; only from places
earlier in an order; decaying with distance; and the distance kernel
switching on or off over time."""


class PlaceMixing(RandomVariable):
    r"""The kernel between strata and its share on each day.

    Places without cases (`quiet`) neither give nor take infections, and
    each place's kernel row is normalised to sum to 1 where it has any
    weight.

    Parameters
    ----------
    name : str
        Name of the component.
    kind : str
        One of `KINDS`.
    share : float
        $\varepsilon$, the share of transmission that moves (at full
        strength for `time_varying`).
    order : sequence of int, optional
        For `asymmetric`, place indices (0-based) in the order infection
        moves: a place takes only from places before it. Indices past the
        number of places are ignored.
    xy : array_like, optional
        For `distance` and `time_varying`, each place's coordinates,
        `(n_places, 2)` or longer.
    distance_range : float, optional
        The kernel's decay length, $e^{-d / \text{range}}$.
    quiet : sequence of int
        Places (0-based) without cases.
    switch : str
        For `time_varying`, `"on"` or `"off"`.
    switch_at : float
        For `time_varying`, the share of the way through at which the
        switch starts (rounded to a day).
    switch_days : float
        For `time_varying`, the days the switch takes.

    Raises
    ------
    ValueError
        If `kind` is not one of `KINDS`.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        share: float = 0.0,
        order: Sequence[int] | None = None,
        xy: ArrayLike | None = None,
        distance_range: float | None = None,
        quiet: Sequence[int] = (),
        switch: str = "on",
        switch_at: float = 0.0,
        switch_days: float = 1.0,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        super().__init__(name=name)
        self.kind = kind
        self.share = share
        self.order = order
        self.xy = None if xy is None else np.asarray(xy, dtype=float)
        self.distance_range = distance_range
        self.quiet = [int(q) for q in quiet]
        self.switch = switch
        self.switch_at = switch_at
        self.switch_days = switch_days

    def place_kernel(self, n_places: int) -> np.ndarray:
        """Return the kernel between places, rows receive.

        Parameters
        ----------
        n_places : int
            Places.

        Returns
        -------
        numpy.ndarray
            `(n_places, n_places)`, each row with weight summing to 1.
        """
        kernel = np.zeros((n_places, n_places))
        if n_places < 2 or self.kind == "none":
            return kernel
        if self.kind == "symmetric":
            kernel = 1 - np.eye(n_places)
        elif self.kind == "asymmetric":
            order = [i for i in (self.order or ()) if i < n_places]
            at = np.array([order.index(i) for i in range(n_places)])
            kernel = (at[:, None] > at[None, :]).astype(float)
        else:
            if self.xy is None or self.distance_range is None:
                raise ValueError(f"{self.kind} mixing needs xy and a range")
            xy = self.xy[:n_places]
            dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            kernel = distance_kernel(dist, self.distance_range, "receiver")
        quiet = [q for q in self.quiet if q < n_places]
        kernel[quiet, :] = 0
        kernel[:, quiet] = 0
        return normalise_kernel(kernel, "receiver")

    def shares(self, n: int) -> np.ndarray:
        r"""Return $\varepsilon_t$ on each of `n` days.

        Parameters
        ----------
        n : int
            Days.

        Returns
        -------
        numpy.ndarray
            Shape `(n,)`.
        """
        if self.kind == "none":
            return np.zeros(n)
        if self.kind != "time_varying":
            return np.full(n, float(self.share))
        start = int(np.round(self.switch_at * n))
        ramp = np.clip((np.arange(1, n + 1) - start) / self.switch_days, 0, 1)
        return self.share * (ramp if self.switch == "on" else 1 - ramp)

    def sample(
        self, n: int, place: ArrayLike, age: ArrayLike, **kwargs: object
    ) -> tuple[Array, Array]:
        r"""Return the kernel between strata and the share of each day.

        Parameters
        ----------
        n : int
            Days.
        place : array_like
            Each stratum's place index (0-based).
        age : array_like
            Each stratum's age group.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        tuple of Array
            $K$, strata by strata (rows receive), and $\varepsilon$,
            shape `(n,)`.
        """
        idx = np.asarray(place, dtype=int)
        ages = np.asarray(age)
        kernel = self.place_kernel(int(idx.max()) + 1)
        K = kernel[np.ix_(idx, idx)] * (ages[:, None] == ages[None, :])
        return jnp.asarray(K), jnp.asarray(self.shares(n))
