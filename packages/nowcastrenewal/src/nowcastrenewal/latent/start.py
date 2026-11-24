# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Each stratum's infections over the generation time's span before day 1.

`InitializeStrata` is a PyRenew `InfectionInitializationMethod` for
strata: an outbreak that starts in the series has its seed on the last
day before day 1, grown into it over the span at the daily growth factor
of its starting R (as PyRenew's exponential seeding does), so the renewal
carries on that growth from day 1 rather than falling from a one-day
seed; an established or endemic one runs at its level over the whole
span. It is used through PyRenew's `InfectionInitializationProcess`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.infection_initialization_method import (
    InfectionInitializationMethod,
)

__all__ = ["InitializeStrata"]


class InitializeStrata(InfectionInitializationMethod):
    """A seed grown into the last day, or a constant level, per stratum.

    Parameters
    ----------
    n_timepoints : int
        Days before day 1 (the generation time's span).
    early : bool
        Whether the outbreak starts in the series (a seed of at least 1
        on the last day, in strata with a positive value) or runs at its
        level before it.
    active : int, optional
        The span's last days that hold the seed or level, the generation
        time's own span; all of them if None. Earlier days are 0, padding
        the span to a shared shape.
    growth : float
        For an outbreak that starts in the series, the daily growth factor
        its seed grows by into the last day: the seed is `growth ** -a` of
        it `a` days before the last.
    """

    def __init__(
        self,
        n_timepoints: int,
        early: bool,
        active: int | None = None,
        growth: float = 1.0,
    ) -> None:
        super().__init__(n_timepoints)
        self.early = early
        self.active = n_timepoints if active is None else int(active)
        self.growth = float(growth)

    def initialize_infections(self, I_pre_init: ArrayLike) -> Array:
        """Return the starting infections.

        Parameters
        ----------
        I_pre_init : array_like
            Each stratum's seed or level, 0 for strata that start
            without infections.

        Returns
        -------
        Array
            `(n_timepoints, k)`.
        """
        x = jnp.atleast_1d(jnp.asarray(I_pre_init, dtype=float))
        days = jnp.arange(self.n_timepoints)
        on = days >= self.n_timepoints - self.active
        if not self.early:
            return jnp.where(on[:, None], x[None, :], 0.0)
        seed = jnp.where(x > 0, jnp.maximum(1.0, x), 0.0)
        before = self.n_timepoints - 1 - days
        grown = seed[None, :] * self.growth ** -before[:, None]
        return jnp.where(on[:, None], grown, 0.0)
