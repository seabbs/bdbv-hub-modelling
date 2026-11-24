# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Expected reference events (onsets, tests, admissions) from infections.

`delayed_events` spreads each place's infections over an
infection-to-reference delay with PyRenew's
`compute_delay_ascertained_incidence` and scales them by an
ascertainment. A place may have run at a constant level before the first
day, so its first days also hold the delayed part of that history. The
fitted models' onsets (`model.observation.OnsetStreams`, one random delay
shared by every place) and the simulator's reference events
(`ReferenceEvents`, one fixed delay per stratum) both use it.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.metaclass import RandomVariable

__all__ = ["ReferenceEvents", "delayed_events"]


def _one(x: Array, pmf: Array, asc: Array, before: Array) -> Array:
    # The history before the first day, long enough that the "valid"
    # convolution returns one value per day.
    past = jnp.full(pmf.shape[0] - 1, before, dtype=x.dtype)
    out, _ = compute_delay_ascertained_incidence(
        jnp.concatenate([past, x]), pmf, asc
    )
    return jnp.asarray(out)


def delayed_events(
    infections: ArrayLike,
    pmfs: ArrayLike | Sequence[ArrayLike],
    ascertainment: ArrayLike = 1.0,
    before: ArrayLike | None = None,
) -> Array:
    r"""Return each place's expected reference events.

    With infections $I_k(t)$, delay pmf $f_k$ and ascertainment
    $\alpha_k$,

    $$E_k(t) = \alpha_k \sum_{d \ge 0} f_k(d) I_k(t - d),$$

    where $I_k(t) = b_k$ on the days before the first.

    Parameters
    ----------
    infections : array_like
        Days by places.
    pmfs : array_like or sequence of array_like
        One pmf on 0, 1, ... days for every place, or one per place (which
        may differ in length).
    ascertainment : array_like
        $\alpha$, a scalar or one per place.
    before : array_like, optional
        Each place's constant infections before the first day, $b$; 0 by
        default.

    Returns
    -------
    Array
        Days by places, the shape of `infections`.

    Raises
    ------
    ValueError
        If there is not one pmf per place.
    """
    x = jnp.asarray(infections)
    k = x.shape[1]
    b = jnp.zeros(k, x.dtype) if before is None else jnp.asarray(before)
    asc = jnp.broadcast_to(jnp.asarray(ascertainment), (k,))
    if not isinstance(pmfs, (list, tuple)):
        pmf = jnp.asarray(pmfs)
        return jax.vmap(
            lambda col, a, b0: _one(col, pmf, a, b0),
            in_axes=(1, 0, 0),
            out_axes=1,
        )(x, asc, b)
    if len(pmfs) != k:
        raise ValueError("there must be one delay per place")
    cols = [
        _one(x[:, j], jnp.asarray(pmf), asc[j], b[j])
        for j, pmf in enumerate(pmfs)
    ]
    return jnp.stack(cols, axis=1)


class ReferenceEvents(RandomVariable):
    r"""Each stratum's expected reference events (`delayed_events`).

    With infections $I_k(t)$, delay pmf $f_k$ and ascertainment $\alpha_k$,

    $$E_k(t) = \alpha_k \sum_{d \ge 0} f_k(d) I_k(t - d),$$

    where $I_k(t) = b_k$ on the days before the first, $b_k$ the
    stratum's level before the span (0 for an outbreak that starts in
    it).

    Parameters
    ----------
    name : str
        Name of the component. The events are recorded as the
        deterministic site `"{name}::expected"`.
    delay_rvs : sequence of RandomVariable
        One delay pmf per stratum on 0, 1, ... days, such as PyRenew's
        `DeterministicPMF`. The pmfs may differ in length.
    ascertainment_rv : RandomVariable
        Returns $\alpha$, a scalar or one value per stratum.
    """

    def __init__(
        self,
        name: str,
        delay_rvs: Sequence[RandomVariable],
        ascertainment_rv: RandomVariable,
    ) -> None:
        super().__init__(name=name)
        self.delay_rvs = list(delay_rvs)
        self.ascertainment_rv = ascertainment_rv

    def sample(
        self,
        infections: ArrayLike,
        before: ArrayLike | None = None,
        **kwargs: object,
    ) -> Array:
        """Return the expected reference events.

        Parameters
        ----------
        infections : array_like
            Days by strata.
        before : array_like, optional
            Each stratum's constant infections before the first day.
            `None` is none.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Days by strata, the shape of `infections`.

        Raises
        ------
        ValueError
            If there is not one delay per stratum.
        """
        x = jnp.asarray(infections)
        if len(self.delay_rvs) != x.shape[1]:
            raise ValueError("there must be one delay per stratum")
        out = delayed_events(
            x,
            [rv() for rv in self.delay_rvs],
            self.ascertainment_rv(),
            before,
        )
        numpyro.deterministic(f"{self.name}::expected", out)
        return out
