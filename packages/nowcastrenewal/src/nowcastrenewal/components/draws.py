# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Large draws made on a flat array of a few sizes, for shared compiles.

JAX compiles a sampler once per shape, and the count samplers (Poisson,
gamma-Poisson, binomial, Bernoulli) take seconds each. A simulation's
reporting triangle is strata by days by delays, so nearly every
simulation drew them at a shape of its own. `padded_draws` is a numpyro
handler that draws such a site on one flat array of a few lengths
(`flat_size`: four a doubling, so at most a quarter of the cells are
padding), and keeps the first cells in the site's shape: the
parameters are padded with their last value, so each cell has its own
distribution, and only which draw lands in which cell
changes. The site's shape, `log_prob` and any observed value are as
before, so a model under it can still be conditioned.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array
from numpyro.primitives import Messenger

__all__ = ["MIN_SIZE", "PaddedDraws", "flat_size", "padded_draws"]

MIN_SIZE = 1024
"""The smallest flat length a draw is made on."""


def flat_size(n: int) -> int:
    """Return the flat length `n` cells are drawn on.

    `MIN_SIZE` for at most that many; otherwise `n` rounded up to a
    multiple of an eighth of the next power of two, so 5, 6, 7 or 8
    eighths of it: four lengths a doubling, at most a quarter padding.

    Parameters
    ----------
    n : int
        The site's cells.

    Returns
    -------
    int
        The flat length, at least `n`.
    """
    if n <= MIN_SIZE:
        return MIN_SIZE
    step = 1 << ((n - 1).bit_length() - 3)
    return -(-n // step) * step


_PARAMETERS: tuple[tuple[type, tuple[str, ...]], ...] = (
    (dist.GammaPoisson, ("concentration", "rate")),
    (dist.Poisson, ("rate",)),
    (dist.BinomialProbs, ("total_count", "probs")),
    (dist.BernoulliProbs, ("probs",)),
)
"""The samplers drawn flat, with the parameters that rebuild each."""


def _family(d: dist.Distribution) -> tuple[type, tuple[str, ...]] | None:
    for cls, names in _PARAMETERS:
        if isinstance(d, cls):
            return cls, names
    return None


class PaddedDraws(dist.Distribution):
    """A distribution sampled on a flat array of a `flat_size` length.

    Parameters
    ----------
    base : numpyro.distributions.Distribution
        A Poisson, gamma-Poisson (such as `NegativeBinomial2`), binomial
        or Bernoulli with no event shape.
    """

    def __init__(self, base: dist.Distribution) -> None:
        family = _family(base)
        if family is None or base.event_shape:
            raise ValueError(f"no flat draws for {type(base).__name__}")
        self.base = base
        self._cls, self._names = family
        super().__init__(batch_shape=base.batch_shape)

    @property
    def support(self) -> Any:
        """The base distribution's support."""
        return self.base.support

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw on a flat array and return the cells in the site's shape.

        Parameters
        ----------
        key : jax.Array
            The PRNG key.
        sample_shape : tuple of int
            Leading sample dimensions.

        Returns
        -------
        jax.Array
            Shape `sample_shape + batch_shape`.
        """
        shape = tuple(sample_shape) + tuple(self.batch_shape)
        n = math.prod(shape)
        if n == 0:
            return jnp.asarray(self.base.sample(key, sample_shape))
        size = flat_size(n)
        params = {
            name: jnp.pad(
                jnp.broadcast_to(getattr(self.base, name), shape).ravel(),
                (0, size - n),
                mode="edge",
            )
            for name in self._names
        }
        flat = jnp.asarray(self._cls(**params).sample(key))
        return flat[:n].reshape(shape)

    def log_prob(self, value: Array) -> Array:
        """Return the base distribution's log density.

        Parameters
        ----------
        value : jax.Array
            Values in the site's shape.

        Returns
        -------
        jax.Array
            The base's `log_prob`.
        """
        return jnp.asarray(self.base.log_prob(value))


class padded_draws(Messenger):  # noqa: N801 (a numpyro handler)
    """Draw sites of at least `min_ndim` batch dimensions flat.

    Sample sites that are not observed and whose distribution, or
    masked distribution, is one `PaddedDraws` takes are drawn through
    it; others are left as they are.

    Parameters
    ----------
    fn : callable, optional
        The model, as for any numpyro handler.
    min_ndim : int
        The fewest batch dimensions a site needs to be drawn flat.
    """

    def __init__(self, fn: Any = None, min_ndim: int = 3) -> None:
        self.min_ndim = min_ndim
        super().__init__(fn)

    def process_message(self, msg: dict[str, Any]) -> None:
        """Wrap a large sample site's distribution in `PaddedDraws`.

        A masked distribution (as numpyro's `mask` handler makes around
        the count noise) keeps its mask over the wrapped one.
        """
        fn = msg.get("fn")
        if msg["type"] != "sample" or msg["is_observed"]:
            return
        mask = None
        if isinstance(fn, dist.MaskedDistribution):
            fn, mask = fn.base_dist, fn._mask
        if (
            isinstance(fn, dist.Distribution)
            and len(fn.batch_shape) >= self.min_ndim
            and not fn.event_shape
            and _family(fn) is not None
        ):
            padded = PaddedDraws(fn)
            msg["fn"] = padded if mask is None else padded.mask(mask)
