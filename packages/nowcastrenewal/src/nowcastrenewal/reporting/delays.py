# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Discretised reporting delays: the parametric families and their pmfs.

A delay $T$ from reference to report has a continuous distribution $F$ of
one of `FAMILIES`. Two discretisations are used:

- `double_censored_log_pmf` and `double_censored_logit_hazard`: both the
  reference and the report day are censored to whole days, as in
  primarycensored (a uniform reference time over its day, daily report
  censoring, truncation at $D$), which is how epinowcast discretises at
  the simulator's pin. This is the delay of the reporting triangle; its
  logit hazard is the baseline that the hazard effects shift.
- `interval_pmf`: $p_d \propto F(d + 1) - F(d)$, a delay from the start of
  the reference day. The simulator uses it only to size the NB1
  dispersion (its $\sum_d p_d^2$).

`ParametricDelay` is a PyRenew `RandomVariable` that returns the double
censored pmf for locations and scales from other components, one per
group and day.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.scipy.special import gammainc, gammaincc, ndtr
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable
from scipy import stats

__all__ = [
    "FAMILIES",
    "DelayParameters",
    "ParametricDelay",
    "cdf_and_survival",
    "double_censored_log_pmf",
    "double_censored_logit_hazard",
    "interval_pmf",
    "log_cdf",
]

FAMILIES = ("exponential", "lognormal", "gamma")
r"""The parametric families, each by a location $\mu$ and scale $\sigma$:
exponential with rate $e^{-\mu}$; lognormal with meanlog $\mu$ and sdlog
$\sigma$; gamma with shape $e^{\mu}$ and rate $\sigma$."""


def _log1m_exp(x: Array) -> Array:
    return jnp.log(-jnp.expm1(x))


def cdf_and_survival(
    family: str, mu: ArrayLike, sigma: ArrayLike, x: ArrayLike
) -> tuple[Array, Array]:
    r"""Return $F(x)$ and $S(x) = 1 - F(x)$, each to full precision.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale, broadcast against `x`.
    x : array_like
        Positive delays.

    Returns
    -------
    tuple of Array
        $F(x)$ and $S(x)$.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """
    mu, sigma, x = jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(x)
    if family == "exponential":
        rate_x = jnp.exp(-mu) * x
        return -jnp.expm1(-rate_x), jnp.exp(-rate_x)
    if family == "lognormal":
        z = (jnp.log(x) - mu) / sigma
        return ndtr(z), ndtr(-z)
    if family == "gamma":
        shape = jnp.exp(mu)
        return gammainc(shape, sigma * x), gammaincc(shape, sigma * x)
    raise ValueError(f"family must be one of {FAMILIES}")


def log_cdf(
    family: str, mu: ArrayLike, sigma: ArrayLike, x: ArrayLike
) -> Array:
    r"""Return $\log F(x)$ for a family's location and scale.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale, broadcast against `x`.
    x : array_like
        Positive delays.

    Returns
    -------
    Array
        $\log F(x)$.
    """
    return jnp.log(cdf_and_survival(family, mu, sigma, x)[0])


def _partial_means(
    family: str, mu: Array, sigma: Array, x: Array
) -> tuple[Array, Array]:
    r"""Return $E[T; T \le x]$ and $E[T; T > x]$."""
    if family == "lognormal":
        mean = jnp.exp(mu + sigma**2 / 2)
        z = (jnp.log(x) - mu - sigma**2) / sigma
        return mean * ndtr(z), mean * ndtr(-z)
    if family in ("gamma", "exponential"):
        shape = jnp.exp(mu) if family == "gamma" else jnp.ones_like(mu)
        rate = sigma if family == "gamma" else jnp.exp(-mu)
        mean = shape / rate
        return (
            mean * gammainc(shape + 1, rate * x),
            mean * gammaincc(shape + 1, rate * x),
        )
    raise ValueError(f"family must be one of {FAMILIES}")


def _censored_mass(
    family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
) -> Array:
    r"""Return $F_c(d + 1) - F_c(d)$ for $d = 0, \dots, D - 1$.

    $F_c(q) = \int_{q - 1}^{q} F(t)\,dt$ is the delay's cdf with the
    reference time uniform over its day. With $G(t) = \int_0^t F$ and
    $U(t) = \int_t^\infty S$, the mass is $G(d + 1) - 2 G(d) + G(d - 1)$,
    taken below the median, and $U(d - 1) - 2 U(d) + U(d + 1)$ above it,
    so the tail keeps its digits.
    """
    if dmax < 3:
        raise ValueError("dmax must be at least 3")
    mu = jnp.asarray(mu)[..., None]
    sigma = jnp.asarray(sigma)[..., None]
    t = jnp.arange(1, dmax + 1, dtype=float)
    f, s = cdf_and_survival(family, mu, sigma, t)
    lower, upper = _partial_means(family, mu, sigma, t)
    # G(t) and U(t) at t = 1, ..., D; G(0) = 0 and U(0) = E[T].
    g = t * f - lower
    u = upper - t * s
    lead = (*g.shape[:-1], 1)
    zero = jnp.zeros(lead, g.dtype)
    g = jnp.concatenate([zero, zero, g], axis=-1)  # t = -1, 0, ..., D
    u0 = (lower + upper)[..., :1]
    u = jnp.concatenate([u0 + 1, u0, u], axis=-1)  # U(-1) = E[T] + 1
    d = jnp.arange(dmax)
    below = g[..., d + 2] - 2 * g[..., d + 1] + g[..., d]
    above = u[..., d] - 2 * u[..., d + 1] + u[..., d + 2]
    use_below = (d == 0) | (f[..., :dmax] < 0.5)
    return jnp.where(use_below, below, above)


def double_censored_log_pmf(
    family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
) -> Array:
    r"""Return the double censored delay's log pmf on 0, ..., dmax - 1.

    primarycensored's pmf with a uniform reference time over a day, daily
    report censoring and truncation at $D$:
    $p_d = \{F_c(d + 1) - F_c(d)\} / F_c(D)$, with
    $F_c(q) = \int_{q - 1}^{q} F(t)\,dt$.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale; leading dimensions are kept.
    dmax : int
        Delays in the window, $D$ (at least 3).

    Returns
    -------
    Array
        Shape `(..., dmax)`.
    """
    mass = _censored_mass(family, mu, sigma, dmax)
    return jnp.log(mass) - jnp.log(mass.sum(axis=-1, keepdims=True))


def double_censored_logit_hazard(
    family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
) -> Array:
    r"""Return the double censored delay's logit hazard on 0, ..., dmax - 1.

    $h_d = p_d / \sum_{i \ge d} p_i$, so
    $\operatorname{logit} h_d = \log p_d - \log \sum_{i > d} p_i$, and
    $h_{D - 1} = 1$ (a logit of $+\infty$), so every report falls in the
    window.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale; leading dimensions are kept.
    dmax : int
        Delays in the window, $D$ (at least 3).

    Returns
    -------
    Array
        Shape `(..., dmax)`.
    """
    mass = _censored_mass(family, mu, sigma, dmax)
    after = jnp.flip(jnp.cumsum(jnp.flip(mass, -1), axis=-1), -1)
    rest = jnp.concatenate(
        [after[..., 1:], jnp.zeros_like(after[..., :1])], axis=-1
    )
    logit = jnp.log(mass) - jnp.log(rest)
    return logit.at[..., -1].set(jnp.inf)


@dataclass(frozen=True)
class DelayParameters:
    r"""A family's location $\mu$ and scale $\sigma$ from the grid's values.

    Attributes
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : float
        Location and scale (`FAMILIES`); the exponential's scale is unused.
    """

    family: str
    mu: float
    sigma: float

    @classmethod
    def from_values(
        cls, family: str, values: Mapping[str, Any]
    ) -> DelayParameters:
        """Return the parameters from a simulation's values.

        Lognormal from `delay_median` and `delay_sdlog`; gamma from
        `delay_mean` and `delay_cv` (shape $1 / cv^2$); exponential from
        `delay_mean`.

        Parameters
        ----------
        family : str
            One of `FAMILIES`.
        values : mapping
            The values.

        Returns
        -------
        DelayParameters
            The parameters.

        Raises
        ------
        ValueError
            If `family` is not one of `FAMILIES`.
        """
        v = values
        if family == "lognormal":
            return cls(family, math.log(v["delay_median"]), v["delay_sdlog"])
        if family == "gamma":
            shape = 1 / v["delay_cv"] ** 2
            return cls(family, math.log(shape), shape / v["delay_mean"])
        if family == "exponential":
            return cls(family, math.log(v["delay_mean"]), 1.0)
        raise ValueError(f"family must be one of {FAMILIES}")


def interval_pmf(
    family: str, values: Mapping[str, Any], max_delay: int
) -> np.ndarray:
    r"""Return $p_d \propto F(d + 1) - F(d)$ on 0, ..., max_delay - 1.

    For `"none"` (a hazard with no parametric part) the delay is geometric
    with hazard $1 / (1 + \text{delay\_mean})$.

    Parameters
    ----------
    family : str
        One of `FAMILIES` or `"none"`.
    values : mapping
        The values (`DelayParameters.from_values`).
    max_delay : int
        Delays in the window.

    Returns
    -------
    numpy.ndarray
        The pmf, summing to 1.

    Raises
    ------
    ValueError
        If `family` is unknown.
    """
    d = np.arange(max_delay + 1, dtype=float)
    v = values
    if family == "lognormal":
        cdf = stats.lognorm.cdf(d, v["delay_sdlog"], scale=v["delay_median"])
    elif family == "gamma":
        shape = 1 / v["delay_cv"] ** 2
        cdf = stats.gamma.cdf(d, shape, scale=v["delay_mean"] / shape)
    elif family == "exponential":
        cdf = stats.expon.cdf(d, scale=v["delay_mean"])
    elif family == "none":
        hazard = 1 / (1 + v["delay_mean"])
        cdf = 1 - (1 - hazard) ** d
    else:
        raise ValueError(f"unknown family {family!r}")
    pmf = np.diff(cdf)
    return np.asarray(pmf / pmf.sum(), dtype=float)


class ParametricDelay(RandomVariable):
    """The double censored delay pmf for given locations and scales.

    Parameters
    ----------
    name : str
        Name of the component; the pmf is recorded as `"{name}::pmf"`.
    family : str
        One of `FAMILIES`.
    mu_rv, sigma_rv : RandomVariable
        Return the location and scale, of any matching shape (for example
        one per group and reference day).
    dmax : int
        Delays in the window.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """

    def __init__(
        self,
        name: str,
        family: str,
        mu_rv: RandomVariable,
        sigma_rv: RandomVariable,
        dmax: int,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        super().__init__(name=name)
        self.family = family
        self.mu_rv = mu_rv
        self.sigma_rv = sigma_rv
        self.dmax = dmax

    def sample(self, **kwargs: object) -> Array:
        """Return the pmf.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(*mu.shape, dmax)`.
        """
        pmf = jnp.exp(
            double_censored_log_pmf(
                self.family, self.mu_rv(), self.sigma_rv(), self.dmax
            )
        )
        numpyro.deterministic(f"{self.name}::pmf", pmf)
        return pmf
