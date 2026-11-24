# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Double interval censored delay pmfs, for every delay in the models.

The generation time, the incubation period, the delay from infection to
the reference event and the delay from reference to report are each a
continuous delay $T$ (`Delay`) recorded on whole days. Each pmf here is
primarycensored's: the primary event uniform over its day, the secondary
event censored to its day, and right truncation at $D$ days,

$$p_d = \frac{F_c(d + 1) - F_c(d)}{F_c(D)}, \qquad
F_c(q) = \int_{q - 1}^{q} F(t)\,dt,$$

(`primarycensored::dpcens(d, ..., pwindow = 1, swindow = 1, D = D)`).
With $G(t) = \int_0^t F = t F(t) - E[T; T \le t]$ and
$U(t) = \int_t^\infty S = E[T; T > t] - t S(t)$, the mass is the second
difference $G(d + 1) - 2 G(d) + G(d - 1)$ below the median and
$U(d - 1) - 2 U(d) + U(d + 1)$ above it, so the tail keeps its digits.
The partial means $E[T; T \le t]$ are closed form for every family.

primarycensored has no Python package, so its formulation is written here
in JAX (differentiable in the parameters, for the fitted models) and
checked live against the R package (the package's `tests/test_pmfs.py`).
Fixed pmfs are the values of PyRenew `DeterministicPMF` components;
`CensoredPMF` is the pmf of random parameters, a PyRenew `RandomVariable`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.scipy.special import gammainc, gammaincc, gammaln, ndtr
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable
from scipy import stats

__all__ = [
    "FAMILIES",
    "CensoredPMF",
    "Delay",
    "censored_log_pmf",
    "censored_logit_hazard",
    "censored_pmf",
    "delay_pmf",
    "generation_time_pmf",
]

FAMILIES = ("exponential", "lognormal", "gamma", "gengamma")
r"""The families, each by a location $\mu$ and scale $\sigma$ (and a
shape $Q$ for the generalised gamma): exponential with rate $e^{-\mu}$;
lognormal with meanlog $\mu$ and sdlog $\sigma$; gamma with shape
$e^{\mu}$ and rate $\sigma$ (epinowcast's parameters); generalised gamma
as flexsurv's `pgengamma(q, mu, sigma, Q)` with $Q > 0$."""

_QUANTILE = 0.999
"""Where a pmf of unstated length is cut: its 99.9% quantile."""


@dataclass(frozen=True)
class Delay:
    r"""A continuous delay $T$ by family and parameters (`FAMILIES`).

    The parameters may be arrays (traced or not) that broadcast together.

    Attributes
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale; the exponential's scale is unused.
    q : array_like
        The generalised gamma's shape $Q > 0$; unused otherwise.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """

    family: str
    mu: ArrayLike
    sigma: ArrayLike = 1.0
    q: ArrayLike = 1.0

    def __post_init__(self) -> None:
        """Check the family."""
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")

    @classmethod
    def from_mean_cv(
        cls, family: str, mean: ArrayLike, cv: ArrayLike = 1.0
    ) -> Delay:
        """Return the delay with a given mean and coefficient of variation.

        Parameters
        ----------
        family : str
            `"exponential"` (whose cv is 1), `"lognormal"` or `"gamma"`.
        mean, cv : array_like
            Mean (days) and coefficient of variation.

        Returns
        -------
        Delay
            The delay.

        Raises
        ------
        ValueError
            For the generalised gamma, which has no such form here.
        """
        # Concrete values stay in float64 NumPy, so a fixed pmf built in
        # float64 keeps its digits; traced values stay in JAX.
        traced = isinstance(mean, jax.Array) or isinstance(cv, jax.Array)
        xp = jnp if traced else np
        mean, cv = xp.asarray(mean, dtype=None), xp.asarray(cv, dtype=None)
        if family == "gamma":
            shape = 1 / xp.square(cv)
            return cls(family, xp.log(shape), shape / mean)
        if family == "lognormal":
            sdlog = xp.sqrt(xp.log1p(xp.square(cv)))
            return cls(family, xp.log(mean) - xp.square(sdlog) / 2, sdlog)
        if family == "exponential":
            return cls(family, xp.log(mean))
        raise ValueError(f"no mean and cv form for {family!r}")

    @classmethod
    def from_values(cls, family: str, values: Mapping[str, Any]) -> Delay:
        """Return the reporting delay from a simulation's values.

        Lognormal from `delay_median` and `delay_sdlog`; gamma from
        `delay_mean` and `delay_cv`; exponential from `delay_mean`.

        Parameters
        ----------
        family : str
            `"exponential"`, `"lognormal"` or `"gamma"`.
        values : mapping
            The values.

        Returns
        -------
        Delay
            The delay, with float parameters.
        """
        v = values
        if family == "lognormal":
            return cls(family, math.log(v["delay_median"]), v["delay_sdlog"])
        if family == "gamma":
            shape = 1 / v["delay_cv"] ** 2
            return cls(family, math.log(shape), shape / v["delay_mean"])
        if family == "exponential":
            return cls(family, math.log(v["delay_mean"]))
        raise ValueError(f"no grid values for {family!r}")

    def cdf_and_survival(self, x: ArrayLike) -> tuple[Array, Array]:
        r"""Return $F(x)$ and $S(x) = 1 - F(x)$, each to full precision.

        Parameters
        ----------
        x : array_like
            Positive delays, broadcast against the parameters.

        Returns
        -------
        tuple of Array
            $F(x)$ and $S(x)$.
        """
        mu, sigma, x = jnp.asarray(self.mu), jnp.asarray(self.sigma), x
        x = jnp.asarray(x)
        if self.family == "exponential":
            rate_x = jnp.exp(-mu) * x
            return -jnp.expm1(-rate_x), jnp.exp(-rate_x)
        if self.family == "lognormal":
            z = (jnp.log(x) - mu) / sigma
            return ndtr(z), ndtr(-z)
        k, y = self._gamma_form(x)
        return gammainc(k, y), gammaincc(k, y)

    def partial_means(self, x: ArrayLike) -> tuple[Array, Array]:
        r"""Return $E[T; T \le x]$ and $E[T; T > x]$.

        Parameters
        ----------
        x : array_like
            Positive delays, broadcast against the parameters.

        Returns
        -------
        tuple of Array
            The partial means below and above `x`.
        """
        mu, sigma = jnp.asarray(self.mu), jnp.asarray(self.sigma)
        x = jnp.asarray(x)
        if self.family == "lognormal":
            mean = jnp.exp(mu + sigma**2 / 2)
            z = (jnp.log(x) - mu - sigma**2) / sigma
            return mean * ndtr(z), mean * ndtr(-z)
        k, y = self._gamma_form(x)
        # E[T; T <= x] = E[T] P(k + s, y), with s = 1 except for the
        # generalised gamma (`_gamma_form`).
        s = self._power()
        return self.mean() * gammainc(k + s, y), self.mean() * gammaincc(
            k + s, y
        )

    def _power(self) -> Array:
        if self.family == "gengamma":
            return jnp.asarray(self.sigma) / jnp.asarray(self.q)
        return jnp.asarray(1.0)

    def _gamma_form(self, x: Array) -> tuple[Array, Array]:
        r"""Return the shape $k$ and the gamma variate $y$ at `x`.

        $T = c\,Y^{s}$ with $Y \sim \text{Gamma}(k, 1)$: for the gamma and
        exponential $s = 1$; for the generalised gamma, $k = Q^{-2}$,
        $s = \sigma / Q$ and $y = k e^{Q (\log x - \mu) / \sigma}$.
        """
        mu, sigma = jnp.asarray(self.mu), jnp.asarray(self.sigma)
        if self.family == "gamma":
            return jnp.exp(mu), sigma * x
        if self.family == "exponential":
            return jnp.ones_like(mu), jnp.exp(-mu) * x
        q = jnp.asarray(self.q)
        k = 1 / jnp.square(q)
        return k, k * jnp.exp(q * (jnp.log(x) - mu) / sigma)

    def mean(self) -> Array:
        """Return $E[T]$.

        Returns
        -------
        Array
            The mean, the parameters' shape.
        """
        mu, sigma = jnp.asarray(self.mu), jnp.asarray(self.sigma)
        if self.family == "lognormal":
            return jnp.exp(mu + sigma**2 / 2)
        if self.family == "gamma":
            return jnp.exp(mu) / sigma
        if self.family == "exponential":
            return jnp.exp(mu)
        k, s = 1 / jnp.square(jnp.asarray(self.q)), self._power()
        return jnp.exp(mu - s * jnp.log(k) + gammaln(k + s) - gammaln(k))

    def quantile(self, p: float) -> float:
        """Return the `p` quantile, for concrete scalar parameters.

        Parameters
        ----------
        p : float
            Probability.

        Returns
        -------
        float
            The quantile, in days.
        """
        mu, sigma = float(np.asarray(self.mu)), float(np.asarray(self.sigma))
        if self.family == "lognormal":
            d = stats.lognorm(sigma, scale=math.exp(mu))
        elif self.family == "gamma":
            d = stats.gamma(math.exp(mu), scale=1 / sigma)
        elif self.family == "exponential":
            d = stats.expon(scale=math.exp(mu))
        else:
            q = float(np.asarray(self.q))
            k, s = 1 / q**2, sigma / q
            d = stats.gengamma(k, 1 / s, scale=math.exp(mu) * k ** (-s))
        return float(d.ppf(p))


def _censored_mass(delay: Delay, n: int) -> Array:
    r"""Return $F_c(d + 1) - F_c(d)$ for $d = 0, \dots, n - 1$."""
    if n < 1:
        raise ValueError("n must be at least 1")
    expand = {
        k: jnp.asarray(getattr(delay, k))[..., None]
        for k in ("mu", "sigma", "q")
    }
    delay = Delay(delay.family, **expand)
    t = jnp.arange(1, n + 1, dtype=float)
    f, s = delay.cdf_and_survival(t)
    lower, upper = delay.partial_means(t)
    # G(t) and U(t) at t = 1, ..., n; G(0) = 0 and U(0) = E[T].
    g = t * f - lower
    u = upper - t * s
    shape = jnp.broadcast_shapes(g.shape, u.shape)
    g, u = jnp.broadcast_to(g, shape), jnp.broadcast_to(u, shape)
    zero = jnp.zeros((*shape[:-1], 1), g.dtype)
    g = jnp.concatenate([zero, zero, g], axis=-1)  # t = -1, 0, ..., n
    u0 = jnp.broadcast_to(delay.mean(), (*shape[:-1], 1))
    u = jnp.concatenate([u0 + 1, u0, u], axis=-1)  # U(-1) = E[T] + 1
    d = jnp.arange(n)
    below = g[..., d + 2] - 2 * g[..., d + 1] + g[..., d]
    above = u[..., d] - 2 * u[..., d + 1] + u[..., d + 2]
    use_below = (d == 0) | (jnp.broadcast_to(f, shape)[..., :n] < 0.5)
    mass = jnp.where(use_below, below, above)
    # Rounding can leave a far-tail cell at or below 0 in float32; a floor
    # well above the smallest normal float keeps its log finite after
    # normalising (XLA flushes subnormals to 0).
    floor = 1e-30 if mass.dtype == jnp.float32 else 1e-300
    return jnp.maximum(mass, floor)


def censored_pmf(delay: Delay, n: int) -> Array:
    r"""Return the double censored pmf on $0, \dots, n - 1$, truncated at $n$.

    `primarycensored::dpcens(0:(n - 1), ..., pwindow = 1, swindow = 1,
    D = n)`.

    Parameters
    ----------
    delay : Delay
        The delay; leading dimensions of its parameters are kept.
    n : int
        Delays in the window, $D$.

    Returns
    -------
    Array
        Shape `(..., n)`, summing to 1 on the last axis.
    """
    mass = _censored_mass(delay, n)
    return mass / mass.sum(axis=-1, keepdims=True)


def censored_log_pmf(delay: Delay, n: int) -> Array:
    """Return the log of `censored_pmf`.

    Parameters
    ----------
    delay : Delay
        The delay.
    n : int
        Delays in the window.

    Returns
    -------
    Array
        Shape `(..., n)`.
    """
    mass = _censored_mass(delay, n)
    return jnp.log(mass) - jnp.log(mass.sum(axis=-1, keepdims=True))


def censored_logit_hazard(delay: Delay, n: int) -> Array:
    r"""Return the logit hazard of `censored_pmf`.

    $h_d = p_d / \sum_{i \ge d} p_i$, so
    $\operatorname{logit} h_d = \log p_d - \log \sum_{i > d} p_i$, and
    $h_{n - 1} = 1$ (a logit of $+\infty$).

    Parameters
    ----------
    delay : Delay
        The delay.
    n : int
        Delays in the window.

    Returns
    -------
    Array
        Shape `(..., n)`.
    """
    mass = _censored_mass(delay, n)
    after = jnp.flip(jnp.cumsum(jnp.flip(mass, -1), axis=-1), -1)
    rest = jnp.concatenate(
        [after[..., 1:], jnp.zeros_like(after[..., :1])], axis=-1
    )
    logit = jnp.log(mass) - jnp.log(rest)
    return logit.at[..., -1].set(jnp.inf)


def _longest(delay: Delay, at_least: int) -> int:
    return max(at_least, math.ceil(delay.quantile(_QUANTILE)))


def delay_pmf(delay: Delay, longest: int | None = None) -> np.ndarray:
    """Return a fixed delay pmf on 0, ..., `longest` days, in float64.

    For an incubation period or a delay from infection to the reference
    event: `censored_pmf` with $D$ = `longest` + 1.

    Parameters
    ----------
    delay : Delay
        The delay, with concrete scalar parameters.
    longest : int, optional
        The longest delay; by default the 99.9% quantile rounded up, and
        at least 1.

    Returns
    -------
    numpy.ndarray
        The pmf, element 0 at a delay of 0 days.
    """
    n = _longest(delay, 1) if longest is None else longest
    with jax.enable_x64(True):
        return np.asarray(censored_pmf(delay, n + 1), dtype=float)


def generation_time_pmf(delay: Delay, longest: int | None = None) -> np.ndarray:
    """Return a fixed generation time pmf on lags 1, ..., `longest`.

    `censored_pmf` with $D$ = `longest` + 1, without lag 0 (left
    truncated at a day, `dpcens(..., L = 1)`), renormalised.

    Parameters
    ----------
    delay : Delay
        The generation time, with concrete scalar parameters.
    longest : int, optional
        The longest lag; by default the 99.9% quantile rounded up, and at
        least 2.

    Returns
    -------
    numpy.ndarray
        The pmf, element 0 at lag 1 (PyRenew's convention).
    """
    pmf = delay_pmf(delay, _longest(delay, 2) if longest is None else longest)
    return np.asarray(pmf[1:] / pmf[1:].sum(), dtype=float)


class CensoredPMF(RandomVariable):
    """The double censored pmf of a delay with random parameters.

    Give either the mean and SD (`mean_rv`, `sd_rv`; exponential,
    lognormal or gamma) or the location and scale (`mu_rv`, `sigma_rv`;
    `FAMILIES`). The parameters may have any matching shape, for example
    one per group and day.

    Parameters
    ----------
    name : str
        Name of the component.
    family : str
        One of `FAMILIES`.
    longest : int
        The longest delay; the pmf is on 0, ..., `longest`.
    mean_rv, sd_rv : RandomVariable, optional
        The mean and SD in days.
    mu_rv, sigma_rv : RandomVariable, optional
        The location and scale (`FAMILIES`).
    record : bool
        Whether to record the pmf as `"{name}::pmf"`.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`, or not exactly one pair of
        parameters is given.
    TypeError
        If a parameter is not a `RandomVariable`.
    """

    def __init__(
        self,
        name: str,
        family: str,
        longest: int,
        *,
        mean_rv: RandomVariable | None = None,
        sd_rv: RandomVariable | None = None,
        mu_rv: RandomVariable | None = None,
        sigma_rv: RandomVariable | None = None,
        record: bool = False,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        moments = (mean_rv, sd_rv)
        natural = (mu_rv, sigma_rv)
        given = [p for p in (moments, natural) if any(x is not None for x in p)]
        if len(given) != 1 or any(x is None for x in given[0]):
            raise ValueError("give mean_rv and sd_rv, or mu_rv and sigma_rv")
        pair = [x for x in given[0] if isinstance(x, RandomVariable)]
        if len(pair) != 2:
            raise TypeError("the parameters must be RandomVariables")
        super().__init__(name=name)
        self.family = family
        self.longest = longest
        self.by_moments = given[0] is moments
        self.first_rv, self.second_rv = pair
        self.record = record

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CensoredPMF(name={self.name!r}, family={self.family!r}, "
            f"longest={self.longest})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample the parameters and return the pmf.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(*parameters, longest + 1)`.
        """
        a, b = jnp.asarray(self.first_rv()), jnp.asarray(self.second_rv())
        delay = (
            Delay.from_mean_cv(self.family, a, b / a)
            if self.by_moments
            else Delay(self.family, a, b)
        )
        pmf = censored_pmf(delay, self.longest + 1)
        if self.record:
            numpyro.deterministic(f"{self.name}::pmf", pmf)
        return pmf
