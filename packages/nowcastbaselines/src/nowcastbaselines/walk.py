# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""The random walk, one function for both models.

From start value $x_i$ per sample, for $h \ge 1$,

$$
\log(1 + \lambda_i(h)) = \max\Big(0, \log(1 + x_i)
+ \sum_{j=1}^{h} \epsilon_{i,j}\Big),
\quad \epsilon \sim \mathcal{N}(0, \sigma^2),
$$

$$
y_i(h) \sim \mathrm{Poisson}(\lambda_i(h)).
$$

Steps are cumulative within a sample, so each sample is one trajectory.

The spread $\sigma$ comes from the weekly changes $\Delta_t$ of
$\log(1 + y_t)$ in reported counts, net of their Poisson noise,

$$
\sigma^2 = \widehat{\mathrm{Var}}(\Delta_t)
- \overline{v(y_{t-1}) + v(y_t)},
\quad v(y) = \frac{y}{(1 + y)^2},
$$

where $v$ is the delta-method variance of $\log(1 + Y)$ for
$Y \sim \mathrm{Poisson}(\lambda)$, $\lambda / (1 + \lambda)^2$, with the
observed count for $\lambda$ (`walk_variance`). The walk draws its Poisson
counts on top, so the noise is not counted twice.

A walk in days (`DailySteps`) has $\epsilon_{i,j} \sim
\mathcal{N}(\mu_{d(j)}, \sigma^2)$, where $d(j)$ is the weekday of
step $j$'s day. $\mu_d$ is the mean change into weekday $d$
(`weekday_means`), centred so the seven sum to 0: the walk repeats the
weekly pattern and has no drift. $\sigma^2$ is the variance of the
changes over 7 days, net of their Poisson noise, over 7
(`same_weekday_variance`), in which a recurring weekday effect cancels.
A walk in weeks has one step type, $\mu = 0$.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nowcastbaselines.draws import poisson
from nowcastutils.shapes import bucket

__all__ = [
    "DailySteps",
    "random_walk",
    "random_walks",
    "same_weekday_variance",
    "walk_noise",
    "walk_variance",
    "weekday_means",
]


def walk_noise(y: np.ndarray) -> np.ndarray:
    r"""Return the Poisson noise variance of each change of log(1 + y).

    Parameters
    ----------
    y : numpy.ndarray
        Weekly counts, `[W]`.

    Returns
    -------
    numpy.ndarray
        $v(y_{t-1}) + v(y_t)$ for each change, `[W - 1]`, with
        $v(y) = y / (1 + y)^2$ the delta-method variance of
        $\log(1 + Y)$ for $Y \sim \mathrm{Poisson}(y)$.
    """
    y = np.asarray(y, dtype=float)
    v = y / (1.0 + y) ** 2
    return v[1:] + v[:-1]


def walk_variance(changes: np.ndarray, noise: np.ndarray) -> float:
    """Return the variance of the walk's steps net of Poisson noise.

    Parameters
    ----------
    changes : numpy.ndarray
        Weekly changes of log(1 + y), `[N]`.
    noise : numpy.ndarray
        Their Poisson noise variances (`walk_noise`), `[N]`.

    Returns
    -------
    float
        The sample variance of `changes` less the mean of `noise`. It
        may be 0 or below, which says the changes hold no more than
        Poisson noise. NaN with fewer than 2 changes.
    """
    if changes.size < 2:
        return float("nan")
    return float(np.var(changes, ddof=1)) - float(np.mean(noise))


def weekday_means(changes: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """Return the mean daily step into each weekday.

    Parameters
    ----------
    changes : numpy.ndarray
        Daily changes of log(1 + y), from the day before, `[N]`.
    weekday : numpy.ndarray
        Weekday of each change's later day, Monday 0, `[N]`.

    Returns
    -------
    numpy.ndarray
        The mean change into each weekday, `[7]`, NaN for a weekday with
        none.
    """
    return np.array(
        [
            changes[weekday == d].mean() if (weekday == d).any() else np.nan
            for d in range(7)
        ]
    )


def same_weekday_variance(series: Sequence[np.ndarray]) -> float:
    """Return a daily walk step's variance from same-weekday changes.

    The changes of log(1 + y) over 7 days of each daily series, pooled
    and net of their Poisson noise (`walk_variance`), over 7: a
    recurring weekday effect cancels in them, and a random walk's
    variance over 7 steps is 7 times a step's.

    Parameters
    ----------
    series : sequence of numpy.ndarray
        Daily counts, each `[T]`.

    Returns
    -------
    float
        As `walk_variance` gives it, over 7.
    """
    logs = [np.log1p(np.asarray(y, dtype=float)) for y in series]
    noise = [np.asarray(y, dtype=float) for y in series]
    noise = [v / (1.0 + v) ** 2 for v in noise]
    return (
        walk_variance(
            np.concatenate([lg[7:] - lg[:-7] for lg in logs]),
            np.concatenate([v[7:] + v[:-7] for v in noise]),
        )
        / 7.0
    )


@dataclass(frozen=True)
class DailySteps:
    """The steps of a walk in days, by the weekday of the day stepped into.

    Attributes
    ----------
    mean : numpy.ndarray
        Each place's mean step into each weekday, Monday 0, centred to sum
        to 0 over the week, `[L, 7]`.
    sigma : numpy.ndarray
        Each place's step spread, `[L]`.
    """

    mean: np.ndarray
    sigma: np.ndarray

    def at(self, first: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and spread of `h` steps, `[L, h]` each.

        `first` `[L]` is the weekday of each place's first step's day.
        """
        weekday = (np.asarray(first)[:, None] + np.arange(h)[None, :]) % 7
        n_loc = self.mean.shape[0]
        mean = np.take_along_axis(
            self.mean, np.broadcast_to(weekday, (n_loc, h)), axis=1
        )
        return mean, np.repeat(self.sigma[:, None], h, axis=1)


def random_walk(
    key: Array,
    x0: Array,
    sigma: Array | float,
    h: int,
    mean: Array | None = None,
) -> tuple[Array, Array]:
    r"""Draw the walk forward on the log scale with Poisson counts.

    Parameters
    ----------
    key : jax.Array
        Random key.
    x0 : jax.Array
        Start value of each sample, shape `[S]`.
    sigma : float or jax.Array
        Standard deviation of a step, or of each step, `[h]`.
    h : int
        Steps to draw.
    mean : jax.Array, optional
        Mean of each step, `[h]`; without it the steps have mean 0.

    Returns
    -------
    tuple of jax.Array
        The rate $\lambda$ and the Poisson count, each `[S, h]`.
    """
    k_step, k_count = jax.random.split(key)
    steps = sigma * jax.random.normal(k_step, (x0.shape[0], h), x0.dtype)
    if mean is not None:
        steps = steps + mean
    level = jnp.log1p(x0)[:, None] + jnp.cumsum(steps, axis=1)
    lam = jnp.expm1(jnp.maximum(level, 0.0))
    return lam, jax.random.poisson(k_count, lam).astype(x0.dtype)


def random_walks(
    keys: Array,
    x0: Array | np.ndarray,
    sigma: Array | np.ndarray,
    h: int,
    mean: Array | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `random_walk` for each place, compiled as one program.

    The steps are padded to `nowcastutils.shapes.bucket` of `h`, so one
    program serves every `h` of a size. The draws are those of
    `random_walk` over `h` steps (`_padded_walks`).

    Parameters
    ----------
    keys : jax.Array
        One random key per place, `[L]`.
    x0 : jax.Array
        Start values, `[L, S]`.
    sigma : jax.Array
        Step standard deviation of each place, `[L]`, or of each place's
        each step, `[L, h]`.
    h : int
        Steps to draw.
    mean : jax.Array, optional
        Mean of each place's each step, `[L, h]`.

    Returns
    -------
    tuple of numpy.ndarray
        The rates and the Poisson counts, each `[L, S, h]`.
    """
    x = np.asarray(x0)
    n_loc, width = x.shape[0], bucket(h)
    spread = np.asarray(sigma, x.dtype)
    if spread.ndim == 1:
        spread = np.repeat(spread[:, None], h, axis=1)
    pad = ((0, 0), (0, width - h))
    shift = (
        np.zeros((n_loc, width), x.dtype)
        if mean is None
        else np.pad(np.asarray(mean, x.dtype), pad)
    )
    lam, y = _padded_walks(
        keys, x, np.pad(spread, pad), shift, np.int32(h), mean is not None
    )
    return np.asarray(lam)[:, :, :h], np.asarray(y)[:, :, :h]


@partial(jax.jit, static_argnums=5)
def _padded_walks(
    keys: Array,
    x0: Array,
    sigma: Array,
    mean: Array,
    h: Array,
    shifted: bool,
) -> tuple[Array, Array]:
    r"""Draw `random_walk` over `h` steps of `[L, H]` padded ones.

    JAX draws each element of an array from its flat index alone, so the
    draws of `random_walk`'s `[S, h]` steps are the first $S h$ of `[S H]`
    flat draws, read at index $s h + j$. The steps after `h` read draws
    of no step, and are cut off by `random_walks`. The counts are drawn
    as the unpadded ones (`draws.poisson`). `h` is traced, so one program
    serves every `h` up to `H`.
    """
    n, width = x0.shape[1], sigma.shape[1]
    j = jnp.arange(width)[None, :]
    at = jnp.arange(n)[:, None] * h + j
    real = j < h
    flat = jnp.where(real, at, n * width)
    cells = jnp.arange(n * width)

    def one(key: Array, x: Array, s: Array, m: Array) -> tuple[Array, Array]:
        k_step, k_count = jax.random.split(key)
        z = jax.random.normal(k_step, (n * width,), x.dtype)
        steps = s * jnp.take(z, at, mode="clip")
        if shifted:
            steps = steps + m
        level = jnp.log1p(x)[:, None] + jnp.cumsum(steps, axis=1)
        lam = jnp.expm1(jnp.maximum(level, 0.0))
        rates = jnp.zeros(n * width, x.dtype).at[flat].set(lam, mode="drop")
        drawn = poisson(k_count, rates, cells < n * h).astype(x.dtype)
        return lam, jnp.where(real, jnp.take(drawn, at, mode="clip"), 0.0)

    return jax.vmap(one)(keys, x0, sigma, mean)
