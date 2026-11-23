# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""One renewal step for the fitted models and the simulator.

`run_renewal` is a pure function of the starting infections, the
reproduction numbers and the generation time, in the form of PyRenew's
renewal scans (`compute_infections_from_rt`). Each day, with $F$ the
generation-time-weighted past infections and $W_t$ the mixing matrix
(`components.mixing.mixing_matrix`), the raw incidence is

$$y_t = R_t\,(W_t F_t) + \sigma_t,$$

with $\sigma_t$ seeding added to the force, and the new infections are
`depletion(y_t, S_t, N, imports_t)`, with imports $\iota_t$ added after
depletion. It is unit free: the simulator runs it in counts ($N$ each
stratum's population, $W$ in counts), the fitted models per resident ($N$
the share at risk $\nu$, $W$ per resident).

It adds to PyRenew's scans what they lack:

- a choice of susceptible depletion: none, `LinearDepletion` (the
  simulator's default) or `ExponentialDepletion`
  (the fitted models' form, PyRenew's `logistic_susceptibility_adjustment`
  written with `expm1`, see below);
- mixing between places with a share that may change each day;
- seeding before depletion and imports after it;
- shifts in R that may react to the previous day's effective R
  (`Shifts`);
- variants that raise R and return immune people to the susceptibles
  (`Variants`).

PyRenew's `compute_infections_from_rt_with_mobility` mixes where people
are present with linear depletion only and a fixed matrix, and
`compute_infections_from_rt_with_feedback` adjusts R by a convolution of
past infections; neither has a depleting pool with mixing, imports,
shifts or variants. `logistic_susceptibility_adjustment` computes
$1 - e^{-y / N}$, which loses most of its digits in float32 at
per-resident rates near $10^{-6}$ and stalls NUTS, so the exponential
form here uses `expm1`; a test checks it equals PyRenew's in float64.

The effective reproduction number is recorded after each day's
infections, $R_t S_{t+1} / N$, and shifts react to it on the next day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "DEPLETIONS",
    "Depletion",
    "ExponentialDepletion",
    "LinearDepletion",
    "NoDepletion",
    "RenewalRun",
    "Shifts",
    "Variants",
    "run_renewal",
]

# The logistic in shifts and variants runs over `days`, from about 10% to
# 90% of the way: 2 * log(9) is about 4.4.
_SPREAD = 4.4


class Depletion(Protocol):
    """New infections from the raw incidence, susceptibles and imports."""

    @property
    def depletes(self) -> bool:
        """Whether new infections leave the susceptibles."""
        ...

    def __call__(
        self, raw: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each place's new infections."""
        ...


@dataclass(frozen=True)
class NoDepletion:
    r"""$y + \iota$: every infection finds a susceptible."""

    depletes: bool = False

    def __call__(
        self, raw: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each place's new infections."""
        return raw + imports


@dataclass(frozen=True)
class LinearDepletion:
    r"""$\min(y S / N + \iota, S)$: linear in the susceptible share.

    The simulator's default.
    """

    depletes: bool = True

    def __call__(
        self, raw: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each place's new infections, at most its susceptibles."""
        return jnp.minimum(raw * s / pop + imports, s)


@dataclass(frozen=True)
class ExponentialDepletion:
    r"""$\min(S \{1 - e^{-y / N}\} + \iota, S)$.

    PyRenew's `logistic_susceptibility_adjustment` (Bhatt et al. 2023,
    eq. 6) with the raw incidence $y$, written with `expm1` so small rates
    keep their digits. To first order in $y / N$ it is `LinearDepletion`.
    The fitted models' form.
    """

    depletes: bool = True

    def __call__(
        self, raw: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each place's new infections, at most its susceptibles."""
        return jnp.minimum(-s * jnp.expm1(-raw / pop) + imports, s)


DEPLETIONS: dict[str, Depletion] = {
    "none": NoDepletion(),
    "linear": LinearDepletion(),
    "exponential": ExponentialDepletion(),
}
"""The depletions by name."""


def _vec(x: Sequence[float] | ArrayLike) -> Array:
    return jnp.atleast_1d(jnp.asarray(x, dtype=float))


@dataclass(frozen=True)
class Shifts:
    r"""Shifts in R, each along a logistic over `days` centred on `at`.

    Shift $j$ multiplies R by $e^{\pm s_j}$ at full strength. Its direction
    is set on the first day at or after `at[j] - days[j] / 2`: a reactive
    shift lowers R in places whose previous day's effective R is at least
    1 and raises it where it is below; the others go up where `up[j]`.

    Attributes
    ----------
    at, size, days : array_like
        Each shift's centre (day), size $s_j$ and length (days).
    reactive, up : array_like of bool
        Whether each shift reacts to the effective R, and if not whether
        it goes up.
    """

    at: Sequence[float] | ArrayLike = field(default=())
    size: Sequence[float] | ArrayLike = field(default=())
    days: Sequence[float] | ArrayLike = field(default=())
    reactive: Sequence[bool] | ArrayLike = field(default=())
    up: Sequence[bool] | ArrayLike = field(default=())

    def arrays(self) -> dict[str, Array]:
        """Return the shifts as float arrays, one entry per shift."""
        return {
            k: _vec(getattr(self, k))
            for k in ("at", "size", "days", "reactive", "up")
        }


@dataclass(frozen=True)
class Variants:
    r"""Variants, each raising R along a logistic and escaping immunity.

    Variant $j$ multiplies R by $1 + (m_j - 1)\,\sigma((t - a_j) /
    (d_j / 4.4))$, and on the first day at or after `at[j]` returns a share
    `escape[j]` of the immune, $N - S$, to the susceptibles.

    Attributes
    ----------
    at, escape, multiplier, days : array_like
        Each variant's centre (day), escape share, R multiplier at full
        strength and length (days).
    """

    at: Sequence[float] | ArrayLike = field(default=())
    escape: Sequence[float] | ArrayLike = field(default=())
    multiplier: Sequence[float] | ArrayLike = field(default=())
    days: Sequence[float] | ArrayLike = field(default=())

    def multiplier_on(self, t: ArrayLike) -> Array:
        """Return the variants' combined R multiplier on days `t`.

        Parameters
        ----------
        t : array_like
            Days, 1 the first.

        Returns
        -------
        Array
            The product over variants, the shape of `t`.
        """
        t = jnp.asarray(t, dtype=float)
        at, mult, days = _vec(self.at), _vec(self.multiplier), _vec(self.days)
        rise = jax.nn.sigmoid(
            (t[:, None] - at[None, :]) / (days[None, :] / _SPREAD)
        )
        return jnp.prod(1 + (mult[None, :] - 1) * rise, axis=1)


class RenewalRun(NamedTuple):
    """A renewal's output.

    Attributes
    ----------
    infections : Array
        `(n_start + n, k)`, the starting infections then each day's.
    r : Array
        R after shifts and variants, `(n, k)`.
    reff : Array
        $R_t S_{t+1} / N$, `(n, k)`; $R_t$ without depletion.
    susceptible : Array
        Susceptibles after each day, `(n, k)`.
    imported : Array
        The new infections from other places through the mixing: their
        share of $y_t$ times the new infections, `(n, k)`.
    """

    infections: Array
    r: Array
    reff: Array
    susceptible: Array
    imported: Array


def run_renewal(
    start: ArrayLike,
    gt: ArrayLike,
    rt: ArrayLike,
    population: ArrayLike,
    mixing: ArrayLike | None = None,
    seeding: ArrayLike | None = None,
    imports: ArrayLike | None = None,
    shifts: Shifts | None = None,
    variants: Variants | None = None,
    depletion: Depletion | None = None,
) -> RenewalRun:
    r"""Run each place's renewal (module docstring).

    Parameters
    ----------
    start : array_like
        Infections before day 1, `(n_start, k)` with `n_start` at least the
        generation time's length. All count towards depletion; the last
        `len(gt)` rows start the renewal.
    gt : array_like
        Generation time pmf on 1, ..., G days (PyRenew's convention).
    rt : array_like
        Reproduction numbers before shifts and variants, `(n, k)`.
    population : array_like
        Each place's population $N$, or the share at risk per resident; a
        scalar or `(k,)`.
    mixing : array_like, optional
        $W$, `(n, k, k)` or `(k, k)` (`components.mixing.mixing_matrix`).
        `None` is none.
    seeding : array_like, optional
        Added to the raw incidence before depletion, `(n, k)`.
    imports : array_like, optional
        Added to the new infections after depletion, `(n, k)`.
    shifts : Shifts, optional
        Shifts in R.
    variants : Variants, optional
        Variants.
    depletion : Depletion, optional
        `LinearDepletion` (the default), `ExponentialDepletion` or
        `NoDepletion` (`DEPLETIONS`).

    Returns
    -------
    RenewalRun
        The infections, R, effective R, susceptibles and imported
        infections.
    """
    start_ = jnp.asarray(start, dtype=float)
    rt_ = jnp.asarray(rt, dtype=float)
    n, k = rt_.shape
    pop = jnp.broadcast_to(jnp.asarray(population, dtype=float), (k,))
    deplete = LinearDepletion() if depletion is None else depletion
    rev_gt = jnp.flip(jnp.asarray(gt, dtype=float))
    W = (
        jnp.broadcast_to(jnp.eye(k), (n, k, k))
        if mixing is None
        else jnp.broadcast_to(jnp.asarray(mixing), (n, k, k))
    )
    off = W * (1 - jnp.eye(k))
    zeros = jnp.zeros((n, k))
    seed = zeros if seeding is None else jnp.asarray(seeding, dtype=float)
    imp = zeros if imports is None else jnp.asarray(imports, dtype=float)
    sh = (shifts or Shifts()).arrays()
    va = variants or Variants()
    va_at, va_escape = _vec(va.at), _vec(va.escape)
    n_sh = sh["at"].shape[0]
    days = jnp.arange(1, n + 1, dtype=float)
    va_mult = va.multiplier_on(days)
    s0 = jnp.maximum(pop - start_.sum(axis=0), 0)

    def step(carry: Any, inputs: Any) -> tuple[Any, tuple[Array, ...]]:
        hist, s, prev, mult, set_, escaped = carry
        t, r, w, w_off, sd, im, vm = inputs
        # Shifts: the direction is fixed on the first day each is due.
        due = (~set_) & (t >= sh["at"] - sh["days"] / 2)
        up = jnp.where(
            sh["reactive"][:, None] > 0,
            prev[None, :] < 1,
            (sh["up"][:, None] > 0) & jnp.ones((n_sh, k), dtype=bool),
        )
        new_mult = jnp.exp(jnp.where(up, 1.0, -1.0) * sh["size"][:, None])
        mult = jnp.where(due[:, None], new_mult, mult)
        set_ = set_ | due
        weight = jax.nn.sigmoid((t - sh["at"]) / (sh["days"] / _SPREAD))
        factor = jnp.where(set_[:, None], 1 + (mult - 1) * weight[:, None], 1.0)
        r = r * jnp.prod(factor, axis=0) * vm
        # Variants: escape once, on the first day at or after `at`.
        go = (~escaped) & (t >= va_at)
        for j in range(va_at.shape[0]):
            s = jnp.where(go[j], s + va_escape[j] * (pop - s), s)
        escaped = escaped | go
        force = rev_gt @ hist
        raw = r * (w @ force) + sd
        new = deplete(raw, s, pop, im)
        if deplete.depletes:
            s = s - new
        from_off = r * (w_off @ force)
        imported = new * jnp.where(
            raw > 0, from_off / jnp.where(raw > 0, raw, 1), 0
        )
        reff = r * s / pop if deplete.depletes else r
        hist = jnp.concatenate([hist[1:], new[None, :]])
        return (hist, s, reff, mult, set_, escaped), (new, r, reff, s, imported)

    carry = (
        start_[-rev_gt.shape[0] :],
        s0,
        jnp.ones(k),
        jnp.ones((n_sh, k)),
        jnp.zeros(n_sh, dtype=bool),
        jnp.zeros(va_at.shape[0], dtype=bool),
    )
    _, (new, r, reff, s, imported) = jax.lax.scan(
        step, carry, (days, rt_, W, off, seed, imp, va_mult)
    )
    return RenewalRun(jnp.concatenate([start_, new]), r, reff, s, imported)
