# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Structure of the reporting delay across strata and time.

The parts `effects.ReferenceEffects` adds to the delay's location (and,
for a non-parametric delay, to its logit hazard) beyond epinowcast's
regression, each a PyRenew `RandomVariable` of standard normal sites
scaled by their standard deviations (non-centred):

- `NoiseLayers`: noise on a delay term in three layers, by week (shared
  by every stratum), by stratum (static) and by stratum and week, each a
  standard normal per level times its sd. One structure for every delay
  term, so each term's noise is any mix of the three layers.
- `StructuredChange`: a change over time by stratum. Strata share one
  change (`shared`), mix it with a change of their own (`related`) or
  each follow their own (`independent`), and the mix can switch on or
  off part way through (`diverge`, `converge`). A change is one kind or
  two combined (`effects.change_path`).
- `CountCoupling`: a term linear in a stratum's log expected counts by
  reference date, centred over its observed days, with a slope by
  stratum that may move from week to week (`NoiseLayers`), of either
  sign.

PyRenew has no reporting-delay structure of this kind and epinowcast's
regression has no counterpart of these terms, so their forms are our
own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from pyrenew.metaclass import RandomVariable

from .effects import CHANGE_TERMS, ReferenceFrame, change_path

__all__ = [
    "COUPLING_LAYERS",
    "NOISE_LAYERS",
    "SWITCH_WIDTH",
    "CountCoupling",
    "NoiseLayers",
    "StructuredChange",
    "noise_layers",
    "own_change",
    "switch_weight",
]

SWITCH_WIDTH = 2 * np.log(9)
"""A switch's logistic runs from 10% to 90% over its days."""

NOISE_LAYERS: dict[str, tuple[str, ...]] = {
    "none": (),
    "strata": ("strata",),
    "strata_time": ("strata_time",),
    "both": ("strata", "strata_time"),
}
"""The stratum noise layers of each `delay_strata_noise` level; the time
layer is `delay_jitter`'s."""

COUPLING_LAYERS: dict[str, tuple[str, ...]] = {
    "none": (),
    "shared": (),
    "strata": ("strata",),
    "strata_time": ("strata", "strata_time"),
}
"""The noise layers on a count coupling's slope by level: one slope
(`shared`), a static slope by stratum (`strata`) or one that also moves
from week to week (`strata_time`)."""


def noise_layers(
    design: Mapping[str, Any], values: Mapping[str, Any]
) -> tuple[float, float, float]:
    """Return the delay terms' noise sds by time, stratum and both.

    Parameters
    ----------
    design : mapping
        The design (`delay_jitter`, `delay_strata_noise`; either absent
        is no noise).
    values : mapping
        The values (`delay_jitter_sd`, `delay_strata_sd`,
        `delay_strata_jitter_sd`).

    Returns
    -------
    tuple of float
        The time, stratum and stratum-by-time sds; 0 for a layer that is
        off.
    """
    d, v = design, values
    on = d.get("delay_jitter") == "yes"
    time = float(v["delay_jitter_sd"]) if on else 0.0
    layers = NOISE_LAYERS[str(d.get("delay_strata_noise", "none"))]
    strata = float(v["delay_strata_sd"]) if "strata" in layers else 0.0
    both = (
        float(v["delay_strata_jitter_sd"]) if "strata_time" in layers else 0.0
    )
    return time, strata, both


class NoiseLayers(RandomVariable):
    r"""Noise on a delay term by week, stratum, and stratum and week.

    For term $k$, stratum $g$ and reference day $t$ in week $w(t)$:

    $$\epsilon^k_{g,t} = s_\tau z^k_{w(t)} + s_g z^k_g
    + s_{g\tau} z^k_{g,w(t)},$$

    standard normal sites `"{name}::jitter_{k}"` (by week),
    `"{name}::strata_{k}"` (by stratum) and `"{name}::strata_jitter_{k}"`
    (by stratum and week), each drawn only when its sd is positive.
    Centred over the frame's observed days, as the random effects, unless
    `centre` is False.

    Parameters
    ----------
    name : str
        Name of the component; its sites are named after it.
    frame : ReferenceFrame
        The triangle's strata and dates.
    time_sd, strata_sd, strata_time_sd : float
        The layers' sds, $s_\tau$, $s_g$ and $s_{g\tau}$.
    centre : bool
        Centre the noise over the observed days.
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        time_sd: float = 0.0,
        strata_sd: float = 0.0,
        strata_time_sd: float = 0.0,
        centre: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        self.time_sd = float(time_sd)
        self.strata_sd = float(strata_sd)
        self.strata_time_sd = float(strata_time_sd)
        self.centre = centre

    def __bool__(self) -> bool:
        """Whether any layer is on."""
        return bool(self.time_sd or self.strata_sd or self.strata_time_sd)

    def sample(self, term: str = "", **kwargs: object) -> Array:
        """Return the term's noise by stratum and reference day.

        Parameters
        ----------
        term : str
            The delay term, which names its sites.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            `(G, T)`; 0 if every layer is off.
        """
        G, T = len(self.frame.place), len(self.frame.dates)
        week = np.asarray(self.frame.week)
        W = int(week.max()) + 1
        total = jnp.zeros((G, T))
        if self.time_sd:
            z = numpyro.sample(
                f"{self.name}::jitter_{term}", dist.Normal(0.0, 1.0).expand([W])
            )
            total = total + self.time_sd * z[week][None, :]
        if self.strata_sd:
            z = numpyro.sample(
                f"{self.name}::strata_{term}", dist.Normal(0.0, 1.0).expand([G])
            )
            total = total + self.strata_sd * z[:, None]
        if self.strata_time_sd:
            z = numpyro.sample(
                f"{self.name}::strata_jitter_{term}",
                dist.Normal(0.0, 1.0).expand([G, W]),
            )
            total = total + self.strata_time_sd * z[:, week]
        if self.centre:
            total = total - total[:, : self.frame.observed].mean()
        return total


def own_change(
    v: Mapping[str, Any], prefix: str, g: int
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return stratum `g`'s own change: its kinds and values.

    Parameters
    ----------
    v : mapping
        The values: `{prefix}change` (each stratum's kinds, a combined
        change as `"drift+step"`) and `{prefix}{setting}` by stratum for
        the settings its kinds use.
    prefix : str
        The values' prefix, as `"own_"`.
    g : int
        The stratum, 0-based.

    Returns
    -------
    tuple
        The kinds and the change's values (`effects.change_path`).
    """
    kinds = tuple(str(np.atleast_1d(v[f"{prefix}change"])[g]).split("+"))
    out = {"weeks": v["weeks"]}
    for k in OWN_SETTINGS:
        if f"{prefix}{k}" in v:
            out[k] = float(np.atleast_1d(v[f"{prefix}{k}"])[g])
    return kinds, out


OWN_SETTINGS = (
    "drift",
    "step",
    "step_at",
    "periodic",
    "periodic_weeks",
    "smooth_amp",
    "smooth_period",
    "smooth_phase",
    "slow_at",
    "slow_weeks",
    "slow_mult",
)
"""The settings of a stratum's own change, each a value by stratum."""


def switch_weight(
    frame: ReferenceFrame, switch: str, at: float, days: float
) -> np.ndarray:
    r"""Return the weight of the strata's own change by reference day.

    1 throughout with no switch (`"none"`); with `"diverge"` a logistic
    from 0 to 1 centred `at` of the way through the observed days and
    rising from 10% to 90% over `days`; `"converge"` is its reverse.

    Returns
    -------
    numpy.ndarray
        `(T,)`, on padded days the last observed day's.
    """
    T = len(frame.dates)
    if switch == "none":
        return np.ones(T)
    day = np.minimum(np.arange(T), frame.observed - 1)
    centre = at * (frame.observed - 1)
    z = -SWITCH_WIDTH * (day - centre) / max(days, 1.0)
    r = np.asarray(1 / (1 + np.exp(z)), dtype=float)
    if switch == "diverge":
        return r
    if switch == "converge":
        return np.asarray(1 - r, dtype=float)
    raise ValueError(f"unknown switch {switch}")


class StructuredChange(RandomVariable):
    r"""A change over time by stratum: shared, related or independent.

    With the shared change $S_t$ and stratum $g$'s own change $O_{g,t}$:

    $$\Delta_{g,t} = (1 - u_t) S_t + u_t O_{g,t},$$

    with $u_t$ 0 for `shared` strata, $(1 - \rho)$ for `related` ones and
    1 for `independent` ones, times the switch's weight
    (`switch_weight`). A stratum's own walk is its own: weekly standard
    normal steps, `"{name}::walk"` by stratum and week, times `walk_sd`,
    read between weeks as the shared walk is.

    Parameters
    ----------
    name : str
        Name of the component.
    frame : ReferenceFrame
        The triangle's strata and dates.
    shared : callable
        `shared(sign)` returns the shared change $S_t$, `(T,)`.
    own : sequence
        Each stratum's own kinds and values (`own_change`).
    relation : str
        `shared`, `related` or `independent`.
    rho : float
        Related strata's share of the shared change.
    switch : str
        `none`, `diverge` or `converge`.
    switch_at, switch_days : float
        Where the switch is centred (share of the observed days) and the
        days it takes.
    walk_sd : float
        Own walks' weekly sd.
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        shared: Any,
        own: Any = (),
        relation: str = "shared",
        rho: float = 1.0,
        switch: str = "none",
        switch_at: float = 0.5,
        switch_days: float = 14.0,
        walk_sd: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        self.shared = shared
        self.own = list(own)
        self.relation = relation
        weight = {"shared": 0.0, "related": 1 - rho, "independent": 1.0}
        many = len(frame.place) > 1
        self.u = (
            weight[relation]
            * switch_weight(frame, switch, switch_at, switch_days)
            if many
            else np.zeros(len(frame.dates))
        )
        self.walk_sd = float(walk_sd)

    def _walks(self, walking: np.ndarray) -> Array:
        """Each stratum's own walk by day, `(G, T)`, 0 where none."""
        G, T = len(self.frame.place), len(self.frame.dates)
        obs = self.frame.cropped()
        W = int(obs.week.max()) + 2
        z = numpyro.sample(
            f"{self.name}::walk", dist.Normal(0.0, 1.0).expand([G, W])
        )
        steps = jnp.concatenate([jnp.zeros((G, 1)), z[:, :-1]], axis=1)
        weekly = self.walk_sd * jnp.cumsum(steps, axis=1)
        day = np.minimum(np.arange(T), self.frame.observed - 1)
        x = day / 7
        w = np.floor(x).astype(int)
        f = x - w
        daily = (
            weekly[:, np.minimum(w, W - 1)] * (1 - f)
            + weekly[:, np.minimum(w + 1, W - 1)] * f
        )
        return jnp.where(jnp.asarray(walking)[:, None], daily, 0.0)

    def sample(self, sign: float = 1.0, **kwargs: object) -> Array:
        """Return the change by stratum and reference day, uncentred.

        Parameters
        ----------
        sign : float
            1 for the location, -1 for the logit hazard.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            `(G, T)`.
        """
        G, T = len(self.frame.place), len(self.frame.dates)
        S = jnp.broadcast_to(jnp.asarray(self.shared(sign)), (G, T))
        if not np.any(self.u):
            return S
        own = np.stack(
            [
                change_path(self.frame, kinds, v, sign, walk=False)
                for kinds, v in self.own
            ]
        )
        mine = jnp.asarray(own)
        walking = np.array(
            [
                "walk_daily" in [t for k in kinds for t in CHANGE_TERMS[k]]
                for kinds, _ in self.own
            ]
        )
        if walking.any():
            mine = mine + sign * self._walks(walking)
        u = jnp.asarray(self.u)[None, :]
        return (1 - u) * S + u * mine


class CountCoupling(RandomVariable):
    r"""A term linear in each stratum's log expected counts.

    For stratum $g$ and reference day $t$ with expected counts
    $\lambda_{g,t}$:

    $$c_{g,t} = \kappa_g e^{\epsilon_{g,t}}
    \left(x_{g,t} - \bar x_g\right), \quad x_{g,t} = \log(1 +
    \lambda_{g,t}),$$

    with $\bar x_g$ the stratum's mean over the observed days, the slope
    $\kappa_g$ of either sign, and $\epsilon$ noise on the log slope
    (`NoiseLayers`, uncentred, sites `"{name}::strata_slope"` and
    `"{name}::strata_jitter_slope"`). Padded days take the last observed
    day's value.

    Parameters
    ----------
    name : str
        Name of the component.
    frame : ReferenceFrame
        The triangle's strata and dates.
    slope : array_like
        Each stratum's slope $\kappa_g$, `(G,)` (or one for all).
    level : str
        `shared`, `strata` or `strata_time` (`COUPLING_LAYERS`).
    sd : float
        The sd of each layer of $\epsilon$.
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        slope: Any,
        level: str = "shared",
        sd: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        G = len(frame.place)
        self.slope = np.broadcast_to(
            np.atleast_1d(np.asarray(slope, dtype=float)), (G,)
        )
        layers = COUPLING_LAYERS[level]
        self.noise = NoiseLayers(
            name,
            frame,
            strata_sd=sd if "strata" in layers else 0.0,
            strata_time_sd=sd if "strata_time" in layers else 0.0,
            centre=False,
        )

    def sample(self, log_lambda: Any = None, **kwargs: object) -> Array:
        """Return the term by stratum and reference day.

        Parameters
        ----------
        log_lambda : array_like
            Log expected counts by stratum and reference day, `(G, T)`.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            `(G, T)`.
        """
        if log_lambda is None:
            raise ValueError(f"{self.name} needs the log expected counts")
        obs = self.frame.observed
        x = jnp.logaddexp(0.0, jnp.asarray(log_lambda))
        x = x - x[:, :obs].mean(axis=1, keepdims=True)
        T = x.shape[1]
        x = x[:, np.minimum(np.arange(T), obs - 1)]
        slope = jnp.asarray(self.slope)[:, None]
        if self.noise:
            slope = slope * jnp.exp(self.noise(term="slope"))
        return slope * x
