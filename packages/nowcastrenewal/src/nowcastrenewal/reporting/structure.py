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

from .effects import ReferenceFrame

__all__ = ["NOISE_LAYERS", "NoiseLayers", "noise_layers"]

NOISE_LAYERS: dict[str, tuple[str, ...]] = {
    "none": (),
    "strata": ("strata",),
    "strata_time": ("strata_time",),
    "both": ("strata", "strata_time"),
}
"""The stratum noise layers of each `delay_strata_noise` level; the time
layer is `delay_jitter`'s."""


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
