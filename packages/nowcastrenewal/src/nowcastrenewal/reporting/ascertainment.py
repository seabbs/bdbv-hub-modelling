# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""Ascertainment that changes over time, by stratum.

The latent step ascertains a static share $a_g$ of each stratum's
reference events. `AscertainmentChange` moves it on the logit scale:

$$\operatorname{logit} a_{g,t} = \operatorname{logit} a_g + \eta_{g,t},$$

so the expected counts by reference date become
$\lambda_{g,t} a_{g,t} / a_g$. The shift $\eta$ has three parts, each the
delay's structure applied to ascertainment (`structure`):

- a change over time (drift, step, periodic, smooth or walk), shared by
  the strata, related or independent, with a switch between shared and
  own (`structure.StructuredChange`);
- a term linear in the stratum's log expected counts by reference date,
  lower or higher ascertainment at peaks (`structure.CountCoupling`);
- noise by week, by stratum and by both (`structure.NoiseLayers`).

$\eta$ is centred over the strata and observed days, so the drawn
ascertainment stays the level. Ascertainment changes how many cases are
ever reported, the delay's terms only when; the two are kept apart.
Its sites are `"{name}::change::walk"`, `"{name}::counts::..."` and
`"{name}::{layer}_ascertainment"`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logit
from pyrenew.metaclass import RandomVariable

from .effects import ReferenceFrame, change_path
from .structure import CountCoupling, NoiseLayers, StructuredChange, own_change

__all__ = [
    "ASCERTAINMENT_NOISE",
    "AscertainmentChange",
    "ascertainment_change",
]

ASCERTAINMENT_NOISE: dict[str, tuple[str, ...]] = {
    "none": (),
    "time": ("time",),
    "strata": ("strata",),
    "strata_time": ("strata_time",),
    "all": ("time", "strata", "strata_time"),
}
"""The noise layers of each `ascertainment_noise` level."""

_SHARED = {
    "drift": "asc_drift",
    "step": "asc_step",
    "step_at": "asc_step_at",
    "periodic": "asc_periodic",
    "periodic_weeks": "asc_periodic_weeks",
    "smooth_amp": "asc_smooth_amp",
    "smooth_period": "asc_smooth_period",
    "smooth_phase": "asc_smooth_phase",
    "delay_walk": "asc_walk",
}
"""The shared change's settings (`effects.change_path`) and their values."""


def _active(design: Mapping[str, Any]) -> bool:
    return any(
        design.get(k, off) != off
        for k, off in (
            ("ascertainment_change", "static"),
            ("ascertainment_change_strata", "shared"),
            ("ascertainment_counts", "none"),
            ("ascertainment_noise", "none"),
        )
    )


class AscertainmentChange(RandomVariable):
    """Ascertainment that changes over time (module docstring).

    Parameters
    ----------
    name : str
        Name of the component.
    frame : ReferenceFrame
        The triangle's strata and dates.
    design, values : mapping
        The simulation's design (`ascertainment_change`,
        `ascertainment_change_strata`, `ascertainment_change_switch`,
        `ascertainment_counts`, `ascertainment_noise`) and values
        (`ascertainment` and the `asc_` settings).
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        design: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        v = values
        G = len(frame.place)
        self.level = np.broadcast_to(
            np.atleast_1d(np.asarray(v["ascertainment"], dtype=float)), (G,)
        )
        kinds = (str(design.get("ascertainment_change", "static")),)
        shared = {k: v[name_] for k, name_ in _SHARED.items()}
        shared["weeks"] = v["weeks"]
        self.change = StructuredChange(
            f"{name}::change",
            frame,
            shared=lambda sign: change_path(frame, kinds, shared, sign),
            own=[own_change(v, "asc_own_", g) for g in range(G)],
            relation=str(design.get("ascertainment_change_strata", "shared")),
            rho=float(v["asc_rho"]),
            switch=str(design.get("ascertainment_change_switch", "none")),
            switch_at=float(v["asc_switch_at"]),
            switch_days=float(v["asc_switch_days"]),
            walk_sd=float(v["asc_walk_sd"]),
        )
        counts = str(design.get("ascertainment_counts", "none"))
        self.coupling = None
        if counts != "none":
            sign = np.broadcast_to(np.atleast_1d(v["asc_counts_sign"]), (G,))
            self.coupling = CountCoupling(
                f"{name}::counts",
                frame,
                slope=float(v["asc_counts_slope"]) * sign,
                level=counts,
                sd=float(v["asc_counts_sd"]),
            )
        layers = ASCERTAINMENT_NOISE[
            str(design.get("ascertainment_noise", "none"))
        ]
        sd = float(v["asc_noise_sd"])
        self.noise = NoiseLayers(
            name,
            frame,
            time_sd=sd if "time" in layers else 0.0,
            strata_sd=sd if "strata" in layers else 0.0,
            strata_time_sd=sd if "strata_time" in layers else 0.0,
            centre=False,
        )

    def shift(self, log_lambda: Any) -> Array:
        r"""Return $\eta$ by stratum and reference day, `(G, T)`."""
        eta = jnp.asarray(self.change(sign=1.0))
        if self.coupling is not None:
            eta = eta + self.coupling(log_lambda=log_lambda)
        if self.noise:
            eta = eta + self.noise(term="ascertainment")
        obs = self.frame.observed
        return eta - eta[:, :obs].mean()

    def sample(self, log_lambda: Any, **kwargs: object) -> Array:
        """Return the log expected counts with the changing ascertainment.

        Parameters
        ----------
        log_lambda : array_like
            Log expected counts at the static ascertainment, `(G, T)`.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            `(G, T)`.
        """
        a = jnp.clip(jnp.asarray(self.level), 1e-6, 1 - 1e-6)[:, None]
        base = logit(a)
        eta = self.shift(log_lambda)
        # log a_{g,t} = log sigmoid(logit a_g + eta).
        log_a = -jnp.logaddexp(0.0, -(base + eta))
        return jnp.asarray(log_lambda) + log_a - jnp.log(a)


def ascertainment_change(
    name: str,
    frame: ReferenceFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
) -> AscertainmentChange | None:
    """Return the design's `AscertainmentChange`, or None if static."""
    if not _active(design):
        return None
    return AscertainmentChange(name, frame, design, values)
