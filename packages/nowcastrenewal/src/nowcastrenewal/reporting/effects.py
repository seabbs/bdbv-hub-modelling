# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Reference-date effects on the reporting delay, as numpyro sites.

The delay of a report from reference day $t$ in stratum $g$ has a
parametric part, a delay with location $\mu_{g,t}$ and scale $\sigma$
(`components.pmfs.Delay`), and a non-parametric part on the logit hazard
by delay $d$. Each is a regression on the reference day's covariates
(`change_covariates`: trend, step, periodic, a weekly walk, a smooth
cycle, a slowdown, backlogs) and random effects by place, age, weekday or
weekday within place, with standard normal sites scaled by their standard
deviations (non-centred):

$$\mu_{g,t} = \mu_0 + \sum_c \beta_c \tilde x_{c,g,t}
+ \sum_e s_e \tilde z_{e}[g, t],$$

where a tilde is a value less its mean over the frame's rows, as
epinowcast centres its design matrices (`centring_offsets`) when the
module has an intercept. The non-parametric part is a random walk over
delays, $s_w \sum_{j \le d} z_{w,j}$ (epinowcast's `rw(delay)`), plus an
iid effect by delay, plus, when there is no parametric part, an
intercept, the covariates and the random effects (centred).

The forms, and the simulator's values for each term, are the simulator R
engine's (`report_outbreak`) at the epinowcast pin; two differences are
listed in `ReferenceEffects`. PyRenew has no reporting-delay regression,
so this is our own `RandomVariable`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from pyrenew.metaclass import RandomVariable

from ..components.pmfs import Delay, censored_logit_hazard

__all__ = [
    "CHANGE_TERMS",
    "ReferenceEffects",
    "ReferenceFrame",
    "change_betas",
    "change_covariates",
]

CHANGE_TERMS: dict[str, tuple[str, ...]] = {
    "static": (),
    "drift": ("trend",),
    "step": ("step",),
    "periodic": ("periodic",),
    "walk": ("walk_daily",),
    "smooth": ("smooth",),
    "slowdown": ("slowdown",),
}
"""The covariates of each kind of delay change (`design["delay_change"]`);
a `backlog` covariate is added for backlog shocks."""


@dataclass(frozen=True)
class ReferenceFrame:
    """The strata and reference days of a reporting triangle.

    Attributes
    ----------
    dates : numpy.ndarray
        Reference dates, `datetime64[D]`, consecutive, `(T,)`.
    place, age : numpy.ndarray
        Each stratum's place and age group, 0-based, `(G,)`.
    """

    dates: np.ndarray
    place: np.ndarray
    age: np.ndarray

    @property
    def day(self) -> np.ndarray:
        """Days since the first reference date, `(T,)`."""
        d = np.asarray(self.dates, dtype="datetime64[D]")
        return np.asarray((d - d[0]).astype(int))

    @property
    def week(self) -> np.ndarray:
        """Weeks since the first reference date (epinowcast's `week`)."""
        return self.day // 7

    @property
    def day_of_week(self) -> np.ndarray:
        """Weekday of each reference date, Monday 0, `(T,)`."""
        d = np.asarray(self.dates, dtype="datetime64[D]")
        # 1970-01-01 was a Thursday (3).
        return (d.astype(int) + 3) % 7

    @property
    def n_places(self) -> int:
        """Places."""
        return int(np.max(self.place)) + 1

    @property
    def n_ages(self) -> int:
        """Age groups."""
        return int(np.max(self.age)) + 1


def _window(
    start_share: float, weeks: float, v: Mapping[str, Any]
) -> tuple[int, int]:
    # R's round() rounds half to even, as NumPy's does.
    start = int(np.round(start_share * v["weeks"] * 7))
    return start, start + 7 * int(weeks) - 1


def change_covariates(
    frame: ReferenceFrame, design: Mapping[str, Any], v: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return the reference-date covariates of a simulation's delay change.

    Parameters
    ----------
    frame : ReferenceFrame
        The triangle's strata and dates.
    design : mapping
        The design (`delay_change`, `shocks`).
    v : mapping
        The values.

    Returns
    -------
    dict
        Each covariate by stratum and reference day, `(G, T)`.
    """
    G, T = len(frame.place), len(frame.dates)
    week, day = frame.week, frame.day
    n_weeks = week.max()
    terms = list(CHANGE_TERMS[design["delay_change"]])
    if design["shocks"] == "backlog":
        terms.append("backlog")
    out: dict[str, np.ndarray] = {}
    for term in terms:
        if term == "trend":
            x = week / n_weeks - 0.5
        elif term == "step":
            x = (week >= v["step_at"] * n_weeks).astype(float)
        elif term == "periodic":
            x = np.sin(2 * np.pi * week / v["periodic_weeks"])
        elif term == "walk_daily":
            from ..latent.rt import daily_walk

            x = daily_walk(day, v["delay_walk"])
        elif term == "smooth":
            x = np.sin(2 * np.pi * day / v["smooth_period"] + v["smooth_phase"])
        elif term == "slowdown":
            a, b = _window(v["slow_at"], v["slow_weeks"], v)
            x = ((day >= a) & (day <= b)).astype(float)
        else:
            # A backlog's covariate is the log of its multiplier, in the
            # shocked place over the shock's reference dates.
            x2 = np.zeros((G, T))
            places = np.atleast_1d(v["shock_places"])
            for k, p in enumerate(places):
                a, b = _window(
                    np.atleast_1d(v["shock_at"])[k],
                    np.atleast_1d(v["shock_weeks"])[k],
                    v,
                )
                on = (frame.place == int(p) - 1)[:, None] & (
                    (day >= a) & (day <= b)
                )[None, :]
                x2 += on * np.log(np.atleast_1d(v["shock_mult"])[k])
            out[term] = x2
            continue
        out[term] = np.broadcast_to(np.asarray(x, dtype=float), (G, T)).copy()
    return out


def change_betas(
    terms: Any, design: Mapping[str, Any], v: Mapping[str, Any], sign: float
) -> dict[str, float]:
    """Return each change covariate's coefficient.

    A backlog, a slowdown, the walk and the smooth cycle multiply the mean
    delay: their coefficients are on the location (`sign` 1) or, with the
    opposite sign, on the logit hazard (`sign` -1). Trend, step and
    periodic take the drawn size.

    Parameters
    ----------
    terms : iterable of str
        The covariates.
    design : mapping
        The design (`delay_change`).
    v : mapping
        The values.
    sign : float
        1 for the location, -1 for the logit hazard.

    Returns
    -------
    dict
        Coefficient by covariate.
    """
    size = {"drift": "drift", "step": "step", "periodic": "periodic"}.get(
        design["delay_change"]
    )
    change = float(v[size]) if size else 0.0
    out = {}
    for term in terms:
        if term in ("trend", "step", "periodic"):
            out[term] = change
        elif term in ("backlog", "walk_daily"):
            out[term] = sign
        elif term == "slowdown":
            out[term] = float(np.log(v["slow_mult"])) * sign
        elif term == "smooth":
            out[term] = float(v["smooth_amp"]) * sign
    return out


def _effects(
    frame: ReferenceFrame, design: Mapping[str, Any], v: Mapping[str, Any]
) -> dict[str, tuple[np.ndarray, int, float]]:
    """Return the random effects: index `(G, T)`, levels and sd, by name."""
    G, T = len(frame.place), len(frame.dates)
    place = np.broadcast_to(frame.place[:, None], (G, T))
    age = np.broadcast_to(frame.age[:, None], (G, T))
    dow = np.broadcast_to(frame.day_of_week[None, :], (G, T))
    multi_place, multi_age = frame.n_places > 1, frame.n_ages > 1
    out = {}
    if multi_place and design["delay_places"] != "shared":
        out["place"] = (place, frame.n_places, float(v["delay_place_sd"]))
    if multi_age:
        out["age"] = (age, frame.n_ages, float(v["delay_age_sd"]))
    if design["delay_dow"] == "yes":
        sd = float(v["delay_dow_sd"])
        if multi_place and design["delay_places"] == "independent":
            out["day_of_week_place"] = (
                dow * frame.n_places + place,
                7 * frame.n_places,
                sd,
            )
        else:
            out["day_of_week"] = (dow, 7, sd)
    return out


class ReferenceEffects(RandomVariable):
    r"""The reporting delay's logit hazards from reference-date effects.

    Returns the parametric and non-parametric logit hazards by stratum,
    reference day and delay, `(G, T, D)` each, for
    `triangle.combine_logit_hazards`. The sites are standard normal
    (non-centred): `"{name}::mean_{effect}"` for the location's random
    effects, `"{name}::hazard_walk"` and `"{name}::hazard_delay"` (one
    per delay), and `"{name}::hazard_{effect}"` for the hazard's random
    effects when there is no parametric part.

    Two differences from the R engine, which cannot run them as meant:

    - the weekday effect within place (`delay_places = "independent"`
      with `delay_dow = "yes"` and several places) is one effect per
      weekday and place; the engine's `(1 | day_of_week:place)` fails at
      the epinowcast pin (its fixed design drops a weekday while its
      random design keeps all), so those draws never ran;
    - the scale has no random effects: the engine sets them to 0
      (`refp_sd_beta`), so the scale is its intercept, as here.

    Parameters
    ----------
    name : str
        Name of the component.
    frame : ReferenceFrame
        The triangle's strata and dates.
    design, values : mapping
        The simulation's design and values.
    max_delay : int
        Delays in the window, $D$.
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        design: Mapping[str, Any],
        values: Mapping[str, Any],
        max_delay: int,
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        self.design = design
        self.values = values
        self.max_delay = max_delay
        self.parametric = design["delay_form"] != "non_parametric"
        self.hazard = design["delay_form"] != "parametric"
        self.covariates = change_covariates(frame, design, values)
        self.effects = _effects(frame, design, values)

    def _random(self, prefix: str) -> Array:
        """Each stratum and day's random effects, centred, `(G, T)`."""
        G, T = len(self.frame.place), len(self.frame.dates)
        total = jnp.zeros((G, T))
        for effect, (idx, n, sd) in self.effects.items():
            z = numpyro.sample(
                f"{self.name}::{prefix}_{effect}",
                dist.Normal(0.0, 1.0).expand([n]),
            )
            x = sd * z[idx]
            total = total + x - x.mean()
        return total

    def _fixed(self, sign: float, centre: bool) -> Array:
        """Each stratum and day's change covariates times coefficients."""
        betas = change_betas(self.covariates, self.design, self.values, sign)
        G, T = len(self.frame.place), len(self.frame.dates)
        total = jnp.zeros((G, T))
        for term, x in self.covariates.items():
            xc = x - x.mean() if centre else x
            total = total + betas[term] * jnp.asarray(xc)
        return total

    def location(self) -> Array:
        r"""Return $\mu_{g,t}$, `(G, T)` (centred effects)."""
        delay = Delay.from_values(self.design["delay_family"], self.values)
        return (
            jnp.asarray(delay.mu)
            + self._fixed(1.0, centre=True)
            + self._random("mean")
        )

    def non_parametric(self) -> Array:
        """Return the non-parametric logit hazard, `(G, T, D)`."""
        v, D = self.values, self.max_delay
        G, T = len(self.frame.place), len(self.frame.dates)
        z_w = numpyro.sample(
            f"{self.name}::hazard_walk", dist.Normal(0.0, 1.0).expand([D])
        )
        z_d = numpyro.sample(
            f"{self.name}::hazard_delay", dist.Normal(0.0, 1.0).expand([D])
        )
        walk = float(v["hazard_walk_sd"]) * jnp.cumsum(z_w)
        by_delay = walk + float(v["hazard_sd"]) * z_d
        if self.parametric:
            # No intercept: epinowcast does not centre the module.
            return jnp.broadcast_to(by_delay, (G, T, D))
        by_delay = by_delay - by_delay.mean()
        intercept = jnp.log(1 / v["delay_mean"])  # logit(1 / (1 + mean))
        by_day = self._fixed(-1.0, centre=True) + self._random("hazard")
        return jnp.asarray(intercept + by_day[..., None] + by_delay)

    def sample(self, **kwargs: object) -> tuple[Array, Array]:
        """Return the parametric and non-parametric logit hazards.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        tuple of Array
            Each `(G, T, D)`; a part that is absent is 0.
        """
        G, T, D = len(self.frame.place), len(self.frame.dates), self.max_delay
        zeros = jnp.zeros((G, T, D))
        parametric = zeros
        if self.parametric:
            family = self.design["delay_family"]
            sigma = Delay.from_values(family, self.values).sigma
            mu = self.location()
            numpyro.deterministic(f"{self.name}::location", mu)
            parametric = censored_logit_hazard(Delay(family, mu, sigma), D)
        non_parametric = self.non_parametric() if self.hazard else zeros
        return parametric, non_parametric
