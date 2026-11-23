# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""External Rt estimates as a pseudo-observation of a latent Rt."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


def group_rt(rt: ArrayLike, weight: ArrayLike, members: ArrayLike) -> Array:
    r"""Return the Rt of groups of places, weighted by their infections.

    $$R_G = \frac{\sum_{p \in G} w_p R_p}{\sum_{p \in G} w_p}.$$

    A place with no weight adds nothing, whatever its Rt. A group with
    no weight has no Rt (NaN).

    Parameters
    ----------
    rt
        Rt of each place, `(..., P)`.
    weight
        Infections of each place, `(..., P)`.
    members
        Membership of each group, `(G, P)`.

    Returns
    -------
    Array
        `(..., G)` group Rt.
    """
    weight = jnp.asarray(weight)
    members = jnp.asarray(members, dtype=weight.dtype)
    num = jnp.where(weight > 0, weight * jnp.asarray(rt), 0.0)
    return (num @ members.T) / (weight @ members.T)


class RtPseudoObservation(RandomVariable):
    r"""External Rt estimates as a pseudo-observation with inflated SDs.

    For groups $G$ of provinces (the nation, or BVDOutbreakSize patches)
    and the day $s_b$ of the external estimate, the model's group Rt is
    infection weighted over the week of $s_b$,

    $$R_G(s_b) = \frac{\sum_{p \in G} I_p R_p(s_b)}{\sum_{p \in G} I_p},$$

    and the summaries enter as

    $$\mu_N \sim N(\log R_N(s_b), \kappa \sigma_N), \qquad
    d_k \sim N(\log R_k(s_b) - \log R_N(s_b), \kappa \tau_k).$$

    The SD inflation $\kappa \ge 1$ says how far we trust an external
    estimate built partly on the same data: its SDs are widened by
    $\kappa$. It is a power prior with tempering $1 / \kappa^2$. With
    $\kappa = \infty$ nothing is added, so the model is unchanged.
    PyRenew routes only infections to observation processes, so this is
    a `RandomVariable` that the latent process calls, as it holds both
    Rt and infections. Its sites are `{name}_log_rt_model`,
    `{name}_z` and `{name}_rt_obs`.
    """

    def __init__(
        self,
        name: str,
        sd_inflation: float,
        day: int,
        groups: ArrayLike,
        values: ArrayLike,
        scales: ArrayLike,
        kinds: ArrayLike,
    ) -> None:
        r"""Initialize the pseudo-observation.

        Parameters
        ----------
        name
            Prefix for the sites.
        sd_inflation
            $\kappa$, at least 1; `math.inf` adds no term.
        day
            $s_b$ as a day index from the first data day.
        groups
            Boolean membership, `(G, P)`.
        values
            Log-scale summaries, `(G,)`.
        scales
            Their SDs, positive, `(G,)`.
        kinds
            0 for the national level, 1 for a deviation from it, `(G,)`.

        Raises
        ------
        ValueError
            If `sd_inflation` is below 1 or NaN, `day` is negative, the
            shapes do not match, a scale is not positive or a kind is not
            0 or 1.
        """
        groups = jnp.asarray(groups, dtype=float)
        values = jnp.asarray(values, dtype=float)
        scales = jnp.asarray(scales, dtype=float)
        kinds = jnp.asarray(kinds, dtype=int)
        if not float(sd_inflation) >= 1.0:
            raise ValueError(
                f"sd_inflation must be at least 1, got {sd_inflation}."
            )
        if int(day) < 0:
            raise ValueError(f"day must be non-negative, got {day}.")
        if groups.ndim != 2:
            raise ValueError(
                f"groups must have shape (G, P), got {groups.shape}."
            )
        n_groups = groups.shape[0]
        for label, x in (
            ("values", values),
            ("scales", scales),
            ("kinds", kinds),
        ):
            if x.shape != (n_groups,):
                raise ValueError(
                    f"{label} must have shape ({n_groups},), got {x.shape}."
                )
        if bool(jnp.any(scales <= 0)):
            raise ValueError("scales must be positive.")
        if not bool(jnp.all((kinds == 0) | (kinds == 1))):
            raise ValueError("kinds must be 0 (national) or 1 (deviation).")
        super().__init__(name=name)
        self.sd_inflation = float(sd_inflation)
        self.day = int(day)
        self.groups = groups
        self.values = values
        self.scales = scales
        self.kinds = kinds

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RtPseudoObservation(name={self.name!r}, "
            f"sd_inflation={self.sd_inflation}, "
            f"day={self.day}, n_groups={self.groups.shape[0]})"
        )

    def log_group_rt(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
    ) -> tuple[Array, Array]:
        """Return log $R_G(s_b)$ for each group and for the nation.

        Infections are per resident, so the weights are scaled by the
        population `fractions` to be proportional to counts.

        Parameters
        ----------
        rt
            Daily Rt by province.
        infections
            Daily infections per resident by province.
        n_init
            Seeding days at the start of the axis.
        fractions
            Population share of each province.

        Returns
        -------
        tuple[Array, Array]
            `(G,)` group values and the national scalar.
        """
        week = n_init + 7 * (self.day // 7)
        w = infections[week : week + 7].sum(axis=0)
        if fractions is not None:
            w = w * fractions
        r = rt[n_init + self.day]

        group = jnp.log(group_rt(r, w, self.groups))
        national = jnp.log(group_rt(r, w, jnp.ones((1, w.shape[-1]))))[0]
        return group, national

    def sample(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
        **kwargs: object,
    ) -> None:
        """Add the pseudo-likelihood and record its diagnostics.

        Parameters
        ----------
        rt
            Daily Rt by province.
        infections
            Daily infections per resident by province.
        n_init
            Seeding days at the start of the axis.
        fractions
            Population share of each province.
        **kwargs
            Ignored, for the PyRenew interface.
        """
        group, national = self.log_group_rt(rt, infections, n_init, fractions)
        model = jnp.where(self.kinds == 0, group, group - national)
        numpyro.deterministic(f"{self.name}_log_rt_model", model)
        numpyro.deterministic(
            f"{self.name}_z", (self.values - model) / self.scales
        )
        if math.isfinite(self.sd_inflation):
            numpyro.sample(
                f"{self.name}_rt_obs",
                dist.Normal(model, self.sd_inflation * self.scales),
                obs=self.values,
            )
