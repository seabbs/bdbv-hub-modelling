# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Subpopulation renewal coupled by importation, with susceptible depletion.

`MetapopulationInfections` is a PyRenew `BaseSubpopulationInfections` with
the shared renewal step (`components.renewal.run_renewal`, per resident,
with exponential depletion of the share at risk) and the per-resident
mixing matrix from the shared `components.mixing.Mixing`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent import BaseSubpopulationInfections
from pyrenew.latent.base import LatentSample, PopulationStructure
from pyrenew.metaclass import RandomVariable

from ..components.renewal import (
    ExponentialDepletion,
    NoDepletion,
    run_renewal,
)
from .pooled import pooled_walk


class MetapopulationInfections(BaseSubpopulationInfections):
    r"""Province renewal with importation and susceptible depletion.

    Rt follows `BaseSubpopulationInfections`: a national temporal process
    plus sum-to-zero province deviations,
    $\log R_p(t) = \log R(t) + \delta_p(t)$, here sampled on the
    sum-to-zero basis (`pooled_walk`).
    Seeding is exponential at the growth rate of `seed_growth_rv`, anchored
    so that `I0_rv` is the level on the first data day rather than the
    first seeding day.
    The renewal is `components.renewal.run_renewal`, which sends a share
    $\varepsilon$ of transmission to other provinces through the mixing
    matrix of `mixing_rv` and depletes each province's effective
    susceptible pool.
    `sample` runs the steps of `BaseSubpopulationInfections.sample`, with
    the renewal in `_infections_from_rt`.

    Infections are per resident, so the population fractions passed at
    sample time set the transfer weights. Only a share $\nu$
    (`effective_population_rv`) of each province is at risk, and it is
    that pool that depletes. Because $\nu$ only enters through depletion,
    it does not trade off against the seed or the ascertainment.
    """

    def __init__(
        self,
        *,
        mixing_rv: RandomVariable | None = None,
        effective_population_rv: RandomVariable | None = None,
        rt_observation: RandomVariable | None = None,
        seed_growth_rv: RandomVariable | None = None,
        seed_days: ArrayLike | None = None,
        seeded: ArrayLike | None = None,
        **kwargs: object,
    ) -> None:
        r"""Initialize the coupled infections process.

        Parameters
        ----------
        mixing_rv
            Returns the mixing matrix $W$ given the population fractions
            (`components.mixing.Mixing`). `None` turns coupling off.
        effective_population_rv
            $\nu$. `None` turns susceptible depletion off.
        rt_observation
            Called with Rt, infections, the seeding length and the
            fractions, to add an observation of Rt
            (`RtPseudoObservation`). `None` adds none.
        seed_growth_rv
            Daily growth rate of the seed over the seeding period, before
            the data, shared by every place. `None` holds the seed flat
            (rate 0).
        seed_days
            Day (from the first data day) each province's seeding starts.
            Provinces with a day after 0 start (almost) empty and get
            their seed, the `I0_rv` level, as a constant daily input over
            the 7 days from it, plus whatever importation brings. The
            day leads the first case week by the incubation period
            (`model.seed_lead`). A province with day `inf` is never
            seeded and gains infections only by importation. `None`
            seeds every province at the start.
        seeded
            Boolean, one per province: provinces with an Rt deviation.
            The rest follow the pooled trend. `None` is all.
        **kwargs
            Passed to `BaseSubpopulationInfections`.
        """
        super().__init__(**kwargs)
        self.mixing_rv = mixing_rv
        self.effective_population_rv = effective_population_rv
        self.rt_observation = rt_observation
        self.seed_growth_rv = seed_growth_rv
        self.seed_days = None if seed_days is None else np.asarray(seed_days)
        self.seeded = None if seeded is None else np.asarray(seeded)

    def _initial_infections(
        self, I0: Array, n_days: int
    ) -> tuple[Array, Array | None]:
        """Seed infections, and the later seeding input of late provinces.

        The seed `I0` is set on the first data day and projected back
        over the seeding period at the seed growth rate
        (`seed_growth_rv`). That rate is its own parameter rather than
        the growth implied by the first Rt, which the data pin as the
        level of the fitted walk. A positive rate, as its prior gives,
        means the seed only falls going back, so it never exceeds its
        level on the first data day. Late provinces start from 1e-6 of
        their seed level (PyRenew needs a positive start) and take the
        seed over the 7 days from their seed day.

        Returns
        -------
        tuple
            Infections per resident over the seeding period,
            `(n_init, P)`, and the seeding input after it, `(n_days, P)`,
            or None.
        """
        n_init = self.n_initialization_points
        r_seed = 0.0 if self.seed_growth_rv is None else self.seed_growth_rv()
        I0_all = I0[None, :] * jnp.exp(
            r_seed * (jnp.arange(n_init) - n_init)[:, None]
        )
        if self.seed_days is None:
            return I0_all, None
        late = jnp.asarray(self.seed_days > 0)
        I0_all = jnp.where(late[None, :], 1e-6 * I0_all, I0_all)
        days = np.arange(n_days)[:, None]
        window = (days >= self.seed_days) & (days < self.seed_days + 7)
        seeding = jnp.asarray(window & late) * I0[None, :]
        return I0_all, seeding

    def _infections_from_rt(
        self,
        rt_raw: Array,
        I0: Array,
        gen_int: Array,
        pop: PopulationStructure,
        seeding: Array | None = None,
    ) -> tuple[Array, Array]:
        r"""Infections and adjusted Rt after seeding, by `run_renewal`.

        Records the infections from importation as `infections_imported`.

        Returns
        -------
        tuple of Array
            Infections per resident and $R_p(t) S_p(t) / \nu$, both
            `(n_days, P)`.
        """
        mixing = (
            jnp.eye(pop.n_subpops)
            if self.mixing_rv is None
            else self.mixing_rv(fractions=pop.fractions)
        )
        pool = (
            None
            if self.effective_population_rv is None
            else self.effective_population_rv()
        )
        run = run_renewal(
            I0,
            gen_int,
            rt_raw,
            1.0 if pool is None else pool,
            mixing=mixing,
            seeding=seeding,
            depletion=NoDepletion() if pool is None else ExponentialDepletion(),
        )
        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic(
                "infections_imported",
                jnp.vstack([jnp.zeros_like(I0), run.imported]),
            )
        if pool is None:
            return run.infections[I0.shape[0] :], rt_raw
        # R S / nu on each day before its infections, from the step's
        # susceptibles after each day.
        s0 = jnp.clip(pool - I0.sum(axis=0), 0.0, pool)
        before = jnp.vstack([s0[None, :], run.susceptible[:-1]])
        return run.infections[I0.shape[0] :], rt_raw * before / pool

    def _latent(
        self,
        log_rt: ArrayLike,
        log_rt_baseline: ArrayLike,
        n_init: int,
        n_days_post_init: int,
        pop: PopulationStructure,
    ) -> LatentSample:
        """Infections from each place's log Rt, with the Rt sites recorded.

        The seeding and renewal shared by this model and the zones' model
        (`nowcastrenewal.model.zones.ZoneInfections`).
        """
        rt = jnp.exp(log_rt)
        gen_int = self.gen_int_rv()
        I0 = self._validate_and_prepare_I0(jnp.asarray(self.I0_rv()), pop)
        I0_all, seeding = self._initial_infections(I0, n_days_post_init)
        post, r_eff = self._infections_from_rt(
            rt[n_init:], I0_all, gen_int, pop, seeding
        )
        infections = jnp.vstack([I0_all, post])
        aggregate = infections @ pop.fractions

        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
            numpyro.deterministic("rt_subpop", rt)
            numpyro.deterministic(
                "rt_subpop_adjusted", jnp.vstack([rt[:n_init], r_eff])
            )
        if self.rt_observation is not None:
            self.rt_observation(
                rt=rt,
                infections=infections,
                n_init=n_init,
                fractions=pop.fractions,
            )
        return LatentSample(aggregate=aggregate, all_subpops=infections)

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: Array | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """Sample per-capita infections for each province.

        Parameters
        ----------
        n_days_post_init
            Days after the seeding period.
        subpop_fractions
            Population share of each province.
        first_day_dow
            Weekday of the first day, passed by PyRenew.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        LatentSample
            `aggregate` of shape `(n_total,)` and `all_subpops` of shape
            `(n_total, n_subpops)`.
        """
        pop = self._parse_and_validate_fractions(subpop_fractions)
        n_init = self.n_initialization_points
        log_rt_baseline, log_rt = pooled_walk(
            self.baseline_rt_process,
            self.subpop_rt_deviation_process,
            self.log_rt_time_0_rv(),
            n_init + n_days_post_init,
            pop.n_subpops,
            "log_rt",
            first_day_dow,
            active=self.seeded,
        )
        return self._latent(
            log_rt, log_rt_baseline, n_init, n_days_post_init, pop
        )
