# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Assemble, fit and forecast the province model with `PyrenewBuilder`."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import Array, random
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    PowerTransform,
)
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.model import MultiSignalModel
from pyrenew.observation import NegativeBinomialNoise
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from ..components.pmfs import Delay, delay_pmf, generation_time_pmf
from ..components.walks import NonCentredRandomWalk
from .composition_noise import DirichletMultinomialNoise
from .pooled import PooledInitialValue

SUNDAY = 6
"""Weekday number of Sunday (Monday is 0), the first day of a hub week."""
GEN_INT_LENGTH = 42
"""Lags of the generation interval, about its 98.5% quantile (a whole
number of weeks, as the model axis starts this many days early)."""
GI_MEAN = 15.3
"""Gamma generation interval mean in days. See `GI_SD`."""
GI_SD = 9.3
"""Gamma generation interval mean and SD in days, BVDOutbreakSize's prior
means (from the NEJM serial interval). After double censoring, dropping
lag 0 and truncating at 42 days the effective mean is 14.8 and SD 8.4."""
INCUBATION_MEAN = 6.3
"""Log-normal incubation period prior mean in days, Bundibugyo virus disease
in Uganda in 2007 (MacNeil et al. 2010, n = 24), as BVDOutbreakSize uses."""
INCUBATION_MEAN_SD = 0.54
"""Prior SD of the incubation period mean, BVDOutbreakSize's, which gives
MacNeil et al.'s 95% CI of 5.2 to 7.3 days."""
INCUBATION_SD = 3.5
"""Log-normal incubation period prior SD in days, BVDOutbreakSize's (MacNeil
et al. give no interval on the spread)."""
INCUBATION_SD_SD = 0.8
"""Prior SD of the incubation period SD, BVDOutbreakSize's."""
INCUBATION_LENGTH = 28
"""Longest incubation period in days, its 99.9% quantile at the prior means
rounded up."""
PHI_FLOOR = 1e-3
"""Added to $1/\\phi$ so the concentration stays at or below 1000."""
SEED_DOUBLING_DAYS = 11.7
"""Median doubling time in days of the seed's growth before the data,
BVDOutbreakSize's cryptic growth prior (`M_PRIOR_DOUBLING_DAYS`, from the
BEAST X exponential growth estimate, 95% HPD 6.8 to 17.5 days)."""
SEED_GROWTH_LOG_SD = 0.30
"""Log-scale SD of the seed growth rate prior, BVDOutbreakSize's."""
ALPHA0 = 0.6
"""Ascertainment that turns the seed's expected onsets into infections."""


def generation_interval(
    mean: float = GI_MEAN, sd: float = GI_SD, length: int = GEN_INT_LENGTH
) -> np.ndarray:
    r"""Double interval censored Gamma generation interval on lags 1..length.

    The daily pmf on $0..L$ from `components.pmfs`, truncated at $L$. The
    renewal uses lags 1 to $L$, so lag 0 (about 0.05% of the mass) is
    dropped and the rest renormalised, as BVDOutbreakSize left-truncates
    its generation interval at one day. Computed once, in float64.
    Matches `primarycensored::dpcens(1:L, ..., L = 1, D = L + 1)`.

    Parameters
    ----------
    mean, sd : float
        Gamma mean and SD in days. Default to `GI_MEAN` and `GI_SD`.
    length : int
        Longest lag in days. Defaults to `GEN_INT_LENGTH`.

    Returns
    -------
    numpy.ndarray
        Pmf on lags 1..length (PyRenew order, element 0 at lag 1).
    """
    return generation_time_pmf(
        Delay.from_mean_cv("gamma", mean, sd / mean), length
    )


def incubation_period(
    mean: float = INCUBATION_MEAN,
    sd: float = INCUBATION_SD,
    length: int = INCUBATION_LENGTH,
) -> np.ndarray:
    r"""Double interval censored log-normal incubation period on 0..length.

    The daily pmf from infection to onset, from `components.pmfs`, truncated
    at $L$ and renormalised. Computed once, in float64.
    Matches `primarycensored::dpcens(0:L, plnorm, D = L + 1)`.

    Parameters
    ----------
    mean, sd : float
        Log-normal mean and SD in days. Default to `INCUBATION_MEAN` and
        `INCUBATION_SD`.
    length : int
        Longest lag in days. Defaults to `INCUBATION_LENGTH`.

    Returns
    -------
    numpy.ndarray
        Pmf on lags 0..length.
    """
    return delay_pmf(Delay.from_mean_cv("lognormal", mean, sd / mean), length)


def seed_lead(incubation: np.ndarray) -> int:
    """Days to seed a late province before its first case week.

    The lead that puts the most onsets in the first case week from a week
    of constant seeding: the argmax over $L$ of the mass of the week-long
    seed convolved with the incubation pmf on days $L$ to $L + 6$ after
    the seed starts. At the prior mean incubation period it is 5 days,
    with 67% of onsets in the week and 9% before it.

    Parameters
    ----------
    incubation : numpy.ndarray
        Incubation pmf on lags 0..H.

    Returns
    -------
    int
        The lead in days, at most 6 so the seed starts after the
        previous case week began.
    """
    onsets = np.convolve(np.full(7, 1 / 7), incubation)
    in_week = [onsets[lead : lead + 7].sum() for lead in range(7)]
    return int(np.argmax(in_week))


MAX_PUBLICATION_LAG: Final = 7
"""Longest report to publication lag, in days. Rows later than this at
the earlier release of a pair count towards that pair's background share
(`ReleasePairLikelihood`)."""


@dataclass(frozen=True)
class ModelConfig:
    r"""Structural choices of the model, for the ladder and the variants.

    Attributes
    ----------
    national : bool
        Fit one national renewal instead of provinces (ladder step 0).
    streams : {"split", "total"}
        Fit known and hub-imputed onsets as two streams with estimated
        delays and missingness, or one total stream.
    truncation : bool
        Right truncation by vintage. For the total stream it uses the
        hub's delay pmf as a fixed reporting delay.
    coupling : {"gravity", "none"}
        Importation between modelled provinces through the fixed gravity
        kernel, or none.
    depletion : bool
        Susceptible depletion in the renewal.
    effective_population : float or None
        Share $\nu$ of each province at risk, which depletion uses.
        `None` (default) estimates it with a log-normal prior; a number
        fixes it.
    bvd : bool
        Add BVDOutbreakSize Rt as a pseudo-observation
        (`RtPseudoObservation`), when the vintage carries it.
    bvd_sd_inflation : float
        Factor $\kappa \ge 1$ on the SDs of the BVDOutbreakSize Rt
        estimates. It sets how much we trust an external estimate built
        partly on the same case data as ours: at 1 we would count those
        data twice at full weight, and larger values widen the estimates
        so they move our Rt mainly where our own data are weak. It is set
        by that judgement, like a prior, and is not tuned on scores.
        Sam set 1.5 (#768 item 5), the power prior with tempering
        $1 / 1.5^2 \approx 0.44$, a little weaker than the earlier
        $\sqrt{2}$ (tempering 0.5). `math.inf` gives the model without
        the term, exactly.
    dmax : int
        Maximum onset-to-report delay in days. Longer delays of known
        onsets are counted at `dmax`, the last cell of the hazard pmf,
        so the delay counts keep every case. At 60 days this lumps few
        cases: in the 2026-11-15 release 0.47% of known-onset delays
        exceed 60 days (99% are within 37). A longer axis would add
        hazard cells that those few cases barely inform.
    delay_counts : {"multinomial", "dirichlet-multinomial"}
        Likelihood of the delay file's counts given the weekly totals.
        The Dirichlet-multinomial (default) has $1/\sqrt{\psi} \sim
        \text{HalfNormal}(0.1)$, with $\psi$ capped at $10^4$, so that
        thousands of pooled delays do not make the delay overconfident.
    centred_rt : bool
        Sample the levels of the shared log Rt walk (`log_rt_baseline_x`)
        rather than its standardised steps (`NonCentredRandomWalk`).
        Every variant sets it the same way, per fit, where the counts pin
        the walk's weekly changes (`forecaster.fit_config`).
    rt_sd_lognormal : tuple of float or None
        Median and log-scale SD of a LogNormal prior on that walk's step
        SD, shared by every variant (Sam, 2026-11-24). A centred walk
        funnels where its step SD nears 0, which HalfNormal(0.2) allowed:
        at the 2026-11-22
        release, CD-NK fitted alone mixed the step SD at bulk ESS 29 and
        E-BFMI 0.27 to 0.36 under HalfNormal(0.2), and at 72 and 0.49 to
        0.70 under LogNormal(log 0.12, 0.5), in the same time and with no
        divergences. The prior moves its posterior median from 0.04 to
        0.08 a week.
    """

    national: bool = False
    streams: Literal["split", "total"] = "split"
    truncation: bool = True
    coupling: Literal["gravity", "none"] = "gravity"
    depletion: bool = True
    effective_population: float | None = None
    bvd: bool = False
    bvd_sd_inflation: float = 1.5
    dmax: int = 60
    delay_counts: Literal["multinomial", "dirichlet-multinomial"] = (
        "dirichlet-multinomial"
    )
    centred_rt: bool = False
    rt_sd_lognormal: tuple[float, float] | None = (0.12, 0.5)


MAIN = ModelConfig()
"""The main model configuration, with every default of `ModelConfig`."""
BVD = ModelConfig(bvd=True)
"""Main model informed by BVDOutbreakSize Rt (`hubrenewalbvd`)."""


@dataclass
class Vintage:
    """One data vintage, on weeks starting `start_date` (a Sunday).

    Attributes
    ----------
    start_date : datetime.date
        First day of the first data week. Must be a Sunday.
    cutoff : int
        Data cut-off as a day index from `start_date`.
    known, imputed : numpy.ndarray
        Weekly onset counts, `(n_weeks, P)`, NaN where unobserved.
    delay_counts : numpy.ndarray
        Known-onset cases by onset week, province and delay,
        `(n_weeks, P, dmax + 1)`.
    hub_delay_pmf : numpy.ndarray
        The hub's imputation delay pmf for this release, `(dmax + 1,)`.
    populations : numpy.ndarray
        Province populations, `(P,)`.
    kernel : numpy.ndarray or None
        Importation kernel between modelled provinces, rows receive,
        normalised per origin (`inputs.gravity_kernel`).
    horizon_weeks : int
        Forecast horizon in weeks.
    names : list of str
        Modelled province names.
    fit_mask : numpy.ndarray or None
        Boolean `(P,)`. False switches a province's likelihood off (it
        stays in the latent process and is still predicted).
    seed_days : numpy.ndarray or None
        Day (from `start_date`) each province's seeding starts: 0 for
        provinces with cases in the first data week, `seed_lead` days
        before the first case week for later ones, and `inf` (never) for
        provinces with no case.
    bvd : dict or None
        External Rt summaries: `day` (index of $s_b$ from the first data
        day), `groups` `(G, P)`, `values`, `scales` and `kinds` `(G,)`.
    release_pairs : dict or None
        Rows of each earlier delay file and how many the file before it
        had (`nowcastutils.releases.release_pairs`), or None.
    """

    start_date: dt.date
    cutoff: int
    known: np.ndarray
    imputed: np.ndarray
    delay_counts: np.ndarray
    hub_delay_pmf: np.ndarray
    populations: np.ndarray
    kernel: np.ndarray | None = None
    horizon_weeks: int = 3
    names: list[str] = field(default_factory=list)
    fit_mask: np.ndarray | None = None
    bvd: dict[str, Any] | None = None
    seed_days: np.ndarray | None = None
    release_pairs: dict[str, np.ndarray] | None = None

    @property
    def n_weeks(self) -> int:
        """Number of data weeks."""
        return int(self.known.shape[0])

    @property
    def n_provinces(self) -> int:
        """Number of provinces."""
        return int(self.known.shape[1])

    @property
    def seeded(self) -> np.ndarray:
        """Boolean, one per province: seeded, with a sampled seed.

        Provinces never seeded follow the pooled trends.
        """
        if self.seed_days is None:
            return np.ones(self.n_provinces, dtype=bool)
        return np.isfinite(self.seed_days)


def _pooled(
    name: str,
    n_groups: int,
    loc: float,
    scale: float,
    pooling_sd: float,
    centred: bool = True,
    active: np.ndarray | None = None,
) -> PooledInitialValue:
    r"""Group values around $\mu \sim N(m, s)$, spread $N^+(0, \tau)$."""
    return PooledInitialValue(
        name,
        n_groups,
        DistributionalVariable(f"{name}_mean", dist.Normal(loc, scale)),
        DistributionalVariable(f"{name}_sd", dist.HalfNormal(pooling_sd)),
        centred=centred,
        active=active,
    )


def _named_walk(
    name: str,
    sd: float,
    n_weeks: int,
    ar: bool = False,
    centred: bool = False,
    sd_prior: dist.Distribution | None = None,
) -> TemporalProcess:
    """Weekly walk frozen over the seeding period, fitted on the data weeks.

    `n_weeks` data weeks set the fit axis; later steps are drawn in the
    forecast pass. With `ar`, the walk is AR(1) with a half-life of
    LogNormal(log 6 weeks, 0.6) (BVDOutbreakSize deviations: 42 days),
    in the fit and the forecast alike. With `centred`, the fitted levels
    are sampled rather than the standardised steps. `sd_prior`, when
    given, is the step SD's prior in place of HalfNormal(`sd`).
    """
    sd_rv = DistributionalVariable(
        name, dist.HalfNormal(sd) if sd_prior is None else sd_prior
    )
    frozen = GEN_INT_LENGTH // 7
    half_life = None
    if ar:
        half_life = DistributionalVariable(
            f"{name}_half_life", dist.LogNormal(np.log(6.0), 0.6)
        )
    inner = NonCentredRandomWalk(
        sd_rv,
        frozen=frozen,
        n_fit=frozen + n_weeks,
        half_life_rv=half_life,
        centred=centred,
    )
    return WeeklyTemporalProcess(inner, start_dow=SUNDAY)


def fit_weeks(data: Vintage, config: ModelConfig) -> int:
    """Weeks on the fit axis: the data weeks, extended to the BVD day.

    The BVD pseudo-observation sits on $s_b$, which can fall after the
    cut-off; the fit axis then runs to the end of that week (with no
    data) so the observation informs Rt there.

    Parameters
    ----------
    data : Vintage
        The model inputs at one vintage.
    config : ModelConfig
        The model configuration.

    Returns
    -------
    int
        Number of weeks.
    """
    if config.bvd and data.bvd is not None:
        return max(data.n_weeks, int(data.bvd["day"]) // 7 + 1)
    return data.n_weeks


def capped_concentration(
    name: str, inv_sqrt_rv: DistributionalVariable, floor: float
) -> TransformedVariable:
    r"""Concentration $1 / (x^2 + c)$ from $x = 1 / \sqrt{\text{conc}}$.

    The prior on $x$ shrinks towards the Poisson or multinomial limit, as
    EpiNow2's `obs_opts`, and the floor $c$ caps the concentration at
    $1 / c$: as the concentration grows the float32 log density loses
    precision, which stalls NUTS.

    Parameters
    ----------
    name : str
        Name of the concentration.
    inv_sqrt_rv : DistributionalVariable
        $x$.
    floor : float
        $c$.

    Returns
    -------
    TransformedVariable
        The concentration.
    """
    capped = ComposeTransform(
        [PowerTransform(2.0), AffineTransform(floor, 1.0), PowerTransform(-1.0)]
    )
    return TransformedVariable(name, inv_sqrt_rv, capped)


def _nb_noise(stream: str) -> NegativeBinomialNoise:
    r"""Negative binomial with $1 / \sqrt{\phi} \sim$ HalfNormal(0.25).

    $\phi \le 1000$: the float32 negative binomial log density loses
    precision as $\phi \to \infty$ (second-difference noise 1.5 at
    $x = 3 \times 10^{-4}$ on real data), and at $\phi = 1000$ it is
    Poisson for these counts.
    """
    return NegativeBinomialNoise(
        capped_concentration(
            f"phi_{stream}",
            DistributionalVariable(
                f"inv_sqrt_phi_{stream}", dist.HalfNormal(0.25)
            ),
            PHI_FLOOR,
        )
    )


def _dm_noise() -> DirichletMultinomialNoise:
    r"""Dirichlet-multinomial with $1 / \sqrt{\psi} \sim$ HalfNormal(0.1).

    $\psi \le 10^4$, so that the pooled delays of thousands of cases do
    not make the delay overconfident.
    """
    return DirichletMultinomialNoise(
        capped_concentration(
            "delay_concentration",
            DistributionalVariable(
                "delay_overdispersion", dist.HalfNormal(0.1)
            ),
            1e-4,
        )
    )


def model_args(
    model: MultiSignalModel,
    data: Vintage,
    forecast: bool = False,
    as_of: int | None = None,
) -> dict[str, Any]:
    """Arguments for `model.run` and `Predictive` for one vintage.

    The model axis runs from `n_init` days before `start_date` to the end
    of the forecast horizon. Weekly data are padded with NaN for the
    seeding weeks and the horizon.

    Parameters
    ----------
    model : MultiSignalModel
        The model, from `build_model`.
    data : Vintage
        The model inputs at one vintage.
    forecast : bool, default False
        Extend the axis by the horizon and pass the data as the vintage,
        not as observations, to draw forecasts.
    as_of : int, optional
        With `forecast`, a later cut-off in days from `start_date`, to
        draw counts as reported by it.

    Returns
    -------
    dict
        Keyword arguments for `MultiSignalModel.sample`.
    """
    if data.start_date.weekday() != SUNDAY:
        raise ValueError("start_date must be a Sunday.")
    n_init = model.latent.n_initialization_points
    # The fit runs on the data weeks only; the forecast pass extends the
    # axis by the horizon and draws the new steps from their priors.
    extra = fit_weeks(data, model.config) - data.n_weeks
    horizon = max(data.horizon_weeks, extra) if forecast else extra
    n_weeks = data.n_weeks + horizon

    def pad(x: np.ndarray) -> Array:
        lead = np.full((n_init // 7, *x.shape[1:]), np.nan)
        tail = np.full((horizon, *x.shape[1:]), np.nan)
        return jnp.asarray(np.vstack([lead, x, tail]))

    obs: dict[str, Any] = {
        "cutoff": n_init + data.cutoff,
        "hub_delay_pmf": jnp.asarray(data.hub_delay_pmf),
        "forecast": forecast,
        **({"as_of": n_init + as_of} if forecast and as_of is not None else {}),
    }
    known, imputed = data.known.copy(), data.imputed.copy()
    if data.fit_mask is not None:
        known[:, ~data.fit_mask] = np.nan
        imputed[:, ~data.fit_mask] = np.nan
    if forecast:
        obs["vintage"] = jnp.stack([pad(known), pad(imputed)])
    else:
        obs |= {
            "known": pad(known),
            "imputed": pad(imputed),
            "delay_counts": jnp.nan_to_num(pad(data.delay_counts)),
        }
        if data.release_pairs is not None:
            obs["release_pairs"] = {
                k: jnp.asarray(v) for k, v in data.release_pairs.items()
            }
    pops = np.asarray(data.populations, dtype=float)
    return {
        "n_days_post_init": 7 * n_weeks,
        "population_size": float(pops.sum()),
        "subpop_fractions": jnp.asarray(pops / pops.sum()),
        "obs_start_date": data.start_date,
        "onsets": obs,
    }


EXTRA_FIELDS = (
    "diverging",
    "num_steps",
    "accept_prob",
    "energy",
    "adapt_state.step_size",
)
"""Sampler fields kept for the diagnostics."""


def fit(
    model: MultiSignalModel,
    data: Vintage,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
) -> MultiSignalModel:
    """Fit by NUTS and return the model with its MCMC attached.

    As `MultiSignalModel.run`, but keeping `EXTRA_FIELDS` for the
    diagnostics. Chains run in parallel when devices allow.

    Parameters
    ----------
    model : MultiSignalModel
        The model, from `build_model`.
    data : Vintage
        The model inputs at one vintage.
    num_warmup, num_samples : int, default 500
        Warm-up and kept iterations per chain.
    num_chains : int, default 1
        Number of chains.
    seed : int, default 0
        Random seed.
    target_accept : float, default 0.9
        NUTS target acceptance probability.
    max_tree_depth : int, default 10
        NUTS maximum tree depth.

    Returns
    -------
    MultiSignalModel
        The same model, with `model.mcmc` holding the posterior.
    """
    args = model_args(model, data)
    model.validate_data(
        **{k: v for k, v in args.items() if k != "population_size"}
    )
    kernel = NUTS(
        model.sample,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_sample,
    )
    model.mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    model.mcmc.run(random.key(seed), extra_fields=EXTRA_FIELDS, **args)
    return model


def forecast_trajectories(
    model: MultiSignalModel,
    data: Vintage,
    n_trajectories: int = 100,
    seed: int = 1,
    posterior: dict[str, Any] | None = None,
    as_of: Sequence[int] = (),
) -> dict[str, Any]:
    """Draw joint trajectories of final weekly onsets (known plus imputed).

    Each trajectory uses one posterior draw for every province and week,
    so national counts are the sum over provinces within a trajectory.

    Parameters
    ----------
    model : MultiSignalModel
        A fitted model, from `fit`.
    data : Vintage
        The model inputs it was fitted to.
    n_trajectories : int, default 100
        Number of trajectories.
    seed : int, default 1
        Random seed.
    posterior : dict, optional
        Posterior draws to use instead of `model.mcmc`.
    as_of : sequence of int
        Later cut-offs, in days from `start_date`, to draw counts as
        reported by.

    Returns
    -------
    dict
        `province`: `(n_trajectories, n_weeks + horizon, P)`,
        `national`: their sum, `insample`, the predictive
        draws of the fitted streams at the vintage (see `ppc_draws`),
        `draw`, the index of the posterior draw behind each trajectory,
        and `as_reported`: for each later cut-off in `as_of` (days from
        `start_date`), the counts as reported by it, with the same
        posterior draws, as `{"province", "national"}`. These score
        forecasts against a later vintage as reported, like for like.
    """
    if posterior is None:
        posterior = model.mcmc.get_samples()
    n_draws = next(iter(posterior.values())).shape[0]
    rng = np.random.default_rng(seed)
    keep = rng.choice(n_draws, n_trajectories, replace=n_draws < n_trajectories)
    thinned = {k: v[keep] for k, v in posterior.items()}
    pred = Predictive(model.sample, posterior_samples=thinned)(
        random.key(seed), **model_args(model, data, forecast=True)
    )
    n_init = model.latent.n_initialization_points
    n_lead = n_init // 7
    province = np.asarray(pred["onsets_final_total"])[:, n_lead:]
    sites = [
        s for s in ("known", "imputed", "total") if f"onsets_{s}_obs" in pred
    ]
    insample = {
        s: np.asarray(pred[f"onsets_{s}_obs"])[:, 0, n_lead:] for s in sites
    }
    as_reported = {}
    for cut in as_of:
        later = Predictive(model.sample, posterior_samples=thinned)(
            random.key(seed + cut),
            **model_args(model, data, forecast=True, as_of=cut),
        )
        prov = np.asarray(later["onsets_final_total"])[:, n_lead:]
        as_reported[cut] = {"province": prov, "national": prov.sum(axis=2)}
    return {
        "province": province,
        "national": province.sum(axis=2),
        "insample": insample,
        "draw": keep,
        "as_reported": as_reported,
    }


RT_SITE: Final = "latent::rt_subpop_adjusted"
"""Each place's effective Rt: Rt times its susceptible share."""
INFECTIONS_SITE: Final = "latent_infections_by_subpop"
"""Each place's infections per resident."""


def weekly_rt(
    model: MultiSignalModel,
    data: Vintage,
    posterior: dict[str, Any] | None = None,
    seed: int = 1,
    chunk: int = 200,
) -> dict[str, Any]:
    r"""Draw each place's weekly effective Rt, by week of infection.

    Every posterior draw is used, so quantiles carry the full posterior
    uncertainty. The axis is the forecast axis (`model_args` with
    `forecast`), so the weeks after the data take the forecast Rt. Each
    place's weekly Rt weights its daily effective Rt (Rt times its
    susceptible share) by the infections of each day,

    $$R_{p,w} = \frac{\sum_{t \in w} I_p(t) R_p(t)}{\sum_{t \in w} I_p(t)},$$

    and the weight is $\sum_{t \in w} I_p(t)$, as counts, so groups of
    places pool with `rt_observation.group_rt`.

    Parameters
    ----------
    model : MultiSignalModel
        A fitted model, from `fit`.
    data : Vintage
        The model inputs it was fitted to.
    posterior : dict, optional
        Posterior draws to use instead of `model.mcmc`.
    seed : int, default 1
        Random seed of the first chunk; chunk `k` uses `seed + k`.
    chunk : int, default 200
        Draws taken at once, to bound memory.

    Returns
    -------
    dict
        `rt` and `weight`, `(n_draws, n_weeks, P)` (`rt` is NaN where a
        place has no infections in a week), and `start`, the first day
        of the axis, `n_init` days before `start_date`.
    """
    if posterior is None:
        posterior = model.mcmc.get_samples()
    # The fit's own Rt and infections stop at the data, so they are
    # redrawn on the forecast axis.
    sampled = {
        k: v
        for k, v in posterior.items()
        if k not in (RT_SITE, INFECTIONS_SITE)
    }
    n = next(iter(sampled.values())).shape[0]
    args = model_args(model, data, forecast=True)
    pops = np.asarray(data.populations, dtype=float)
    rts, weights = [], []
    for k, lo in enumerate(range(0, n, chunk)):
        part = {key: v[lo : lo + chunk] for key, v in sampled.items()}
        pred = Predictive(
            model.sample,
            posterior_samples=part,
            return_sites=[RT_SITE, INFECTIONS_SITE],
        )(random.key(seed + k), **args)
        rt = np.asarray(pred[RT_SITE], dtype=float)
        counts = np.asarray(pred[INFECTIONS_SITE], dtype=float) * pops
        m, days, p = rt.shape
        weeks = days // 7
        rt = rt[:, : 7 * weeks].reshape(m, weeks, 7, p)
        counts = counts[:, : 7 * weeks].reshape(m, weeks, 7, p)
        weight = counts.sum(axis=2)
        total = (counts * rt).sum(axis=2)
        rts.append(
            np.divide(
                total, weight, out=np.full_like(total, np.nan), where=weight > 0
            )
        )
        weights.append(weight)
    n_init = model.latent.n_initialization_points
    return {
        "rt": np.concatenate(rts),
        "weight": np.concatenate(weights),
        "start": data.start_date - dt.timedelta(days=n_init),
    }
