# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Hub metadata text for each model that may be submitted.

`bdbvhub metadata --model <model>` renders it with
`bdbvhub.submission.hub.render_model_metadata`.
"""

from __future__ import annotations

__all__ = ["MODEL_METADATA"]

_RENEWAL_RT = (
    " We also provide the wk R_t target. "
    "It is the model's weekly effective reproduction number (Rt times the "
    "susceptible share), indexed to the week of infection, for each health "
    "zone the model fits, each province and the country. "
    "A province or the country weights its zones' Rt by their infections. "
    "The generation interval and incubation period come from "
    "BVDOutbreakSize, with their uncertainty, and the quantiles carry the "
    "full posterior uncertainty. "
    "Please read it with two points in mind. "
    "First, it is estimated from the hub target data (confirmed cases by "
    "symptom onset, including the hub's imputed onsets). "
    "Because of how onsets are imputed "
    "(https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61), it is "
    "likely to react more slowly than transmission and to be out of phase "
    "with the outbreak. "
    "Second, because this target is collected as quantiles, the submission "
    "gives each week and location's distribution on its own. "
    "The joint Rt trajectories behind them, with their dependence across "
    "weeks and locations, cannot be represented, and quantiles joined "
    "across weeks or locations do not describe a trajectory. "
    "For an Rt estimate less affected by these issues, please see the "
    "BVDOutbreakSize summary "
    "(https://epiforecasts.io/BVDOutbreakSize/stable/)."
)
_RENEWAL_OBSERVATION = (
    "Rt is dated by infection. "
    "Onsets are infections convolved with a log-normal incubation period, "
    "with BVDOutbreakSize's priors on its mean and SD. "
    "A time-varying share of onsets is confirmed. "
    "The onset-to-report delay is a discrete hazard, smooth over delay, "
    "with a national weekly shift and province offsets and drifts pooled "
    "towards it. "
    "Known onsets are right truncated by this delay and by a 0 to 7 day "
    "report-to-publication lag learnt from consecutive releases. "
    "The share of reports with no onset date is modelled by province. "
    "The hub's imputed onsets are modelled as the hub makes them: cases "
    "with no onset date, back-sampled from report day with the hub's delay. "
    "Weekly known and imputed counts are negative binomial, and the delays "
    "of known onsets are Dirichlet-multinomial. "
    "Log Rt reverts towards Rt = 1 with a fitted half-life, in the fit "
    "and the forecast weeks alike. "
    "The model is fitted by NUTS (NumPyro) to each data release, and each "
    "sample is one joint posterior trajectory. "
    "See https://samabbott.co.uk/bdbv-hub-modelling/methods/model/."
    + _RENEWAL_RT
)
_ZONE_SUMS = (
    "Health zones are fitted. "
    "Each province is the sum of its zones and national is the sum of the "
    "provinces, sample by sample. "
    "Zone WorldPop populations and the centroids of each zone's largest "
    "polygon set the importation kernel between zones. "
)
_ZONE_SPLIT = (
    "The national value is the sum of the provinces plus the national cases "
    "in no province. "
    "Health zones are not fitted. Each province sample is split across its "
    "zones by shares drawn once per sample from a Dirichlet on recent and "
    "longer-run zone counts, keeping each zone's published count. "
)
_RENEWAL_INPUTS = (
    "Hub target data (weekly confirmed cases by onset week, observed plus "
    "imputed, for the country, provinces and health zones), the hub's "
    "published onset-to-report delays by vintage, province populations and "
    "population-weighted province centres. The generation interval is "
    "BVDOutbreakSize's."
)

MODEL_METADATA = {
    "randomwalk": {
        "model_name": "RandomWalk baseline",
        "methods": (
            "Latest complete onset week carried forward per place, with "
            "log-scale random walk noise and Poisson draws. Health zones "
            "and provinces, with national the sum of provinces."
        ),
        "methods_long": (
            "For each place, the latest onset week with at least 95% of its "
            "final count reported is carried forward. The share reported "
            "comes from a baselinenowcast chain ladder on the daily release "
            "snapshots. Later weeks are drawn on the log scale as a random "
            "walk whose weekly SD is the SD of the last 8 weekly changes of "
            "log(1 + cases), with Poisson noise. A place uses its own delay "
            "factor at an age if that factor rests on at least 10 cases, and "
            "otherwise its "
            "parent's (zone, then province, then national). A place under "
            "10 cases a week uses the walk spread of its parent. Each sample "
            "is a joint trajectory and the national value is the sum of "
            "province samples. See "
            "https://samabbott.co.uk/bdbv-hub-modelling/methods/baselines/."
        ),
        "data_inputs": (
            "Hub daily onsets by release (observed plus imputed) for "
            "provinces and health zones, and the hub's weekly target data."
        ),
    },
    # The PyRenew models, submitted as seabbs-renewal. The gate writes the
    # metadata of the model the chain submits (`metadata --model <variant>`).
    # The committed hub/model-metadata/seabbs-renewal.yml is that of
    # `[submission] model` in config/hub.toml: regenerate it with
    #   bdbvhub metadata --model <renewal|renewalsp|renewalbvd>
    # when that changes (tests/submission/test_fitting.py checks it).
    "renewal": {
        "model_name": "Pooled zone renewal model (PyRenew)",
        "methods": (
            "A renewal model per health zone with pooled Rt, importation "
            "between zones and depletion, fitted to known and imputed onsets "
            "with modelled delays. Provinces are zone sums."
        ),
        "methods_long": (
            "We model infections in each health zone with recent cases with "
            "a renewal process in PyRenew. "
            "Log Rt is a national weekly AR(1) plus partially pooled "
            "province and zone deviations. "
            "Infections move between zones through a gravity kernel, and "
            "susceptibles are depleted. "
            "Zones with no recent case are not fitted: they take infections "
            "only through the fitted importation from the fitted zones. "
            + _ZONE_SUMS
            + _RENEWAL_OBSERVATION
        ),
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalsp": {
        "model_name": "Single-province renewal models (PyRenew)",
        "methods": (
            "A renewal model fitted to each province on its own, with no "
            "pooling or importation, to known and imputed onsets with "
            "modelled delays. Silent provinces wait for an introduction."
        ),
        "methods_long": (
            "We fit a renewal process in PyRenew to each province's series "
            "on its own. "
            "Each has its own weekly AR(1) on log Rt and susceptible "
            "depletion, with no pooling or importation. "
            "A province with no case in the last nine weeks is zero until an "
            "introduction, then a random walk. "
            + _ZONE_SPLIT
            + _RENEWAL_OBSERVATION
        ),
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalbvd": {
        "model_name": (
            "Pooled zone renewal model informed by BVDOutbreakSize Rt (PyRenew)"
        ),
        "methods": (
            "A renewal model per health zone with pooled Rt, importation and "
            "depletion, with Rt tied to the newest BVDOutbreakSize estimates "
            "with their SDs inflated. Provinces are zone sums."
        ),
        "methods_long": (
            "We model infections in each health zone with recent cases with "
            "a renewal process in PyRenew. "
            "Log Rt is a national weekly AR(1) plus partially pooled "
            "province and zone deviations. "
            "Infections move between zones through a gravity kernel, and "
            "susceptibles are depleted. "
            "Zones with no recent case are not fitted: they take infections "
            "only through the fitted importation from the fitted zones. "
            "The model's national, province and zone Rt in the week of the "
            "newest BVDOutbreakSize release published by the cut-off are tied "
            "to that release's Rt estimates by a likelihood term with their "
            "SDs inflated by 1.5. " + _ZONE_SUMS + _RENEWAL_OBSERVATION
        ),
        "data_inputs": (
            _RENEWAL_INPUTS + " The newest BVDOutbreakSize release published "
            "by the forecast's cut-off, for its Rt estimates."
        ),
    },
    # A hub model of its own (seabbs-rwnowcast), submitted every round.
    "randomwalknowcast": {
        "model_name": "Random walk from a chain-ladder nowcast "
        "(baselinenowcast)",
        "methods": (
            "A chain-ladder nowcast of each place's recent weeks from the "
            "daily release snapshots (baselinenowcast), carried forward as a "
            "log-scale random walk. National is the province sum."
        ),
        "methods_long": (
            "We nowcast and forecast weekly confirmed cases by onset week "
            "for each health zone and province. "
            "The nowcast is the baselinenowcast method (Johnson et al. 2025, "
            "Wellcome Open Research, doi:10.12688/wellcomeopenres.25027.2), "
            "from a JAX port of baselinenowcast 0.2.2. "
            "A chain ladder on the hub's daily release snapshots fits one "
            "daily reporting delay to every release pair in a 42-day window, "
            "by non-negative least squares on the log scale. "
            "Report days no release observes are skipped, and downward "
            "revisions are moved onto earlier releases. "
            "Daily nowcasts are summed to hub weeks. "
            "The cases still to be reported are negative binomial, with a "
            "dispersion fitted per horizon from retrospective nowcasts. "
            "Each nowcast draw is carried forward from its latest week as a "
            "random walk on the log scale with Poisson noise. "
            "A place with fewer than 10 cases a week uses the delay, "
            "dispersion and walk spread pooled over its parent (zone, then "
            "province, then country). "
            "Within each week the provinces' draws are paired by a Gaussian "
            "copula whose correlation is estimated from the model's own past "
            "errors (Schaake shuffle), so each province keeps its draws. "
            "The national value is the sum of the provinces. "
            "See https://samabbott.co.uk/bdbv-hub-modelling/methods/baselines/."
        ),
        "data_inputs": (
            "Hub daily onsets by release (observed plus imputed) for "
            "provinces and health zones, and the hub's weekly target data."
        ),
    },
    # Fine-tuned Chronos-2, the hub model seabbs-nowcastfm. `{release}`
    # is the checkpoint release the model loads, filled in by `metadata`.
    # The committed hub/model-metadata/seabbs-nowcastfm.yml is that of
    # the model in config/hub.toml (tests/submission/test_fitting.py).
    "fm_chronos_daily_ragged_ft": {
        "model_name": "Nowcast foundation (fine-tuned Chronos-2, daily "
        "releases)",
        "methods": (
            "A time series foundation model (Chronos-2) fine-tuned on "
            "simulated outbreaks, forecasting the country, provinces and "
            "health zones jointly from the hub's daily releases."
        ),
        "methods_long": (
            "We forecast the hub target with Chronos-2 (Ansari et al. 2025, "
            "arXiv:2510.15821) and a LoRA adapter fine-tuned on outbreaks "
            "simulated with epinowcast's generative model (renewal process, "
            "drifting reporting delays, day-of-week effects, negative "
            "binomial noise). "
            "The adapter is release {release} of "
            "https://github.com/seabbs/bdbv-hub-modelling, the newest when "
            "this was written. "
            "Each daily release is laid out by onset day and release lag. "
            "The country, each province and each health zone are target "
            "series in one block, forecast jointly. "
            "Each place with cases has its counts at release lags under "
            "28 days as covariates. "
            "Each of 100 samples draws its own uniform level per series and "
            "day from Chronos-2's 21 quantiles, rounded to counts. "
            "The model is not fitted to these data. "
            "The country, provinces and health zones are each the model's "
            "own forecast of that series: nothing is summed or split. "
            "See https://samabbott.co.uk/bdbv-hub-modelling/methods/foundation/."
        ),
        "data_inputs": (
            "The hub's daily onsets by release (observed plus imputed) "
            "for the country, provinces and health zones."
        ),
    },
    "fm_chronos_weekly_ragged_ft": {
        "model_name": "Nowcast foundation (fine-tuned Chronos-2, weekly "
        "releases)",
        "methods": (
            "A time series foundation model (Chronos-2) fine-tuned on "
            "simulated outbreaks, forecasting the country, provinces and "
            "health zones jointly from the hub's weekly releases."
        ),
        "methods_long": (
            "We forecast the hub target with Chronos-2 (Ansari et al. 2025, "
            "arXiv:2510.15821) and a LoRA adapter fine-tuned on outbreaks "
            "simulated with epinowcast's generative model (renewal process, "
            "drifting reporting delays, day-of-week effects, negative "
            "binomial noise). "
            "The adapter is release {release} of "
            "https://github.com/seabbs/bdbv-hub-modelling, the newest when "
            "this was written. "
            "Each weekly release is laid out by onset week and release lag. "
            "The country, each province and each health zone are target "
            "series in one block, forecast jointly. "
            "Each place with cases has its counts at release lags under "
            "4 weeks as covariates. "
            "Each of 100 samples draws its own uniform level per series and "
            "week from Chronos-2's 21 quantiles, rounded to counts. "
            "The model is not fitted to these data. "
            "The country, provinces and health zones are each the model's "
            "own forecast of that series: nothing is summed or split. "
            "See https://samabbott.co.uk/bdbv-hub-modelling/methods/foundation/."
        ),
        "data_inputs": (
            "The hub's weekly target data by release "
            "for the country, provinces and health zones."
        ),
    },
}
"""Metadata text for each model that may be submitted."""
