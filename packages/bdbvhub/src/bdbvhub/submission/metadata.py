# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Hub metadata text for each model that may be submitted.

`bdbvhub metadata --model <model>` renders it with
`bdbvhub.submission.hub.render_model_metadata`.
"""

from __future__ import annotations

__all__ = ["MODEL_METADATA"]

_RENEWAL_RT = (
    " We also submit the wk R_t target. "
    "It is the model's weekly effective reproduction number by week of "
    "infection. "
    "Please note two points. "
    "First, it is estimated from the hub's onsets, including imputed onsets "
    "(https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61), so it "
    "is likely to react more slowly than transmission and to be out of "
    "phase with the outbreak. "
    "Second, the hub collects quantiles, so quantiles joined across weeks "
    "or locations do not describe a trajectory. "
    "For a less affected Rt estimate, see the BVDOutbreakSize summary "
    "(https://epiforecasts.io/BVDOutbreakSize/stable/)."
)
_RENEWAL_REST = (
    "The model includes the incubation period, reporting delays, right "
    "truncation, the share confirmed and the hub's onset imputation. "
    "It is fitted with NUTS in NumPyro to each release. "
    "Provinces and the country are sums of zones. "
    "seabbs-renewal submits the first model to pass its checks, in order: "
    "the zone "
    "model with BVDOutbreakSize Rt, single-zone models, the zone model, a "
    "random walk from a nowcast and a random walk. "
    "See https://samabbott.co.uk/bdbv-hub-modelling/methods/model/."
    + _RENEWAL_RT
)
_ZONE_MODEL = (
    "We fit a renewal model in PyRenew to the health zones with recent "
    "cases. "
    "Log Rt is a national weekly AR(1) with pooled province and zone "
    "deviations. "
    "Infections move between zones through a gravity kernel, and "
    "susceptibles are depleted. "
    "Zones with no recent case only receive imported infections. "
)
_RENEWAL_INPUTS = (
    "Hub weekly cases by onset week (observed plus imputed) and reporting "
    "delays by release. Zone populations and locations. BVDOutbreakSize's "
    "generation interval and incubation period priors."
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
    # The committed packages/bdbvsubmission/model-metadata/seabbs-renewal.yml is
    # that of
    # `[submission] model` in packages/bdbvhub/config/hub.toml: regenerate it
    # with
    #   bdbvhub metadata --model <renewal|renewalsp|renewalbvd>
    # when that changes (bdbvhub's tests/test_fitting.py checks it).
    "renewal": {
        "model_name": "Pooled zone renewal model (PyRenew)",
        "methods": (
            "A pooled renewal model of health zones with importation and "
            "depletion. Provinces and the country are zone sums."
        ),
        "methods_long": _ZONE_MODEL + _RENEWAL_REST,
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalsp": {
        "model_name": "Single-zone renewal models (PyRenew)",
        "methods": (
            "A renewal model fitted to each health zone on its own, with no "
            "pooling or importation. Provinces and the country are zone sums."
        ),
        "methods_long": (
            "We fit a renewal model in PyRenew to each health zone on its "
            "own. "
            "Each has its own weekly AR(1) on log Rt and susceptible "
            "depletion, with no pooling or importation. "
            "A zone with no case in the last nine weeks is zero until an "
            "introduction, then a random walk. " + _RENEWAL_REST
        ),
        "data_inputs": _RENEWAL_INPUTS,
    },
    "renewalbvd": {
        "model_name": (
            "Pooled zone renewal model informed by BVDOutbreakSize Rt (PyRenew)"
        ),
        "methods": (
            "A pooled renewal model of health zones with importation and "
            "depletion, and Rt informed by BVDOutbreakSize. Provinces and the "
            "country are zone sums."
        ),
        "methods_long": (
            _ZONE_MODEL + "Rt in the week of the newest BVDOutbreakSize "
            "release is fitted to that release's estimates, with their SDs "
            "inflated by 1.5. " + _RENEWAL_REST
        ),
        "data_inputs": (
            _RENEWAL_INPUTS + " Rt from the newest BVDOutbreakSize release "
            "before the cut-off."
        ),
    },
    # A hub model of its own (seabbs-rwnowcast), submitted every round.
    "randomwalknowcast": {
        "model_name": "Random walk from a chain-ladder nowcast "
        "(baselinenowcast)",
        "methods": (
            "A chain-ladder nowcast (baselinenowcast) of each province and "
            "zone, carried forward as a log-scale random walk. The country "
            "sums the provinces."
        ),
        "methods_long": (
            "We nowcast each province and health zone with a JAX port of "
            "baselinenowcast (Johnson et al. 2025, "
            "doi:10.12688/wellcomeopenres.25027.2). "
            "The delay is fitted to release pairs up to 42 days apart. "
            "Cases still to be reported are negative binomial, with "
            "dispersion fitted to past nowcasts. "
            "Each draw follows a log-scale random walk with Poisson "
            "noise. "
            "Zones are forecast from their own data. "
            "Places under 10 cases a week use their parent's delay and "
            "spread. "
            "A Gaussian copula pairs the province draws within each week. "
            "See https://samabbott.co.uk/bdbv-hub-modelling/methods/baselines/."
        ),
        "data_inputs": (
            "Hub daily onsets by release (observed plus imputed) for "
            "provinces and health zones."
        ),
    },
    # Fine-tuned Chronos-2, the hub model seabbs-foundation. `{release}`
    # is the checkpoint release the model loads, filled in by `metadata`.
    # The committed packages/bdbvsubmission/model-metadata/seabbs-foundation.yml
    # is that of
    # the model in packages/bdbvhub/config/hub.toml (bdbvhub's
    # tests/test_fitting.py).
    "foundation": {
        "model_name": "Nowcast foundation (fine-tuned Chronos-2)",
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
}
"""Metadata text for each model that may be submitted."""
