# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The province renewal model, built on PyRenew (JAX, NumPyro).

- `pooled`: pooled group values and sum-to-zero group deviations.
- `renewal`: the coupled renewal with importation, seeding and
  depletion.
- `hazard_delay`: a delay pmf that varies with the reference day.
- `delays`: a fixed time-varying delay.
- `observation`: the known and imputed onset streams, the missing-onset
  fraction, the hub's imputation and importation-only zeros.
- `composition_noise`: multinomial and Dirichlet-multinomial noise for
  counts by category given totals (the delay counts).
- `distributions`: a Dirichlet-multinomial scored on its non-zero cells.
- `rt_observation`: external Rt estimates as a pseudo-observation with
  inflated SDs (the BVDOutbreakSize term).
- `model`: `ModelConfig`, the configurations, fitting and forecasting.
- `zones`: the model builder, with health zones nested in provinces or
  each place its own patch (`build_model`).
- `inputs`: model inputs (`Vintage`) from the tidy hub tables.

The package imports without JAX until this subpackage is imported.
"""

from ..components.walks import NonCentredRandomWalk
from .delays import (
    FixedTimeVarying,
)
from .hazard_delay import HazardDelay
from .model import (
    BVD,
    MAIN,
    ModelConfig,
    Vintage,
    fit,
    forecast_trajectories,
    generation_interval,
    incubation_period,
    model_args,
    weekly_rt,
)
from .observation import (
    MissingOnsetFraction,
    OnsetStreams,
    onset_expectations,
)
from .pooled import PooledInitialValue, pooled_walk
from .renewal import MetapopulationInfections
from .rt_observation import RtPseudoObservation
from .zones import build_model

__all__ = [
    "BVD",
    "MAIN",
    "FixedTimeVarying",
    "HazardDelay",
    "MetapopulationInfections",
    "MissingOnsetFraction",
    "ModelConfig",
    "NonCentredRandomWalk",
    "OnsetStreams",
    "PooledInitialValue",
    "RtPseudoObservation",
    "Vintage",
    "build_model",
    "fit",
    "forecast_trajectories",
    "generation_interval",
    "incubation_period",
    "model_args",
    "onset_expectations",
    "pooled_walk",
    "weekly_rt",
]
