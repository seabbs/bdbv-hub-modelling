"""JAX port of R baselinenowcast 0.2.2, with release snapshots.

Reporting triangles are ``(triangle, mask)`` pairs.
``mask`` is True where R has an observed cell and False where R has
NA.
The chain ladder also takes its factors from pairs of consecutive
release snapshots, skipping the cells a release does not observe
(``delays.pair_sums``); on daily releases this is R's chain ladder.

Modules: ``delays`` (the chain ladder), ``triangle`` (structure and
preprocessing), ``aggregate`` (rolling sums and hub weeks),
``uncertainty`` (retrospective errors and negative binomial draws) and
``nowcast`` (the nowcast of a triangle or of release snapshots).
"""

from .aggregate import (
    hub_week_matrix,
    hub_weeks,
    identity,
    rolling_sum,
    row_sums,
)
from .delays import (
    apply_delay,
    estimate_and_apply_delay,
    estimate_delay,
    growth,
    log_factors,
    pair_sums,
    window_pairs,
)
from .nowcast import baselinenowcast, fill_row
from .triangle import (
    allocate_reference_times,
    apply_reporting_structure,
    get_reporting_structure,
    preprocess_negative_values,
    truncate_to_rows,
)
from .uncertainty import (
    combine_obs_with_pred,
    estimate_uncertainty,
    estimate_uncertainty_retro,
    fit_by_horizon,
    fit_nb,
    sample_nb,
    sample_nowcast,
    sample_nowcasts,
    sample_prediction,
)

__all__ = [
    "allocate_reference_times",
    "apply_delay",
    "apply_reporting_structure",
    "baselinenowcast",
    "combine_obs_with_pred",
    "estimate_and_apply_delay",
    "estimate_delay",
    "estimate_uncertainty",
    "estimate_uncertainty_retro",
    "fill_row",
    "fit_by_horizon",
    "fit_nb",
    "get_reporting_structure",
    "growth",
    "hub_week_matrix",
    "hub_weeks",
    "identity",
    "log_factors",
    "pair_sums",
    "preprocess_negative_values",
    "rolling_sum",
    "row_sums",
    "sample_nb",
    "sample_nowcast",
    "sample_nowcasts",
    "sample_prediction",
    "truncate_to_rows",
    "window_pairs",
]
