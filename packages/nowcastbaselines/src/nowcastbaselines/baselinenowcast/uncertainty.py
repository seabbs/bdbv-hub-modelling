"""Uncertainty: retrospective errors and their negative binomial fit.

As baselinenowcast's: point nowcasts made from earlier reference times
against what was later reported (`estimate_uncertainty_retro`), a negative
binomial size by horizon (`fit_by_horizon`, `fit_nb`), and draws of the
counts still to be reported (`sample_nb`, `sample_nowcasts`).
The release-pair nowcast fits the same sizes, pooled to parents
(`fit_sizes`, `sample_count`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.scipy.special import gammaln, xlogy

from ._checks import assert_integerish, coerce, validate_reporting_triangle
from .aggregate import identity, row_sums
from .delays import estimate_and_apply_delay
from .triangle import (
    Triangle,
    apply_reporting_structure,
    get_reporting_structure,
    truncate_to_rows,
)

__all__ = [
    "MAX_SIZE",
    "PRIOR_SIZE",
    "DelayAggregator",
    "FitModel",
    "RefTimeAggregator",
    "Sampler",
    "combine_obs_with_pred",
    "estimate_uncertainty",
    "estimate_uncertainty_retro",
    "fit_by_horizon",
    "fit_nb",
    "fit_sizes",
    "sample_count",
    "sample_nb",
    "sample_nowcast",
    "sample_nowcasts",
    "sample_prediction",
]


Sampler = Callable[[Array, Array, Array], Array]
"""Draws counts: `sampler(key, mean, uncertainty_params)`."""


def sample_nb(key: Array, pred: Any, uncertainty_params: Any) -> Array:
    """Negative binomial draws with mean ``pred`` and size ``params``.

    R sample_distrib.R:18 (``rnbinom(size, mu)``), drawn as a gamma
    mixed Poisson.

    Parameters
    ----------
    key : jax.Array
        Random key.
    pred : array_like
        Means.
    uncertainty_params : array_like
        Sizes, of length 1 or matching ``pred``.

    Returns
    -------
    jax.Array
        One draw per mean.
    """
    mu = jnp.asarray(pred, dtype=jnp.result_type(pred, jnp.float32))
    size = jnp.asarray(uncertainty_params, dtype=mu.dtype)
    if size.ndim > 0 and size.size not in (1, mu.size):
        raise ValueError(
            "`uncertainty_params` must have length 1 or match length(pred)."
        )
    size = size.reshape(()) if size.size == 1 else size.reshape(mu.shape)
    k_gamma, k_pois = jax.random.split(key)
    lam = jax.random.gamma(k_gamma, size, mu.shape, dtype=mu.dtype) * mu / size
    draw = jax.random.poisson(k_pois, lam).astype(mu.dtype)
    # R's rnbinom gives NaN for a NaN mean or size, not a count.
    ok = jnp.isfinite(mu) & jnp.isfinite(size)
    return jnp.where(ok, draw, jnp.nan)


class _Prepared:
    """Deterministic parts of a nowcast draw."""

    def __init__(
        self,
        mean_pred: Array,
        params_use: Array,
        obs_counts: Array,
        n_agg: int,
    ) -> None:
        self.mean_pred = mean_pred
        self.params_use = params_use
        self.obs_counts = obs_counts
        self.n_agg = n_agg


def _prepare(
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    ref_time_aggregator: RefTimeAggregator,
    delay_aggregator: DelayAggregator,
    validate: bool = True,
) -> _Prepared:
    """Aggregate and split the nowcast (R sample_prediction, :55)."""
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    nc = jnp.asarray(point_nowcast_matrix)
    params = jnp.atleast_1d(jnp.asarray(uncertainty_params, dtype=nc.dtype))
    if params.shape[0] > nc.shape[0]:
        raise ValueError(
            "Vector of uncertainty parameters is greater than the number of "
            "reference times in `point_nowcast_matrix`."
        )
    if nc.shape != values.shape:
        raise ValueError(
            "`point_nowcast_matrix` and `reporting_triangle` must have the "
            "same shape."
        )
    # .extract_predictions (sample_predictions.R:138)
    # The check needs concrete values, so it is skipped under tracing.
    traced = isinstance(nc, jax.core.Tracer) or isinstance(
        values, jax.core.Tracer
    )
    if not traced and not bool(
        jnp.all(jnp.where(jnp.asarray(m), nc == values, True))
    ):
        raise ValueError(
            "`reporting_triangle` is not a subset of `point_nowcast_matrix`."
        )
    agg_nc, _ = ref_time_aggregator(nc, np.ones(nc.shape, dtype=bool))
    agg_rt, agg_m = ref_time_aggregator(values, m)
    n_agg = agg_rt.shape[0]
    n_horizons = int(np.sum(~agg_m.all(axis=1)))
    if params.shape[0] < n_horizons:
        raise ValueError(
            "Vector of uncertainty parameters is less than the number of "
            "horizons in the `reporting_triangle`."
        )
    pred_cells = jnp.where(jnp.asarray(~agg_m), agg_nc, 0.0)
    mean_pred_all = delay_aggregator(pred_cells)
    if mean_pred_all.shape != (n_agg,):
        raise ValueError("`delay_aggregator` must return a vector.")
    obs_counts = delay_aggregator(agg_rt)
    if n_horizons == 0:
        return _Prepared(mean_pred_all[:0], params[:0], obs_counts, n_agg)
    # tail(params, n_horizons), reversed (sample_predictions.R:106)
    params_use = params[-n_horizons:][::-1]
    return _Prepared(
        mean_pred_all[n_agg - n_horizons :], params_use, obs_counts, n_agg
    )


def _draw(key: Array, prep: _Prepared, sampler: Sampler) -> Array:
    """One padded prediction draw, zero for fully observed rows."""
    n_h = prep.mean_pred.shape[0]
    if n_h == 0:
        return jnp.zeros((prep.n_agg,), prep.obs_counts.dtype)
    drawn = sampler(key, prep.mean_pred, prep.params_use)
    return jnp.concatenate([jnp.zeros((prep.n_agg - n_h,), drawn.dtype), drawn])


def sample_prediction(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """One draw of the counts still to be reported (R sample_predictions.R:55).

    The last ``n`` of the uncertainty parameters are used, as R's
    ``tail``, most recent horizon first.

    Parameters
    ----------
    key : jax.Array
        Random key.
    point_nowcast_matrix : array_like
        The filled triangle, as from `apply_delay`.
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    uncertainty_params : array_like
        One parameter per horizon.
    uncertainty_sampler : Sampler, default sample_nb
        Draws counts given a mean and the parameters.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.

    Returns
    -------
    jax.Array
        One value per aggregated reference time, zero for rows that are
        fully observed.
    """
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    return _draw(key, prep, uncertainty_sampler)


def combine_obs_with_pred(
    predicted_counts: Any,
    triangle: Any,
    mask: Any,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """Observed totals plus predicted counts (R sample_predictions.R:189).

    Parameters
    ----------
    predicted_counts : array_like
        Counts still to be reported, as from `sample_prediction`.
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.

    Returns
    -------
    jax.Array
        One total per aggregated reference time.
    """
    values, m = coerce(triangle, mask)
    agg, _ = ref_time_aggregator(values, m)
    return delay_aggregator(agg) + jnp.asarray(predicted_counts)


def sample_nowcast(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """One nowcast draw over aggregated reference times (R :312).

    Parameters
    ----------
    key : jax.Array
        Random key.
    point_nowcast_matrix : array_like
        The filled triangle, as from `apply_delay`.
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    uncertainty_params : array_like
        One parameter per horizon.
    uncertainty_sampler : Sampler, default sample_nb
        Draws counts given a mean and the parameters.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.

    Returns
    -------
    jax.Array
        One total per aggregated reference time.
    """
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    return prep.obs_counts + _draw(key, prep, uncertainty_sampler)


def sample_nowcasts(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    draws: int = 1000,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """Draws of the nowcast (R :372).

    Draws are ``vmap``ped over split keys.
    With a rolling aggregator the leading rows R pads with NA are
    dropped, so the columns are the last ``n_aggregated`` reference
    times.

    Parameters
    ----------
    key : jax.Array
        Random key.
    point_nowcast_matrix : array_like
        The filled triangle, as from `apply_delay`.
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    uncertainty_params : array_like
        One parameter per horizon.
    draws : int, default 1000
        Number of draws.
    uncertainty_sampler : Sampler, default sample_nb
        Draws counts given a mean and the parameters.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.

    Returns
    -------
    jax.Array
        Shape ``(draws, n_aggregated)``.
    """
    if draws < 1:
        raise ValueError("`draws` must be at least 1")
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    keys = jax.random.split(key, draws)
    drawn = jax.vmap(lambda k: _draw(k, prep, uncertainty_sampler))(keys)
    return prep.obs_counts[None, :] + drawn


RefTimeAggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]


DelayAggregator = Callable[[Array], Array]


FitModel = Callable[[Any, Any], Array]


def _nb_logpmf(x: Array, mu: Array, size: Array) -> Array:
    """Log pmf of the negative binomial with mean ``mu`` (R dnbinom)."""
    return (
        gammaln(x + size)
        - gammaln(size)
        - gammaln(x + 1)
        - size * jnp.log1p(mu / size)
        + xlogy(x, mu / (size + mu))
    )


def _brent(
    f: Callable[[Array], Array], lo: float, hi: float, dtype: Any
) -> Array:
    """Brent's minimiser as R's ``optimize`` (R optimize.c, Brent_fmin).

    Non-finite function values are replaced by the largest finite
    number, as R's wrapper does.
    """
    eps = jnp.finfo(dtype).eps
    tol = eps**0.25
    c = (3.0 - jnp.sqrt(jnp.asarray(5.0, dtype))) * 0.5
    sqrt_eps = jnp.sqrt(eps)
    big = jnp.finfo(dtype).max

    def ff(x: Array) -> Array:
        v = f(x)
        return jnp.where(jnp.isfinite(v), v, big)

    a = jnp.asarray(lo, dtype)
    b = jnp.asarray(hi, dtype)
    v0 = a + c * (b - a)
    fx0 = ff(v0)
    zero = jnp.zeros((), dtype)
    tol3 = tol / 3.0
    # state: a b d e v w x fv fw fx
    init = (a, b, zero, zero, v0, v0, v0, fx0, fx0, fx0)

    def stop(s: tuple[Array, ...]) -> Array:
        a, b, _, _, _, _, x, *_ = s
        xm = (a + b) * 0.5
        tol1 = sqrt_eps * jnp.abs(x) + tol3
        t2 = tol1 * 2.0
        return jnp.abs(x - xm) <= t2 - (b - a) * 0.5

    def body(s: tuple[Array, ...]) -> tuple[Array, ...]:
        a, b, d, e, v, w, x, fv, fw, fx = s
        xm = (a + b) * 0.5
        tol1 = sqrt_eps * jnp.abs(x) + tol3
        t2 = tol1 * 2.0
        fit = jnp.abs(e) > tol1
        r = jnp.where(fit, (x - w) * (fx - fv), zero)
        q = jnp.where(fit, (x - v) * (fx - fw), zero)
        p = jnp.where(fit, (x - v) * q - (x - w) * r, zero)
        q = jnp.where(fit, (q - r) * 2.0, zero)
        p = jnp.where(fit & (q > 0), -p, p)
        q = jnp.where(fit, jnp.abs(q), zero)
        r_old = jnp.where(fit, e, zero)
        e_prev = jnp.where(fit, d, e)
        golden = (
            (jnp.abs(p) >= jnp.abs(q * 0.5 * r_old))
            | (p <= q * (a - x))
            | (p >= q * (b - x))
        )
        e_gold = jnp.where(x < xm, b - x, a - x)
        d_gold = c * e_gold
        safe_q = jnp.where(q == 0, 1.0, q)
        d_par = p / safe_q
        u_par = x + d_par
        close = (u_par - a < t2) | (b - u_par < t2)
        d_par = jnp.where(close, jnp.where(x >= xm, -tol1, tol1), d_par)
        d_new = jnp.where(golden, d_gold, d_par)
        e_new = jnp.where(golden, e_gold, e_prev)
        u = jnp.where(
            jnp.abs(d_new) >= tol1,
            x + d_new,
            jnp.where(d_new > 0, x + tol1, x - tol1),
        )
        fu = ff(u)
        better = fu <= fx
        a1 = jnp.where(better, jnp.where(u < x, a, x), jnp.where(u < x, u, a))
        b1 = jnp.where(better, jnp.where(u < x, x, b), jnp.where(u < x, b, u))
        shift_w = (~better) & ((fu <= fw) | (w == x))
        shift_v = (~better) & ~shift_w & ((fu <= fv) | (v == x) | (v == w))
        v1 = jnp.where(
            better, w, jnp.where(shift_w, w, jnp.where(shift_v, u, v))
        )
        fv1 = jnp.where(
            better, fw, jnp.where(shift_w, fw, jnp.where(shift_v, fu, fv))
        )
        w1 = jnp.where(better, x, jnp.where(shift_w, u, w))
        fw1 = jnp.where(better, fx, jnp.where(shift_w, fu, fw))
        x1 = jnp.where(better, u, x)
        fx1 = jnp.where(better, fu, fx)
        return (a1, b1, d_new, e_new, v1, w1, x1, fv1, fw1, fx1)

    final = lax.while_loop(lambda s: ~stop(s), body, init)  # type: ignore[arg-type]
    return final[6]


@jax.jit
def _fit_nb_kernel(x: Array, mu: Array) -> Array:
    def nll(size: Array) -> Array:
        return -jnp.sum(_nb_logpmf(x, mu, size))

    return _brent(nll, 0.1, 1000.0, x.dtype)


def fit_nb(x: Any, mu: Any) -> Array:
    """Fit the negative binomial size by maximum likelihood.

    R estimate_uncertainty.R:445.
    Minimises the negative log likelihood on [0.1, 1000] with Brent's
    method, as ``optimize`` does.

    Parameters
    ----------
    x : array_like
        Observed counts.
    mu : array_like
        Their predicted means.

    Returns
    -------
    jax.Array
        The size, or NaN for empty input.
    """
    xa = jnp.atleast_1d(jnp.asarray(x))
    mua = jnp.atleast_1d(jnp.asarray(mu, dtype=xa.dtype))
    if xa.size == 0:
        return jnp.asarray(jnp.nan)
    if bool(jnp.any(xa < 0)):
        raise ValueError(
            "Negative values detected in observations for uncertainty "
            "estimation. fit_nb() requires non-negative integer "
            "observations."
        )
    if bool(jnp.any(mua < 0)):
        raise ValueError(
            "Negative values detected in predictions for uncertainty "
            "estimation. fit_nb() requires non-negative predictions."
        )
    assert_integerish(np.asarray(xa), "x")
    return _fit_nb_kernel(xa, mua)


def _check_obs_and_pred(obs: Any, pred: Any) -> tuple[Array, Array]:
    """R ``.check_obs_and_pred`` (estimate_uncertainty.R:409)."""
    if obs is None or pred is None:
        raise ValueError("Missing `obs` and/or `pred`")
    o = jnp.atleast_2d(jnp.asarray(obs))
    p = jnp.atleast_2d(jnp.asarray(pred))
    if o.shape != p.shape:
        raise ValueError("`obs` and `pred` must have the same dimensions")
    return o, p


def fit_by_horizon(obs: Any, pred: Any, fit_model: FitModel = fit_nb) -> Array:
    """Fit one uncertainty parameter per horizon (column).

    R estimate_uncertainty.R:291.

    Parameters
    ----------
    obs : array_like
        Observed counts, retrospective nowcasts by horizons.
    pred : array_like
        Their predictions, the same shape.
    fit_model : FitModel, default fit_nb
        Fits one column.

    Returns
    -------
    jax.Array
        One parameter per horizon.
    """
    o, p = _check_obs_and_pred(obs, pred)
    return jnp.stack(
        [jnp.asarray(fit_model(o[:, i], p[:, i])) for i in range(o.shape[1])]
    )


@partial(jax.jit, static_argnames=("delay_aggregator",))
def _horizon_sums(
    nowcast: Array,
    observed: Array,
    unobserved_retro: Array,
    observed_final: Array,
    delay_aggregator: DelayAggregator,
) -> tuple[Array, Array]:
    """Return predicted and observed totals per horizon (R :230)."""
    sel = unobserved_retro & observed_final
    pred = delay_aggregator(jnp.where(sel, nowcast, 0.0))
    obs = delay_aggregator(jnp.where(sel, observed, 0.0))
    return pred, obs


def _check_list_length(
    obj: Sequence[Any], name: str, required: int, empty_check: bool = True
) -> None:
    """R ``.check_list_length`` (estimate_uncertainty.R:383)."""
    if len(obj) < required:
        raise ValueError(
            f"Insufficient elements in `{name}` for the `n` desired."
        )
    if empty_check and len(obj) < 1:
        raise ValueError(f"`{name}` is an empty list")


def _count_retro_times(
    obs_list: Sequence[Triangle],
    n_possible_horizons: int,
    ref_time_aggregator: RefTimeAggregator,
) -> int:
    """R ``.calc_n_retro_nowcast_times`` (estimate_uncertainty.R:320)."""
    if len(obs_list) == 0:
        return 0
    first_values, first_mask = obs_list[0]
    nrow_orig = first_mask.shape[0]
    nrow_agg = ref_time_aggregator(first_values, first_mask)[1].shape[0]
    required = n_possible_horizons + (nrow_orig - nrow_agg)
    return sum(1 for _, m in obs_list if m.shape[0] >= required)


def estimate_uncertainty(
    point_nowcast_matrices: Sequence[Any],
    truncated_reporting_triangles: Sequence[Triangle],
    retro_reporting_triangles: Sequence[Triangle],
    n: int | None = None,
    uncertainty_model: Callable[..., Array] = fit_by_horizon,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
    validate: bool = True,
) -> Array:
    """Estimate uncertainty parameters per horizon.

    R estimate_uncertainty.R:71.

    Parameters
    ----------
    point_nowcast_matrices : sequence of array_like or None
        Retrospective point nowcasts, ``None`` for failed ones.
    truncated_reporting_triangles : sequence of Triangle
        The ``(triangle, mask)`` pairs those nowcasts were made from.
    retro_reporting_triangles : sequence of Triangle
        The same rows as later observed.
    n : int, optional
        Nowcasts to use. Defaults to all.
    uncertainty_model : callable, default fit_by_horizon
        Fits the parameters from observed and predicted counts.
    ref_time_aggregator : RefTimeAggregator, default identity
        ``ref_time_aggregator(values, mask)`` returns aggregated values
        and mask, where a cell is unobserved if any input cell is.
    delay_aggregator : DelayAggregator, default row_sums
        ``delay_aggregator(values)`` sums over delays, with unobserved
        cells already zero.
    validate : bool, default True
        Check the inputs first.

    Returns
    -------
    jax.Array
        One parameter per horizon.
    """
    if n is None:
        n = len(point_nowcast_matrices)
    assert_integerish(n, "n", lower=0)
    n = int(n)
    _check_list_length(point_nowcast_matrices, "point_nowcast_matrices", n)
    _check_list_length(
        truncated_reporting_triangles, "truncated_reporting_triangles", n
    )
    _check_list_length(
        retro_reporting_triangles,
        "retro_reporting_triangles",
        n,
        empty_check=False,
    )
    keep = [i for i in range(n) if point_nowcast_matrices[i] is not None]
    n_iters = len(keep)
    ncs = [jnp.asarray(point_nowcast_matrices[i]) for i in keep]
    obs_list = [coerce(*truncated_reporting_triangles[i]) for i in keep]
    rt_list = [coerce(*retro_reporting_triangles[i]) for i in keep]
    obs_list = [(v, m) for v, m in obs_list]
    if validate:
        for _, m in obs_list:
            validate_reporting_triangle(m)
        for _, m in rt_list:
            validate_reporting_triangle(m)
    if n_iters == 0:
        raise ValueError(
            "No valid retrospective nowcasts were found, therefore "
            "uncertainty can not be estimated using the reporting "
            "triangle passed in."
        )
    if any(bool(jnp.any(jnp.isnan(x))) for x in ncs):
        raise ValueError("`point_nowcast_matrices` contains NAs")
    if all(m.all() for _, m in obs_list):
        warnings.warn(
            "`truncated_reporting_triangles` does not contain any NAs",
            stacklevel=2,
        )
    if any(x.shape != v.shape for x, (v, _) in zip(ncs, obs_list, strict=True)):
        raise ValueError(
            "Dimensions of the first `n` matrices in `point_nowcast_matrices` "
            "and `truncated_reporting_triangles` are not the same."
        )

    # Count horizons on the aggregated retrospective triangle, which
    # equals R's daily count for rolling sums.
    n_possible_horizons = int(
        np.sum(~np.asarray(ref_time_aggregator(*rt_list[0])[1]).all(axis=1))
    )
    n_iters = _count_retro_times(
        obs_list, n_possible_horizons, ref_time_aggregator
    )
    if n_iters == 0:
        raise ValueError(
            "No valid retrospective nowcast times after reference time "
            "aggregation."
        )
    if n_iters < n:
        warnings.warn(
            f"Only the first {n_iters} retrospective nowcast times were used.",
            stacklevel=2,
        )
    agg_first = ref_time_aggregator(*obs_list[0])
    if agg_first[0].shape[1] != obs_list[0][0].shape[1]:
        raise ValueError(
            "`ref_time_aggregator` must return a matrix with the same "
            "number of columns as the input observations"
        )

    exp_to_add = np.zeros((n_iters, n_possible_horizons))
    already_observed = np.zeros((n_iters, n_possible_horizons))
    h = n_possible_horizons
    for i in range(n_iters):
        nowcast_i = ncs[i]
        obs_v, obs_m = obs_list[i]
        rt_v, rt_m = rt_list[i]
        agg_obs, agg_obs_m = ref_time_aggregator(obs_v, obs_m)
        agg_nc, _ = ref_time_aggregator(
            nowcast_i, np.ones(nowcast_i.shape, dtype=bool)
        )
        _, agg_rt_m = ref_time_aggregator(rt_v, rt_m)
        if h > agg_obs.shape[0]:
            raise ValueError(
                f"n_possible_horizons ({h}) exceeds matrix rows "
                f"({agg_obs.shape[0]})"
            )
        pred_h, obs_h = _horizon_sums(
            agg_nc[-h:],
            agg_obs[-h:],
            jnp.asarray(~agg_rt_m[-h:]),
            jnp.asarray(agg_obs_m[-h:]),
            delay_aggregator,
        )
        if pred_h.shape != (h,) or obs_h.shape != (h,):
            raise ValueError(
                "`delay_aggregator` must return a vector of length "
                f"{h}, got {pred_h.shape}"
            )
        exp_to_add[i] = np.asarray(pred_h)[::-1]
        already_observed[i] = np.asarray(obs_h)[::-1]

    obs = jnp.asarray(already_observed, dtype=ncs[0].dtype)
    pred = jnp.asarray(exp_to_add, dtype=ncs[0].dtype)
    _check_obs_and_pred(obs, pred)
    return jnp.asarray(uncertainty_model(obs, pred))


def _validate_inputs_uncertainty(
    n_ref_times: int,
    n_min_delay: int,
    n_history_delay: int,
    n_retrospective_nowcasts: int,
    n_min_retro_nowcasts: int = 2,
) -> None:
    """R ``.validate_inputs_uncertainty`` (estimate_uncertainty_retro.R:124)."""
    for name, val in (
        ("n_ref_times", n_ref_times),
        ("n_min_delay", n_min_delay),
        ("n_history_delay", n_history_delay),
        ("n_retrospective_nowcasts", n_retrospective_nowcasts),
    ):
        assert_integerish(val, name, lower=0)
    if n_ref_times < n_history_delay + n_retrospective_nowcasts:
        raise ValueError(
            "Insufficient reference times in reporting triangle for "
            "specified `n_history_delay` and `n_retrospective_nowcasts`."
        )
    if n_history_delay < n_min_delay:
        raise ValueError("Insufficient `n_history_delay`.")
    if n_retrospective_nowcasts < n_min_retro_nowcasts:
        raise ValueError("Insufficient `n_retrospective_nowcasts`.")


def estimate_uncertainty_retro(
    triangle: Any,
    mask: Any,
    n_history_delay: int,
    n_retrospective_nowcasts: int,
    structure: Any = None,
    delay_pmf: Any = None,
    validate: bool = True,
    **kwargs: Any,
) -> Array:
    """Estimate uncertainty from retrospective nowcasts.

    R estimate_uncertainty_retro.R:51.

    Truncates the last ``n_retrospective_nowcasts`` rows one at a time,
    applies the reporting structure, nowcasts each and compares with
    what was later observed.
    A retrospective nowcast that raises ``ValueError`` is skipped
    (R's ``.safelydoesit``).

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    n_history_delay : int
        Rows used to estimate each delay.
    n_retrospective_nowcasts : int
        Retrospective nowcasts to make.
    structure : int or array_like, optional
        Reporting structure. Defaults to `get_reporting_structure`.
    delay_pmf : array_like, optional
        A delay to use instead of estimating one for each nowcast.
    validate : bool, default True
        Check the triangle and mask first.
    **kwargs
        Passed to `estimate_uncertainty`.

    Returns
    -------
    jax.Array
        One parameter per horizon.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if structure is None:
        structure = get_reporting_structure(m)
    n_ref = m.shape[0]
    min_ref_times_delay = int(np.sum(~m.all(axis=1))) + 1
    _validate_inputs_uncertainty(
        n_ref, min_ref_times_delay, n_history_delay, n_retrospective_nowcasts
    )
    truncated = truncate_to_rows(
        values, m, n=n_retrospective_nowcasts, validate=False
    )
    retro: list[Triangle] = []
    for v, tm in truncated:
        new_mask = apply_reporting_structure(v, tm, structure, validate=False)
        retro.append((jnp.where(jnp.asarray(new_mask), v, 0.0), new_mask))
    pmfs = (
        list(delay_pmf)
        if isinstance(delay_pmf, list)
        else [delay_pmf] * len(retro)
    )
    if len(pmfs) != len(retro):
        raise ValueError(
            "List of `delay_pmf` is not the same length as the retrospective "
            "reporting triangles."
        )
    nowcasts: list[Array | None] = []
    for (v, rm), pmf in zip(retro, pmfs, strict=True):
        try:
            nowcasts.append(
                estimate_and_apply_delay(
                    v, rm, n=n_history_delay, delay_pmf=pmf, validate=False
                )
            )
        except ValueError as err:
            warnings.warn(
                "Point nowcast matrix could not be generated for a "
                f"retrospective nowcast: {err}",
                stacklevel=2,
            )
            nowcasts.append(None)
    if all(x is None for x in nowcasts):
        raise ValueError(
            f"Errors occurred in all {len(retro)} reporting triangles. "
            "Check if input triangles have valid data structure or contain "
            "zeros in the first column."
        )
    return estimate_uncertainty(
        nowcasts,
        truncated,
        retro,
        n=n_retrospective_nowcasts,
        validate=False,
        **kwargs,
    )


MAX_SIZE = 1000.0
"""Upper bound of `fit_nb`'s size, the Poisson end of its range."""


PRIOR_SIZE = 1.0 / 0.6745**2
r"""Size with no errors to fit one, about 2.2.

It is the median of epinowcast's prior $1 / \sqrt{\phi} \sim
\mathcal{N}^+(0, 1)$.
"""


def fit_sizes(added: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    r"""Fit the size $\phi_k$ of each week back by maximum likelihood.

    Parameters
    ----------
    added : numpy.ndarray
        New reports after each retrospective nowcast, `[N, K]`.
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[N, K]`. A cell with 0 and
        no new reports adds nothing to the likelihood, so it pads.

    Returns
    -------
    numpy.ndarray
        $\phi_k$, `[K]`.
    """
    fitted = np.asarray(fit_by_horizon(added, remaining))
    return np.where((remaining > 0).any(axis=0), fitted, PRIOR_SIZE)


def sample_count(
    key: Array, reported: Array, remaining: Array, size: Array
) -> Array:
    r"""Draw the count at the later release.

    Parameters
    ----------
    key : jax.Array
        Random key.
    reported : jax.Array
        $Y$, of any shape.
    remaining : jax.Array
        $\mathrm{rem}$, the shape of `reported`.
    size : jax.Array
        $\phi_k$, the shape of `reported`.

    Returns
    -------
    jax.Array
        $X$, the shape of `reported`.
    """
    return reported + sample_nb(key, remaining, size)
