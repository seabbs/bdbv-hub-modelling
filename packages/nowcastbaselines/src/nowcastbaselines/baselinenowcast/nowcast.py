r"""The nowcast: the reports still to come, with their uncertainty.

`baselinenowcast` nowcasts a reporting triangle as the R package does.
`build_origin` nowcasts from release snapshots with the same chain ladder,
filling each week from what it has reported (`fill_row`, `week_nowcast`).

Daily chain ladder on release snapshots, with its uncertainty.

Counts $N_r(t)$ are the daily onsets of release $r$, on the onset clock.
Release $r$ has cut-off $c_r$, and day $t$ is $a = c_r - t$ days old.
A pair of consecutive releases $(r, r + 1)$ has step
$s = c_{r+1} - c_r$ days and is in window when $s \le D$.

**Delay.** Releases are batched reports of one daily delay.
A day of age $a$ at release $r$ grows by $\exp \sum_{j=a}^{a+s-1} g(j)$
to release $r + 1$, with $g(j) \ge 0$ the daily log development factor
and $g(j) = 0$ for $j \ge D$.
For each gap $s$ and age $a$, the sums $\sum N_{r+1}(t)$ and
$\sum N_r(t)$, over the in-window pairs of that gap and the days of age
$a$ that both releases observe, are the cases at the later release and
those of them shown at the earlier.
Each case was shown at the earlier with chance
$\exp(-\sum_{j=a}^{a+s-1} g(j))$, a binomial, and $g$ maximises the
likelihood of all of them (`log_factors`).
Where the earlier release shows 0, 1 case counts as shown at both, as
baselinenowcast's chain ladder divides by at least 1, so the later cases
of days first shown as 0 count.
Cells a release does not observe are skipped, not filled.
A place with at least 10 cases a week uses its own $g$, and otherwise
the pooled $g$ of its nearest parent with them, fitted to the summed
pair sums of that parent's places with 10 cases a week (`pooling`), as
for the walk spread and the nowcast size.
Only onset days within `Settings.history_delay` (63 for $D = 42$) of the
latest cut-off are summed, and downward revisions are first moved onto
earlier releases of the day, as baselinenowcast does.

**Nowcast.** At the latest release $o$, day $t$ of age $a$ grows to final
by $G(a) = \exp \sum_{j=a}^{D-1} g(j)$.
Each hub week is a row of a weekly reporting triangle, with a column every
7 days, and the mean over its observed days of $1 / G$ at each column is
the share of the week reported by it.
baselinenowcast's `apply_delay` fills the row from the week's reported
count (`week_nowcast`), with its $+1$, so a week with nothing reported yet
still expects some cases.

**Uncertainty.** A release $r < o$ with an in-window pair at $r$, and a
cut-off at most `Settings.retro_delay` (63) days before $c_o$, nowcasts
the count expected at $o$ for days up to $c_r$, with the same weekly fill
from $c_r$ to $c_o$ and the $g$ of $r$.
Its new reports by weeks back fit the observation model (`obs`), on
counts whose downward revisions are moved onto earlier releases, as for
the delay.

**Introductions.** A place with no case in the latest `walk_weeks` weeks
is silent.
Its weekly probability of an introduction is the rate of a first case,
or of a re-introduction if it was seen before, over the `walk_weeks`
weeks to its start week. Its siblings' pooled rate and that of the rest
of its level combine as independent sources
(`introductions.introduction_rates`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from nowcastbaselines.introductions import introduction_rates, silent_places
from nowcastbaselines.pooling import pool_matrix, source_index
from nowcastbaselines.walk import walk_noise, walk_variance
from nowcastutils.base import ModelUnavailableError
from nowcastutils.snapshots import SnapshotArrays, week_sum, weekly_sums

from ._checks import coerce, validate_reporting_triangle
from .aggregate import identity, row_sums
from .delays import (
    DelayFit,
    NoDelayPair,
    Settings,
    _delay_counts,
    apply_delay,
    estimate_delay,
    final_growth,
    fit_delay,
    growth,
    weeks_back,
)
from .triangle import (
    allocate_reference_times,
    preprocess_negative_values,
)
from .uncertainty import (
    DelayAggregator,
    RefTimeAggregator,
    Sampler,
    estimate_uncertainty_retro,
    fit_by_horizon,
    fit_sizes,
    sample_nb,
    sample_nowcasts,
)

__all__ = [
    "DelayFn",
    "Errors",
    "NoCompleteWeek",
    "NoWalkSpread",
    "Origin",
    "baselinenowcast",
    "build_origin",
    "delay_summary",
    "fill_row",
    "retro_errors",
    "start_weeks",
    "walk_changes",
    "week_nowcast",
]


def fill_row(x: Any, cdf: Any) -> np.ndarray:
    r"""Return a row's count by its last column, as R's ``apply_delay``.

    Each later column $j$ adds $(x_{j-1} + 1 - F_{j-1}) / F_{j-1}\,
    (F_j - F_{j-1})$, with $x_{j-1}$ the count up to column $j - 1$
    (R ``.calc_modified_expectation``).
    The $+1$ gives a row with nothing reported yet a positive count.

    Parameters
    ----------
    x : array_like
        Count reported by the first column.
    cdf : array_like
        Share reported by each column, the first that of ``x``, on the
        last axis. It broadcasts with ``x`` on the others.

    Returns
    -------
    numpy.ndarray
        The count by the last column.
    """
    cdf = np.asarray(cdf, float)
    total = np.asarray(x, float)
    for j in range(1, cdf.shape[-1]):
        prev = cdf[..., j - 1]
        total = total + (total + 1 - prev) / prev * (cdf[..., j] - prev)
    return total


def baselinenowcast(
    triangle: Any,
    mask: Any,
    key: Array | None = None,
    scale_factor: float = 3,
    prop_delay: float = 0.5,
    output_type: Literal["samples", "point"] = "samples",
    draws: int = 1000,
    uncertainty_model: Callable[..., Array] = fit_by_horizon,
    uncertainty_sampler: Sampler = sample_nb,
    delay_pmf: Any = None,
    uncertainty_params: Any = None,
    preprocess: Callable[..., Array] | None = preprocess_negative_values,
    structure: Any = None,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
    validate: bool = True,
) -> Array:
    """Nowcast a reporting triangle.

    Estimates the delay and its uncertainty from the triangle, fills
    the unobserved cells and, for samples, draws the counts still to
    be reported.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    key : jax.Array, optional
        Random key. Needed for ``output_type="samples"``.
    scale_factor, prop_delay : float
        Passed to `allocate_reference_times`.
    output_type : {"samples", "point"}, default "samples"
        Draws, or the point nowcast.
    draws : int, default 1000
        Number of draws.
    uncertainty_model : callable, default fit_by_horizon
        Fits the uncertainty parameters.
    uncertainty_sampler : Sampler, default sample_nb
        Draws counts given a mean and the parameters.
    delay_pmf, uncertainty_params : array_like, optional
        Use these instead of estimating them.
    preprocess : callable or None, default preprocess_negative_values
        Applied to the triangle first.
    structure : int or array_like, optional
        Reporting structure. Defaults to `get_reporting_structure`.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times, such as `rolling_sum`.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        For ``"point"``, the point nowcast per reference time (row sums
        of the filled triangle). For ``"samples"``, draws of shape
        ``(draws, n_aggregated)``.

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> from nowcastbaselines.baselinenowcast import baselinenowcast
    >>> rng = np.random.default_rng(1)
    >>> triangle = rng.poisson(10, size=(20, 4)).astype(float)
    >>> mask = np.add.outer(np.arange(20), np.arange(4)) < 20
    >>> draws = baselinenowcast(
    ...     triangle * mask, mask, key=jax.random.key(1), draws=100
    ... )
    >>> draws.shape
    (100, 20)
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if output_type not in ("samples", "point"):
        raise ValueError("`output_type` must be 'samples' or 'point'")
    n_req = 2 if output_type == "samples" else 0
    n_history_delay, n_retro = allocate_reference_times(
        m,
        scale_factor=scale_factor,
        prop_delay=prop_delay,
        n_min_retro_nowcasts=n_req,
        validate=False,
    )
    processed = values
    if preprocess is not None:
        processed = preprocess(values, m, validate=False)
    if delay_pmf is None:
        delay_pmf = estimate_delay(
            processed, m, n=n_history_delay, validate=False
        )
    pt = apply_delay(processed, m, delay_pmf, validate=False)
    if output_type == "point":
        return jnp.sum(pt, axis=1)
    if key is None:
        raise ValueError("`key` is required for output_type='samples'")
    if uncertainty_params is None:
        uncertainty_params = estimate_uncertainty_retro(
            processed,
            m,
            n_history_delay=n_history_delay,
            n_retrospective_nowcasts=n_retro,
            structure=structure,
            uncertainty_model=uncertainty_model,
            ref_time_aggregator=ref_time_aggregator,
            delay_aggregator=delay_aggregator,
            validate=False,
        )
    return sample_nowcasts(
        key,
        pt,
        processed,
        m,
        uncertainty_params,
        draws=draws,
        uncertainty_sampler=uncertainty_sampler,
        ref_time_aggregator=ref_time_aggregator,
        delay_aggregator=delay_aggregator,
    )


log = logging.getLogger(__name__)


class NoCompleteWeek(ModelUnavailableError):  # noqa: N818
    """No week of a place reaches the completeness that starts the walk."""


class NoWalkSpread(ModelUnavailableError):  # noqa: N818
    """A place has no walk spread of its own or from any parent."""


def delay_summary(origin: Origin, model: str, vintage: str) -> pd.DataFrame:
    """Return the share of a day's final count reported by each age.

    The share at age $a$ is $1 / G(a)$, from the delay fitted at the
    origin, for every place, in the form of the renewal models' delay
    summaries (`nowcastrenewal.forecaster.fit_delays`).

    Parameters
    ----------
    origin : Origin
        The fit.
    model : str
        Model name.
    vintage : str
        The origin.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `vintage`, `location`, `kind` (`"cumulative"`),
        `x` (age in days), `quantity` (`"reported_share"`) and `q50`.
    """
    ages = np.arange(origin.settings.max_delay + 1)
    share = 1.0 / growth(origin.fit.g, ages, origin.settings.max_delay)
    places = origin.snap.locations
    return pd.DataFrame(
        {
            "model": model,
            "vintage": vintage,
            "location": np.repeat(places, ages.size),
            "kind": "cumulative",
            "x": np.tile(ages, len(places)),
            "quantity": "reported_share",
            "q50": share.ravel(),
        }
    )


@dataclass
class Errors:
    """Retrospective nowcasts by release, place and weeks back.

    Attributes
    ----------
    added : numpy.ndarray
        Rise of each week's count between each release and the origin,
        `[R, L, K]`, after downward revisions are moved onto earlier
        releases (`_delay_counts`).
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[R, L, K]`.
    reported : numpy.ndarray
        Count reported at each release, `[R, L, K]`.
    valid : numpy.ndarray
        Whether each cell is an error, `[R, L, K]`.
    releases : list of str
        Releases that gave errors.
    """

    added: np.ndarray
    remaining: np.ndarray
    reported: np.ndarray
    valid: np.ndarray
    releases: list[str] = field(default_factory=list)


def week_nowcast(
    snap: SnapshotArrays, r: int, g: np.ndarray, age: np.ndarray, gap: int
) -> np.ndarray:
    """Return baselinenowcast's count of each hub week `gap` days after `r`.

    Each hub week is a reference time of a weekly reporting triangle,
    with a column every `snap.period` days from the cut-off of release
    `r` and a last one at `gap`.
    The share of the week reported by a column is the mean, over the
    week's days observed at `r`, of their growth from that column to
    `gap` inverted.
    R's `apply_delay` fills each row from these shares (`fill_row`).

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        The release.
    g : numpy.ndarray
        Daily log factors at `r`, `[L, D]`.
    age : numpy.ndarray
        Age of each day at `r`, `[T]`.
    gap : int
        Days from the cut-off of `r` to the count nowcast. `max_delay`
        or more gives the final count.

    Returns
    -------
    numpy.ndarray
        `[L, W]`, 0 for weeks with no day observed at `r`.
    """
    seen = np.broadcast_to(snap.observed[r], snap.counts[:, r, :].shape)
    x = week_sum(snap, np.where(seen, snap.counts[:, r, :], 0.0))
    n = week_sum(snap, seen.astype(float))
    steps = np.r_[np.arange(0, gap, snap.period), gap]
    share = np.stack(
        [
            week_sum(
                snap, np.where(seen, 1.0 / growth(g, age + s, age + gap), 0)
            )
            for s in steps
        ],
        axis=-1,
    )
    cdf = np.where(n[..., None] > 0, share / np.maximum(n, 1)[..., None], 1.0)
    return np.where(n > 0, fill_row(x, cdf), 0.0)


class DelayFn(Protocol):
    """Fits the delay at a release, as `fit_delay`."""

    def __call__(
        self,
        snap: SnapshotArrays,
        r: int,
        settings: Settings,
        longer: bool = False,
    ) -> DelayFit | None:
        """Return the fit at release `r`, or `None` with no pair to fit."""
        ...


def retro_errors(
    snap: SnapshotArrays, o: int, settings: Settings, delay: DelayFn = fit_delay
) -> Errors:
    """Collect the errors of nowcasts made at earlier releases.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    settings : Settings
        Settings.
    delay : callable
        Fits the delay at each earlier release.

    Returns
    -------
    Errors
        Sums over the releases that give errors.
    """
    n_loc = len(snap.locations)
    k_n = weeks_back(snap, o, settings.max_delay)
    # As baselinenowcast preprocesses its triangle, a fall is taken from
    # the earlier releases' counts, so a week's count never falls.
    pre = replace(snap, counts=_delay_counts(snap, o))
    cells: list[tuple[np.ndarray, ...]] = []
    releases: list[str] = []
    for r in range(1, o):
        if snap.cut[o] - snap.cut[r] > settings.retro_delay:
            continue
        fit = delay(snap, r, settings)
        if fit is None:
            continue
        age = snap.cut[r] - np.arange(snap.week.size)
        ahead = week_nowcast(pre, r, fit.g, age, snap.cut[o] - snap.cut[r])
        wa, k_r = snap.cut_week(r), weeks_back(snap, r, settings.max_delay)
        cols = wa - np.arange(k_n)
        take = (cols >= 0) & (np.arange(k_n) < k_r)
        safe = np.clip(cols, 0, None)
        obs = weekly_sums(pre, r)[:, safe]
        pred = ahead[:, safe]
        # The week's change to the origin, on days up to the cut-off of r.
        added = weekly_sums(pre, o, upto=int(snap.cut[r]))[:, safe] - obs
        valid = take[None, :] & ((obs > 0) | (pred > 0))
        cells.append((added, pred - obs, obs, valid))
        releases.append(snap.releases[r])
    c = np.array(cells, dtype=float).reshape(-1, 4, n_loc, k_n)
    return Errors(c[:, 0], c[:, 1], c[:, 2], c[:, 3] > 0, releases)


@dataclass
class Origin:
    r"""Everything the random walk models need at one release.

    Attributes
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    settings : Settings
        Settings.
    fit : DelayFit
        Delay factors at the origin.
    growth : numpy.ndarray
        $G$ for each place and day, `[L, T]`.
    reported : numpy.ndarray
        Counts reported at the origin by hub week, `[L, W]`.
    point : numpy.ndarray
        Nowcast mean by hub week, `[L, W]`.
    complete : numpy.ndarray
        Estimated share reported, $\pi$, `[L, W]`.
    start : numpy.ndarray
        Week index where `randomwalk` starts, `[L]`.
    sigma : numpy.ndarray
        Spread of the walk, `[L]`.
    sigma_source : numpy.ndarray
        Index of the estimate behind each spread, `[L]`, as for
        `source_index`.
    silent : numpy.ndarray
        Places with no case reported in the latest `walk_weeks` weeks,
        `[L]`.
    intro : numpy.ndarray
        Weekly probability of an introduction, a first case or a
        re-introduction, pooled over the place's siblings
        (`introduction_rates`), `[L]`.
    errors : Errors or None
        Retrospective errors, or None where they were not asked for.
    size : numpy.ndarray or None
        Negative binomial size $\phi_k$ of the new reports of each place
        by weeks back from the cut-off week, `[L, K]`.
    obs_n : numpy.ndarray or None
        Errors behind each fit, `[L, K]`.
    obs_source : numpy.ndarray or None
        Index of the estimate behind each fit, `[L, K]`.
    """

    snap: SnapshotArrays
    o: int
    settings: Settings
    fit: DelayFit
    growth: np.ndarray
    reported: np.ndarray
    point: np.ndarray
    complete: np.ndarray
    start: np.ndarray
    sigma: np.ndarray
    sigma_source: np.ndarray
    silent: np.ndarray
    intro: np.ndarray
    errors: Errors | None = None
    size: np.ndarray | None = None
    obs_n: np.ndarray | None = None
    obs_source: np.ndarray | None = None

    @property
    def start_nowcast(self) -> int:
        """Latest week whose seven days are all on or before the cut-off."""
        sat = np.array(
            [
                np.flatnonzero(self.snap.week == w).max()
                for w in range(self.snap.n_weeks)
            ]
        )
        full = np.flatnonzero(sat <= self.snap.cut[self.o])
        return int(full.max()) if full.size else 0


def _latest_fit(
    snap: SnapshotArrays, settings: Settings, delay: DelayFn
) -> tuple[int, DelayFit]:
    """Return the latest release and its delay fit.

    With no pair in the delay window, the delay is fitted to the pairs
    with a longer step. Nowcasts made at earlier releases, whose
    errors set the observation model, keep the window.

    Raises
    ------
    NoDelayPair
        If no two releases are a day or more apart.
    """
    o = len(snap.releases) - 1
    fit = delay(snap, o, settings)
    if fit is None:
        fit = delay(snap, o, settings, longer=True)
    if fit is None:
        raise NoDelayPair(
            f"No pair of releases up to {snap.releases[o]} is a day or "
            "more apart."
        )
    return o, fit


def _completeness(
    snap: SnapshotArrays, o: int, fit: DelayFit, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return $G$, the share reported by week and each place's start week.

    The start week is the latest week at least `Settings.completeness`
    complete, or the first week where none is.
    """
    g = final_growth(snap, o, fit, settings)
    inv = np.where(snap.observed[o][None, :], 1.0 / g, 0.0)
    complete = week_sum(snap, inv) / snap.period
    good = complete >= settings.completeness
    start = np.where(
        good.any(axis=1), snap.n_weeks - 1 - np.argmax(good[:, ::-1], axis=1), 0
    )
    return g, complete, start


def start_weeks(
    snap: SnapshotArrays, settings: Settings, delay: DelayFn = fit_delay
) -> np.ndarray:
    """Return the week each place's walk starts from, from the delay alone.

    It is `Origin.start` with no walk spread fitted, so it exists where
    `build_origin` raises `NoWalkSpread`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays. The origin is the latest release.
    settings : Settings
        Settings.
    delay : callable
        Fits the delay at a release.

    Returns
    -------
    numpy.ndarray
        Week index of each place, `[L]`.

    Raises
    ------
    NoDelayPair
        If no two releases are a day or more apart.
    """
    # The walk's weeks are blocks of the snapshots' period.
    settings = replace(
        settings, walk_weeks=settings.walk_weeks * 7 // snap.period
    )
    o, fit = _latest_fit(snap, settings, delay)
    return _completeness(snap, o, fit, settings)[2]


def build_origin(
    snap: SnapshotArrays,
    settings: Settings,
    errors: bool,
    delay: DelayFn = fit_delay,
) -> Origin:
    """Fit the delay and the walk spread at the latest release.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays. The origin is the latest release.
    settings : Settings
        Settings.
    errors : bool
        Whether to fit the observation model.
    delay : callable
        Fits the delay at a release.

    Returns
    -------
    Origin
        The fit.

    Raises
    ------
    NoDelayPair
        If no two releases are a day or more apart.
    """
    o, fit = _latest_fit(snap, settings, delay)
    g, complete, start = _completeness(snap, o, fit, settings)
    reported = weekly_sums(snap, o)
    age = snap.cut[o] - np.arange(snap.week.size)
    point = week_nowcast(snap, o, fit.g, age, settings.max_delay)
    own, pooled = _walk_spreads(reported, start, snap, fit.level, settings)
    # A variance of 0 or below is no estimate, so the place takes its
    # nearest parent's, as one without the level does.
    sigma_source = source_index(
        fit.level,
        snap.parent,
        (own > 0).astype(int)[:, None],
        (pooled > 0).astype(int)[:, None],
        settings.min_level,
        1,
    )[:, 0]
    var = np.concatenate([own, pooled])[sigma_source]
    if not (var > 0).all():
        none = [
            c for c, v in zip(snap.locations, var, strict=True) if not v > 0
        ]
        roots = [
            f"{snap.locations[n]} ({pooled[n]:.3g})"
            for n in np.flatnonzero(snap.parent < 0)
        ]
        raise NoWalkSpread(
            f"No walk spread above 0 at {snap.releases[o]} for "
            f"{', '.join(none)}, of its own or from any parent. The "
            f"national pool, {', '.join(roots)}, nets its weekly changes "
            "of their Poisson noise to 0 or below, or has fewer than 2 of "
            f"them, and so does each of its children with "
            f"{settings.min_level:g} or more cases a week."
        )
    sigma = np.sqrt(var)
    k = settings.walk_weeks
    silent = silent_places(reported, snap.cut_week(o), k)
    intro = introduction_rates(reported, start, snap.parent, k)
    origin = Origin(
        snap, o, settings, fit, g, reported, point, complete, start, sigma,
        sigma_source, silent, intro,
    )  # fmt: skip
    if errors:
        _add_spread(origin)
    return origin


def walk_changes(
    reported: np.ndarray, start: np.ndarray, settings: Settings
) -> list[np.ndarray]:
    """Return each place's weekly changes of log(1 + y) up to its start week.

    Parameters
    ----------
    reported : numpy.ndarray
        Weekly reported counts, `[L, W]`.
    start : numpy.ndarray
        Start week of each place, `[L]`.
    settings : Settings
        Settings, for `walk_weeks`.

    Returns
    -------
    list of numpy.ndarray
        The changes of each place over the `walk_weeks` weeks ending at
        its start week.
    """
    diffs = []
    for n in range(reported.shape[0]):
        lo = max(int(start[n]) - settings.walk_weeks + 1, 0)
        diffs.append(np.diff(np.log1p(reported[n, lo : int(start[n]) + 1])))
    return diffs


def _walk_spreads(
    reported: np.ndarray,
    start: np.ndarray,
    snap: SnapshotArrays,
    level: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each place's own and pooled walk variance, net of Poisson noise.

    A variance of 0 or below, or NaN, is no estimate (`walk.walk_variance`).
    """
    n_loc = reported.shape[0]
    diffs = walk_changes(reported, start, settings)
    noise = []
    for n in range(n_loc):
        lo = max(int(start[n]) - settings.walk_weeks + 1, 0)
        noise.append(walk_noise(reported[n, lo : int(start[n]) + 1]))

    def variance(members: np.ndarray) -> float:
        return walk_variance(
            np.concatenate([diffs[m] for m in members]),
            np.concatenate([noise[m] for m in members]),
        )

    own = np.array([variance(np.array([n])) for n in range(n_loc)])
    pool = pool_matrix(level, snap.parent, settings.min_level)
    pooled = np.array([variance(np.flatnonzero(pool[n])) for n in range(n_loc)])
    # A country with no pooled estimate pools only the children that
    # qualify with an estimate of their own (`pooling`), else its own
    # changes, so every place below it has the country's spread.
    qualify = pool_matrix(
        np.where(own > 0, level, 0.0), snap.parent, settings.min_level
    )
    for country in np.flatnonzero((snap.parent < 0) & ~(pooled > 0)):
        pooled[country] = variance(np.flatnonzero(qualify[country]))
    return own, pooled


def _add_spread(origin: Origin) -> None:
    """Fit the observation model by weeks back.

    The size follows the one rule. Each place's own estimate uses its
    errors, and its pooled estimate the errors of the places in its
    pool. The size is fitted only for the estimates some place uses.
    """
    snap, settings = origin.snap, origin.settings
    errors = retro_errors(snap, origin.o, settings)
    n_loc = len(snap.locations)
    pool = pool_matrix(origin.fit.level, snap.parent, settings.min_level)
    member = np.concatenate([np.eye(n_loc), pool]) > 0  # [2L, L]
    # A cell with nothing nowcast to come says nothing about the size.
    fit = errors.valid & (errors.remaining > 0)
    n = np.einsum("el,rlk->ek", member, fit, dtype=int)
    source = np.repeat(
        source_index(
            origin.fit.level, snap.parent, min_level=settings.min_level
        )[:, None],
        n.shape[1],
        axis=1,
    )
    size = np.zeros(n.shape)
    for e in np.unique(source):
        keep = fit & member[e][None, :, None]
        k_n = keep.shape[-1]
        size[e] = fit_sizes(
            np.where(keep, errors.added, 0.0).reshape(-1, k_n),
            np.where(keep, errors.remaining, 0.0).reshape(-1, k_n),
        )
    k = np.arange(n.shape[1])[None, :]
    origin.errors = errors
    origin.size = size[source, k]
    origin.obs_n, origin.obs_source = n[source, k], source
