# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Coherent sample trajectories from any model's quantile forecasts.

A quantile forecast gives each series and step its own marginal and
nothing about how they move together. `couple` draws trajectories whose
cells each follow the cell's quantile function, piecewise linear between
the levels and flat beyond them (so the end quantiles carry the tail
mass), and joins the cells by a copula:

- `independent`: every cell draws its own level;
- `comonotone`: one level per sample and series across its steps, the
  quantile-only analogue of ensemble copula coupling;
- `gaussian`: levels from a Gaussian copula with AR(1) correlation `rho`
  between steps and exchangeable correlation `s` between series, given
  in `fit`;
- `schaake`: the Schaake shuffle, each sample's levels the ranks of one
  past window of values (`fit["windows"]`), observed (`history_windows`)
  or simulated.

The parameters come from a `fit`; nothing is hard-coded. `fit_coupling`
estimates them from normal scores of the model's own matured forecasts
(`pit_scores`), and `choose_coupling` picks the coupling and how many of
those origins to fit it from by the energy score (`energy_score`) of each
candidate's draws on the matured earlier forecasts.
This module depends on numpy, scipy and `nowcastutils.dates` only, so any
model can use it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from typing import Any, Final

import numpy as np
from scipy.special import ndtr, ndtri

from nowcastutils.dates import as_date

__all__ = [
    "CANDIDATES",
    "COUPLING_DAYS",
    "METHODS",
    "MIN_EARLIER",
    "choose_coupling",
    "couple",
    "coupling_releases",
    "energy_score",
    "fit_coupling",
    "history_windows",
    "pit_scores",
    "read_quantiles",
]

METHODS: Final = ("independent", "comonotone", "gaussian", "schaake")
"""How `couple` joins the cells."""


def read_quantiles(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read quantile functions at uniform draws.

    Linear between the levels and flat beyond them.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    values : numpy.ndarray
        Quantiles at those levels, `[..., Q]`, sorted first so crossed
        quantiles give a valid function.
    u : numpy.ndarray
        Uniform draws, the shape of `values` less its last axis.

    Returns
    -------
    numpy.ndarray
        The draws, the shape of `u`.
    """
    return _read_sorted(levels, np.sort(values, axis=-1), u)


def _read_sorted(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read sorted quantiles `values` at uniform draws `u`."""
    hi = np.clip(np.searchsorted(levels, u), 1, len(levels) - 1)
    lo = hi - 1
    weight = np.clip((u - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
    at_lo = np.take_along_axis(values, lo[..., None], axis=-1)[..., 0]
    at_hi = np.take_along_axis(values, hi[..., None], axis=-1)[..., 0]
    return np.asarray(at_lo + weight * (at_hi - at_lo))


def _gaussian_levels(
    n_samples: int,
    series: int,
    steps: int,
    rho: float,
    s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return uniform levels from the AR(1) x exchangeable Gaussian copula.

    Each series' steps follow a stationary AR(1) with correlation `rho`;
    a shared factor gives correlation `s` between any two series at the
    same step. Both pieces have unit variance, so the levels are uniform.
    """
    shared = rng.standard_normal((n_samples, 1, steps))
    own = rng.standard_normal((n_samples, series, steps))
    z = np.sqrt(s) * shared + np.sqrt(1 - s) * own
    path = np.empty_like(z)
    path[..., 0] = z[..., 0]
    scale = np.sqrt(1 - rho**2)
    for t in range(1, steps):
        path[..., t] = rho * path[..., t - 1] + scale * z[..., t]
    return np.asarray(ndtr(path))


def _schaake_levels(
    windows: np.ndarray,
    n_samples: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return uniform levels with the ranks of past windows.

    Each sample takes one window, without replacement while there are
    enough and reused otherwise. A cell's level is its window's rank among
    the samples' windows at that cell, ties broken at random, plus a
    uniform jitter, over `n_samples`: uniform in every cell, and ordered
    across samples as the windows are.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[1:] != shape:
        raise ValueError(
            f"schaake windows must be [windows, {shape[0]}, {shape[1]}], "
            f"got {windows.shape}"
        )
    if len(windows) == 0 or not np.isfinite(windows).all():
        raise ValueError("schaake windows must be finite and not empty")
    pick = rng.choice(len(windows), n_samples, replace=len(windows) < n_samples)
    chosen = windows[pick]
    tie = rng.random(chosen.shape)
    order = np.lexsort((tie, chosen), axis=0)
    rank = np.empty_like(order)
    place = np.broadcast_to(np.arange(n_samples)[:, None, None], order.shape)
    np.put_along_axis(rank, order, place, axis=0)
    return np.asarray((rank + rng.random(chosen.shape)) / n_samples)


def couple(
    quantiles: np.ndarray,
    levels: np.ndarray,
    method: str,
    fit: Mapping[str, Any] | None,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return coherent sample trajectories from quantile forecasts.

    Parameters
    ----------
    quantiles : numpy.ndarray
        Quantiles `[series, steps, Q]` at `levels`.
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    method : str
        One of `METHODS`.
    fit : mapping or None
        The method's parameters: `rho` and `s` for `gaussian`, `windows`
        `[windows, series, steps]` for `schaake`.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the levels.

    Returns
    -------
    numpy.ndarray
        Draws `[n_samples, series, steps]`, each cell's marginal its
        piecewise-linear quantile function.

    Raises
    ------
    ValueError
        For an unknown method, a `gaussian` fit without `rho` and `s`
        or with them out of range, or a `schaake` fit without finite
        windows of the forecast's shape.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    series, steps = quantiles.shape[:2]
    rng = np.random.default_rng(seed)
    if method == "independent":
        u = rng.random((n_samples, series, steps))
    elif method == "comonotone":
        u = np.repeat(rng.random((n_samples, series, 1)), steps, axis=2)
    elif method == "gaussian":
        if not fit or "rho" not in fit or "s" not in fit:
            raise ValueError(
                "a gaussian coupling needs fit['rho'] and fit['s']"
            )
        rho, s = float(fit["rho"]), float(fit["s"])
        if not (-1 < rho < 1 and 0 <= s <= 1):
            raise ValueError(
                "a gaussian coupling needs -1 < rho < 1 and 0 <= s <= 1, "
                f"got rho={rho}, s={s}"
            )
        u = _gaussian_levels(n_samples, series, steps, rho, s, rng)
    elif method == "schaake":
        if not fit or "windows" not in fit:
            raise ValueError("a schaake coupling needs fit['windows']")
        u = _schaake_levels(fit["windows"], n_samples, (series, steps), rng)
    else:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    values = np.sort(quantiles, axis=-1)
    values = np.broadcast_to(values, (n_samples, *values.shape))
    return _read_sorted(levels, values, u)


def _knots(
    levels: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the quantile function's knots: levels `[K]`, values `[..., K]`.

    The function `read_quantiles` reads: flat from 0 to the first level
    and from the last level to 1, linear between.
    """
    u = np.concatenate([[0.0], levels, [1.0]])
    v = np.concatenate([values[..., :1], values, values[..., -1:]], axis=-1)
    return u, v


def pit_scores(
    quantiles: np.ndarray,
    levels: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return normal scores of observed values in quantile forecasts.

    The score is $\Phi^{-1}$ of the randomised PIT,
    $F(y^-) + v\,(F(y) - F(y^-))$ with $v$ uniform, in the quantile
    function `read_quantiles` reads. So a value on a flat stretch (zero
    counts with every quantile zero, or the mass beyond the end levels)
    draws its level rather than taking a fixed one.

    Parameters
    ----------
    quantiles : numpy.ndarray
        Quantiles `[..., Q]` at `levels`.
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    observed : numpy.ndarray
        Observed values, the shape of `quantiles` less its last axis; NaN
        where none is observed yet.
    rng : numpy.random.Generator
        Draws the uniforms.

    Returns
    -------
    numpy.ndarray
        Scores, the shape of `observed`, NaN where it is.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.sort(np.asarray(quantiles, dtype=float), axis=-1)
    y = np.asarray(observed, dtype=float)
    u, v = _knots(levels, values)
    k = len(u)
    filled = np.where(np.isnan(y), 0.0, y)[..., None]

    def between(j: np.ndarray) -> np.ndarray:
        """Return the level at `filled` on the segment from knot j to j + 1."""
        j = np.clip(j, 0, k - 2)
        v0 = np.take_along_axis(v, j[..., None], axis=-1)[..., 0]
        v1 = np.take_along_axis(v, j[..., None] + 1, axis=-1)[..., 0]
        width = np.where(v1 > v0, v1 - v0, 1.0)
        t = np.clip((filled[..., 0] - v0) / width, 0.0, 1.0)
        return np.asarray(u[j] + (u[j + 1] - u[j]) * t)

    # F(y): the last knot at or below y; F(y-): the first at or above it.
    last = (v <= filled).sum(axis=-1) - 1
    upper = np.where(last >= k - 1, 1.0, np.where(last < 0, 0.0, between(last)))
    first = (v < filled).sum(axis=-1)
    lower = np.where(
        first >= k, 1.0, np.where(first <= 0, 0.0, between(first - 1))
    )
    pit = lower + rng.random(y.shape) * (upper - lower)
    tiny = np.finfo(float).eps
    z = ndtri(np.clip(pit, tiny, 1 - tiny))
    return np.where(np.isnan(y), np.nan, z)


def _pair_sums(
    a: np.ndarray, b: np.ndarray, weight_a: np.ndarray, weight_b: np.ndarray
) -> np.ndarray:
    """Return each origin's sums over pairs: n, a, a^2, b, b^2.

    `a` and `b` are `[origins, ...]` with NaN filled by 0; each value
    enters as many pairs as its weight.
    """
    axes = tuple(range(1, a.ndim))
    return np.stack(
        [
            weight_a.sum(axis=axes),
            (weight_a * a).sum(axis=axes),
            (weight_a * a**2).sum(axis=axes),
            (weight_b * b).sum(axis=axes),
            (weight_b * b**2).sum(axis=axes),
        ],
        axis=1,
    )


def _step_sums(z: np.ndarray) -> np.ndarray:
    """Return each origin's sums over pairs of consecutive steps.

    `[origins, 6]`: n, a, a^2, b, b^2 and ab, with a a series' score at a
    step and b at the next, over pairs with both scores.
    """
    a, b = z[:, :, :-1], z[:, :, 1:]
    ok = (np.isfinite(a) & np.isfinite(b)).astype(float)
    a, b = np.where(ok > 0, a, 0.0), np.where(ok > 0, b, 0.0)
    ab = (ok * a * b).sum(axis=(1, 2))
    return np.column_stack([_pair_sums(a, b, ok, ok), ab])


def _series_sums(z: np.ndarray) -> np.ndarray:
    """Return each origin's sums over pairs of series at one step.

    As `_step_sums`, with a series i's score and b a later series j's, over
    every pair i < j with both scores. Sums over the series give them
    without listing the pairs: a value of series i enters a pair with each
    later series that has a score (`after`), one of series j with each
    earlier one (`before`), and the products sum to half the square of
    the step's sum less its sum of squares.
    """
    ok = np.isfinite(z)
    x = np.where(ok, z, 0.0)
    seen = np.cumsum(ok, axis=1)
    after = ok * (seen[:, -1:] - seen)
    before = ok * (seen - ok)
    total = x.sum(axis=1)
    ab = 0.5 * (total**2 - (x**2).sum(axis=1)).sum(axis=1)
    return np.column_stack([_pair_sums(x, x, after, before), ab])


def _shrunk_correlation(sums: np.ndarray) -> tuple[float, float, int]:
    """Return the pairs' correlation shrunk to zero, its shrinkage and n.

    From each origin's sums over its pairs (`_step_sums`, `_series_sums`).
    The Ledoit-Wolf intensity for a target of zero, var(r) / r^2 capped at
    1, with var(r) from the pairs clustered by origin (pairs of one
    forecast are not independent). Fewer than two origins give no evidence:
    zero, shrinkage 1.
    """
    n_o, sa, saa, sb, sbb, sab = sums.T
    n = float(n_o.sum())
    if n < 2 or (n_o > 0).sum() < 2:
        return 0.0, 1.0, int(n)
    ma, mb = sa.sum() / n, sb.sum() / n
    va, vb = saa.sum() / n - ma**2, sbb.sum() / n - mb**2
    if va <= 0 or vb <= 0:
        return 0.0, 1.0, int(n)
    scale = np.sqrt(va * vb)
    r = float((sab.sum() / n - ma * mb) / scale)
    own = (sab - mb * sa - ma * sb + n_o * ma * mb) / scale - r * n_o
    var = float((own**2).sum() / n**2)
    shrink = 1.0 if r == 0 else min(1.0, var / r**2)
    return (1 - shrink) * r, shrink, int(n)


def fit_coupling(scores: np.ndarray, method: str) -> dict[str, Any]:
    r"""Return a coupling's fit from normal scores of matured forecasts.

    For `gaussian`, $\rho$ is the correlation of a series' scores at
    consecutive steps and $s$ that of two series at the same step, each
    over every pair in the scores, shrunk towards zero (independence) by
    the Ledoit-Wolf intensity estimated from the pairs themselves
    (`_shrunk_correlation`). $s$ is floored at zero, as `couple`'s
    exchangeable copula needs, and both stay below 1. No parameter is
    fixed in advance.

    Parameters
    ----------
    scores : numpy.ndarray
        `pit_scores` of the model's earlier forecasts against what was
        later reported, `[origins, series, steps]`, NaN where a cell has
        not matured.
    method : str
        One of `METHODS`.

    Returns
    -------
    dict
        `method`; for `gaussian` also `rho`, `s`, their `shrinkage_rho`
        and `shrinkage_s`, `n_origins` and the pairs behind each,
        `n_step_pairs` and `n_series_pairs`.

    Raises
    ------
    ValueError
        For an unknown method, or `schaake`, whose fit is its windows.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    if method == "schaake":
        raise ValueError(
            "a schaake fit is its windows, not scores: use history_windows "
            "or simulated windows"
        )
    if method != "gaussian":
        return {"method": method}
    z = np.asarray(scores, dtype=float)
    n_origins = z.shape[0]
    rho, shrink_rho, n_steps = _shrunk_correlation(_step_sums(z))
    s, shrink_s, n_series = _shrunk_correlation(_series_sums(z))
    bound = np.nextafter(1.0, 0.0)
    return {
        "method": method,
        "rho": float(np.clip(rho, -bound, bound)),
        "s": float(np.clip(s, 0.0, bound)),
        "shrinkage_rho": shrink_rho,
        "shrinkage_s": shrink_s,
        "n_origins": int(n_origins),
        "n_step_pairs": int(n_steps),
        "n_series_pairs": int(n_series),
    }


def history_windows(
    values: np.ndarray, steps: int, n_windows: int
) -> np.ndarray:
    """Return the latest complete stretches of past values, as windows.

    The Schaake shuffle's historical windows: every stretch of `steps`
    consecutive steps of all series, newest first, one step apart, skipping
    any with a value missing.

    Parameters
    ----------
    values : numpy.ndarray
        Past values `[series, time]`, oldest first, NaN where missing.
    steps : int
        Steps a window spans, the forecast's.
    n_windows : int
        Most windows to return.

    Returns
    -------
    numpy.ndarray
        Windows `[n, series, steps]`, `n <= n_windows`.
    """
    values = np.asarray(values, dtype=float)
    out = []
    for end in range(values.shape[1], steps - 1, -1):
        window = values[:, end - steps : end]
        if np.isfinite(window).all():
            out.append(window)
            if len(out) == n_windows:
                break
    return np.asarray(out).reshape(len(out), values.shape[0], steps)


def energy_score(draws: np.ndarray, observed: np.ndarray) -> float:
    r"""Return the energy score of trajectories on the observed cells.

    $\mathrm{ES} = E\|X - y\| - \tfrac12 E\|X - X'\|$ over the cells
    where `observed` is finite, the second term from each draw and the
    next (Gneiting et al. 2008), so the cost is linear in the draws. It
    estimates the same score as `nowcasteval.scoring.energy_sample`'s fair
    estimator, without its cost quadratic in the draws, and is noisier.

    Parameters
    ----------
    draws : numpy.ndarray
        Trajectories `[n, ...]`.
    observed : numpy.ndarray
        Values, the shape of one trajectory, NaN where not observed.

    Returns
    -------
    float
        The score (lower is better), NaN with no observed cell.
    """
    draws = np.asarray(draws, dtype=float)
    observed = np.asarray(observed, dtype=float)
    ok = np.isfinite(observed)
    if not ok.any():
        return float("nan")
    x = draws[:, ok]
    to_truth = np.linalg.norm(x - observed[ok], axis=1).mean()
    spread = np.linalg.norm(x - np.roll(x, 1, axis=0), axis=1).mean()
    return float(to_truth - 0.5 * spread)


CANDIDATES: Final = (
    ("gaussian", 14),
    ("gaussian", 28),
    ("gaussian", 56),
    ("schaake", None),
    ("comonotone", None),
)
"""The couplings `choose_coupling` compares by default: the gaussian
copula fitted over the origins of the latest 14, 28 or 56 days, the
Schaake shuffle on the series' own history, and comonotone draws. Which
one is used, and so the look-back window, is chosen from the data at
each forecast."""

MIN_EARLIER: Final = 2
"""Earlier origins an origin needs to be scored on in `choose_coupling`:
the fewest a gaussian fit has evidence from (one origin's pairs are not
independent)."""

COUPLING_DAYS: Final = (
    max(w for _, w in CANDIDATES if w is not None) + 7 * MIN_EARLIER
)
"""How many days of earlier origins `choose_coupling` reads
(`coupling_releases`): the longest gaussian window (`CANDIDATES`) and
`MIN_EARLIER` weekly origins before it, so that window can be scored."""


def coupling_releases(
    vintages: Iterable[Any], days: int = COUPLING_DAYS
) -> list[str]:
    """Return the earlier origins a forecast's coupling is chosen from.

    Every release before the latest of `vintages` and at most `days`
    before it.

    Parameters
    ----------
    vintages : sequence of str
        The task's release dates, ISO formatted.
    days : int
        How many days back.

    Returns
    -------
    list of str
        The releases, oldest first, each once.
    """
    dates = sorted({as_date(str(v)) for v in vintages})
    if not dates:
        return []
    edge = dates[-1] - dt.timedelta(days=days)
    return [d.isoformat() for d in dates[:-1] if d >= edge]


def _label(method: str, window: int | None) -> str:
    """Return a candidate's name, as `choose_coupling` reports it."""
    return method if window is None else f"{method}-{window}"


def _candidate_fit(
    method: str,
    window: int | None,
    scores: np.ndarray,
    ages: np.ndarray,
    history: np.ndarray | None,
    steps: int,
    n_samples: int,
) -> dict[str, Any] | None:
    """Return a candidate's fit from what came before, or None if it has none.

    A gaussian fit from the origins of `scores` at most `window` days old
    (`ages`); Schaake windows from `history` (`history_windows`), None
    without any.
    """
    if method == "schaake":
        if history is None:
            return None
        windows = history_windows(history, steps, n_samples)
        if not len(windows):
            return None
        return {"method": method, "windows": windows}
    if method == "gaussian":
        latest = scores if window is None else scores[ages <= window]
        return fit_coupling(latest, method)
    return fit_coupling(scores, method)


def choose_coupling(
    quantiles: np.ndarray,
    levels: np.ndarray,
    observed: np.ndarray,
    days: np.ndarray,
    day: int,
    history: np.ndarray | None,
    starts: np.ndarray | None,
    n_samples: int,
    seed: int,
    candidates: tuple[tuple[str, int | None], ...] = CANDIDATES,
) -> dict[str, Any]:
    """Return the coupling, with its fit, whose draws scored best of late.

    Each earlier origin with at least `MIN_EARLIER` before it and a value
    reported since is scored: every candidate is fitted from what came
    before it (the gaussian from the normal scores, `pit_scores`, of the
    earlier origins at most `window` days before it; the Schaake shuffle
    from windows of
    `history` before the origin's first step), that origin's quantiles are
    coupled (`couple`) on the steps up to its last reported one, with the
    same seed for every candidate, and the draws are scored against what
    was reported (`energy_score`). The candidate with the lowest mean
    score over origins every candidate could be fitted at wins; ties go
    to the earlier candidate; a candidate that could not be fitted at one
    of the origins scored drops out. The winner is then fitted from
    everything before the forecast. With nothing to compare on, the
    gaussian over the longest window is used when there are `MIN_EARLIER`
    origins to fit it from, else the Schaake shuffle when the history has
    windows, else the gaussian (independence), and the record says so
    (`chosen_by`).

    Parameters
    ----------
    quantiles : numpy.ndarray
        The model's earlier forecasts' quantiles `[origins, series, steps,
        Q]` at `levels`, oldest first, on the forecast's series and steps.
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    observed : numpy.ndarray
        What has been reported of their targets, `[origins, series,
        steps]`, NaN where not final.
    days : numpy.ndarray
        Each origin's day, `[origins]`, increasing (any integer count of
        days, such as an ordinal date).
    day : int
        The forecast's day, on the same count.
    history : numpy.ndarray or None
        The series' reported values `[series, time]` before the forecast's
        first step, NaN where missing; None for no Schaake shuffle.
    starts : numpy.ndarray or None
        Each origin's first step as a column of `history`, `[origins]`.
    n_samples : int
        Trajectories, and the most Schaake windows.
    seed : int
        Seed for the scores' and the draws' uniforms.
    candidates : tuple of (str, int or None)
        Each candidate's method and gaussian window in days.

    Returns
    -------
    dict
        The chosen fit, which `couple` takes, with `window`, the mean
        `energy` of each candidate compared, by name, `n_eval` (origins
        scored on),
        `chosen_by` (`energy` or `fallback`) and, for the Schaake shuffle,
        `n_windows`.

    Raises
    ------
    ValueError
        With no candidates, `days` not one per origin, or a history
        without `starts` for each origin.
    """
    if not candidates:
        raise ValueError("choose_coupling needs candidates")
    quantiles = np.asarray(quantiles, dtype=float)
    observed = np.asarray(observed, dtype=float)
    levels = np.asarray(levels, dtype=float)
    n_origins, _, steps = observed.shape
    if history is not None:
        history = np.asarray(history, dtype=float)
        if starts is None or len(starts) != n_origins:
            raise ValueError("a history needs starts, one per origin")
    first = np.zeros(n_origins, dtype=int) if starts is None else starts
    days = np.asarray(days, dtype=int)
    if len(days) != n_origins:
        raise ValueError("days needs one day per origin")
    rng = np.random.default_rng([seed, 1])
    scores = pit_scores(quantiles, levels, observed, rng)
    now = {
        c: _candidate_fit(*c, scores, day - days, history, steps, n_samples)
        for c in candidates
    }
    usable = [c for c in candidates if now[c] is not None]
    got: dict[tuple[str, int | None], list[float]] = {c: [] for c in usable}
    n_eval = 0
    for k in range(MIN_EARLIER, n_origins):
        reported = np.flatnonzero(np.isfinite(observed[k]).any(axis=0))
        if not len(reported):
            continue
        n = int(reported[-1]) + 1
        past = None if history is None else history[:, : int(first[k])]
        n_eval += 1
        for c in usable:
            ages = days[k] - days[:k]
            fit = _candidate_fit(*c, scores[:k], ages, past, steps, n_samples)
            if fit is None:
                got[c].append(np.nan)
                continue
            if "windows" in fit:
                fit["windows"] = fit["windows"][:, :, :n]
            draws = couple(
                quantiles[k][:, :n], levels, c[0], fit, n_samples, seed + k
            )
            got[c].append(energy_score(draws, observed[k][:, :n]))
    # Compared on the same origins: a candidate with no fit at one of them
    # (the Schaake shuffle with too little history then) drops out.
    scored = [c for c in usable if n_eval and np.isfinite(got[c]).all()]
    if scored:
        mean = {c: float(np.mean(got[c])) for c in scored}
        chosen = min(scored, key=lambda c: (mean[c], scored.index(c)))
        chosen_by = "energy"
    else:
        mean, chosen_by = {}, "fallback"
        gaussian = [c for c in usable if c[0] == "gaussian"]
        schaake = [c for c in usable if c[0] == "schaake"]
        longest = max(gaussian, key=lambda c: c[1] or 0, default=None)
        if longest is not None and (n_origins >= MIN_EARLIER or not schaake):
            chosen = longest
        else:
            chosen = (schaake or usable)[0]
    fit = now[chosen] or _candidate_fit(
        *chosen, scores, day - days, history, steps, n_samples
    )
    assert fit is not None
    out = {
        **fit,
        "window": chosen[1],
        "energy": {_label(*c): e for c, e in mean.items()},
        "n_eval": n_eval,
        "chosen_by": chosen_by,
    }
    if "windows" in fit:
        out["n_windows"] = len(fit["windows"])
    return out
