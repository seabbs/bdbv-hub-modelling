"""Delay estimation by the chain ladder.

baselinenowcast's chain ladder on a reporting triangle (`estimate_delay`,
`apply_delay`; Johnson, Abbott et al., the R package this ports) and the
same factors from pairs of consecutive releases (`pair_sums`,
`log_factors`, `growth`, `window_pairs`).
The release pairs use every cell two releases both observe, so they also
take releases that are batched or that begin after the onsets they report
(left-truncated, as BDBV's).
On daily releases the two give the same delay.
`fit_delay` fits each place's factors at a release, pooled to its parent
below `MIN_LEVEL` cases a week.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from scipy.optimize import minimize, nnls

from nowcastbaselines.pooling import MIN_LEVEL, pool_matrix, source_index
from nowcastutils.base import ModelUnavailableError
from nowcastutils.shapes import round_up
from nowcastutils.snapshots import SnapshotArrays, weekly_sums

from ._checks import (
    assert_integerish,
    coerce,
    n_unobserved_rows,
    validate_reporting_triangle,
)
from .triangle import preprocess_negative_values

__all__ = [
    "PAIR_BUCKET",
    "DelayFit",
    "NoDelayPair",
    "Settings",
    "apply_delay",
    "delay_sums",
    "estimate_and_apply_delay",
    "estimate_delay",
    "final_growth",
    "fit_delay",
    "growth",
    "levels",
    "log_factors",
    "pair_sums",
    "weeks_back",
    "window_pairs",
]


@jax.jit
def _estimate_kernel(values: Array, mask: Array) -> Array:
    """Chain ladder fill and pmf (R .chainladder_fill_triangle, .calculate_pmf).

    ``rs`` holds the row sums of the filled columns to the left.
    The multiplication factor is the observed column total over the
    left row sums of the rows that observe the column.
    """
    zero = jnp.zeros((), values.dtype)

    def step(rs: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        col, m = x
        top = jnp.sum(jnp.where(m, col, zero))
        left = jnp.sum(jnp.where(m, rs, zero))
        mult = top / jnp.maximum(left, 1.0)
        filled = jnp.where(m, col, mult * rs)
        return rs + filled, filled

    _, rest = lax.scan(step, values[:, 0], (values[:, 1:].T, mask[:, 1:].T))
    colsum = jnp.concatenate([values[:, :1].sum(axis=0), rest.sum(axis=1)])
    return colsum / colsum.sum()


@jax.jit
def _apply_kernel(values: Array, mask: Array, pmf: Array) -> Array:
    """Fill unobserved cells (R .calc_expectation, apply_delay.R:160)."""
    cdf = jnp.cumsum(pmf)

    def step(
        rs: Array, x: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        col, m, p, cdf_prev = x
        expected = (rs + 1 - cdf_prev) / cdf_prev * p
        filled = jnp.where(m, col, expected)
        return rs + filled, filled

    xs = (values[:, 1:].T, mask[:, 1:].T, pmf[1:], cdf[:-1])
    _, rest = lax.scan(step, values[:, 0], xs)
    return jnp.concatenate([values[:, :1], rest.T], axis=1)


def _validate_for_delay_estimation(
    values: Array, mask: np.ndarray, n: int
) -> None:
    """R ``.validate_for_delay_estimation`` (estimate_delay.R:89)."""
    assert_integerish(n, "n")
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError(
            "Invalid reporting triangle structure. Each column must have "
            "at least one non-NA value."
        )
    n_rows = mask.shape[0]
    if n_rows < n:
        raise ValueError(
            f"Reporting triangle has {n_rows} reference times but n = {n} "
            "was requested."
        )
    if n < 1:
        raise ValueError(
            "Insufficient `n`, must be greater than or equal to 1."
        )
    last = slice(n_rows - n, n_rows)
    if not np.any(mask[last].all(axis=1)):
        raise ValueError(
            "The rows used for delay estimation in the reporting triangle "
            "must contain at least one row with no missing observations."
        )
    last_row = mask[-1]
    if not last_row[0]:
        raise ValueError(
            "The last row of the reporting triangle contains an NA in the "
            "first column. There must be at least an entry for the first "
            "delay at the latest reference time."
        )
    if not last_row.all() and not isinstance(values, jax.core.Tracer):
        first_na = int(np.argmin(last_row))
        lhs = np.asarray(values)[last][:, :first_na]
        if np.all(lhs == 0):
            raise ValueError(
                "The values for the recent reference times and delays only "
                "contain 0s, so the chain ladder fill and delay pmf would "
                "be invalid. Consider increasing `n`."
            )


def estimate_delay(
    triangle: Any,
    mask: Any,
    n: int | None = None,
    validate: bool = True,
) -> Array:
    """Estimate the delay pmf from the last ``n`` rows.

    R estimate_delay.R:51.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    n : int, optional
        Rows to use. Defaults to all rows.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        Probability of each delay, summing to 1.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if n is None:
        n = m.shape[0]
    assert_integerish(n, "n")
    _validate_for_delay_estimation(values, m, int(n))
    n = int(n)
    return _estimate_kernel(values[-n:], jnp.asarray(m[-n:]))


def _validate_delay_and_triangle(mask: np.ndarray, pmf: Array) -> None:
    """R ``.validate_delay_and_triangle`` (apply_delay.R:109).

    Checks on values are skipped under ``vmap`` and ``jit``, where
    they are not concrete.
    """
    if mask.shape[1] != pmf.shape[0]:
        raise ValueError(
            "Length of the delay PMF is not the same as the number of "
            "delays in the triangle to be nowcasted."
        )
    if not mask[:, 0].all():
        raise ValueError("The first delay must be observed in every row.")
    if isinstance(pmf, jax.core.Tracer):
        return
    if not mask[-1, 1] and float(pmf[0]) == 0:
        raise ValueError(
            "Value of delay PMF at delay = 0 is 0, and the latest reference "
            "time only contains a value at delay = 0. There is insufficient "
            "information to generate a point nowcast for the latest "
            "reference time."
        )
    if float(pmf[0]) < 0:
        raise ValueError(
            "First entry of delay PMF (delay = 0) is negative "
            f"({float(pmf[0])})."
        )


def apply_delay(
    triangle: Any,
    mask: Any,
    delay_pmf: Any,
    validate: bool = True,
) -> Array:
    """Fill the unobserved cells with expected counts (R apply_delay.R:43).

    Rows are independent, so only the rows with a missing cell are
    passed to the kernel.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    delay_pmf : array_like
        Probability of each delay, as from `estimate_delay`.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        The point nowcast matrix, with no missing cells.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    pmf = jnp.asarray(delay_pmf, dtype=values.dtype)
    _validate_delay_and_triangle(m, pmf)
    k = n_unobserved_rows(m)
    if k == 0:
        raise ValueError(
            "`reporting_triangle` doesn't contain any missing values, "
            "there is nothing to nowcast."
        )
    n_rows = m.shape[0]
    filled = _apply_kernel(
        values[n_rows - k :], jnp.asarray(m[n_rows - k :]), pmf
    )
    return jnp.concatenate([values[: n_rows - k], filled], axis=0)


def estimate_and_apply_delay(
    triangle: Any,
    mask: Any,
    n: int | None = None,
    delay_pmf: Any = None,
    validate: bool = True,
) -> Array:
    """Estimate the delay (unless given) and nowcast.

    R estimate_and_apply_delay.R:44.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    n : int, optional
        Rows used to estimate the delay. Defaults to all rows.
    delay_pmf : array_like, optional
        A delay to use instead of estimating one.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        The point nowcast matrix, as from `apply_delay`.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if n is None:
        n = m.shape[0]
    if n > m.shape[0]:
        raise ValueError(
            "The number of observations (rows) specified for delay "
            "estimation must be less than or equal to the number of rows "
            "of the reporting triangle."
        )
    if not np.any(m[m.shape[0] - n :].all(axis=1)):
        raise ValueError(
            "The rows used for delay estimation in the reporting triangle "
            "must contain at least one row with no missing observations."
        )
    if delay_pmf is None:
        delay_pmf = estimate_delay(values, m, n=n, validate=False)
    return apply_delay(values, m, delay_pmf, validate=False)


@partial(jax.jit, static_argnames="max_delay")
def _pair_sums(
    counts: Array,
    observed: Array,
    cut: Array,
    pairs: Array,
    max_delay: int,
) -> tuple[Array, Array]:
    n_day = counts.shape[-1]
    t = cut[pairs][:, None] - jnp.arange(max_delay)[None, :]
    inside = (t >= 0) & (t < n_day)
    tc = jnp.clip(t, 0, n_day - 1)
    r = pairs[:, None]
    both = inside & observed[r, tc] & observed[r + 1, tc]
    den = jnp.where(both, counts[:, r, tc], 0.0)
    num = jnp.where(both, counts[:, r + 1, tc], 0.0)
    return num, den


def pair_sums(
    counts: Any,
    observed: Any,
    cut: Any,
    pairs: Any,
    max_delay: int,
) -> tuple[Array, Array]:
    """Numerator and denominator per pair and age (extension E1).

    For pair ``p = (r, r + 1)`` and age ``x = 0 .. max_delay - 1`` at
    release r, the cell is day ``t = cut[r] - x``.
    ``den`` is its count in release r and ``num`` its count in
    release r + 1.
    Cells not observed in both releases, or outside the days, are
    skipped and add 0.

    Parameters
    ----------
    counts : array_like
        Counts by location, release and day.
    observed : array_like of bool
        True where a count is observed, the same shape.
    cut : array_like of int
        Each release's cut-off day.
    pairs : array_like of int
        First release index of each pair.
    max_delay : int
        Ages to use.

    Returns
    -------
    tuple of jax.Array
        ``(num, den)``, each ``(n_loc, n_pair, max_delay)``.
    """
    c = jnp.asarray(counts)
    return _pair_sums(
        c,
        jnp.asarray(observed, bool),
        jnp.asarray(cut, int),
        jnp.asarray(pairs, int),
        int(max_delay),
    )


def log_factors(num: Any, den: Any, gaps: Any) -> np.ndarray:
    r"""Daily log development factors shared by all pairs (extension E1).

    A day of age $a$ at release $r$ is age $a + s$ at release $r + 1$,
    with $s$ the pair's gap in days.
    Releases are batched reports, so its growth over the gap is
    $\exp x$, $x = \sum_{j=a}^{a+s-1} g(j)$, with $g(j) = 0$ from
    ``max_delay``.
    The pairs of each gap are summed by age, as baselinenowcast sums over
    reference times, to $N$ cases at the later release, of which $D$
    were shown at the earlier.
    Each case at the later release was shown at the earlier with chance
    $e^{-x}$, so $D$ is binomial, with log likelihood
    $-D x + (N - D) \log(1 - e^{-x})$.
    Where the days were shown as 0, $D = 0$, so 1 case counts as shown
    at both releases, as baselinenowcast's chain ladder divides by at
    least 1, and their new reports still count.
    $g \ge 0$ maximises the sum over gaps and ages, from the least
    squares fit of $\log(N / D)$ weighted by $D$.
    With a gap of 1 only, $g(a)$ is the log of the chain ladder factor
    $N / D$.

    Parameters
    ----------
    num, den : array_like
        ``(n_loc, n_pair, max_delay)``, from `pair_sums`.
    gaps : array_like of int
        Each pair's gap in days.

    Returns
    -------
    numpy.ndarray
        $g$, ``(n_loc, max_delay)``.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    gaps = np.asarray(gaps, int)
    n_loc, _, d = num.shape
    uniq = np.unique(gaps)
    member = (gaps[:, None] == uniq[None, :]).astype(float)
    n_sum = np.einsum("lpa,ps->lsa", num, member).reshape(n_loc, -1)
    d_sum = np.einsum("lpa,ps->lsa", den, member).reshape(n_loc, -1)
    age = np.arange(d)
    j = np.arange(d)[None, None, :]
    lo = age[None, :, None]
    rows = ((j >= lo) & (j < lo + uniq[:, None, None])).reshape(-1, d)
    rows = rows.astype(float)
    # A count that fell is no removal in the delay (`_delay_counts`).
    new = np.clip(n_sum - d_sum, 0.0, None)
    # baselinenowcast's chain ladder divides by at least 1 case.
    shown = np.maximum(d_sum, 1.0)
    out = np.zeros((n_loc, d))
    for n in range(n_loc):
        w = shown[n]
        y = np.log1p(new[n] / w)
        root = np.sqrt(w)
        start = nnls(rows * root[:, None], y * root)[0]
        out[n] = _binomial_fit(rows, w, new[n], start)
    return out


def _binomial_fit(
    rows: np.ndarray,
    den: np.ndarray,
    new: np.ndarray,
    start: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Maximise the binomial likelihood of `log_factors` over `g >= 0`.

    The likelihood is concave in `g`, and L-BFGS-B climbs it from the
    start, which it keeps when it is already the maximum, as with a gap of
    1 only.

    Parameters
    ----------
    rows : numpy.ndarray
        Which factors each equation sums, ``(n_eq, max_delay)``.
    den, new : numpy.ndarray
        Each equation's cases shown at the earlier release, and new by
        the later, ``(n_eq,)``.
    start : numpy.ndarray
        Starting factors, ``(max_delay,)``.
    tol : float
        Gradient per case to stop at.

    Returns
    -------
    numpy.ndarray
        $g$, ``(max_delay,)``.
    """
    scale = max(float(np.sum(den + new)), 1.0)

    def loss(g: np.ndarray) -> tuple[float, np.ndarray]:
        # The likelihood is 0 at x = 0 where a case is still to come, so
        # x has a floor that keeps the search's trial points finite.
        x = np.maximum(rows @ g, 1e-12)
        ll = -den * x + new * np.log(-np.expm1(-x))
        grad = rows.T @ (den - new / np.expm1(x))
        return -float(ll.sum()) / scale, grad / scale

    fit = minimize(
        loss,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * start.size,
        options={"ftol": 0.0, "gtol": tol, "maxiter": 10_000},
    )
    out: np.ndarray = fit.x
    return out


def growth(g: Any, start: Any, stop: Any) -> np.ndarray:
    r"""Growth $\exp \sum_{j=\mathrm{start}}^{\mathrm{stop}-1} g(j)$.

    Parameters
    ----------
    g : array_like
        Daily log factors, ``(n_loc, max_delay)``, from `log_factors`.
    start, stop : array_like of int
        Ages that broadcast together, clipped to ``0 .. max_delay``, so
        growth from ``max_delay`` on is 1.

    Returns
    -------
    numpy.ndarray
        ``(n_loc, *shape)``.
    """
    g = np.asarray(g, float)
    d = g.shape[1]
    total = np.concatenate([np.zeros((g.shape[0], 1)), np.cumsum(g, 1)], 1)
    a, b = np.broadcast_arrays(
        np.clip(np.asarray(start, int), 0, d),
        np.clip(np.asarray(stop, int), 0, d),
    )
    return np.exp(total[:, b] - total[:, a])


def window_pairs(cut: Any, max_delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs whose step is at most ``max_delay``, and their steps.

    A longer step has no age inside the window to attach a factor to.
    Its releases stay in the data and in the counts.

    Parameters
    ----------
    cut : array_like of int
        Each release's cut-off day.
    max_delay : int
        Longest step kept, in days.

    Returns
    -------
    tuple of numpy.ndarray
        The first release index and the step of each pair.
    """
    steps = np.diff(np.asarray(cut, dtype=int))
    keep = np.flatnonzero((steps >= 1) & (steps <= max_delay))
    return keep, steps[keep]


class NoDelayPair(ModelUnavailableError):  # noqa: N818
    """No pair of releases has a step within the delay window."""


@dataclass(frozen=True)
class Settings:
    """Settings shared by the random walk models.

    Attributes
    ----------
    max_delay : int
        $D$, the days a count may still grow, baselinenowcast's
        `max_delay`.
    min_level : float
        Cases a week for a place to use its own delay, walk spread and
        nowcast size (`pooling`).
    level_weeks : tuple of int
        First and last weeks back averaged for a place's level.
    completeness : float
        Share of a week's final count that must be reported for it to
        start the random walk.
    walk_weeks : int
        Weeks, ending at the start week, whose changes set the spread.
    start_share : float
        Share of a week reported for `randomwalknowcast` to start
        its walk from the week's nowcast.
    start_count : float
        Cases reported in a week for `randomwalknowcast` to start its walk
        from the week's nowcast.
    """

    max_delay: int = 42
    min_level: float = MIN_LEVEL
    level_weeks: tuple[int, int] = (3, 6)
    completeness: float = 0.95
    walk_weeks: int = 9
    start_share: float = 0.05
    start_count: float = 5.0

    @property
    def history_delay(self) -> int:
        """Onset days used for the delay, baselinenowcast's default.

        R uses `floor(0.5 * round(3 * D))` reference times for the delay,
        at least `D + 1`, and the rest for the uncertainty.
        """
        n_used = round(3 * self.max_delay)
        return max(n_used // 2, self.max_delay + 1)

    @property
    def retro_delay(self) -> int:
        """Days back from the origin's cut-off for retrospective nowcasts.

        R gives the `round(3 * D)` reference times left after
        `history_delay` to its retrospective nowcasts, so a release whose
        cut-off is more than this many days before the origin's gives
        none.
        """
        return round(3 * self.max_delay) - self.history_delay


@dataclass
class DelayFit:
    """The daily delay at a release, as each place uses it.

    Attributes
    ----------
    pairs : numpy.ndarray
        Earlier and later release of each in-window pair, `[P, 2]`.
    g : numpy.ndarray
        Daily log development factor $g(a)$ of each place, `[L, D]`, the
        own or pooled estimate of `source`.
    level : numpy.ndarray
        Cases a week of each place, `[L]`.
    source : numpy.ndarray
        Index of the delay each place uses, `[L]`, as for
        `source_index`.
    own : numpy.ndarray
        Whether each place uses its own factor, `[L, D]`. The choice is
        made per place (`source`), so it is the same at every age.
    """

    pairs: np.ndarray
    g: np.ndarray
    level: np.ndarray
    source: np.ndarray
    own: np.ndarray


def weeks_back(snap: SnapshotArrays, r: int, max_delay: int) -> int:
    r"""Return the number of weeks back from the cut-off week with young days.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index.
    max_delay : int
        $D$.

    Returns
    -------
    int
        $K$: weeks back $k = 0, \dots, K - 1$ hold a day younger than
        $D$.
    """
    youngest = max(int(snap.cut[r]) - max_delay + 1, 0)
    return snap.cut_week(r) - int(snap.week[youngest]) + 1


def levels(snap: SnapshotArrays, r: int, settings: Settings) -> np.ndarray:
    """Return mean weekly counts over the level weeks at release `r`."""
    weekly = weekly_sums(snap, r)
    wa = snap.cut_week(r)
    lo, hi = settings.level_weeks
    cols = [wa - k for k in range(lo, hi + 1)]
    vals = [weekly[:, c] if c >= 0 else np.zeros(weekly.shape[0]) for c in cols]
    return np.mean(vals, axis=0)


def _pairs_at(
    snap: SnapshotArrays, r: int, max_delay: int, longer: bool = False
) -> np.ndarray:
    """Return the in-window pairs among releases up to `r`.

    With `longer` and no in-window pair, every pair with a longer step.
    """
    keep, _ = window_pairs(snap.cut[: r + 1], max_delay)
    if longer and keep.size == 0:
        keep = np.flatnonzero(np.diff(np.asarray(snap.cut[: r + 1])) >= 1)
    return np.stack([keep, keep + 1], axis=1).astype(np.int64)


PAIR_BUCKET = 16
"""The pairs a delay fit sums are padded to a multiple of this."""


def _delay_counts(snap: SnapshotArrays, r: int) -> np.ndarray:
    """Return counts with downward revisions moved onto earlier releases.

    For each day the increments between its observed releases up to `r`
    go through baselinenowcast's `preprocess_negative_values`, so a
    negative revision is set to zero and taken from the earlier
    increments. The cumulative counts are then rebuilt, and never fall
    from one release to the next. Releases after `r` are left as they
    are, and are not read by a fit at `r`.

    Every release's increments are passed, those after `r` unobserved,
    so the compiled scan sees one shape for every `r`: an unobserved
    cell is zero and passes no amount on, so the releases up to `r` are
    as they would be alone.
    """
    c = np.asarray(snap.counts, dtype=float)
    n_loc, n_rel, n_day = c.shape
    obs = np.asarray(snap.observed).copy()
    obs[r + 1 :] = False
    prev = np.concatenate(
        [np.zeros_like(c[:, :1]), c[:, :-1] * obs[None, :-1]], axis=1
    )
    inc = np.where(obs[None], c - prev, 0.0)
    rows = inc.transpose(0, 2, 1).reshape(n_loc * n_day, n_rel)
    mask = np.tile(obs.T, (n_loc, 1))
    fixed = np.asarray(preprocess_negative_values(rows, mask, validate=False))
    fixed = fixed.reshape(n_loc, n_day, n_rel).transpose(0, 2, 1)[:, : r + 1]
    out = c.copy()
    out[:, : r + 1] = np.where(
        obs[None, : r + 1], np.cumsum(fixed, axis=1), c[:, : r + 1]
    )
    return out


def delay_sums(
    snap: SnapshotArrays, r: int, settings: Settings, longer: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the pair sums the delay is fitted to at release `r`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index. Pairs of later releases are not used.
    settings : Settings
        Settings.
    longer : bool
        With no in-window pair, use the pairs with a longer step, on the
        onset days of every age at their first release.

    Returns
    -------
    tuple of numpy.ndarray or None
        The pairs `[P, 2]`, `num` and `den` from `pair_sums` `[L, P, D]`,
        and each pair's gap `[P]`, or None with no pair.
    """
    pairs = _pairs_at(snap, r, settings.max_delay, longer)
    if pairs.size == 0:
        return None
    counts = _delay_counts(snap, r)
    day = np.arange(snap.week.size)
    first = int(snap.cut[r]) - settings.history_delay
    gaps = snap.cut[pairs[:, 1]] - snap.cut[pairs[:, 0]]
    if (gaps > settings.max_delay).all():
        # Every age is final at the second release, so reach back
        # `max_delay` days before the first.
        oldest = int(snap.cut[pairs[:, 0]].min()) - settings.max_delay
        first = min(first, oldest)
    window = day > first
    observed = snap.observed & window[None, :]
    # The pairs are padded to a multiple of PAIR_BUCKET with copies of the
    # first, and their sums cut off after, so the compiled sums see a few
    # shapes rather than one a release.
    n = len(pairs)
    first_of = np.full(round_up(n, PAIR_BUCKET), pairs[0, 0], dtype=np.int64)
    first_of[:n] = pairs[:, 0]
    num, den = pair_sums(
        jnp.asarray(counts),
        jnp.asarray(observed),
        jnp.asarray(snap.cut),
        jnp.asarray(first_of),
        settings.max_delay,
    )
    return pairs, np.asarray(num)[:, :n], np.asarray(den)[:, :n], gaps


def fit_delay(
    snap: SnapshotArrays, r: int, settings: Settings, longer: bool = False
) -> DelayFit | None:
    """Estimate the delay factors from the pairs up to release `r`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index. Pairs of later releases are not used.
    settings : Settings
        Settings.
    longer : bool
        With no in-window pair, fit the pairs with a longer step
        (`delay_sums`).

    Returns
    -------
    DelayFit or None
        The factors, or None with no pair to fit.
    """
    sums = delay_sums(snap, r, settings, longer)
    if sums is None:
        return None
    pairs, num, den, gaps = sums
    level = levels(snap, r, settings)
    pool = pool_matrix(level, snap.parent, settings.min_level)
    own_g = log_factors(num, den, gaps)
    pooled_g = log_factors(
        np.einsum("nm,mpa->npa", pool, num),
        np.einsum("nm,mpa->npa", pool, den),
        gaps,
    )
    source = source_index(level, snap.parent, min_level=settings.min_level)
    g = np.concatenate([own_g, pooled_g])[source]
    own = np.broadcast_to((source == np.arange(level.size))[:, None], g.shape)
    return DelayFit(pairs, g, level, source, np.array(own))


def final_growth(
    snap: SnapshotArrays, o: int, fit: DelayFit, settings: Settings
) -> np.ndarray:
    """Return the growth to final of each place and day at release `o`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    fit : DelayFit
        The delay at the origin.
    settings : Settings
        Settings.

    Returns
    -------
    numpy.ndarray
        $G(a)$ at each day's age, `[L, T]`.
    """
    age = snap.cut[o] - np.arange(snap.week.size)
    return growth(fit.g, age, settings.max_delay)
