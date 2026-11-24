"""Reporting triangles: structure, preprocessing and allocation.

As baselinenowcast's: negative reports moved onto earlier ones
(`preprocess_negative_values`), the reporting structure, and the reference
times given to the delay and to the retrospective nowcasts
(`allocate_reference_times`).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, lax

from ._checks import (
    assert_integerish,
    coerce,
    n_unobserved_rows,
    validate_reporting_triangle,
)

__all__ = [
    "Triangle",
    "allocate_reference_times",
    "apply_reporting_structure",
    "daily_triangle",
    "get_reporting_structure",
    "preprocess_negative_values",
    "truncate_to_rows",
]


Triangle = tuple[Array, np.ndarray]
"""A reporting triangle and its mask: values, and True where observed."""


@jax.jit
def _preprocess(values: Array, mask: Array) -> Array:
    zero = jnp.zeros((), values.dtype)

    def row(v: Array, m: Array) -> Array:
        def step(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            val, ok = x
            shifted = val + carry
            neg = shifted < 0
            out = jnp.where(ok & ~neg, shifted, zero)
            new_carry = jnp.where(ok, jnp.where(neg, shifted, zero), carry)
            return new_carry, out

        return lax.scan(step, zero, (v, m), reverse=True)[1]

    return jnp.trunc(jax.vmap(row)(values, mask))


def preprocess_negative_values(
    triangle: Any, mask: Any, validate: bool = True
) -> Array:
    """Move negative reports onto earlier delays (R preprocess.R:47).

    Scanning each row from the last delay back, a negative value is set
    to zero and its amount is taken from the next earlier observed
    cell.
    Every cell is truncated to an integer.
    R truncates only the first ``ncol`` cells in column-major order
    because of an upstream indexing slip.
    Non-integer inputs therefore differ from R.
    Unobserved cells stay zero.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        The triangle with no negative cells.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    return _preprocess(values, jnp.asarray(m))


def truncate_to_rows(
    triangle: Any,
    mask: Any,
    n: int | None = None,
    validate: bool = True,
) -> list[Triangle]:
    """Drop 1..n rows from the bottom (R truncate.R:182).

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    n : int, optional
        Most rows to drop. Defaults to the number of rows without a
        missing cell minus one.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    list of Triangle
        Element ``t - 1`` has its last ``t`` rows removed.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if n is None:
        n = m.shape[0] - n_unobserved_rows(m) - 1
    assert_integerish(n, "n", lower=0)
    out: list[Triangle] = []
    for t in range(1, int(n) + 1):
        n_obs = m.shape[0]
        if t >= n_obs:
            raise ValueError(
                "The as of time point is greater than or equal to the "
                "number of rows in the original triangle."
            )
        out.append((values[: n_obs - t], m[: n_obs - t]))
    return out


def _expand_structure_vec(structure: float, cols: int) -> np.ndarray:
    """R ``.expand_structure_vec`` (apply_reporting_structure.R:123)."""
    if structure <= 0:
        raise ValueError("Structure must be positive")
    if structure > cols:
        raise ValueError("Structure cannot be larger than number of columns")
    adjusted = cols - 1
    n_groups = adjusted // int(structure)
    vec = [structure] * n_groups
    remainder = adjusted % int(structure)
    if remainder > 0:
        vec.append(remainder)
    return np.asarray(vec, dtype=float)


def apply_reporting_structure(
    triangle: Any,
    mask: Any,
    structure: Any = 1,
    validate: bool = True,
) -> np.ndarray:
    """Return the mask with a reporting structure applied.

    R apply_reporting_structure.R:79.
    The last row keeps ``structure[0]`` delays, the next row up keeps
    ``structure[0] + structure[1]``, and so on.
    A scalar structure is expanded to groups of that many delays.
    Values under the new False cells are zero in
    ``triangle * mask``.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    structure : int or array_like of int, default 1
        Delays added per row, from the last row up.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    numpy.ndarray of bool
        The new mask.
    """
    _, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    rows, cols = m.shape
    s = np.atleast_1d(np.asarray(structure, dtype=float))
    if s.size == 1:
        vec = _expand_structure_vec(float(s[0]), cols)
    else:
        vec = s
        if np.any(vec < 0) or not np.all(np.isfinite(vec)):
            raise ValueError("All elements of `structure` must be positive")
        if vec.sum() >= cols:
            raise ValueError(
                "Sum of structure vector must not be greater than or equal "
                "to the number of columns in matrix"
            )
    if vec.size > rows:
        raise ValueError("`structure` has more groups than the triangle rows")
    cutoff = np.cumsum(vec).astype(int) + 1
    out = m.copy()
    for i in range(vec.size):
        out[rows - i - 1, cutoff[i] - 1 :] = False
    return out


def get_reporting_structure(mask: Any) -> float | np.ndarray:
    """Infer the reporting structure of a triangle.

    R reporting_triangle-getters.R:331.

    Parameters
    ----------
    mask : array_like of bool
        True where a cell is observed.

    Returns
    -------
    float or numpy.ndarray
        0 when nothing is missing, a scalar when the structure is a
        regular grouping and a vector otherwise.
    """
    m = np.asarray(mask, dtype=bool)
    n_row_nas = n_unobserved_rows(m)
    if n_row_nas == 0:
        return 0.0
    n_prev = 0.0
    structure = []
    for i in range(1, n_row_nas + 1):
        n_nas = float(m[m.shape[0] - i].sum()) - n_prev
        structure.append(n_nas)
        n_prev += n_nas
    struct = np.asarray(structure)
    expanded = _expand_structure_vec(struct[0], m.shape[1])
    if expanded.shape == struct.shape and np.array_equal(expanded, struct):
        return float(struct[0])
    return struct


def _validate_allocation(
    scale_factor: float, prop_delay: float, n_min_retro_nowcasts: int
) -> None:
    """R ``.validate_inputs_allocation`` (allocate_reference_times.R:88)."""
    assert_integerish(n_min_retro_nowcasts, "n_min_retro_nowcasts", lower=0)
    if not (np.isfinite(prop_delay) and 0 <= prop_delay <= 1):
        raise ValueError("`prop_delay` must be between 0 and 1")
    if not (np.isfinite(scale_factor) and scale_factor >= 0):
        raise ValueError("`scale_factor` must be finite and non-negative")


def allocate_reference_times(
    mask: Any,
    scale_factor: float = 3,
    prop_delay: float = 0.5,
    n_min_retro_nowcasts: int = 2,
    validate: bool = True,
) -> tuple[int, int]:
    """Split reference times between delay and uncertainty estimation.

    R allocate_reference_times.R:57.

    Parameters
    ----------
    mask : array_like of bool
        True where a cell is observed.
    scale_factor : float, default 3
        Reference times to use, as a multiple of the maximum delay.
    prop_delay : float, default 0.5
        Share of them used to estimate the delay.
    n_min_retro_nowcasts : int, default 2
        Fewest retrospective nowcasts to leave for the uncertainty.
    validate : bool, default True
        Check the mask first.

    Returns
    -------
    tuple of int
        ``(n_history_delay, n_retrospective_nowcasts)``.
    """
    m = np.asarray(mask, dtype=bool)
    if validate:
        validate_reporting_triangle(m)
    max_delay = m.shape[1] - 1
    _validate_allocation(scale_factor, prop_delay, n_min_retro_nowcasts)

    n_ref_times = m.shape[0]
    n_min_delay = n_unobserved_rows(m) + 1
    n_required = n_min_delay + n_min_retro_nowcasts
    n_target = round(scale_factor * max_delay)

    # .check_against_requirements (allocate_reference_times.R:189)
    if n_target <= n_ref_times and n_target >= n_required:
        n_used = n_target
    elif n_target > n_ref_times:
        if n_ref_times >= n_required:
            warnings.warn(
                f"{n_ref_times} reference times available and {n_target} "
                f"are specified. All {n_ref_times} will be used.",
                stacklevel=2,
            )
            n_used = n_ref_times
        else:
            raise ValueError(
                f"{n_ref_times} reference times available and {n_required} "
                f"are needed, {n_min_delay} for delay estimation and "
                f"{n_min_retro_nowcasts} for uncertainty estimation."
            )
    else:
        raise ValueError(
            f"{scale_factor * max_delay} reference times specified and "
            f"{n_required} are needed, {n_min_delay} for delay estimation "
            f"and {n_min_retro_nowcasts} for uncertainty estimation."
        )

    # .assign_allocation_from_ns (allocate_reference_times.R:278)
    if n_used >= n_target:
        n_history_delay = max(math.floor(n_used * prop_delay), n_min_delay)
        n_retro = n_used - n_history_delay
    else:
        n_remaining = n_used - n_min_delay
        n_retro = math.ceil(n_remaining * (1 - prop_delay))
        n_history_delay = n_used - n_retro
    if n_retro < n_min_retro_nowcasts:
        n_retro = n_min_retro_nowcasts
        n_history_delay = n_used - n_retro
    return int(n_history_delay), int(n_retro)


def daily_triangle(
    history: pd.DataFrame, place: str, cutoff: dt.date, rows: int, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a place's daily reporting triangle at `cutoff`.

    Parameters
    ----------
    history : pandas.DataFrame
        Daily releases (`ForecastTask.daily_history`): `vintage`,
        `location`, `onset_date` and `observed`, the count of the onset
        day reported by the vintage.
    place : str
        The place.
    cutoff : datetime.date
        The latest release.
    rows : int
        Reference days, the last one the cut-off.
    delay : int
        Longest delay, $D$.

    Returns
    -------
    tuple of numpy.ndarray
        Counts first reported at each delay, `[rows, D + 1]`, and the mask
        of the cells reported by the cut-off.
    """
    own = history.loc[history["location"] == place]
    first = cutoff - dt.timedelta(days=rows - 1)
    days = pd.date_range(first, cutoff).strftime("%Y-%m-%d")
    vintages = pd.date_range(first, cutoff).strftime("%Y-%m-%d")
    cum = (
        own.pivot_table(
            index="onset_date",
            columns="vintage",
            values="observed",
            aggfunc="sum",
        )
        .reindex(index=days, columns=vintages)
        .ffill(axis=1)
        .fillna(0.0)
        .to_numpy()
    )
    t = np.arange(rows)[:, None]
    d = np.arange(delay + 1)[None, :]
    col = t + d
    mask = col <= rows - 1
    seen = np.take_along_axis(cum, np.minimum(col, rows - 1), axis=1)
    seen = np.where(mask, seen, 0.0)
    counts = np.diff(seen, axis=1, prepend=0.0)
    return np.where(mask, counts, 0.0), mask
