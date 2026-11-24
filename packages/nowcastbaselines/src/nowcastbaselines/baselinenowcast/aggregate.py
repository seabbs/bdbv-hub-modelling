"""Aggregating reference times: rolling sums and hub weeks.

baselinenowcast's reference time aggregators (`identity`, `rolling_sum`,
the 7-day rolling sum) and delay sums (`row_sums`), and the hub's weeks of
daily onsets (`hub_weeks`, `hub_week_matrix`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "Aggregator",
    "hub_week_matrix",
    "hub_weeks",
    "identity",
    "rolling_sum",
    "row_sums",
]


Aggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]
"""Aggregates reference times: `(values, mask)` to `(values, mask)`."""


def hub_week_matrix(
    first_onset: Any, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrix of hub weeks (weeks by days) and the Saturday labels.

    Hub weeks run Sunday to Saturday and carry the Saturday's date.
    A first week that starts after its Sunday keeps the days present.
    No onset day is dropped.
    The last week keeps only the days present, so the cut-off week is
    partial unless the last day is a Saturday.

    Parameters
    ----------
    first_onset : datetime.date or str
        Date of the first day.
    n_days : int
        Number of days.

    Returns
    -------
    tuple of numpy.ndarray
        The matrix, weeks by days, and the weeks' Saturdays.
    """
    start = np.datetime64(first_onset, "D")
    days = start + np.arange(n_days)
    # numpy weekday: 1970-01-01 was a Thursday; Sunday = 0 below.
    dow = (days.astype(int) + 4) % 7
    sat = days + (6 - dow)
    saturdays = np.unique(sat)
    w = (sat[None, :] == saturdays[:, None]).astype(float)
    return w, saturdays


def hub_weeks(first_onset: Any) -> Aggregator:
    """Build an aggregator summing onset days into hub weeks.

    The returned callable has the ``ref_time_aggregator`` signature.
    It builds the week matrix from the number of rows of the matrix it
    is given, whose last row is that matrix's cut-off day.
    An aggregated cell is unobserved if any of its inputs is.

    Parameters
    ----------
    first_onset : datetime.date or str
        Date of the first onset day.

    Returns
    -------
    Aggregator
        The aggregator.
    """

    def aggregate(values: Array, mask: np.ndarray) -> tuple[Array, np.ndarray]:
        v = jnp.asarray(values)
        m = np.asarray(mask, dtype=bool)
        w, _ = hub_week_matrix(first_onset, v.shape[0])
        wj = jnp.asarray(w, dtype=v.dtype)
        out = wj @ jnp.where(jnp.asarray(m), v, 0.0)
        ok = (w @ (~m).astype(float)) == 0
        return jnp.where(jnp.asarray(ok), out, 0.0), ok

    return aggregate


def rolling_sum(
    values: Array, mask: Array | np.ndarray, k: int
) -> tuple[Array, np.ndarray]:
    """Trailing sum over ``k`` rows, as ``zoo::rollsum(align = "right")``.

    The first ``k - 1`` rows are dropped.
    A window with any unobserved cell is unobserved.
    Use ``functools.partial(rolling_sum, k=7)`` as a
    ``ref_time_aggregator``.

    Parameters
    ----------
    values : jax.Array
        Matrix with reference times as rows.
    mask : jax.Array or numpy.ndarray of bool
        True where a cell is observed.
    k : int
        Window length in rows.

    Returns
    -------
    tuple
        The summed values and their mask, each with ``k - 1`` fewer rows.
    """
    v = jnp.asarray(values)
    m = np.asarray(mask, dtype=bool)
    n = v.shape[0]
    if k < 1 or k > n:
        raise ValueError("`k` must be between 1 and the number of rows")
    vm = jnp.where(jnp.asarray(m), v, 0.0)
    out = sum(vm[i : n - k + 1 + i] for i in range(k))
    ok = np.stack([m[i : n - k + 1 + i] for i in range(k)]).all(axis=0)
    return jnp.where(jnp.asarray(ok), out, 0.0), ok


def identity(
    values: Array, mask: Array | np.ndarray
) -> tuple[Array, np.ndarray]:
    """Return the input unchanged, the default ``ref_time_aggregator``.

    Parameters
    ----------
    values : jax.Array
        Matrix with reference times as rows.
    mask : jax.Array or numpy.ndarray of bool
        True where a cell is observed.

    Returns
    -------
    tuple
        ``(values, mask)``.
    """
    return jnp.asarray(values), np.asarray(mask, dtype=bool)


def row_sums(x: Array) -> Array:
    """Row sums, the default ``delay_aggregator``.

    Unobserved cells are zero so this equals ``rowSums(na.rm = TRUE)``.

    Parameters
    ----------
    x : jax.Array
        Matrix with reference times as rows.

    Returns
    -------
    jax.Array
        One sum per row.
    """
    return jnp.sum(jnp.asarray(x), axis=1)
