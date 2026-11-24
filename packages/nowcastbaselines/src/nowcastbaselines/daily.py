# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""The random walk nowcast on a daily reporting triangle.

`DailyWalkNowcast` (`randomwalknowcast_daily`) nowcasts each place's
daily counts with baselinenowcast (`baselinenowcast.baselinenowcast`) on
its daily triangle, reference days by delays $0, \dots, D$, from the
task's daily releases (`ForecastTask.daily_history`), with real dates.
The triangle holds the `round(scale_factor * D)` reference days to the
cut-off, the rows baselinenowcast uses, and at least the days back to the
oldest target's sum. $D$ is `max_delay`, capped by the history available
so those rows hold data: days before the first onset in the releases are
not filled in.

It gives two views of one nowcast, each with its own error model, as
baselinenowcast does with an aggregator:

- 7-day sums (`forecast`): the retrospective errors are fitted on 7-day
  rolling sums of reference days (`triangle.rolling_sum`, as R
  baselinenowcast's `ref_time_aggregator`), and the draws are 7-day sums
  ending on each day. As with R's rolling aggregator, a window's delay
  cell counts as reported only when every day of it is, so a partly
  reported cell comes from the draw and a 7-day draw can fall a little
  below the sum reported.
- Daily counts (`daily_quantiles`): the same triangle, errors fitted on
  daily counts.

Days after the cut-off are a random walk on $\log(1 + y)$ with daily steps
(`walk.random_walks`) on the same view: each sample walks on from its own
nowcast of the cut-off day. Its spread is the standard deviation of the
daily changes of $\log(1 + y)$ in the point nowcast over the last
`walk_days` days, net of their Poisson noise (`walk.walk_variance`); a
place whose net variance is 0 or below does not walk. German counts have
weekday effects, which the daily view's spread holds.

National is the sum of the places by sample.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from nowcastbaselines.baselinenowcast import baselinenowcast, rolling_sum
from nowcastbaselines.baselinenowcast.triangle import identity
from nowcastbaselines.walk import random_walks, walk_noise, walk_variance
from nowcastutils.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    MissingSnapshots,
    register_model,
)
from nowcastutils.dates import as_date

__all__ = ["DailyWalkNowcast", "daily_triangle"]


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


@dataclass
class DailyWalkNowcast:
    """baselinenowcast on the daily triangle, then a daily random walk.

    Attributes
    ----------
    name : str
        Model name.
    max_delay : int
        Longest delay $D$ in days.
    scale_factor : float
        Reference days per delay day (baselinenowcast's).
    walk_days : int
        Days of point nowcast the walk spread is estimated on.
    source : str
        "daily": it forecasts every day, not hub weeks.
    """

    name: str = "randomwalknowcast_daily"
    source: str = "daily"
    max_delay: int = 40
    scale_factor: float = 3
    walk_days: int = 56

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return draws of the 7-day sums ending on each target date."""
        return self._samples(task, 7)

    def daily_quantiles(
        self, task: ForecastTask, levels: tuple[float, ...]
    ) -> pd.DataFrame:
        """Return quantiles of each target date's count, by place.

        Columns `location`, `date`, `quantile` and `value`.
        """
        draws = self._samples(task, 1)
        q = (
            draws.groupby(["location", "target_end_date"])["value"]
            .quantile(np.asarray(levels))
            .rename("value")
            .reset_index()
        )
        return q.rename(
            columns={"target_end_date": "date", "level_2": "quantile"}
        )[["location", "date", "quantile", "value"]]

    def _samples(self, task: ForecastTask, k: int) -> pd.DataFrame:
        """Return draws of the `k`-day sums ending on each target date."""
        if task.daily_history is None:
            raise MissingSnapshots(f"{self.name} needs daily releases")
        cutoff = as_date(task.cutoff)
        first = as_date(str(task.daily_history["onset_date"].min()))
        have = (cutoff - first).days + 1
        delay = min(self.max_delay, int(have // self.scale_factor))
        if delay < 3:
            raise MissingSnapshots(f"{self.name}: {have} days of data")
        ends = sorted(task.target_end_dates)
        ahead = max(0, (as_date(ends[-1]) - cutoff).days)
        # The rows reach the oldest target's k days, even where the lag
        # window is capped by a short history.
        back = (cutoff - as_date(ends[0])).days + k
        if back > have:
            raise MissingSnapshots(
                f"{self.name}: {have} days of data, {back} needed"
            )
        rows = max(round(self.scale_factor * delay), back)
        agg = identity if k == 1 else partial(rolling_sum, k=k)
        places = list(task.locations)
        keys = jax.random.split(jax.random.key(task.seed), 2 * len(places))
        nowcast, sigma = [], []
        for i, place in enumerate(places):
            tri, mask = daily_triangle(
                task.daily_history, place, cutoff, rows, delay
            )
            draws = baselinenowcast(
                tri,
                mask,
                key=keys[i],
                draws=task.n_samples,
                ref_time_aggregator=agg,
            )
            nowcast.append(np.asarray(draws, np.float64))
            point = np.asarray(
                baselinenowcast(tri, mask, output_type="point"), np.float64
            )
            y = np.convolve(point, np.ones(k), "valid")[-self.walk_days :]
            var = walk_variance(np.diff(np.log1p(y)), walk_noise(y))
            sigma.append(np.sqrt(var) if var > 0 else 0.0)
        x = np.stack(nowcast)  # [L, S, rows - k + 1], last is the cut-off
        if ahead:
            _, walk = random_walks(
                keys[len(places) :],
                jnp.asarray(x[:, :, -1], jnp.float32),
                jnp.asarray(sigma, jnp.float32),
                ahead,
            )
            x = np.concatenate([x, np.asarray(walk, np.float64)], axis=2)
        last = cutoff + dt.timedelta(days=ahead)
        span = pd.date_range(end=last, periods=x.shape[2]).strftime("%Y-%m-%d")
        at = {d: i for i, d in enumerate(span)}
        cols = [at[d] for d in ends]
        values = x[:, :, cols]
        values = np.concatenate([values.sum(axis=0, keepdims=True), values])
        names = [NATIONAL, *places]
        n_loc, n_s, n_d = values.shape
        return pd.DataFrame(
            {
                "location": np.repeat(names, n_s * n_d),
                "target_end_date": np.tile(ends, n_loc * n_s),
                "sample": np.tile(np.repeat(np.arange(1, n_s + 1), n_d), n_loc),
                "value": values.reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]


register_model(
    "randomwalknowcast_daily",
    DailyWalkNowcast,
    "baselinenowcast on the daily triangle with a daily random walk; "
    "7-day sums and daily counts each with their own error model.",
)
