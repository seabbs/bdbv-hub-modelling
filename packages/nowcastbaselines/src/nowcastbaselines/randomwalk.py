# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""The random walk baselines.

Both models run on the hub's daily release snapshots (`chain`), from one
random walk (`walk.random_walk`).
The places form a hierarchy read from their codes
(`nowcastutils.hierarchy`), so a task may be one series or places of any
depth below a root, such as provinces and zones or age groups. A place
with no rows of its own is the sum of its children
(`nowcastutils.snapshots.snapshot_arrays`), and a task with no places
gives no samples.

`RandomWalk` (`randomwalk`) starts from the reported count of the latest
week judged complete, $\pi \ge 0.95$, where

$$
\pi(w) = \frac{1}{7} \sum_{t \in w} \frac{\mathbb{1}\{t \le c\}}{G(a(t))}
$$

is the share of the week's final count reported, under the delay
estimate of `chain`.
Weeks up to the start are returned as reported.
Later weeks are walk draws.

`RandomWalkNowcast` (`randomwalknowcast`) nowcasts each hub week, with
negative binomial new reports on the remaining count (`obs`).
Weeks with data keep their nowcast draws. If the week holding the cut-off
is partial, its days after the cut-off add a Poisson draw from that week's
walk rate, walked by each sample from its own nowcast of the latest full
week at least `Settings.start_share` reported with at least
`Settings.start_count` cases (`_start_weeks`). Each sample's completed
cut-off week, its nowcast plus that draw, is the base the walk goes on
from for the weeks after the cut-off.

The spread of the walk is the standard deviation of the weekly changes of
$\log(1 + y)$ in reported counts, over the weeks ending at the start of
`randomwalk`, net of their delta-method Poisson noise
(`walk.walk_variance`).
A place whose net variance is 0 or below has no own estimate.
The delay, the observation model and the walk spread each use their own
estimate where a place has at least 10 cases a week, and otherwise the
pooled estimate of the nearest parent that does (`pooling`).
A place with no case in the latest `walk_weeks` weeks is silent. It is
exactly 0 until its first introduction, which comes each week with its
pooled probability times the share of the week not yet reported, and then
walks from a count drawn from Poisson(1) given at least 1
(`introductions.silent_trajectories`).
In `randomwalknowcast` a place with no week to start from is quiet
(`_quiet`): it starts at its latest complete week. A silent quiet place
draws introductions from there. One with recent cases walks from its mean
reported count over the `start_window` weeks to that week (`_level`).
National is the sum of provinces by sample.
Each model raises if the task has no snapshots or no pair of releases
within the delay window.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass, field, replace
from typing import ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from nowcastbaselines.baselinenowcast.delays import (
    DelayFit,
    Settings,
    fit_delay,
)
from nowcastbaselines.baselinenowcast.nowcast import (
    NoCompleteWeek,
    Origin,
    _add_spread,
    build_origin,
    delay_summary,
)
from nowcastbaselines.baselinenowcast.uncertainty import sample_count
from nowcastbaselines.copula import (
    CopulaSettings,
    error_scores,
    estimate_rho,
    pair_walks,
)
from nowcastbaselines.introductions import silent_trajectories
from nowcastbaselines.walk import random_walks
from nowcastutils.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    MissingSnapshots,
    register_model,
)
from nowcastutils.dates import as_date
from nowcastutils.hierarchy import depth
from nowcastutils.snapshots import SnapshotArrays, blocks, snapshot_arrays

__all__ = ["RandomWalk", "RandomWalkNowcast"]

logger = logging.getLogger(__name__)


_KEPT_ORIGINS: Final = 4
"""Origins whose compiled programs are kept in memory at once."""

_ORIGINS: set[tuple[object, ...]] = set()


def _clear_caches(task: ForecastTask) -> None:
    """Drop JAX's compiled programs when a task brings a new origin.

    Each origin compiles its own shapes, and keeping every origin's
    programs over a run fills `vm.max_map_count` (an LLVM "Cannot allocate
    memory" error). The programs of the last `_KEPT_ORIGINS` origins are
    kept, so the models forecasting one task, and the same task forecast
    again (the release backtest after the development one), reuse them.
    """
    origin = (
        task.kind,
        task.vintage,
        task.reference_date,
        tuple(task.locations),
        task.n_samples,
    )
    if origin in _ORIGINS:
        return
    if len(_ORIGINS) >= _KEPT_ORIGINS:
        jax.clear_caches()  # type: ignore[no-untyped-call]
        _ORIGINS.clear()
    _ORIGINS.add(origin)


HUB_WEEKDAY: Final = 5
"""The weekday the hub's weeks end on, Saturday (Monday is 0)."""


def _key(task: ForecastTask) -> jax.Array:
    return jax.random.key(int(task.rng().integers(0, 2**31 - 1)))


def _timeline(origin: Origin, task: ForecastTask) -> list[str]:
    """Return every hub week from the first data week to the last target."""
    weeks = list(origin.snap.week_ends)
    last = as_date(max(task.target_end_dates))
    while as_date(weeks[-1]) < last:
        step = dt.timedelta(days=origin.snap.period)
        weeks.append((as_date(weeks[-1]) + step).isoformat())
    return weeks


def _between_weeks(
    walk: dict[str, np.ndarray], cut: dt.date, days: list[str]
) -> dict[str, np.ndarray]:
    """Return each sample's 7-day sums ending after the cut-off.

    The sums ending on the cut-off's weekday are the walk's; one ending
    $r$ days after such a sum $W_k$ is
    $(1 - r/7) W_k + (r/7) W_{k+1}$, the next sum's days at its rate.
    """
    out = {}
    for d in days:
        k, r = divmod((as_date(d) - cut).days, 7)
        lo = walk[(cut + dt.timedelta(days=7 * k)).isoformat()]
        if r == 0:
            out[d] = lo
            continue
        hi = walk[(cut + dt.timedelta(days=7 * (k + 1))).isoformat()]
        out[d] = lo + (hi - lo) * r / 7
    return out


def _label(origin: Origin, i: int) -> str:
    n = len(origin.snap.locations)
    kind = "own" if i < n else "pooled"
    return f"{kind}:{origin.snap.locations[i % n]}"


def _walks(
    key: jax.Array,
    x0: np.ndarray,
    origin: Origin,
    start: np.ndarray,
    h: int,
    quiet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `h` weeks of walk rates and counts after `start` from `x0`.

    Places in `quiet` `[L]` draw their introductions
    (`silent_trajectories`), and the others walk from `x0` `[L, S]`. The
    rates `[L, S, h]` are the walk's, also for places drawing
    introductions, whose counts do not use them.
    """
    k_walk, k_silent = jax.random.split(key)
    keys = jax.random.split(k_walk, x0.shape[0])
    lam, drawn = random_walks(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(origin.sigma, jnp.float32),
        h,
    )
    silent = silent_trajectories(
        k_silent,
        quiet,
        origin.intro,
        1.0 - np.clip(origin.complete, 0.0, 1.0),
        start,
        origin.sigma,
        h,
        x0.shape[1],
    )
    counts = np.where(
        quiet[:, None, None], silent, np.asarray(drawn, np.float64)
    )
    return np.asarray(lam, np.float64), counts


def start_window(settings: Settings, period: int = 7) -> int:
    """Return the blocks back from the cut-off block a start may be.

    It is `Settings.history_delay`, the onset days the nowcast's delay is
    fitted on (1.5 D), in whole blocks of `period` days.

    Parameters
    ----------
    settings : Settings
        Settings.
    period : int, default 7
        Days in a block (`SnapshotArrays.period`).

    Returns
    -------
    int
        Blocks, counting the cut-off block.
    """
    return -(-settings.history_delay // period)


def _start_ok(origin: Origin) -> np.ndarray:
    """Return the full weeks a walk can start from, `[L, W]`.

    A week can when its share reported is at least `Settings.start_share`,
    its reported count is at least `Settings.start_count`, and it is within
    `start_window` weeks of the cut-off week.
    """
    settings, full = origin.settings, origin.start_nowcast
    ok = (origin.complete[:, : full + 1] >= settings.start_share) & (
        origin.reported[:, : full + 1] >= settings.start_count
    )
    first = (
        origin.snap.cut_week(origin.o)
        - start_window(settings, origin.snap.period)
        + 1
    )
    ok[:, : max(first, 0)] = False
    return ok


def _quiet(origin: Origin) -> np.ndarray:
    """Return the places that draw introductions instead of a walk, `[L]`.

    These are the silent places and those with no week to start from in
    the `start_window`.
    """
    return origin.silent | ~_start_ok(origin).any(axis=1)


def _level(origin: Origin) -> np.ndarray:
    """Return each place's mean reported count before its walk, `[L]`.

    The mean is over the `start_window` weeks up to the place's latest
    complete week (`Origin.start`).
    """
    weeks = np.arange(origin.reported.shape[1])
    last = np.asarray(origin.start)[:, None]
    first = last - start_window(origin.settings, origin.snap.period) + 1
    mask = (weeks >= first) & (weeks <= last)
    out: np.ndarray = (origin.reported * mask).sum(1) / mask.sum(1)
    return out


def _start_weeks(origin: Origin) -> np.ndarray:
    """Return the week each place's walk starts from, `[L]`.

    It is the latest week a walk can start from (`_start_ok`). A quiet
    place (`_quiet`) starts at its latest complete week, after which its
    introductions are drawn.
    """
    ok = _start_ok(origin)
    latest = origin.start_nowcast - np.argmax(ok[:, ::-1], 1)
    return np.where(_quiet(origin), origin.start, latest)


def frame(
    task: ForecastTask, weeks: list[str], values: np.ndarray, rows: list[int]
) -> pd.DataFrame:
    """Return sample rows from `values` of shape `[S, L, H]`.

    A task with no places gives no rows.
    """
    if not task.locations:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    col = {w: i for i, w in enumerate(weeks)}
    targets = list(task.target_end_dates)
    n = values.shape[0]
    frames = []
    for loc, row in zip(task.locations, rows, strict=True):
        picked = np.column_stack(
            [
                values[:, row, col[w]] if w in col else np.zeros(n)
                for w in targets
            ]
        )
        frames.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": np.tile(targets, n),
                    "sample": np.repeat(np.arange(1, n + 1), len(targets)),
                    "value": np.maximum(np.round(picked), 0.0).reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]


@dataclass
class RandomWalk:
    """Carry the latest complete week forward with growing noise.

    Attributes
    ----------
    name : str
        Model name.
    settings : Settings
        Delay window, thresholds and the walk's window.
    copula : CopulaSettings
        How rho_p is estimated (`copula.estimate_rho`).
    fit_diagnostics : list of dict
        One row per place from the last forecast: the start week, the
        errors behind the spread and the source of the delay, spread and
        walk.
    keep_variants : bool
        Keep the last forecast's unpaired draws, for `unpaired`.
    copula_fit : dict or None
        The copula of the last forecast: `model`, `origin`, `n_scores` and
        `copula.estimate_rho`'s `rho_provinces`, `n_pairs` and `fallback`,
        with `reason` where it fell back, or None where there are no
        provinces to pair (a zone task).
    """

    name: str = "randomwalk"
    settings: Settings = field(default_factory=Settings)
    copula: CopulaSettings = field(default_factory=CopulaSettings)
    fit_diagnostics: list[dict[str, object]] = field(default_factory=list)
    keep_variants: bool = field(default=False, repr=False)
    copula_fit: dict[str, object] | None = field(
        init=False, default=None, repr=False
    )
    needs_snapshots: ClassVar[bool] = True

    def pair(
        self,
        origin: Origin,
        task: ForecastTask,
        values: np.ndarray,
    ) -> np.ndarray:
        """Pair the provinces' draws `[S, L, H]` within each week.

        rho_p is estimated from the normal scores of the model's own
        nowcasts at earlier releases (`copula.error_scores`), shrunk to the
        prior (`copula.estimate_rho`), and the draws are paired by it
        (`copula.pair_walks`). A task with no province, a place one level
        below its root, is left unpaired.
        """
        if self.keep_variants:
            self._unpaired = values
            self._kept_origin = origin
        locations = tuple(origin.snap.locations)
        if not any(depth(c) == 1 for c in task.locations):
            self.copula_fit = None
            return values
        scores, reason = self._error_scores(origin, task)
        c = self.copula
        fit = estimate_rho(scores, c.prior, c.weight, max_rho=c.max_rho)
        self.copula_fit = {
            "model": self.name,
            "origin": task.vintage,
            "n_scores": len(scores),
            **fit,
        }
        if fit["fallback"]:
            reason = reason or "no two provinces scored in the same week"
            self.copula_fit["reason"] = reason
            logger.warning(
                "%s at %s: rho_p falls back to the prior %g (%s).",
                self.name,
                task.vintage,
                c.prior,
                reason,
            )
        rng = task.rng(self.name, "copula")
        rho = float(fit["rho_provinces"])
        return pair_walks(values, locations, rho, rng)

    def unpaired(self) -> pd.DataFrame:
        """Return the last forecast's draws before pairing.

        Needs `keep_variants`. The draws are the forecast's own, so only
        their pairing differs.
        """
        weeks, rows, task = self._kept_frame
        return frame(task, weeks, self._unpaired, rows)

    def paired_with(
        self, copula: CopulaSettings
    ) -> tuple[pd.DataFrame, dict[str, object] | None]:
        """Return the last forecast's draws paired under other settings.

        Needs `keep_variants`. The draws are the forecast's own, so only
        their pairing differs, as the release backtest compares them
        (`bdbvhub.pairing.bdbv_candidates`).

        Parameters
        ----------
        copula : CopulaSettings
            The settings to pair by.

        Returns
        -------
        pandas.DataFrame
            The samples.
        dict or None
            The copula record, as `copula_fit`.
        """
        weeks, rows, task = self._kept_frame
        other = replace(self, copula=copula, keep_variants=False)
        values = other.pair(self._kept_origin, task, self._unpaired)
        return frame(task, weeks, values, rows), other.copula_fit

    def _error_scores(
        self, origin: Origin, task: ForecastTask
    ) -> tuple[pd.DataFrame, str | None]:
        """Return the normal scores of the model's own earlier nowcasts.

        With the reason there are none, if so.
        """
        if origin.errors is None:
            try:
                _add_spread(origin)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                return pd.DataFrame(), f"the earlier nowcasts failed: {e}"
        errors, size = origin.errors, origin.size
        if errors is None or size is None or not errors.releases:
            return pd.DataFrame(), "no earlier release to score"
        snap = origin.snap
        at = {r: i for i, r in enumerate(snap.releases)}
        ages = np.array(
            [snap.cut[origin.o] - snap.cut[at[r]] for r in errors.releases]
        )
        scores = error_scores(
            errors.added,
            errors.remaining,
            errors.valid,
            size,
            list(errors.releases),
            ages,
            tuple(snap.locations),
            self.name,
            task.seed,
            self.copula.window_weeks,
        )
        return scores, None

    def settings_for(self, task: ForecastTask) -> Settings:
        """Return the settings for `task`: its study `max_delay`, if any.

        A task's `max_delay` (`ForecastTask.max_delay`, BDBV's study $D$)
        replaces the settings' own.
        """
        if task.max_delay is None:
            return self.settings
        return dataclasses.replace(self.settings, max_delay=int(task.max_delay))

    def snapshots(self, task: ForecastTask) -> SnapshotArrays:
        """Return the task's release snapshots as arrays.

        Parameters
        ----------
        task : ForecastTask
            The task. Its `daily_history` holds the release snapshots.

        Returns
        -------
        SnapshotArrays
            The snapshots up to the task's vintage.

        Raises
        ------
        MissingSnapshots
            If the task has no release snapshots.
        """
        _clear_caches(task)
        daily = task.daily_history
        if daily is None or daily.empty:
            raise MissingSnapshots(
                f"{self.name} needs daily release snapshots, and the task "
                f"for {task.vintage} has none."
            )
        return snapshot_arrays(daily, task.vintage, task.locations)

    def origin_at(self, task: ForecastTask, errors: bool) -> Origin:
        """Fit the delays and the walk at the task's vintage.

        Parameters
        ----------
        task : ForecastTask
            The task. Its `daily_history` holds the release snapshots.
        errors : bool
            Also fit the nowcast errors, which `RandomWalkNowcast` needs.

        Returns
        -------
        Origin
            The fit at the task's vintage.

        Raises
        ------
        MissingSnapshots
            If the task has no release snapshots.
        """
        return build_origin(
            self.snapshots(task), self.settings_for(task), errors
        )

    def rows(self, origin: Origin, task: ForecastTask) -> list[int]:
        """Return the origin's row of each of the task's places.

        Parameters
        ----------
        origin : Origin
            The fit, from `origin_at`.
        task : ForecastTask
            The task.

        Returns
        -------
        list of int
            Row indices, in the order of `task.locations`.
        """
        pos = {c: i for i, c in enumerate(origin.snap.locations)}
        return [pos[c] for c in task.locations]

    def _diagnostics(
        self, origin: Origin, task: ForecastTask, start: np.ndarray
    ) -> None:
        rows: list[dict[str, object]] = []
        for loc, i in zip(task.locations, self.rows(origin, task), strict=True):
            n_err = None if origin.obs_n is None else int(origin.obs_n[i, 0])
            obs_src = (
                None
                if origin.obs_source is None
                else _label(origin, int(origin.obs_source[i, 0]))
            )
            rows.append(
                {
                    "location": loc,
                    "start_week": origin.snap.week_ends[int(start[i])],
                    "n_err": n_err,
                    "delay_source": _label(origin, int(origin.fit.source[i])),
                    "obs_source": obs_src,
                    "sigma_source": _label(origin, int(origin.sigma_source[i])),
                }
            )
        self.fit_diagnostics = rows

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots, no pair of releases within
            the delay window, or a place with no week that is 95 per cent
            complete (`NoCompleteWeek`).

        Examples
        --------
        >>> from bdbvhub.models import get_model
        >>> samples = get_model("randomwalk").forecast(task)
        >>> samples.groupby(["location", "target_end_date"])["value"].median()
        """
        if not task.locations:
            return frame(task, [], np.zeros((task.n_samples, 0, 0)), [])
        origin = self.origin_at(task, errors=False)
        weeks = _timeline(origin, task)
        n_all, n = len(weeks), task.n_samples
        n_loc = len(origin.snap.locations)
        good = origin.complete >= origin.settings.completeness
        if not good.any(axis=1).all():
            has = good.any(axis=1)
            none = [
                x
                for x, ok in zip(origin.snap.locations, has, strict=True)
                if not ok
            ]
            raise NoCompleteWeek(
                f"No week reaches {origin.settings.completeness:.0%} "
                f"complete at {origin.snap.releases[origin.o]} for "
                f"{', '.join(none)}, so the walk has no start week."
            )
        start = origin.start
        x0 = origin.reported[np.arange(n_loc), start]
        _, y = _walks(
            _key(task),
            np.repeat(x0[:, None], n, axis=1),
            origin,
            start,
            n_all,
            origin.silent,
        )
        values = np.zeros((n, n_loc, n_all))
        for loc in range(n_loc):
            for u in range(n_all):
                if u <= start[loc]:
                    values[:, loc, u] = origin.reported[loc, u]
                else:
                    values[:, loc, u] = y[loc, :, u - start[loc] - 1]
        values = self.pair(origin, task, values)
        self._diagnostics(origin, task, start)
        self._kept_frame = (weeks, self.rows(origin, task), task)
        return frame(task, weeks, values, self.rows(origin, task))


@dataclass
class RandomWalkNowcast(RandomWalk):
    """The random walk run forward from a daily nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    error_days : int, default 7
        Days in a block of the observation model. With 7, a task of hub
        weeks is nowcast by week and one whose targets are other days by
        blocks of 7 days ending on each, each with its own observation
        model (baselinenowcast's 7-day rolling `ref_time_aggregator`).
        With 1, the observation model is fitted on daily counts and the
        7-day sums are sums of daily draws.
    delay_summary : pandas.DataFrame or None
        The delay of the last forecast
        (`baselinenowcast.nowcast.delay_summary`), which maps its final
        counts to what later releases show.
    """

    name: str = "randomwalknowcast"
    error_days: int = 7
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )
    _snap: SnapshotArrays | None = field(init=False, default=None, repr=False)
    _fits: dict[tuple[int, bool], DelayFit | None] = field(
        init=False, default_factory=dict, repr=False
    )

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories from a nowcast for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots, or no pair of releases within
            the delay window.
        """
        if not task.locations:
            return frame(task, [], np.zeros((task.n_samples, 0, 0)), [])
        ends = pd.to_datetime(list(task.target_end_dates))
        if self.error_days != 7 or (ends.weekday != HUB_WEEKDAY).any():
            return self._rolling(task)
        origin = self.origin_at(task, errors=True)
        values, weeks = self.trajectories(origin, task, origin.point)
        values = self.pair(origin, task, values)
        self.delay_summary = delay_summary(origin, self.name, task.vintage)
        self._diagnostics(
            origin,
            task,
            _start_weeks(origin),
        )
        self._kept_frame = (weeks, self.rows(origin, task), task)
        return frame(task, weeks, values, self.rows(origin, task))

    def _block_draws(
        self, task: ForecastTask, period: int, end: int = HUB_WEEKDAY
    ) -> tuple[Origin, dict[str, np.ndarray]]:
        """Return the draws of each block of `period` days ending `end`.

        The delay is fitted once, on the hub weeks' snapshots, and shared
        by every block (`_shared_delay`); the observation model, the walk
        and the copula are fitted on the blocks.
        """
        snap = self._snap
        assert snap is not None
        origin = build_origin(
            blocks(snap, period, end),
            self.settings_for(task),
            True,
            self._shared_delay,
        )
        values, labels = self.trajectories(origin, task, origin.point)
        values = self.pair(origin, task, values)
        return origin, {w: values[:, :, i] for i, w in enumerate(labels)}

    def _shared_delay(
        self,
        snap: SnapshotArrays,  # noqa: ARG002 - the blocks, as `DelayFn` gives
        r: int,
        settings: Settings,
        longer: bool = False,
    ) -> DelayFit | None:
        """Fit the delay on the hub weeks' snapshots, once per release."""
        key = (r, longer)
        if key not in self._fits:
            assert self._snap is not None
            self._fits[key] = fit_delay(self._snap, r, settings, longer)
        return self._fits[key]

    def _start(self, task: ForecastTask) -> None:
        """Hold the task's snapshots and clear the delay fits."""
        self._snap = self.snapshots(task)
        self._fits = {}

    def _rolling(self, task: ForecastTask) -> pd.DataFrame:
        """Return draws of the 7-day sums ending on each target date.

        With `error_days` 7 each sum to the cut-off is a block of 7 days
        ending on its weekday, with its own observation model, as
        baselinenowcast's `ref_time_aggregator` of 7-day rolling sums.
        The sums after the cut-off come from the walk of the blocks ending
        on the cut-off's weekday, whose last block is whole at the cut-off
        (`_between_weeks`), so they carry no weekday cycle. With 1, the
        sums are of draws of daily counts, whose observation model is
        fitted on days.
        """
        self._start(task)
        targets = list(task.target_end_dates)
        draws: dict[str, np.ndarray] = {}
        if self.error_days == 1:
            origin, days = self._block_draws(task, 1)
            labels = list(days)
            x = np.stack([days[d] for d in labels], axis=-1)
            c = np.concatenate(
                [np.zeros((*x.shape[:2], 1)), np.cumsum(x, axis=-1)], axis=-1
            )
            week = c[..., 7:] - c[..., :-7]
            draws = {d: week[..., i] for i, d in enumerate(labels[6:])}
        else:
            snap = self._snap
            assert snap is not None
            cut = snap.days[snap.cut[-1]].date()
            past = [d for d in targets if as_date(d) <= cut]
            ends = {as_date(d).weekday() for d in past} - {cut.weekday()}
            for end in sorted(ends):
                _, got = self._block_draws(task, 7, end)
                draws.update({d: got[d] for d in past if d in got})
            origin, walk = self._block_draws(task, 7, cut.weekday())
            draws.update({d: walk[d] for d in past if d in walk})
            ahead = [d for d in targets if as_date(d) > cut]
            draws.update(_between_weeks(walk, cut, ahead))
        self.delay_summary = delay_summary(origin, self.name, task.vintage)
        rows = self.rows(origin, task)
        n = task.n_samples
        values = np.stack(
            [draws.get(d, np.zeros((n, len(origin.snap.locations))))
             for d in targets],
            axis=-1,
        )  # fmt: skip
        self._kept_frame = (targets, rows, task)
        return frame(task, targets, values, rows)

    def daily_quantiles(
        self, task: ForecastTask, levels: tuple[float, ...]
    ) -> pd.DataFrame:
        """Return quantiles of each target date's count, by place.

        The draws are of blocks of one day, with the observation model
        fitted on daily counts.

        Parameters
        ----------
        task : ForecastTask
            The task.
        levels : tuple of float
            Quantile levels.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `date`, `quantile` and `value`.
        """
        self._start(task)
        origin, days = self._block_draws(task, 1)
        targets = list(task.target_end_dates)
        n = task.n_samples
        values = np.stack(
            [days.get(d, np.zeros((n, len(origin.snap.locations))))
             for d in targets],
            axis=-1,
        )  # fmt: skip
        draws = frame(task, targets, values, self.rows(origin, task))
        q = (
            draws.groupby(["location", "target_end_date"])["value"]
            .quantile(np.asarray(levels))
            .rename("value")
            .reset_index()
        )
        return q.rename(
            columns={"target_end_date": "date", "level_2": "quantile"}
        )[["location", "date", "quantile", "value"]]

    def trajectories(
        self, origin: Origin, task: ForecastTask, point: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Return trajectories from a nowcast with mean `point`.

        Parameters
        ----------
        origin : Origin
            The fit at the task's vintage, with its observation model.
        task : ForecastTask
            The task.
        point : numpy.ndarray
            Mean weekly nowcast, `[L, W]`. The model's own is
            `origin.point`. A later release's expected count gives the
            prediction scored in the backtest.

        Returns
        -------
        tuple
            Samples `[S, L, H]` for every hub week from the first data
            week to the last target, and those weeks.
        """
        snap = origin.snap
        assert origin.size is not None
        weeks = _timeline(origin, task)
        n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(snap.locations)
        k_n = origin.size.shape[1]
        back = snap.cut_week(origin.o) - np.arange(n_data)
        # Weeks older than the fitted ones have nothing to come.
        k = np.clip(back, 0, k_n - 1)
        key_x, key_walk, key_part, key_ahead = jax.random.split(_key(task), 4)
        shape = (n, *point.shape)
        reported = jnp.broadcast_to(
            jnp.asarray(origin.reported, jnp.float32), shape
        )
        nowcast = np.asarray(
            sample_count(
                key_x,
                reported,
                jnp.maximum(jnp.asarray(point, jnp.float32) - reported, 0.0),
                jnp.broadcast_to(
                    jnp.asarray(origin.size[:, k], jnp.float32), shape
                ),
            ),
            np.float64,
        )
        # Each sample walks on from its own nowcast of the start week.
        cw = snap.cut_week(origin.o)
        w0 = _start_weeks(origin)
        quiet = _quiet(origin)
        rows = np.arange(n_loc)
        # Only silent places draw introductions: one with recent cases has
        # a level to walk from.
        x0 = nowcast[:, rows, w0].T
        low = quiet & ~origin.silent
        x0[low] = _level(origin)[low, None]
        lam, y = _walks(key_walk, x0, origin, w0, n_all, origin.silent)
        # Days of the cut-off week after the cut-off are drawn from the
        # walk's rate that week. A partial cut-off week's start week is the
        # week before it when that week meets the start rule, so the rate
        # is one step from each sample's nowcast of the week before, and
        # otherwise steps on from the latest earlier week that meets it.
        cut = int(snap.cut[origin.o])
        miss = int(np.flatnonzero(snap.week == cw).max()) - cut
        at_cw = np.clip(cw - w0 - 1, 0, n_all - 1)
        part = np.asarray(
            jax.random.poisson(
                key_part,
                jnp.asarray(
                    lam[rows, :, at_cw] * miss / snap.period, jnp.float32
                ),
            ),
            dtype=np.float64,
        )  # [L, S]
        # Each sample's completed cut-off week is the base for the weeks
        # after it, from which the walk goes on.
        done = nowcast[:, rows, cw].T + np.where(miss > 0, part, 0.0)
        h = n_all - cw - 1
        _, ahead = random_walks(
            jax.random.split(key_ahead, n_loc),
            jnp.asarray(done, jnp.float32),
            jnp.asarray(origin.sigma, jnp.float32),
            max(h, 1),
        )
        ahead = np.asarray(ahead, np.float64)[:, :, :h]
        values = np.zeros((n, n_loc, n_all))
        for loc in range(n_loc):
            s = int(w0[loc])
            if quiet[loc]:
                # A quiet place is its reported counts, then its walk or
                # introductions.
                values[:, loc, : s + 1] = origin.reported[loc, : s + 1]
                values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
                continue
            values[:, loc, :cw] = nowcast[:, loc, :cw]
            values[:, loc, cw] = done[loc]
            values[:, loc, cw + 1 :] = ahead[loc]
        return values, weeks


register_model(
    "randomwalk",
    RandomWalk,
    "Latest complete week carried forward, with log-scale random walk noise.",
    baseline=True,
)
register_model(
    "randomwalknowcast",
    RandomWalkNowcast,
    "Random walk carried forward from a daily nowcast of the recent weeks.",
)
