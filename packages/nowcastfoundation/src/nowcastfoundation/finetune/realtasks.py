# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Training blocks of real data, laid out as the fine-tuned model forecasts.

A real-data block is a forecast task (`ForecastTask`, built by the study's
own task builder) laid out by `FoundationModel` exactly as its forecast
is: the source the task gives (`for_task`), the ragged triangle with each
place's lag series and 7-day rows, and the model's views
(`active_views`, or `block_views` over the block limit). Its lags are
those of the study's maximum delay, as its forecasts fix them, or as
many as the history holds when it is shorter; in a share of blocks the
count is scaled by a draw from Normal(1, sd), as the simulated blocks'
lag jitter (`LAG_JITTER`). Each block's targets after its context hold
the truth, a later task's final counts, masked step by step after the
truth date, so no count trained on is younger than the study's truth
delay.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.forecaster import (
    FINETUNED_CONFIG,
    FROM_DATA,
    FoundationModel,
    _sub,
    active_zone_rule,
    hub_triangle,
    layout_block,
    rolling_totals,
)
from nowcastutils.base import NATIONAL, ForecastTask

RELEASE_LAYER = (
    ft.ROOT / "packages/nowcastfoundation/config/finetune/release-layer.yaml"
)
"""The release layer, whose `lag_jitter` the real blocks' lags take."""


def _jitter() -> tuple[float, float]:
    raw = ft.read_release_layer(RELEASE_LAYER)["lag_jitter"]
    return float(raw["share"]), float(raw["sd"])


LAG_JITTER: Final = _jitter()
"""The share of blocks whose lag count is scaled, and the scale's sd: the
simulated blocks' `lag_jitter`, 10% of blocks by Normal(1, 0.05)."""


BLOCK_LIMIT = 1000
"""Most series in a block: the workflow's `finetune.block_limit`, over
which the model cuts a block into views."""


def real_model(block_limit: int = BLOCK_LIMIT) -> FoundationModel:
    """Return the fine-tuned model's layout for the real-data blocks.

    Ragged, its source read from each task, its active-zone rule
    (packages/nowcastfoundation/config/finetune.toml) and `block_limit`.
    """
    cases, weeks, recent = active_zone_rule(ft.ROOT / FINETUNED_CONFIG)
    return FoundationModel(
        "realdata",
        FROM_DATA,
        "ragged",
        block_limit=block_limit,
        active_zone_cases=cases,
        active_zone_weeks=weeks,
        active_zone_recent_weeks=recent,
    )


@dataclass(frozen=True)
class Truth:
    """A study's final counts, the targets the blocks train on.

    Attributes
    ----------
    places : tuple of str
        The places, the rows of `counts`.
    times : list of str
        The days, the columns of `counts`.
    counts : numpy.ndarray
        Each place's count on each day at the truth task's release.
    until : str
        The last day a target may hold; later days are masked.
    """

    places: tuple[str, ...]
    times: list[str]
    counts: np.ndarray
    until: str

    def rows(self, places: tuple[str, ...], times: list[str]) -> np.ndarray:
        """Return the counts of `places` on `times`, NaN where none."""
        out = np.full((len(places), len(times)), np.nan)
        row = {p: i for i, p in enumerate(self.places)}
        col = {t: j for j, t in enumerate(self.times)}
        cols = [col.get(t, -1) for t in times]
        have = np.array([c >= 0 for c in cols])
        for i, p in enumerate(places):
            if p in row and have.any():
                out[i, have] = self.counts[row[p], np.array(cols)[have]]
        return out


def truth_of(model: FoundationModel, task: ForecastTask, until: str) -> Truth:
    """Return the final counts of `task`'s releases, as `model` reads them."""
    run = model.for_task(task)
    rows, places, weeks = run._rows(task)
    tri = hub_triangle(rows, run.source, places, weeks[-1], max_lag=1)
    return Truth(tuple(places), list(tri.times), tri.final, until)


def lag_scale(rng: np.random.Generator, jitter: tuple[float, float]) -> float:
    """Return a block's lag-count scale: 1, or Normal(1, sd) in a share."""
    share, sd = jitter
    return float(rng.normal(1.0, sd)) if rng.random() < share else 1.0


def levels_of(places: tuple[str, ...]) -> tuple[str, ...]:
    """Return each place's level in a block.

    National, intermediate (a total of places in the block) or leaf.
    """
    return tuple(
        "national"
        if p == NATIONAL
        else "intermediate"
        if any(q.startswith(f"{p}-") for q in places)
        else "leaf"
        for p in places
    )


def task_blocks(
    model: FoundationModel,
    task: ForecastTask,
    truth: Truth,
    rng: np.random.Generator,
    max_delay: int,
    jitter: tuple[float, float] = LAG_JITTER,
    **labels: Any,
) -> list[dict[str, Any]]:
    """Return the training blocks of one task, one per view.

    The task is laid out as `model` forecasts it, at lags 0 to
    `max_delay` days (scaled by `lag_scale`), or as many as its history
    holds (`FoundationModel._triangle`), and each view of its block
    becomes a training block, its targets after the context the truth's,
    missing on each step after `truth.until`. `labels` (such as `data` and
    `cut`) are added to each block. None if the task has under two steps
    of context or no truth.
    """
    run = model.for_task(task)
    rows, places, weeks = run._rows(task)
    scale = lag_scale(rng, jitter)
    run = replace(run, lags=max(1, round((max_delay + 1) * scale)))
    tri = run._triangle(rows, places, weeks[-1])
    lags = int(tri.partial.shape[2])
    start = tri.n_context
    if start < 2 or start >= len(tri.times):
        return []
    late = np.array([t > truth.until for t in tri.times])
    out = []
    for view, mine in run._views(places, tri):
        sub = _sub(tri, places, view)
        period = run._period(sub)
        group = layout_block(sub, run.layout, view, start, period)
        counts = truth.rows(tuple(view), list(tri.times))
        final = counts
        if period is not None:
            final = np.concatenate([counts, rolling_totals(counts, 7)])
        ahead = final[:, start:].copy()
        ahead[:, late[start:]] = np.nan
        if np.isnan(ahead).all():
            continue
        names = list(group["past_covariates"])
        lagged = np.stack(
            [
                np.concatenate(
                    [group["past_covariates"][n], group["future_covariates"][n]]
                )
                for n in names
            ]
        )
        target = np.concatenate([group["target"], ahead], axis=1)
        context = np.concatenate([target, lagged]).astype(np.float32)
        block = {
            "context": context,
            "future_covariates": context[:, start:].copy(),
            "n_targets": len(group["target_places"]),
            "n_covariates": len(names),
            "n_future_covariates": len(names),
            "start": start,
            "horizon": len(tri.times) - start,
            "nowcast": tri.last - start + 1,
        }
        padded = ft._pad(block, ft.CONTEXT)
        levels = levels_of(tuple(view))
        kept = [
            t
            for t, x in zip(tri.times[start:], late[start:], strict=True)
            if not x
        ]
        padded |= {
            "places": tuple(view),
            "target_places": list(group["target_places"]),
            "levels": tuple(levels[k] for k in group["target_places"]),
            "parent": "exact",
            "source": f"{run.source}+rolling7" if period else run.source,
            "view": "joint" if NATIONAL in view else "province",
            "forecasts": tuple(mine),
            "lag": lags,
            "lag_scale": scale,
            "truth_end": max(kept),
            "times": _padded_times(tri.times, start, padded),
            **labels,
        }
        out.append(padded)
    return out


def _padded_times(
    times: list[str], start: int, padded: dict[str, Any]
) -> list[str]:
    """Return the day of each padded column, "" where padded."""
    keep = min(start, ft.CONTEXT)
    width = padded["context"].shape[1]
    days = [""] * (ft.CONTEXT - keep) + list(times[start - keep :])
    return (days + [""] * width)[:width]
