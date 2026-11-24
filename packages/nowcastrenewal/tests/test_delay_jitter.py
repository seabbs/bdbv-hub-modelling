"""Delay jitter (`reporting.effects.ReferenceEffects`): week-to-week noise on
every delay parameter, on top of the other delay settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers

from nowcastrenewal.reporting.effects import ReferenceEffects, ReferenceFrame

MAX_DELAY, N_DAYS, FIRST = 8, 35, "2026-01-05"
VALUES: dict[str, Any] = {
    "weeks": 5,
    "delay_mean": 4.0,
    "delay_median": 3.0,
    "delay_sdlog": 0.7,
    "delay_cv": 0.8,
    "hazard_sd": 0.3,
    "hazard_walk_sd": 0.2,
    "delay_jitter_sd": 0.2,
}


def _design(**kw: str) -> dict[str, str]:
    base = {
        "delay_form": "parametric",
        "delay_family": "lognormal",
        "delay_change": "static",
        "shocks": "none",
        "delay_places": "shared",
        "delay_dow": "no",
        "delay_jitter": "yes",
    }
    return {**base, **kw}


def _run(design: dict[str, str]):
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.array([0]),
        age=np.array([0]),
    )
    effects = ReferenceEffects("ref", frame, design, VALUES, MAX_DELAY)
    with handlers.seed(rng_seed=1), handlers.trace() as tr:
        out = effects()
    return frame, tr, out


def test_jitter_moves_location_and_scale_week_by_week() -> None:
    frame, tr, _ = _run(_design())
    mu = np.asarray(tr["ref::location"]["value"])[0]
    scale = np.asarray(tr["ref::scale"]["value"])[0]
    week = np.asarray(frame.week)
    for series in (mu, scale):
        # Constant within a week, different between weeks.
        for w in np.unique(week):
            assert np.ptp(series[week == w]) == pytest.approx(0)
        assert np.ptp(series) > 0
    assert "ref::jitter_location" in tr
    assert "ref::jitter_scale" in tr


def test_without_jitter_location_and_scale_are_as_before() -> None:
    _, tr, _ = _run(_design(delay_jitter="no"))
    assert "ref::jitter_location" not in tr
    assert np.ptp(np.asarray(tr["ref::location"]["value"])) == pytest.approx(0)
    assert np.ptp(np.asarray(tr["ref::scale"]["value"])) == pytest.approx(0)


def test_jitter_sits_on_top_of_a_delay_change_and_the_hazard() -> None:
    # A non-parametric delay jitters its level and its shape by week.
    _, tr, (_, hazard) = _run(_design(delay_form="non_parametric"))
    assert "ref::jitter_hazard" in tr
    assert "ref::jitter_shape" in tr
    h = np.asarray(hazard)[0]
    assert np.ptp(h[:7, 0] - h[7:14, 0]) == pytest.approx(0)
    assert not np.allclose(h[0], h[7])
