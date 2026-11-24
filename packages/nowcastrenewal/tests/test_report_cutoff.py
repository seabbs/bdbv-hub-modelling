"""Reporting cut-offs (`reporting.expectation`): a hard maximum reporting
delay, after which reports never arrive, and reporting that stops."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpyro import handlers

from nowcastrenewal.reporting.effects import ReferenceFrame
from nowcastrenewal.reporting.expectation import ReportingExpectation

MAX_DELAY, N_DAYS, FIRST = 20, 42, "2026-01-05"
VALUES: dict[str, Any] = {
    "weeks": 6,
    "delay_mean": 8.0,
    "delay_median": 6.0,
    "delay_sdlog": 0.8,
    "delay_cv": 0.8,
    "hazard_sd": 0.3,
    "hazard_walk_sd": 0.2,
    "dow_sd": 0.2,
    "reference_dow_sd": 0.15,
    "report_cap_days": 7,
    "report_stop_at": 0.5,
}


def _design(**kw: str) -> dict[str, str]:
    base = {
        "delay_form": "parametric",
        "delay_family": "lognormal",
        "delay_change": "static",
        "shocks": "none",
        "delay_places": "shared",
        "delay_dow": "no",
        "report_days": "daily",
        "report_dow": "no",
        "report_weekdays": "all",
        "reference_dow": "no",
        "report_cap": "no",
        "report_stop": "no",
    }
    return {**base, **kw}


def _expected(design: dict[str, str]) -> np.ndarray:
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.array([0]),
        age=np.array([0]),
    )
    model = ReportingExpectation("e", frame, design, VALUES, MAX_DELAY)
    with handlers.seed(rng_seed=0):
        out = model(log_lambda=np.zeros((1, N_DAYS)))
    return np.exp(np.asarray(out))[0]


def test_reports_past_a_hard_maximum_delay_never_arrive() -> None:
    free = _expected(_design())
    capped = _expected(_design(report_cap="yes"))
    cap = VALUES["report_cap_days"]
    assert np.all(capped[:, cap + 1 :] == 0)
    np.testing.assert_allclose(capped[:, : cap + 1], free[:, : cap + 1])
    # Unlike the window's last delay, which lumps the tail in.
    assert free[:, cap + 1 :].sum() > 0


def test_reporting_that_stops_reports_nothing_after_it() -> None:
    stopped = _expected(_design(report_stop="yes"))
    stop = round(VALUES["report_stop_at"] * N_DAYS)
    t, d = np.meshgrid(np.arange(N_DAYS), np.arange(MAX_DELAY), indexing="ij")
    assert np.all(stopped[t + d >= stop] == 0)
    assert stopped[t + d < stop].sum() > 0
