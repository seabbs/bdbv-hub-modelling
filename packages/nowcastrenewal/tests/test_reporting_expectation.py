"""A simulation's log expected reports, against epinowcast, live.

Each case runs epinowcast's model at the pin with the simulator's modules
and fixed parameters (`tests/model/r/reporting_cases.R`,
`expectation_case`) and computes every cell's log expected reports with
the model's own `expected_obs_from_snaps`. The port
(`reporting.expectation.ReportingExpectation`) takes the same draws,
mapped by design column name, and the same latent expected counts.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model.r_live import run_r
from numpyro import handlers
from test_reporting_effects import (
    DAYS,
    FIRST,
    MAX_DELAY,
    N_DAYS,
    VALUES,
    _design,
    _frame,
    _names,
    _sites,
)

from nowcastrenewal.components.pmfs import Delay
from nowcastrenewal.reporting.effects import ReferenceFrame, report_days
from nowcastrenewal.reporting.expectation import ReportingExpectation

REPORT_VALUES = {**VALUES, "dow_sd": 0.2, "reference_dow_sd": 0.15}


def _report(**kw: str) -> dict[str, str]:
    base = {
        "report_days": "daily",
        "report_dow": "no",
        "report_weekdays": "Monday;Thursday",
        "reference_dow": "no",
    }
    return {**base, **kw}


CASES: list[dict[str, Any]] = [
    {
        # Hazards: a report weekday effect, and the reference weekday's.
        "design": {
            **_design(
                delay_form="hybrid",
                delay_change="drift",
                shocks="backlog",
                delay_places="independent",
                delay_dow="yes",
            ),
            **_report(report_dow="yes", reference_dow="yes"),
        },
        "n_places": 2,
        "n_ages": 2,
    },
    {
        # Log probabilities (ref_as_p): reports on two weekdays, outages.
        "design": {
            **_design(delay_form="hybrid", shocks="outage"),
            **_report(report_days="weekdays"),
        },
        "n_places": 2,
        "n_ages": 1,
    },
    {
        # Log probabilities, with the reference weekday's effect.
        "design": {
            **_design(delay_family="gamma", delay_change="step"),
            **_report(reference_dow="yes"),
        },
        "n_places": 2,
        "n_ages": 1,
    },
    {
        # No parametric part: hazards, reports on one weekday.
        "design": {
            **_design(
                delay_form="non_parametric",
                delay_change="walk",
                delay_places="shifted",
            ),
            **_report(report_days="weekly", report_weekdays="Wednesday"),
        },
        "n_places": 2,
        "n_ages": 1,
    },
    {
        # Hazards: no parametric part, a report weekday effect, an outage.
        "design": {
            **_design(
                delay_form="non_parametric",
                delay_change="slowdown",
                shocks="outage",
            ),
            **_report(report_dow="yes"),
        },
        "n_places": 3,
        "n_ages": 1,
    },
]


def _log_lambda(i: int, n_strata: int) -> np.ndarray:
    rng = np.random.default_rng(500 + i)
    trend = np.log(np.linspace(2.0, 40.0, N_DAYS))
    return trend[None, :] + rng.normal(0, 0.2, (n_strata, N_DAYS))


def _input(i: int, case: dict[str, Any]) -> dict[str, Any]:
    design = case["design"]
    family = design["delay_family"]
    delay = Delay.from_values(family, REPORT_VALUES)
    n_strata = case["n_places"] * case["n_ages"]
    return {
        **case,
        "values": REPORT_VALUES,
        "seed": 300 + i,
        "max_delay": MAX_DELAY,
        "n_days": N_DAYS,
        "first_date": FIRST,
        "mean_int": float(delay.mu),
        "sd_int": float(delay.sigma),
        "log_lambda": _log_lambda(i, n_strata).tolist(),
    }


@functools.cache
def _epinowcast() -> list[dict[str, Any]]:
    code = """
    source("tests/model/r/reporting_cases.R")
    lapply(input, function(x) expectation_case(jsonlite::fromJSON(x)))
    """
    inputs = [json.dumps(_input(i, c)) for i, c in enumerate(CASES)]
    return run_r(
        code,
        inputs,
        packages=["epinowcast", "cmdstanr", "data.table", "posterior"],
    )


def _report_sites(out: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    p = out["parameters"]
    sites = {}
    for key, cols_key, beta_key in (
        ("day_of_week", "rep_cols", "rep_beta"),
        ("reference_day_of_week", "expl_cols", "expl_beta"),
    ):
        cols = _names(out.get(cols_key))
        if cols:
            z = np.zeros(7)
            for col, b in zip(cols, np.atleast_1d(p[beta_key]), strict=True):
                z[DAYS.index(col[11:])] = b
            sites[f"{prefix}::{key}"] = z
    return sites


def _as_float(xs: list[Any]) -> np.ndarray:
    named = {"Inf": np.inf, "-Inf": -np.inf, None: -np.inf}
    return np.array([named.get(x, x) for x in xs], dtype=float)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_log_expected_reports_match_epinowcast(i: int) -> None:
    out = _epinowcast()[i]
    frame = _frame(out)
    design = CASES[i]["design"]
    model = ReportingExpectation(
        "expected", frame, design, REPORT_VALUES, MAX_DELAY
    )
    assert int(model.as_log_probs) == out["ref_as_p"]
    sites = {
        **_sites(out, frame, "expected::reference"),
        **_report_sites(out, "expected::report"),
    }
    log_lambda = _log_lambda(i, len(frame.place))
    with jax.enable_x64(True):
        got = np.asarray(
            handlers.substitute(model, data=sites)(
                log_lambda=jnp.asarray(log_lambda)
            )
        )
    # epinowcast's cells: snapshot k is stratum sg[k], day st[k], with
    # sdmax[k] delays.
    want = _as_float(out["log_expected"])
    sg, st = np.array(out["sg"]) - 1, np.array(out["st"]) - 1
    sdmax = np.array(out["sdmax"])
    ours = np.concatenate(
        [got[g, t, :n] for g, t, n in zip(sg, st, sdmax, strict=True)]
    )
    assert np.array_equal(np.isneginf(ours), np.isneginf(want))
    on = np.isfinite(want)
    np.testing.assert_allclose(ours[on], want[on], rtol=1e-10, atol=1e-10)


def test_report_days_follow_weekdays_and_outages() -> None:
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(14),
        place=np.array([0, 1]),
        age=np.array([0, 0]),
    )
    design = {**_design(shocks="outage"), **_report(report_days="weekdays")}
    v = {**VALUES, "shock_places": [2], "shock_at": [0.0], "shock_weeks": [1]}
    v["weeks"] = 2
    on = report_days(frame, design, v, max_delay=3)
    # 2026-01-03 is a Saturday: Mondays and Thursdays are days 2, 5, 9, 12.
    assert on.shape == (2, 16)
    assert list(np.flatnonzero(on[0])) == [2, 5, 9, 12]
    # Place 2 reports nothing over its outage's first week (days 0 to 6).
    assert list(np.flatnonzero(on[1])) == [9, 12]
    daily = {**_design(), **_report()}
    assert report_days(frame, daily, VALUES, 3) is None


def test_the_expectation_samples_forward() -> None:
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.array([0, 1]),
        age=np.array([0, 0]),
    )
    model = ReportingExpectation(
        "expected", frame, CASES[0]["design"], REPORT_VALUES, MAX_DELAY
    )
    with handlers.seed(rng_seed=4), handlers.trace() as tr:
        out = model(log_lambda=jnp.zeros((2, N_DAYS)))
    assert out.shape == (2, N_DAYS, MAX_DELAY)
    assert {"expected::report::day_of_week", "expected::log_expected"} <= set(
        tr
    )
    assert not np.any(np.isnan(np.asarray(out)))


def test_ci_runs_the_reporting_expectation_checks() -> None:
    import yaml
    from model.r_live import ROOT

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/tests.yml").read_text()
    )
    steps = workflow["jobs"]["r-live"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    packages = " ".join(s.get("with", {}).get("packages", "") for s in steps)
    assert "packages/nowcastrenewal/tests/test_reporting_expectation.py" in runs
    assert "posterior" in packages
