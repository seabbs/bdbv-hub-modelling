"""The reporting delay's reference-date effects, against epinowcast, live.

Each case builds epinowcast's reference module at the pin with the
simulator's formulas and values (`tests/model/r/reporting_cases.R`), runs
its Stan model with those parameters fixed and reads back the location,
scale and logit hazards. The port takes the same standard normal draws,
mapped by design column name, and must give the same values.
"""

from __future__ import annotations

import functools
import json
import math
from typing import Any

import jax
import numpy as np
import pytest
from numpyro import handlers

from nowcastrenewal.components.pmfs import Delay, censored_log_pmf
from nowcastrenewal.reporting.effects import ReferenceEffects, ReferenceFrame

from .r_live import run_r

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
DAYS = (*DAYS, "Sunday")

VALUES: dict[str, Any] = {
    "weeks": 5,
    "step_at": 0.5,
    "periodic_weeks": 3.0,
    "delay_walk": [0.0, 0.2, -0.1, 0.3, 0.1, 0.0],
    "smooth_period": 10.0,
    "smooth_phase": 0.3,
    "shock_places": [2, 1],
    "shock_at": [0.3, 0.7],
    "shock_weeks": [1, 2],
    "shock_mult": [2.0, 1.4],
    "slow_at": 0.55,
    "slow_weeks": 1,
    "slow_mult": 1.5,
    "drift": 0.3,
    "step": 0.2,
    "periodic": 0.25,
    "smooth_amp": 0.2,
    "delay_place_sd": 0.2,
    "delay_age_sd": 0.1,
    "delay_dow_sd": 0.15,
    "hazard_sd": 0.3,
    "hazard_walk_sd": 0.2,
    "delay_mean": 4.0,
    "delay_median": 3.0,
    "delay_sdlog": 0.7,
    "delay_cv": 0.8,
}


def _design(**kw: str) -> dict[str, str]:
    base = {
        "delay_form": "parametric",
        "delay_family": "lognormal",
        "delay_change": "static",
        "shocks": "none",
        "delay_places": "shared",
        "delay_dow": "no",
    }
    return {**base, **kw}


CASES: list[dict[str, Any]] = [
    {
        "design": _design(
            delay_form="hybrid",
            delay_change="drift",
            shocks="backlog",
            delay_places="independent",
            delay_dow="yes",
        ),
        "n_places": 2,
        "n_ages": 2,
    },
    {
        "design": _design(
            delay_family="gamma", delay_change="step", delay_dow="yes"
        ),
        "n_places": 3,
        "n_ages": 1,
    },
    {
        "design": _design(
            delay_form="non_parametric",
            delay_change="walk",
            delay_places="shifted",
            delay_dow="yes",
        ),
        "n_places": 2,
        "n_ages": 2,
    },
    {
        "design": _design(
            delay_family="exponential",
            delay_change="periodic",
            delay_places="shifted",
        ),
        "n_places": 2,
        "n_ages": 1,
    },
    {
        "design": _design(
            delay_form="hybrid",
            delay_family="gamma",
            delay_change="smooth",
            shocks="backlog",
        ),
        "n_places": 2,
        "n_ages": 1,
    },
    {
        "design": _design(
            delay_form="non_parametric",
            delay_change="slowdown",
            delay_places="independent",
            delay_dow="yes",
        ),
        "n_places": 3,
        "n_ages": 1,
    },
]
MAX_DELAY, N_DAYS, FIRST = 8, 35, "2026-01-03"


def _input(i: int, case: dict[str, Any]) -> dict[str, Any]:
    design = case["design"]
    family = design["delay_family"]
    delay = Delay.from_values(family, VALUES)
    return {
        **case,
        "values": VALUES,
        "seed": 100 + i,
        "max_delay": MAX_DELAY,
        "n_days": N_DAYS,
        "first_date": FIRST,
        "mean_int": float(delay.mu),
        "sd_int": float(delay.sigma),
    }


@functools.cache
def _epinowcast() -> list[dict[str, Any]]:
    # Each case as JSON text, so R reads its nested values as lists.
    code = """
    source("tests/model/r/reporting_cases.R")
    lapply(input, function(x) reference_case(jsonlite::fromJSON(x)))
    """
    inputs = [json.dumps(_input(i, c)) for i, c in enumerate(CASES)]
    return run_r(
        code, inputs, packages=["epinowcast", "cmdstanr", "data.table"]
    )


def _frame(out: dict[str, Any]) -> ReferenceFrame:
    # One row per stratum, in epinowcast's `.group` order.
    groups = out["groups"]
    place = np.array([int(g["place"][1:]) - 1 for g in groups])
    age = np.array([int(g["age"][1:]) - 1 for g in groups])
    dates = np.array(out["dates"], dtype="datetime64[D]")
    return ReferenceFrame(dates=dates, place=place, age=age)


def _names(cols: Any) -> list[str]:
    """Column names; one name comes back as a string (`auto_unbox`)."""
    if cols is None:
        return []
    return [cols] if isinstance(cols, str) else list(cols)


def _sites(
    out: dict[str, Any], frame: ReferenceFrame, prefix: str
) -> dict[str, np.ndarray]:
    """The port's sites from epinowcast's parameters, by column name."""
    n_p, n_a = frame.n_places, frame.n_ages
    sites = {
        "mean_place": np.zeros(n_p),
        "mean_age": np.zeros(n_a),
        "mean_day_of_week": np.zeros(7),
        "mean_day_of_week_place": np.zeros(7 * n_p),
        "hazard_place": np.zeros(n_p),
        "hazard_age": np.zeros(n_a),
        "hazard_day_of_week": np.zeros(7),
        "hazard_day_of_week_place": np.zeros(7 * n_p),
        "hazard_delay": np.zeros(MAX_DELAY),
    }
    p = out["parameters"]
    modules = [("mean", "refp_cols", "refp_mean_beta")]
    modules.append(("hazard", "refnp_cols", "refnp_beta"))
    for kind, cols_key, beta_key in modules:
        cols = _names(out.get(cols_key))
        beta = np.atleast_1d(p.get(beta_key, []))
        for col, b in zip(cols, beta, strict=True):
            if col.startswith("place"):
                sites[f"{kind}_place"][int(col[6:]) - 1] = b
            elif col.startswith("age"):
                sites[f"{kind}_age"][int(col[4:]) - 1] = b
            elif col.startswith("day_of_week"):
                sites[f"{kind}_day_of_week"][DAYS.index(col[11:])] = b
            elif col.startswith("dow_place"):
                day, place = col[9:].split("_P")
                k = DAYS.index(day) * n_p + int(place) - 1
                sites[f"{kind}_day_of_week_place"][k] = b
            elif col.startswith("delay"):
                sites["hazard_delay"][int(col[5:])] = b
    if "refnp_arima_z" in p:
        sites["hazard_walk"] = np.ravel(p["refnp_arima_z"])
    return {f"{prefix}::{k}": v for k, v in sites.items()}


def _ours(i: int) -> tuple[ReferenceEffects, dict[str, Any], Any]:
    out = _epinowcast()[i]
    frame = _frame(out)
    effects = ReferenceEffects(
        "reference", frame, CASES[i]["design"], VALUES, MAX_DELAY
    )
    sites = _sites(out, frame, "reference")
    with jax.enable_x64(True), handlers.trace() as tr:
        parts = handlers.substitute(effects, data=sites)()
    return effects, out, (tr, parts)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_reference_effects_match_epinowcast(i: int) -> None:
    effects, out, (tr, (parametric, non_parametric)) = _ours(i)
    G, T, D = len(effects.frame.place), N_DAYS, MAX_DELAY
    if effects.parametric:
        idx = np.array(out["refp_findex"]) - 1
        want = np.array(out["refp_mean"])[idx].reshape(G, T)
        got = np.asarray(tr["reference::location"]["value"])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        # The scale is its intercept (the engine's scale effects are 0).
        sigma = Delay.from_values(effects.design["delay_family"], VALUES)
        if effects.design["delay_family"] != "exponential":
            np.testing.assert_allclose(out["refp_sd"], float(sigma.sigma))
        # epinowcast's discretised delay (`refp_lh`, delay fastest): log
        # probabilities where it has no report-date model, else logit
        # hazards (the last, 1, is +inf here). Its tail carries its own
        # rounding (`test_reporting_delays.py`), so cells under 1e-8 are
        # left out.
        ref = np.array(out["refp_lh"]).reshape(-1, D)[idx].reshape(G, T, D)
        with jax.enable_x64(True):
            family = effects.design["delay_family"]
            mu = tr["reference::location"]["value"]
            delay = Delay(family, mu, Delay.from_values(family, VALUES).sigma)
            log_p = np.asarray(censored_log_pmf(delay, D))
        if out["ref_as_p"]:
            got, ref = log_p, ref
        else:
            got, ref = np.asarray(parametric)[..., :-1], ref[..., :-1]
            log_p = log_p[..., :-1]
        ok = log_p > np.log(1e-8)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)
    else:
        assert np.all(np.asarray(parametric) == 0)
    if effects.hazard:
        want = np.array(out["refnp_lh"]).reshape(G, T, D)
        np.testing.assert_allclose(
            np.asarray(non_parametric), want, rtol=1e-12, atol=1e-12
        )
    else:
        assert np.all(np.asarray(non_parametric) == 0)


def test_the_frame_reads_weekdays_and_weeks() -> None:
    dates = np.array(
        ["2026-01-03", "2026-01-05", "2026-01-10"], "datetime64[D]"
    )
    frame = ReferenceFrame(
        dates=dates, place=np.zeros(1, int), age=np.zeros(1, int)
    )
    # 2026-01-03 is a Saturday.
    assert list(frame.day_of_week) == [5, 0, 5]
    assert list(frame.week) == [0, 0, 1]


def test_the_effects_sample_forward() -> None:
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.array([0, 1]),
        age=np.array([0, 0]),
    )
    effects = ReferenceEffects(
        "reference", frame, CASES[0]["design"], VALUES, MAX_DELAY
    )
    with handlers.seed(rng_seed=3), handlers.trace() as tr:
        parametric, non_parametric = effects()
    assert parametric.shape == non_parametric.shape == (2, N_DAYS, MAX_DELAY)
    assert np.all(np.isposinf(np.asarray(parametric)[..., -1]))
    assert "reference::mean_day_of_week_place" in tr
    assert tr["reference::mean_day_of_week_place"]["value"].shape == (14,)
    assert math.isfinite(float(np.asarray(non_parametric).sum()))


def test_ci_runs_the_reporting_effect_checks() -> None:
    import yaml

    from .r_live import ROOT

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/heavy.yml").read_text()
    )
    runs = " ".join(
        s.get("run", "") for s in workflow["jobs"]["r-live"]["steps"]
    )
    assert "tests/model/test_reporting_effects.py" in runs


def test_a_weekday_pattern_shapes_the_weekday_draws() -> None:
    import jax.numpy as jnp

    from nowcastrenewal.reporting.effects import ReportEffects

    pattern = np.array([-1.5, 0.5, 0.6, 0.5, 0.4, 0.2, -0.7])
    z = jnp.arange(7.0)
    effects = ReportEffects.__new__(ReportEffects)
    effects.values = {
        "reference_dow_profile": pattern.tolist(),
        "weekday_noise": 0.3,
    }
    np.testing.assert_allclose(
        effects._shaped(z, "reference"),
        pattern + 0.3 * np.arange(7.0),
        rtol=1e-6,
    )
    # Without a pattern the draws are the free profile.
    np.testing.assert_allclose(
        effects._shaped(z, "report"), np.arange(7.0), rtol=1e-6
    )
