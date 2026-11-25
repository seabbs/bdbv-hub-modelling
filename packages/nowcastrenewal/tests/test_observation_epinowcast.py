"""Observation log densities and missing reference dates, against epinowcast.

Each case runs epinowcast's model at the pin with fixed parameters, the
case's reports and the likelihood on (`tests/model/r/reporting_cases.R`,
`observation_case`), and reads its log likelihood by snapshot. The
simulator's expected reports (`ReportingExpectation`) and count noise
(`ObservedReports`) score the same reports, for Poisson, NB2 and NB1.

With a missingness model of one intercept (`enw_missing(~1)`), the
fitted model's expected known onsets and reports with no onset by report
day (`model.observation.onset_expectations`) are checked against
epinowcast's expected cells and its expected reports with no reference
date, from its own functions.
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
from scipy import stats
from test_reporting_effects import FIRST, VALUES, _design, _frame, _sites
from test_reporting_expectation import _as_float, _report

from nowcastrenewal.components.pmfs import Delay, censored_log_pmf
from nowcastrenewal.model.observation import onset_expectations
from nowcastrenewal.reporting.expectation import ReportingExpectation
from nowcastrenewal.reporting.observation import ObservedReports

N_DAYS = 28

CASES: list[dict[str, Any]] = [
    {
        "design": _design(),
        "family": "poisson",
        "n_places": 2,
        "n_ages": 1,
        "max_delay": 8,
    },
    {
        "design": _design(
            delay_form="hybrid", delay_family="gamma", delay_change="drift"
        ),
        "family": "negbin",
        "sqrt_phi": 0.4,
        "n_places": 2,
        "n_ages": 2,
        "max_delay": 10,
    },
    {
        "design": _design(delay_family="exponential", delay_dow="yes"),
        "family": "negbin",
        "sqrt_phi": 1.5,
        "n_places": 1,
        "n_ages": 1,
        "max_delay": 5,
    },
    {
        "design": _design(delay_change="step", delay_places="independent"),
        "family": "negbin1d",
        "sqrt_phi": 0.6,
        "n_places": 3,
        "n_ages": 1,
        "max_delay": 14,
    },
    {
        "design": _design(delay_form="non_parametric", delay_change="walk"),
        "family": "negbin1d",
        "sqrt_phi": 0.3,
        "n_places": 2,
        "n_ages": 1,
        "max_delay": 8,
    },
]

# One stratum each: at the pin, epinowcast's lookup of reports by
# reference date for the missingness model (`enw_reference_by_report`)
# fails with more than one.
MISSING_CASES: list[dict[str, Any]] = [
    {
        "design": _design(),
        "family": "poisson",
        "n_places": 1,
        "n_ages": 1,
        "max_delay": 8,
        "missing_prop": 0.2,
    },
    {
        "design": _design(delay_family="gamma"),
        "family": "negbin",
        "sqrt_phi": 0.5,
        "n_places": 1,
        "n_ages": 1,
        "max_delay": 12,
        "missing_prop": 0.45,
    },
]


def _log_lambda(seed: int, n_strata: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.log(np.linspace(3.0, 30.0, N_DAYS))
    return trend[None, :] + rng.normal(0, 0.2, (n_strata, N_DAYS))


def _counts(seed: int, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reports by place, age, reference day and delay, and with no date."""
    rng = np.random.default_rng(seed)
    shape = (case["n_places"], case["n_ages"], N_DAYS, case["max_delay"])
    mean = 6.0 * 0.6 ** np.arange(case["max_delay"])
    cells = rng.negative_binomial(2, 2 / (2 + mean), size=shape)
    missing = rng.poisson(3.0, size=shape[:3])
    return cells, missing


def _phi(case: dict[str, Any], log_lambda: np.ndarray) -> float:
    """$\\phi$ the simulator gives the case's family (1 for Poisson)."""
    if case["family"] == "poisson":
        return 1.0
    return float(_observed(case, log_lambda).phi[0])


def _observed(case: dict[str, Any], log_lambda: np.ndarray) -> ObservedReports:
    n = log_lambda.shape[0]
    return ObservedReports(
        "obs",
        case["family"],
        np.full(n, case.get("sqrt_phi", 1.0)),
        np.exp(log_lambda).T,
        case["design"]["delay_family"],
        VALUES,
        case["max_delay"],
    )


def _input(i: int, case: dict[str, Any]) -> dict[str, Any]:
    delay = Delay.from_values(case["design"]["delay_family"], VALUES)
    n_strata = case["n_places"] * case["n_ages"]
    log_lambda = _log_lambda(700 + i, n_strata)
    cells, missing = _counts(800 + i, case)
    out = {
        **case,
        "values": VALUES,
        "seed": 900 + i,
        "n_days": N_DAYS,
        "first_date": FIRST,
        "mean_int": float(delay.mu),
        "sd_int": float(delay.sigma),
        "log_lambda": log_lambda.tolist(),
        "phi": _phi(case, log_lambda),
        "new_counts": cells.ravel().tolist(),
    }
    if "missing_prop" in case:
        out["missing_counts"] = missing.ravel().tolist()
    return out


@functools.cache
def _epinowcast(cases: str) -> list[dict[str, Any]]:
    code = """
    source("tests/model/r/reporting_cases.R")
    lapply(input, function(x) observation_case(jsonlite::fromJSON(x)))
    """
    chosen = CASES if cases == "obs" else MISSING_CASES
    offset = 0 if cases == "obs" else len(CASES)
    inputs = [json.dumps(_input(offset + i, c)) for i, c in enumerate(chosen)]
    return run_r(
        code,
        inputs,
        packages=["epinowcast", "cmdstanr", "data.table", "posterior"],
    )


def _log_expected(
    out: dict[str, Any], case: dict[str, Any], log_lambda: np.ndarray
) -> tuple[np.ndarray, Any]:
    """Our log expected reports `(G, T, D)` on epinowcast's draws."""
    frame = _frame(out)
    design = {**case["design"], **_report()}
    D = case["max_delay"]
    model = ReportingExpectation("expected", frame, design, VALUES, D)
    sites = _sites(out, frame, "expected::reference", D)
    with jax.enable_x64(True):
        got = handlers.substitute(model, data=sites)(
            log_lambda=jnp.asarray(log_lambda)
        )
    return np.asarray(got), frame


def _cells_by_stratum(cells: np.ndarray, frame: Any) -> np.ndarray:
    """Reports `(G, T, D)` in epinowcast's stratum order."""
    return cells[frame.place, frame.age]


@pytest.mark.parametrize("i", range(len(CASES)))
def test_observation_log_densities_match_epinowcast(i: int) -> None:
    case, out = CASES[i], _epinowcast("obs")[i]
    log_lambda = _log_lambda(700 + i, case["n_places"] * case["n_ages"])
    got, frame = _log_expected(out, case, log_lambda)
    # The expected cells first, as `test_reporting_expectation.py`.
    sg, st = np.array(out["sg"]) - 1, np.array(out["st"]) - 1
    sdmax = np.array(out["sdmax"])
    ours = np.concatenate(
        [got[g, t, :n] for g, t, n in zip(sg, st, sdmax, strict=True)]
    )
    want = _as_float(out["log_expected"])
    on = np.isfinite(want)
    np.testing.assert_allclose(ours[on], want[on], rtol=1e-10, atol=1e-10)
    # Then each snapshot's log likelihood over its reported cells, those
    # reported by the last day.
    cells, _ = _counts(800 + i, case)
    counts = _cells_by_stratum(cells, frame)
    obs = _observed(case, log_lambda)
    with jax.enable_x64(True), handlers.trace() as tr:
        obs(log_expected=jnp.asarray(got), counts=jnp.asarray(counts))
        site = tr["obs::counts"]
        log_p = np.asarray(site["fn"].log_prob(site["value"]))
    T = got.shape[1]

    def by_snapshot(x: np.ndarray) -> np.ndarray:
        snaps = zip(sg, st, sdmax, strict=True)
        return np.array([x[g, t, : min(n, T - t)].sum() for g, t, n in snaps])

    # epinowcast's to 1e-10 against SciPy's density at our expected cells
    # and phi, so the families and phi agree exactly; numpyro's negative
    # binomial (through JAX's `betaln`) is good to about 1e-7 relative,
    # so ours is held to 1e-6.
    mu, phi = np.exp(got), obs.phi[:, None, None]
    if case["family"] == "poisson":
        exact = stats.poisson.logpmf(counts, mu)
    else:
        size = phi * mu if case["family"] == "negbin1d" else phi + 0 * mu
        exact = stats.nbinom.logpmf(counts, size, size / (size + mu))
    np.testing.assert_allclose(
        by_snapshot(exact), out["log_lik"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        by_snapshot(log_p), out["log_lik"], rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("i", range(len(MISSING_CASES)))
def test_missing_reference_dates_match_epinowcast(i: int) -> None:
    case, out = MISSING_CASES[i], _epinowcast("missing")[i]
    k = len(CASES) + i
    log_lambda = _log_lambda(700 + k, case["n_places"] * case["n_ages"])
    np.testing.assert_allclose(out["missing_prop"], case["missing_prop"])
    frame = _frame(out)
    D = case["max_delay"]
    family = case["design"]["delay_family"]
    with jax.enable_x64(True):
        pmf = np.exp(
            np.asarray(censored_log_pmf(Delay.from_values(family, VALUES), D))
        )
    # Known cells: epinowcast's expected cells times 1 - pi; ours by
    # onset day, summed over the delays reported by the last day.
    want_cells = np.exp(_as_float(out["log_expected"]))
    sg, st = np.array(out["sg"]) - 1, np.array(out["st"]) - 1
    sdmax = np.array(out["sdmax"])
    starts = np.concatenate([[0], np.cumsum(sdmax)[:-1]])
    pi = case["missing_prop"]
    T = N_DAYS
    hub = np.zeros(D)
    hub[0] = 1.0
    lam = np.exp(log_lambda)
    with jax.enable_x64(True):
        res = onset_expectations(
            jnp.asarray(lam.T),
            jnp.broadcast_to(jnp.asarray(pmf), (T, D)),
            jnp.full((T, len(frame.place)), pi),
            jnp.asarray(hub),
            T - 1,
        )
    known = np.asarray(res["known"])
    imputed = np.asarray(res["imputed"])
    for g, t, n, s0 in zip(sg, st, sdmax, starts, strict=True):
        seen = want_cells[s0 : s0 + min(n, T - t)].sum() * (1 - pi)
        np.testing.assert_allclose(known[t, g], seen, rtol=1e-9)
    # Reports with no reference date by report day, where epinowcast
    # models them: report days with every reference date in the window.
    dates = np.array(out["dates"], dtype="datetime64[D]")
    groups = np.array(out["missing_group"]) - 1
    days = np.searchsorted(
        dates, np.array(out["missing_date"], "datetime64[D]")
    )
    want = np.exp(_as_float(out["missing_log_expected"]))
    np.testing.assert_allclose(imputed[days, groups], want, rtol=1e-9)


def test_ci_runs_the_observation_checks() -> None:
    import yaml
    from model.r_live import ROOT

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/tests.yml").read_text()
    )
    steps = workflow["jobs"]["r-live"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    path = "packages/nowcastrenewal/tests/test_observation_epinowcast.py"
    assert path in runs
