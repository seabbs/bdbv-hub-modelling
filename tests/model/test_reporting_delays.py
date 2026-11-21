"""The reporting delay's discretised pmfs, against the R packages, live.

epinowcast's own Stan functions (`enw_stan_to_r`), the simulator's R
engine (`packages/nowcastfoundation/r/R/delays.R`) and primarycensored are
called directly; the tests skip only where R or a package is missing.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyrenew.deterministic import DeterministicVariable

from nowcastrenewal.model.hazard_delay import hazard_to_pmf
from nowcastrenewal.reporting.delays import (
    FAMILIES,
    DelayParameters,
    ParametricDelay,
    double_censored_log_pmf,
    double_censored_logit_hazard,
    interval_pmf,
)

from .r_live import _skip, require_r, run_r

# Parameters across the grid's ranges (about its 5%, 50% and 95% points):
# lognormal median and sdlog, gamma mean and shape, exponential mean.
DMAX = (21, 42, 84)
CASES: list[dict[str, Any]] = (
    [
        {"family": "lognormal", "values": {"delay_median": m, "delay_sdlog": s}}
        for m, s in itertools.product((1.4, 3.2, 7.3), (0.55, 0.9, 1.5))
    ]
    + [
        {"family": "gamma", "values": {"delay_mean": m, "delay_cv": 1 / k**0.5}}
        for m, k in itertools.product((1.9, 5.0, 13.0), (0.6, 1.33, 1.9))
    ]
    + [
        {"family": "exponential", "values": {"delay_mean": m}}
        for m in (1.9, 5.0, 13.0)
    ]
)
DIST = {"exponential": 1, "lognormal": 2, "gamma": 3}


def _rows() -> list[dict[str, Any]]:
    rows = []
    for case, dmax in itertools.product(CASES, DMAX):
        p = DelayParameters.from_values(case["family"], case["values"])
        rows.append({**case, "dmax": dmax, "mu": p.mu, "sigma": p.sigma})
    return rows


ROWS = _rows()


@pytest.fixture(scope="module")
def epinowcast_hazards() -> list[dict[str, Any]]:
    """epinowcast's discretised logit hazards and log pmfs, one R run.

    At the engine's pin (`packages/nowcastfoundation/r/DESCRIPTION`)
    epinowcast discretises the parametric delay with primarycensored
    (`primarycensored_pmf.stan`).
    """
    require_r(["epinowcast", "cmdstanr"])
    has = run_r(
        'file.exists(system.file("stan", "functions", '
        '"primarycensored_pmf.stan", package = "epinowcast"))',
        {},
        packages=["epinowcast"],
    )
    if not has:
        _skip("needs epinowcast at the engine's pin (primarycensored_pmf)")
    code = """
    # A new target directory: functions are exposed only from a fresh
    # build, not from epinowcast's cached model.
    target <- tempfile("enw-fns-")
    dir.create(target)
    suppressMessages(epinowcast::enw_stan_to_r(
      c("primarycensored.stan", "primarycensored_pmf.stan"),
      verbose = FALSE, target_dir = target
    ))
    lapply(seq_len(nrow(input)), function(i) {
      r <- input[i, ]
      list(
        logit_hazard = discretised_pcens_logit_hazard(
          r$mu, r$sigma, r$dmax, r$dist, 0L
        ),
        log_pmf = discretised_pcens_logit_hazard(
          r$mu, r$sigma, r$dmax, r$dist, 1L
        )
      )
    })
    """
    inputs = [
        {
            "mu": r["mu"],
            "sigma": r["sigma"],
            "dmax": r["dmax"],
            "dist": DIST[r["family"]],
        }
        for r in ROWS
    ]
    return run_r(code, inputs, packages=["epinowcast", "cmdstanr"])


def _ours(r: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with jax.enable_x64(True):
        lh = double_censored_logit_hazard(
            r["family"], r["mu"], r["sigma"], r["dmax"]
        )
        lp = double_censored_log_pmf(
            r["family"], r["mu"], r["sigma"], r["dmax"]
        )
    return np.asarray(lh), np.asarray(lp)


def _survival_before(log_pmf: np.ndarray) -> np.ndarray:
    p = np.exp(log_pmf)
    return 1 - np.concatenate([[0.0], np.cumsum(p)[:-1]])


@pytest.mark.parametrize("i", range(len(ROWS)))
def test_double_censored_delays_match_epinowcast(
    i: int, epinowcast_hazards: list[dict[str, Any]]
) -> None:
    # epinowcast's pmf carries absolute errors up to about 2.5e-13 in the
    # tail (ours stays within 6e-14 of a 50-digit reference, a scratch
    # check), so its logit hazard log p_d - log sum_{i > d} p_i can be off
    # by about 1e-12 / p_d + 1e-12 / sum_{i > d} p_i. The hazards are
    # compared to 1e-9 plus that.
    r, want = ROWS[i], epinowcast_hazards[i]
    lh, lp = _ours(r)
    want_lh = np.array(
        [np.inf if x in (None, "Inf") else x for x in want["logit_hazard"]],
        dtype=float,
    )
    assert np.isposinf(lh[-1])
    assert np.isposinf(want_lh[-1])
    p = np.exp(lp)
    after = np.cumsum(p[::-1])[::-1][1:]
    tol = 1e-9 + 1e-12 / p[:-1] + 1e-12 / after
    got, ref = lh[:-1], want_lh[:-1]
    # epinowcast's hazards are NaN once its running survival, built by
    # subtracting each probability, falls below 0: only where under 1e-12
    # of the reports remain (5e-13 at most here).
    nan = np.isnan(ref)
    assert np.all(after[nan] < 1e-12)
    assert np.all(np.abs(got - ref)[~nan] <= tol[~nan])
    want_lp = np.array(
        [-np.inf if x in (None, "-Inf") else x for x in want["log_pmf"]],
        dtype=float,
    )
    # Its log pmf is NaN in places too, only where under 1e-12 is left.
    lost = np.isnan(want_lp)
    assert np.all(p[lost] < 1e-12)
    np.testing.assert_allclose(
        p[~lost], np.exp(want_lp[~lost]), rtol=1e-9, atol=1e-12
    )


def test_the_hazards_give_the_pmf_to_full_precision() -> None:
    # The hazards and the pmf agree in the far tail too.
    for r in ROWS:
        lh, lp = _ours(r)
        with jax.enable_x64(True):
            p = np.asarray(hazard_to_pmf(jnp.asarray(lh[:-1])))
        np.testing.assert_allclose(p, np.exp(lp), rtol=1e-10, atol=1e-300)


def test_the_double_censored_pmf_sums_to_one() -> None:
    for r in ROWS:
        with jax.enable_x64(True):
            p = np.exp(
                double_censored_log_pmf(
                    r["family"], r["mu"], r["sigma"], r["dmax"]
                )
            )
        assert p.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope="module")
def engine_pmfs() -> list[list[float]]:
    """The simulator R engine's delay pmfs (`delay_parameters`)."""
    code = """
    source("packages/nowcastfoundation/r/R/delays.R")
    lapply(seq_len(length(input$family)), function(i) {
      v <- as.list(input$values[i, , drop = FALSE])
      delay_parameters(input$family[i], v, input$dmax[i])$pmf
    })
    """
    cases = [*CASES, {"family": "none", "values": {"delay_mean": 4.0}}]
    inputs = {
        "family": [c["family"] for c in cases for _ in DMAX],
        "values": [c["values"] for c in cases for _ in DMAX],
        "dmax": [d for _ in cases for d in DMAX],
    }
    return run_r(code, inputs)


def test_the_interval_pmf_matches_the_simulator_engine(
    engine_pmfs: list[list[float]],
) -> None:
    cases = [*CASES, {"family": "none", "values": {"delay_mean": 4.0}}]
    pairs = [(c, d) for c in cases for d in DMAX]
    for (case, dmax), want in zip(pairs, engine_pmfs, strict=True):
        got = interval_pmf(case["family"], case["values"], dmax)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_a_parametric_delay_component_returns_its_pmf() -> None:
    p = DelayParameters.from_values("lognormal", CASES[4]["values"])
    delay = ParametricDelay(
        "delay",
        family="lognormal",
        mu_rv=DeterministicVariable("mu", np.array([p.mu, p.mu + 0.3])),
        sigma_rv=DeterministicVariable("sigma", np.array([p.sigma, p.sigma])),
        dmax=21,
    )
    with jax.enable_x64(True):
        out = np.asarray(delay())
    assert out.shape == (2, 21)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # A larger location delays reports.
    assert (out[1] * np.arange(21)).sum() > (out[0] * np.arange(21)).sum()


def test_the_families_are_the_grids() -> None:
    assert set(FAMILIES) == {"exponential", "lognormal", "gamma"}


@pytest.fixture(scope="module")
def primarycensored_pmfs() -> list[list[float]]:
    """primarycensored's double censored pmfs: a uniform primary day,
    daily secondary censoring, right truncated at the window."""
    code = """
    lapply(seq_len(nrow(input)), function(i) {
      r <- input[i, ]
      x <- 0:(r$dmax - 1)
      pc <- function(p, ...) primarycensored::dpcens(
        x, p, pwindow = 1, swindow = 1, D = r$dmax, ...
      )
      switch(r$family,
        lognormal = pc(plnorm, meanlog = r$mu, sdlog = r$sigma),
        gamma = pc(pgamma, shape = exp(r$mu), rate = r$sigma),
        exponential = pc(pexp, rate = exp(-r$mu))
      )
    })
    """
    rows = [{k: r[k] for k in ("family", "mu", "sigma", "dmax")} for r in ROWS]
    return run_r(code, rows, packages=["primarycensored"])


def test_double_censored_delays_match_primarycensored(
    primarycensored_pmfs: list[list[float]],
) -> None:
    for r, want in zip(ROWS, primarycensored_pmfs, strict=True):
        _, lp = _ours(r)
        np.testing.assert_allclose(np.exp(lp), want, rtol=1e-8, atol=1e-8)


def test_ci_runs_the_live_r_checks() -> None:
    import yaml

    from .r_live import ROOT

    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    job = workflow["jobs"]["r-live"]
    assert job["env"]["R_LIVE_REQUIRED"] == "1"
    runs = " ".join(s.get("run", "") for s in job["steps"])
    packages = " ".join(
        s.get("with", {}).get("packages", "") for s in job["steps"]
    )
    assert "tests/model/test_reporting_delays.py" in runs
    assert "install_cmdstan" in runs
    for pkg in ("epinowcast/epinowcast@", "primarycensored@", "cmdstanr"):
        assert pkg in packages, pkg
