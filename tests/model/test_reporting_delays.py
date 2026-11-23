"""The reporting delay's pmfs, against epinowcast, live.

The shared pmfs are checked against primarycensored in `test_pmfs.py`.
Here the reporting delay's hazards are checked against the pinned
epinowcast's own Stan functions (`enw_stan_to_r`); the tests skip only
where R or a package is missing.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import numpy as np
import pytest

from nowcastrenewal.components.pmfs import (
    Delay,
    censored_log_pmf,
    censored_logit_hazard,
)
from nowcastrenewal.reporting.delays import FAMILIES, report_delay_pmf

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
        p = Delay.from_values(case["family"], case["values"])
        rows.append({**case, "dmax": dmax, "mu": p.mu, "sigma": float(p.sigma)})
    return rows


ROWS = _rows()


@pytest.fixture(scope="module")
def epinowcast_hazards() -> list[dict[str, Any]]:
    """epinowcast's discretised logit hazards and log pmfs, one R run.

    At CI's pin (`.github/workflows/test.yml`, job `r-live`)
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
        _skip("needs epinowcast at CI's pin (primarycensored_pmf)")
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
        delay = Delay(r["family"], r["mu"], r["sigma"])
        lh = censored_logit_hazard(delay, r["dmax"])
        lp = censored_log_pmf(delay, r["dmax"])
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


def test_the_report_delay_pmf_is_the_shared_pmf_or_geometric() -> None:
    # A parametric delay is the shared double censored pmf
    # (`test_pmfs.py`); with no parametric part, a geometric delay with
    # hazard 1 / (1 + mean), truncated at the window.
    for case in CASES:
        got = report_delay_pmf(case["family"], case["values"], 21)
        with jax.enable_x64(True):
            delay = Delay.from_values(case["family"], case["values"])
            want = np.exp(np.asarray(censored_log_pmf(delay, 21)))
        np.testing.assert_allclose(got, want, rtol=1e-12)
    h = 1 / (1 + 4.0)
    geometric = h * (1 - h) ** np.arange(21)
    np.testing.assert_allclose(
        report_delay_pmf("none", {"delay_mean": 4.0}, 21),
        geometric / geometric.sum(),
    )


def test_the_families_are_the_grids() -> None:
    assert set(FAMILIES) == {"exponential", "lognormal", "gamma"}


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
