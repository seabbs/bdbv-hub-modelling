"""The shared double censored pmfs, against R primarycensored, live.

Every delay pmf in the models and the simulator comes from
`nowcastrenewal.components.pmfs`; these are its only checks against
primarycensored (`dpcens`, called directly; the tests skip only where R or
a package is missing).
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

from nowcastrenewal.components.pmfs import (
    FAMILIES,
    CensoredPMF,
    Delay,
    censored_log_pmf,
    censored_logit_hazard,
    censored_pmf,
    delay_pmf,
    generation_time_pmf,
)
from nowcastrenewal.model.hazard_delay import hazard_to_pmf

from .r_live import run_r

# Across the simulator grid's ranges and the fitted models' delays: the
# reporting families by location and scale, the hub's imputation delay (a
# generalised gamma in days) and short windows.
CASES: list[dict[str, Any]] = (
    [
        {"family": "lognormal", "mu": np.log(m), "sigma": s}
        for m, s in itertools.product((1.4, 3.2, 7.3), (0.55, 0.9, 1.5))
    ]
    + [
        {"family": "gamma", "mu": np.log(k), "sigma": k / m}
        for m, k in itertools.product((1.9, 5.0, 13.0), (0.6, 1.33, 1.9))
    ]
    + [
        {"family": "exponential", "mu": np.log(m), "sigma": 1.0}
        for m in (1.9, 5.0, 13.0)
    ]
    + [
        {"family": "gengamma", "mu": mu - np.log(24), "sigma": s, "q": q}
        for mu, s, q in (
            (5.12406648030701, 0.97184390713028, 1.07646963898928),
            (4.99016267662049, 0.936895070027868, 0.715805359543864),
        )
    ]
)
WINDOWS = (2, 21, 84)
ROWS = [{"q": 1.0, **c, "n": n} for c, n in itertools.product(CASES, WINDOWS)]


def _delay(r: dict[str, Any]) -> Delay:
    return Delay(r["family"], r["mu"], r["sigma"], r["q"])


@pytest.fixture(scope="module")
def dpcens() -> list[list[float]]:
    """primarycensored's pmfs on 0, ..., n - 1, truncated at n."""
    code = """
    pgg <- function(q, mu, sigma, Q) flexsurv::pgengamma(q, mu, sigma, Q)
    lapply(seq_len(nrow(input)), function(i) {
      r <- input[i, ]
      pc <- function(p, ...) primarycensored::dpcens(
        0:(r$n - 1), p, pwindow = 1, swindow = 1, D = r$n, ...
      )
      switch(r$family,
        lognormal = pc(plnorm, meanlog = r$mu, sdlog = r$sigma),
        gamma = pc(pgamma, shape = exp(r$mu), rate = r$sigma),
        exponential = pc(pexp, rate = exp(-r$mu)),
        gengamma = pc(pgg, mu = r$mu, sigma = r$sigma, Q = r$q)
      )
    })
    """
    return run_r(code, ROWS, packages=["primarycensored", "flexsurv"])


def test_the_pmfs_match_primarycensored(dpcens: list[list[float]]) -> None:
    # primarycensored integrates the generalised gamma numerically (it has
    # no closed form there), good to about 2e-8; ours is closed form and
    # within 1e-13 of a tight quadrature (below).
    for r, want in zip(ROWS, dpcens, strict=True):
        tol = 1e-7 if r["family"] == "gengamma" else 1e-8
        with jax.enable_x64(True):
            got = np.asarray(censored_pmf(_delay(r), r["n"]))
        np.testing.assert_allclose(
            got, np.atleast_1d(want), rtol=tol, atol=1e-10, err_msg=str(r)
        )


@pytest.mark.parametrize("r", [r for r in ROWS if r["family"] == "gengamma"])
def test_the_generalised_gamma_matches_a_tight_quadrature(
    r: dict[str, Any],
) -> None:
    with jax.enable_x64(True):
        got = np.asarray(censored_pmf(_delay(r), r["n"]))
    np.testing.assert_allclose(got, _quad_pmf(_delay(r), r["n"]), atol=1e-13)


@pytest.fixture(scope="module")
def dpcens_fixed() -> dict[str, list[float]]:
    """primarycensored's generation time (left truncated at a day) and
    incubation pmfs, at the lengths `generation_time_pmf` and `delay_pmf`
    choose by default."""
    code = """
    pc <- primarycensored::dpcens
    list(
      gt = pc(1:input$gt_n, pgamma, L = 1, D = input$gt_n + 1,
        shape = 1 / 0.6^2, rate = 1 / 0.6^2 / 15.3),
      gt_ln = pc(1:input$gt_ln_n, plnorm, L = 1, D = input$gt_ln_n + 1,
        meanlog = input$ln_mu, sdlog = input$ln_sigma),
      inc = pc(0:input$inc_n, pgamma, D = input$inc_n + 1,
        shape = 4, rate = 4 / 6.3)
    )
    """
    ln = Delay.from_mean_cv("lognormal", 9.0, 0.4)
    inputs = {
        "gt_n": int(
            np.ceil(Delay.from_mean_cv("gamma", 15.3, 0.6).quantile(0.999))
        ),
        "gt_ln_n": int(np.ceil(ln.quantile(0.999))),
        "ln_mu": float(ln.mu),
        "ln_sigma": float(ln.sigma),
        "inc_n": int(
            np.ceil(Delay.from_mean_cv("gamma", 6.3, 0.5).quantile(0.999))
        ),
    }
    return run_r(code, inputs, packages=["primarycensored"])


def test_fixed_pmfs_match_primarycensored(
    dpcens_fixed: dict[str, list[float]],
) -> None:
    gt = generation_time_pmf(Delay.from_mean_cv("gamma", 15.3, 0.6))
    np.testing.assert_allclose(gt, dpcens_fixed["gt"], rtol=1e-8, atol=1e-12)
    gt_ln = generation_time_pmf(Delay.from_mean_cv("lognormal", 9.0, 0.4))
    np.testing.assert_allclose(
        gt_ln, dpcens_fixed["gt_ln"], rtol=1e-8, atol=1e-12
    )
    inc = delay_pmf(Delay.from_mean_cv("gamma", 6.3, 0.5))
    np.testing.assert_allclose(inc, dpcens_fixed["inc"], rtol=1e-8, atol=1e-12)


def test_fixed_pmfs_run_to_the_999_quantile() -> None:
    d = Delay.from_mean_cv("gamma", 6.3, 0.5)
    n = int(np.ceil(d.quantile(0.999)))
    assert len(delay_pmf(d)) == n + 1
    assert len(generation_time_pmf(d)) == n
    assert len(delay_pmf(d, 10)) == 11
    assert len(generation_time_pmf(d, 10)) == 10
    # Very short delays keep at least lags 1 and 2, or delays 0 and 1.
    short = Delay.from_mean_cv("gamma", 0.05, 0.2)
    assert len(generation_time_pmf(short)) == 2
    assert len(delay_pmf(short)) == 2


def test_the_mean_and_cv_forms_give_the_stated_moments() -> None:
    for family in ("gamma", "lognormal", "exponential"):
        cv = 1.0 if family == "exponential" else 0.7
        d = Delay.from_mean_cv(family, 6.0, cv)
        with jax.enable_x64(True):
            assert float(d.mean()) == pytest.approx(6.0, rel=1e-12)
    with pytest.raises(ValueError, match="no mean and cv"):
        Delay.from_mean_cv("gengamma", 6.0, 0.7)


def test_the_grid_values_give_the_location_and_scale() -> None:
    ln = Delay.from_values(
        "lognormal", {"delay_median": 3.0, "delay_sdlog": 0.8}
    )
    assert (float(ln.mu), float(ln.sigma)) == pytest.approx((np.log(3), 0.8))
    g = Delay.from_values("gamma", {"delay_mean": 5.0, "delay_cv": 0.5})
    assert (float(g.mu), float(g.sigma)) == pytest.approx((np.log(4), 0.8))
    e = Delay.from_values("exponential", {"delay_mean": 5.0})
    assert float(e.mu) == pytest.approx(np.log(5))


def test_log_pmf_and_logit_hazard_agree_with_the_pmf() -> None:
    for r in ROWS:
        if r["n"] < 3:
            continue
        with jax.enable_x64(True):
            d = _delay(r)
            p = np.asarray(censored_pmf(d, r["n"]))
            lp = np.asarray(censored_log_pmf(d, r["n"]))
            lh = np.asarray(censored_logit_hazard(d, r["n"]))
            from_h = np.asarray(hazard_to_pmf(jnp.asarray(lh[:-1])))
        np.testing.assert_allclose(np.exp(lp), p, rtol=1e-12)
        assert np.isposinf(lh[-1])
        np.testing.assert_allclose(from_h, p, rtol=1e-10, atol=1e-300)


def _quad_pmf(delay: Delay, n: int) -> np.ndarray:
    from scipy import integrate

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        with jax.enable_x64(True):
            return float(delay.cdf_and_survival(x)[0])

    pmf = np.array(
        [
            integrate.quad(
                lambda u, d=d: cdf(d + 1 - u) - cdf(d - u),
                0,
                1,
                epsabs=1e-15,
                epsrel=1e-13,
                limit=200,
            )[0]
            for d in range(n)
        ]
    )
    return pmf / pmf.sum()


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("cv", [0.3, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("mean", [2.0, 8.0, 15.0])
def test_the_pmf_is_close_in_float32(
    family: str, cv: float, mean: float
) -> None:
    # The fitted models run in float32: the incomplete gamma is good to
    # about 1e-6 there, and the second difference scales that by the delay.
    with jax.enable_x64(False):
        got = np.asarray(censored_pmf(Delay.from_mean_cv(family, mean, cv), 61))
        assert got.dtype == np.float32
        assert np.all(np.isfinite(np.log(got)))
    with jax.enable_x64(True):
        ref = _quad_pmf(Delay.from_mean_cv(family, mean, cv), 61)
    assert np.max(np.abs(got - ref)) < 4e-5
    big = ref > 1e-4
    assert np.max(np.abs(got[big] / ref[big] - 1)) < 2e-2


def test_the_pmf_has_gradients_in_its_parameters() -> None:
    def mean_delay(mean: jax.Array, cv: jax.Array) -> jax.Array:
        p = censored_pmf(Delay.from_mean_cv("gamma", mean, cv), 40)
        return (p * jnp.arange(40)).sum()

    g = jax.grad(mean_delay, argnums=(0, 1))(5.0, 0.7)
    assert np.all(np.isfinite(g))
    assert float(g[0]) > 0


def test_the_component_samples_its_parameters() -> None:
    rv = CensoredPMF(
        "incubation",
        "lognormal",
        28,
        mean_rv=DistributionalVariable(
            "incubation_mean", dist.TruncatedNormal(6.3, 0.54, low=1.0)
        ),
        sd_rv=DistributionalVariable(
            "incubation_sd", dist.TruncatedNormal(3.5, 0.8, low=1.0)
        ),
        record=True,
    )
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=1):
        pmf = rv()
    assert {"incubation_mean", "incubation_sd", "incubation::pmf"} <= set(tr)
    mean, sd = tr["incubation_mean"]["value"], tr["incubation_sd"]["value"]
    with jax.enable_x64(True):
        want = censored_pmf(
            Delay.from_mean_cv("lognormal", mean, sd / mean), 29
        )
    # The component runs in float32 here (`test_the_pmf_is_close_in_float32`).
    np.testing.assert_allclose(pmf, want, atol=4e-5)


def test_the_component_takes_a_location_and_scale_per_group() -> None:
    d = Delay.from_values(
        "lognormal", {"delay_median": 3.2, "delay_sdlog": 0.9}
    )
    rv = CensoredPMF(
        "delay",
        "lognormal",
        20,
        mu_rv=DeterministicVariable("mu", np.array([d.mu, d.mu + 0.3])),
        sigma_rv=DeterministicVariable("sigma", np.array([d.sigma] * 2)),
    )
    with jax.enable_x64(True):
        out = np.asarray(rv())
    assert out.shape == (2, 21)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # A larger location delays reports.
    assert (out[1] * np.arange(21)).sum() > (out[0] * np.arange(21)).sum()


def test_the_component_needs_one_pair_of_parameters() -> None:
    one = DeterministicVariable("x", 1.0)
    with pytest.raises(ValueError, match="give mean_rv"):
        CensoredPMF("d", "gamma", 5, mean_rv=one)
    with pytest.raises(ValueError, match="give mean_rv"):
        CensoredPMF(
            "d", "gamma", 5, mean_rv=one, sd_rv=one, mu_rv=one, sigma_rv=one
        )
    with pytest.raises(ValueError, match="family"):
        Delay("weibull", 1.0)


def test_the_families() -> None:
    assert set(FAMILIES) == {"exponential", "lognormal", "gamma", "gengamma"}


def test_ci_runs_the_pmf_checks() -> None:
    import yaml

    from .r_live import ROOT

    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    job = workflow["jobs"]["r-live"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    packages = " ".join(
        s.get("with", {}).get("packages", "") for s in job["steps"]
    )
    assert "tests/model/test_pmfs.py" in runs
    for pkg in ("primarycensored@", "flexsurv"):
        assert pkg in packages, pkg
