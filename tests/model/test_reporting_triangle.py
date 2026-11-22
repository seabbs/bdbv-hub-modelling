"""The reporting triangle's expectation, against epinowcast, live.

epinowcast's own Stan functions (`combine_logit_hazards`,
`hazard_to_log_prob`, `expected_obs`, exposed with `enw_stan_to_r`) and its
report-day aggregation lookups (`enw_report(structural = )`) are called
directly; the tests skip only where R or a package is missing.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nowcastrenewal.reporting.delays import double_censored_logit_hazard
from nowcastrenewal.reporting.triangle import (
    combine_logit_hazards,
    log_expected_counts,
    log_report_probs,
    report_day_groups,
)

from .r_live import run_r

G, T, D = 2, 9, 6
REPORT_DAYS = ("Monday", "Thursday")
# 2024-01-01 is a Monday; group 2 has an outage over days 3 and 4.
OUTAGE = (2, 3, 4)


def _inputs() -> dict[str, Any]:
    rng = np.random.default_rng(20261122)
    with jax.enable_x64(True):
        refp = np.asarray(
            double_censored_logit_hazard(
                "lognormal", np.log([2.0, 4.0, 3.0]), [0.6, 0.9, 1.2], D
            )
        )
    return {
        # One parametric pmf per (group, reference day), from three.
        "refp_lh": refp[:, :-1],
        "dpmfs": rng.integers(0, 3, size=(G, T)),
        "refnp_lh": rng.normal(0, 0.5, size=(G, T, D)),
        "srdlh": rng.normal(0, 0.5, size=(G, T + D - 1)),
        "log_lambda": rng.normal(3, 1, size=(G, T)),
    }


INPUTS = _inputs()


def _reports() -> np.ndarray:
    # Report days by report date, 0, ..., T + D - 2.
    days = np.datetime64("2024-01-01") + np.arange(T + D - 1)
    names = [str(np.datetime_as_string(d)) for d in days]
    weekday = [np.datetime64(n).astype("datetime64[D]").item() for n in names]
    on = np.array([w.strftime("%A") in REPORT_DAYS for w in weekday])
    reports = np.tile(on, (G, 1)).astype(int)
    g, a, b = OUTAGE
    reports[g - 1, a : b + 1] = 0
    return reports


REPORTS = _reports()


def _parametric() -> np.ndarray:
    refp = np.concatenate([INPUTS["refp_lh"], np.full((3, 1), np.inf)], axis=1)
    return refp[INPUTS["dpmfs"]]


@pytest.fixture(scope="module")
def epinowcast_expected() -> dict[str, Any]:
    """epinowcast's combined logit hazards and log expected counts."""
    code = """
    target <- tempfile("enw-fns-")
    dir.create(target)
    suppressMessages(epinowcast::enw_stan_to_r(
      c("hazard.stan", "combine_logit_hazards.stan", "expected_obs.stan"),
      verbose = FALSE, target_dir = target
    ))
    suppressMessages(library(data.table))
    G <- input$G; T <- input$T; D <- input$D
    d0 <- as.Date("2024-01-01")
    obs <- CJ(.place = seq_len(G), reference_date = d0 + 0:(T - 1),
      delay = 0:(D - 1))
    obs[, report_date := reference_date + delay][, delay := NULL]
    obs <- obs[report_date <= d0 + T - 1][, confirm := 1L]
    pobs <- suppressMessages(suppressWarnings(
      epinowcast::enw_preprocess_data(obs, by = ".place", max_delay = D)
    ))
    s <- epinowcast::enw_structural_reporting_metadata(pobs)
    rep_on <- input$reports
    s[, report := rep_on[cbind(.group, as.integer(report_date - d0) + 1L)]]
    rep <- epinowcast::enw_report(
      structural = s[, .(.group, date, report_date, report)], data = pobs
    )
    n_sel <- rep$data$rep_agg_n_selected
    sel <- rep$data$rep_agg_selected_idx
    refp <- rbind(t(as.matrix(input$refp_lh)), Inf)  # (D, pmfs)
    dpmfs <- as.vector(t(input$dpmfs)) + 1L
    refnp <- input$refnp_lh
    refnp_flat <- unlist(lapply(seq_len(G), function(g) {
      unlist(lapply(seq_len(T), function(t) refnp[g, t, ]))
    }))
    rep_t <- T + D - 1
    srdlh <- as.vector(t(input$srdlh))
    # Stan's 2-d integer arrays are lists of rows here.
    rows <- function(m) lapply(seq_len(nrow(m)), function(i) m[i, ])
    rdlurd <- rows(matrix(seq_len(G * rep_t), G, rep_t, byrow = TRUE))
    out <- list(lh = list(), lexp = list(), lexp_agg = list())
    for (g in seq_len(G)) for (t in seq_len(T)) {
      i <- (g - 1) * T + t
      l <- min(D, T - t + 1)
      p <- ((g - 1) * T + (t - 1)) * D + 1
      lh <- combine_logit_hazards(
        i, rdlurd, srdlh, refp, dpmfs, 1L, 1L, g, t, l, refnp_flat, 1L, p
      )
      tar <- input$log_lambda[g, t]
      none <- array(0L, 0)
      out$lh[[i]] <- lh
      out$lexp[[i]] <- expected_obs(tar, lh, l, 0L, 0L, none, list())
      out$lexp_agg[[i]] <- expected_obs(
        tar, lh, l, 0L, 1L, n_sel[g, t, 1:l],
        rows(matrix(sel[g, t, 1:l, 1:l], l, l))
      )
    }
    out
    """
    inputs = {
        "G": G,
        "T": T,
        "D": D,
        "reports": REPORTS.tolist(),
        **{k: np.asarray(v).tolist() for k, v in INPUTS.items()},
    }
    return run_r(code, inputs, packages=["epinowcast", "cmdstanr"])


def _cells() -> list[tuple[int, int, int]]:
    # (group, reference day, delays in the triangle), in epinowcast's order.
    return [(g, t, min(D, T - t)) for g in range(G) for t in range(T)]


def _as_float(xs: Any) -> np.ndarray:
    # A one-delay cell comes back as a scalar (`auto_unbox`).
    xs = xs if isinstance(xs, list) else [xs]
    named = {"Inf": np.inf, "-Inf": -np.inf, None: -np.inf}
    return np.array([named.get(x, x) for x in xs], dtype=float)


def _ours(aggregate: bool) -> tuple[np.ndarray, np.ndarray]:
    with jax.enable_x64(True):
        lh = combine_logit_hazards(
            jnp.asarray(_parametric()),
            jnp.asarray(INPUTS["refnp_lh"]),
            jnp.asarray(INPUTS["srdlh"]),
        )
        lexp = log_expected_counts(
            jnp.asarray(INPUTS["log_lambda"]),
            lh,
            jnp.asarray(REPORTS) if aggregate else None,
        )
    return np.asarray(lh), np.asarray(lexp)


def test_combined_logit_hazards_match_epinowcast(
    epinowcast_expected: dict[str, Any],
) -> None:
    lh, _ = _ours(aggregate=False)
    for k, (g, t, n) in enumerate(_cells()):
        want = _as_float(epinowcast_expected["lh"][k])
        np.testing.assert_allclose(lh[g, t, :n], want, rtol=1e-12, atol=0)


def test_log_expected_counts_match_epinowcast(
    epinowcast_expected: dict[str, Any],
) -> None:
    _, lexp = _ours(aggregate=False)
    for k, (g, t, n) in enumerate(_cells()):
        want = _as_float(epinowcast_expected["lexp"][k])
        np.testing.assert_allclose(lexp[g, t, :n], want, rtol=1e-12, atol=0)


def test_report_day_aggregation_matches_epinowcast(
    epinowcast_expected: dict[str, Any],
) -> None:
    _, lexp = _ours(aggregate=True)
    for k, (g, t, n) in enumerate(_cells()):
        want = _as_float(epinowcast_expected["lexp_agg"][k])
        got = lexp[g, t, :n]
        assert np.array_equal(np.isneginf(got), np.isneginf(want)), (g, t)
        on = np.isfinite(want)
        np.testing.assert_allclose(got[on], want[on], rtol=1e-12, atol=0)


def test_report_probs_with_a_last_hazard_of_one_sum_to_one() -> None:
    with jax.enable_x64(True):
        lh = jnp.asarray(_parametric()) + jnp.asarray(INPUTS["refnp_lh"])
        p = np.exp(np.asarray(log_report_probs(lh)))
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


def test_a_finite_last_hazard_leaves_reports_beyond_the_window() -> None:
    # As in epinowcast, a hazard with no parametric part need not end at 1.
    with jax.enable_x64(True):
        p = np.exp(np.asarray(log_report_probs(jnp.zeros((1, 4)))))
    np.testing.assert_allclose(p, [[0.5, 0.25, 0.125, 0.0625]])


def test_report_day_groups_send_each_delay_to_its_next_report() -> None:
    # Reports on delays 1 and 4 of a window of 6: delays 0 and 1 are
    # reported on delay 1, 2 to 4 on delay 4, and 5 never.
    reports = jnp.asarray([[0, 1, 0, 0, 1, 0]])
    to = np.asarray(report_day_groups(reports, window=6))
    np.testing.assert_array_equal(to, [[[1, 1, 4, 4, 4, -1]]])


def test_ci_runs_the_triangle_checks() -> None:
    import yaml

    from .r_live import ROOT

    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    runs = " ".join(
        s.get("run", "") for s in workflow["jobs"]["r-live"]["steps"]
    )
    assert "tests/model/test_reporting_triangle.py" in runs


def test_the_triangle_component_composes_its_parts() -> None:
    from numpyro import handlers
    from pyrenew.deterministic import DeterministicVariable

    from nowcastrenewal.reporting.triangle import ReportingTriangle

    triangle = ReportingTriangle(
        "triangle",
        parametric_rv=DeterministicVariable("p", _parametric()),
        non_parametric_rv=DeterministicVariable("np", INPUTS["refnp_lh"]),
        report_rv=DeterministicVariable("r", INPUTS["srdlh"]),
        reports=REPORTS,
    )
    with jax.enable_x64(True), handlers.trace() as tr:
        got = np.asarray(triangle(log_lambda=jnp.asarray(INPUTS["log_lambda"])))
    _, want = _ours(aggregate=True)
    np.testing.assert_array_equal(got, want)
    assert "triangle::log_expected" in tr
