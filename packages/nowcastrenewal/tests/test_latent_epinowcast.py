"""The simulator's renewal and reference events against epinowcast, live.

epinowcast's own Stan functions (exposed with `enw_stan_to_r`) compute
the same two quantities as our latent step where both models have them:
the renewal from R and a generation time with no depletion
(`log_expected_latent_from_r` with `use_pop = 0`), and expected reference
events from infections through a delay and an ascertainment
(`log_expected_obs_from_latent`, with epinowcast's `convolution_matrix`
and `extract_sparse_matrix`). The tests skip only where R or a package is
missing; CI's `r-live` job fails instead.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from model.r_live import run_r
from numpyro import handlers
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable

from nowcastrenewal.components.pmfs import (
    Delay,
    delay_pmf,
    generation_time_pmf,
)
from nowcastrenewal.components.reference import ReferenceEvents
from nowcastrenewal.components.renewal import NoDepletion, run_renewal

N_DAYS = 90
GT = generation_time_pmf(Delay.from_mean_cv("gamma", 6.0, 0.5))
T = np.arange(1, N_DAYS + 1)
LOG_R = np.stack(
    [0.4 * np.cos(T / 20), 0.2 - T / 400, 0.6 * np.sin(T / 15)], axis=1
)
START = np.zeros((len(GT), 3))
START[-1] = [5.0, 2.0, 1.0]
START[:, 1] += 0.5
DELAYS = [
    delay_pmf(Delay.from_mean_cv("gamma", m, 0.5)) for m in (2.0, 5.0, 9.0)
]
ASCERTAINMENT = np.array([0.5, 1.0, 0.2])
INFECTIONS = np.random.default_rng(1).poisson(50, (40, 3)).astype(float)
BEFORE = np.array([40.0, 0.0, 25.0])

_EXPOSE = """
target <- tempfile("enw-fns-")
dir.create(target)
suppressMessages(epinowcast::enw_stan_to_r(
  c("log_expected_latent_from_r.stan", "log_expected_obs_from_latent.stan"),
  verbose = FALSE, target_dir = target
))
"""


@pytest.fixture(scope="module")
def epinowcast() -> dict[str, Any]:
    """epinowcast's renewal and reference events on the inputs, one run."""
    code = (
        _EXPOSE
        + """
    G <- ncol(input$log_r)
    latent <- log_expected_latent_from_r(
      log(input$start), as.vector(input$log_r), (0:(G - 1)) * nrow(input$log_r),
      nrow(input$log_r), nrow(input$start), length(input$rev_gt),
      log(input$rev_gt), nrow(input$start) + nrow(input$log_r), G,
      rep(1, G), 0L, 1
    )
    events <- lapply(seq_len(ncol(input$x)), function(k) {
      pmf <- input$delays[[k]]
      D <- length(pmf)
      x <- c(rep(input$before[k], D - 1), input$x[, k])
      ft <- length(x)
      sparse <- epinowcast::extract_sparse_matrix(
        epinowcast::convolution_matrix(pmf, ft)
      )
      out <- log_expected_obs_from_latent(
        list(log(x)), D, sparse$w, sparse$v, sparse$u, nrow(input$x), 1L,
        rep(log(input$asc[k]), nrow(input$x))
      )
      exp(out[[1]])
    })
    list(latent = lapply(latent, exp), events = events)
    """
    )
    inputs = {
        "start": START.tolist(),
        "log_r": LOG_R.tolist(),
        "rev_gt": np.flip(GT).tolist(),
        "x": INFECTIONS.tolist(),
        "before": BEFORE.tolist(),
        "delays": [d.tolist() for d in DELAYS],
        "asc": ASCERTAINMENT.tolist(),
    }
    return run_r(code, inputs, packages=["epinowcast", "cmdstanr"])


def test_the_renewal_without_depletion_is_epinowcast_s(
    epinowcast: dict[str, Any],
) -> None:
    # Same R, generation time and starting infections: the infections
    # after the start, to 1e-10 relative (epinowcast works on the log
    # scale and back).
    with jax.enable_x64(True):
        out = run_renewal(
            START, GT, np.exp(LOG_R), np.full(3, 1e12), depletion=NoDepletion()
        )
    want = np.array(epinowcast["latent"], dtype=float).T[len(GT) :]
    got = np.asarray(out.infections)[len(GT) :]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_reference_events_are_epinowcast_s_expected_observations(
    epinowcast: dict[str, Any],
) -> None:
    # Infections through each stratum's delay, with its level before the
    # first day, times its ascertainment, to 1e-10 relative.
    events = ReferenceEvents(
        "ref",
        delay_rvs=[
            DeterministicPMF(f"delay_{i}", p) for i, p in enumerate(DELAYS)
        ],
        ascertainment_rv=DeterministicVariable("asc", ASCERTAINMENT),
    )
    with jax.enable_x64(True):
        out = handlers.seed(events.sample, 0)(INFECTIONS, before=BEFORE)
    want = np.array(epinowcast["events"], dtype=float).T
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-10)
