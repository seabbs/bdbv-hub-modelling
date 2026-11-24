"""The hub's onset imputation delays, refitted live in R (`HUB_GENGAMMA`).

The hub fits a generalised gamma in hours to the delays it published at
2026-10-17 (auxiliary-data/imputation-details.md). Releases before
2026-11-01 used flexsurv with status 1, so each delay is exact; from
2026-11-01 it is double interval censored with primarycensored. This
refits both from the hub's archived delays and compares each
`(mu, sigma, Q)` with the constants in `nowcastbdbv.data.delays`, to a
relative tolerance of 1e-3 for flexsurv and 5e-3 for primarycensored's
`fitdistdoublecens`, whose optimiser stops within about 0.2% of the
optimum (1.6.0 gives sigma 0.9385 and Q 0.7147 for 0.9369 and 0.7158).
"""

from __future__ import annotations

import numpy as np
import pytest
from model.r_live import run_r

from nowcastbdbv.data.delays import HUB_GENGAMMA

URL = (
    "https://raw.githubusercontent.com/InsightNet-US/BDBV-Modeling-Hub/"
    "d67f1b74e0ecd1b32bb3d57554d871502acb2eca/auxiliary-data/"
    "archived-delay-data/weekly-reporting-delays-by-province-2026-10-17.csv"
)
"""The hub's delays at 2026-10-17, at a fixed commit."""

FIT = """
suppressPackageStartupMessages({
  library(flexsurv)
  library(primarycensored)
})
d <- read.csv(input$url)$days_sym_to_report
d <- d[d >= 0]
exact <- flexsurvreg(
  Surv(ifelse(d == 0, 1, 24 * d), 24 * (d + 1) - 1, rep(1, length(d)),
       type = "interval") ~ 1,
  dist = "gengamma"
)
cens <- data.frame(left = 24 * d, right = 24 * (d + 1), pwindow = 24, D = Inf)
double_censored <- fitdistdoublecens(
  cens, distr = "gengamma", start = list(mu = 5, sigma = 1, Q = 0.5)
)
list(
  exact = unname(c(
    exact$coefficients["mu"], exp(exact$coefficients["sigma"]),
    exact$coefficients["Q"]
  )),
  double_censored = unname(double_censored$estimate[c("mu", "sigma", "Q")])
)
"""


@pytest.fixture(scope="module")
def refit() -> dict[str, list[float]]:
    return run_r(FIT, {"url": URL}, packages=["flexsurv", "primarycensored"])


@pytest.mark.parametrize(
    ("fit", "rtol"), [("exact", 1e-3), ("double_censored", 5e-3)]
)
def test_the_hub_s_delay_fit_matches_a_live_refit(
    refit: dict[str, list[float]], fit: str, rtol: float
) -> None:
    np.testing.assert_allclose(refit[fit], HUB_GENGAMMA[fit], rtol=rtol)
