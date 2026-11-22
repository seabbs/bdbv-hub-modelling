# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The reporting triangle's expectation.

For group $g$, reference day $t$ and delay $d$ the logit hazard of a
report is the sum of a parametric part (`delays`), a non-parametric part
(effects by delay and reference day) and a report-date part (effects on
day $t + d$):

$$\operatorname{logit} h_{g,t,d} = \eta^{p}_{g,t,d} + \eta^{np}_{g,t,d}
+ \eta^{r}_{g,t+d}.$$

The probability of a report at delay $d$ is
$p_{g,t,d} = h_{g,t,d} \prod_{j<d} (1 - h_{g,t,j})$, and the expected
count is $\lambda_{g,t}\,p_{g,t,d}$. Where reports are made on some days
only, each delay's probability moves to the next report day in the window
and is lost if there is none.

PyRenew has no reporting triangle, so these are added here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

__all__ = [
    "ReportingTriangle",
    "combine_logit_hazards",
    "log_expected_counts",
    "log_report_probs",
    "report_day_groups",
]


def _by_report_date(x: Array, window: int) -> Array:
    """Return `x[..., t + d]`, shape `(..., T, window)`."""
    n = x.shape[-1] - window + 1
    idx = jnp.arange(n)[:, None] + jnp.arange(window)[None, :]
    return x[..., idx]


def combine_logit_hazards(
    parametric: Array, non_parametric: Array, report: Array
) -> Array:
    r"""Return the logit hazards of report, $(G, T, D)$.

    Parameters
    ----------
    parametric : Array
        $\eta^{p}$, shape $(G, T, D)$; its last delay is usually
        $+\infty$ (`components.pmfs.censored_logit_hazard`).
    non_parametric : Array
        $\eta^{np}$, broadcast against `parametric`.
    report : Array
        $\eta^{r}$ by report day, shape $(G, T + D - 1)$.

    Returns
    -------
    Array
        Shape $(G, T, D)$.
    """
    window = parametric.shape[-1]
    return parametric + non_parametric + _by_report_date(report, window)


def log_report_probs(logit_hazard: Array) -> Array:
    r"""Return $\log p_d = \log h_d + \sum_{j<d} \log(1 - h_j)$.

    Unlike `model.hazard_delay.hazard_to_pmf`, the last hazard is used as
    given, so a finite one leaves some reports beyond the window.

    Parameters
    ----------
    logit_hazard : Array
        Logit hazards on the last axis.

    Returns
    -------
    Array
        Log probabilities, the same shape.
    """
    log_h = jax.nn.log_sigmoid(logit_hazard)
    log_1mh = jax.nn.log_sigmoid(-logit_hazard)
    # Survival to each delay: an exclusive running sum, so a last hazard
    # of 1 (a logit of +inf) never meets its own log(1 - h) = -inf.
    before = jnp.cumsum(log_1mh[..., :-1], axis=-1)
    before = jnp.concatenate([jnp.zeros_like(log_h[..., :1]), before], -1)
    return log_h + before


def report_day_groups(reports: Array, window: int) -> Array:
    r"""Return the delay each delay's reports are made on, or -1 for none.

    Parameters
    ----------
    reports : Array
        1 on report days, 0 otherwise, by report day: shape
        $(G, T + D - 1)$.
    window : int
        Delays in the window, $D$.

    Returns
    -------
    Array
        Integer shape $(G, T, D)$: for each delay, the first report day at
        or after it within the window, or -1.
    """
    on = _by_report_date(jnp.asarray(reports), window) > 0
    d = jnp.arange(window)
    # The next report day at or after each delay: a reverse running min.
    nxt = jnp.where(on, d, window)
    nxt = jnp.flip(jax.lax.cummin(jnp.flip(nxt, -1), axis=nxt.ndim - 1), -1)
    return jnp.where(nxt < window, nxt, -1)


def log_expected_counts(
    log_lambda: Array, logit_hazard: Array, reports: Array | None = None
) -> Array:
    r"""Return $\log \lambda_{g,t} + \log p_{g,t,d}$, shape $(G, T, D)$.

    Parameters
    ----------
    log_lambda : Array
        Log expected counts by reference day, $(G, T)$.
    logit_hazard : Array
        `combine_logit_hazards`, $(G, T, D)$.
    reports : Array, optional
        Report days (`report_day_groups`); `None` for every day. Each
        delay's probability then moves to its report day, and a day with
        no report has probability 0 ($\log 0 = -\infty$).

    Returns
    -------
    Array
        Shape $(G, T, D)$.
    """
    log_p = log_report_probs(logit_hazard)
    if reports is not None:
        window = logit_hazard.shape[-1]
        to = report_day_groups(reports, window)
        hit = to[..., None, :] == jnp.arange(window)[:, None]
        log_p = jax.nn.logsumexp(
            jnp.where(hit, log_p[..., None, :], -jnp.inf), axis=-1
        )
    return log_lambda[..., None] + log_p


class ReportingTriangle(RandomVariable):
    """The log expected counts of a reporting triangle from its parts.

    Parameters
    ----------
    name : str
        Name of the component; the result is recorded as
        `"{name}::log_expected"`.
    parametric_rv, non_parametric_rv, report_rv : RandomVariable
        Return the logit hazards' parts (`combine_logit_hazards`).
    reports : array_like, optional
        Report days by report day, $(G, T + D - 1)$; `None` for every day.
    """

    def __init__(
        self,
        name: str,
        parametric_rv: RandomVariable,
        non_parametric_rv: RandomVariable,
        report_rv: RandomVariable,
        reports: ArrayLike | None = None,
    ) -> None:
        super().__init__(name=name)
        self.parametric_rv = parametric_rv
        self.non_parametric_rv = non_parametric_rv
        self.report_rv = report_rv
        self.reports = None if reports is None else jnp.asarray(reports)

    def sample(self, log_lambda: ArrayLike, **kwargs: object) -> Array:
        r"""Return the log expected counts.

        Parameters
        ----------
        log_lambda : array_like
            Log expected counts by reference day, $(G, T)$.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape $(G, T, D)$.
        """
        lh = combine_logit_hazards(
            jnp.asarray(self.parametric_rv()),
            jnp.asarray(self.non_parametric_rv()),
            jnp.asarray(self.report_rv()),
        )
        out = log_expected_counts(jnp.asarray(log_lambda), lh, self.reports)
        numpyro.deterministic(f"{self.name}::log_expected", out)
        return out
