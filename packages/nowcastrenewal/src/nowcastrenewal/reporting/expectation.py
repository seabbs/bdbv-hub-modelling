# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A simulation's log expected reports by stratum, reference day and delay.

`ReportingExpectation` composes the reporting step's parts:

- the reference-date effects on the delay (`effects.ReferenceEffects`);
- the report-date and reference-weekday effects
  (`effects.ReportEffects`);
- the report days and outages (`effects.report_days`);
- the triangle (`triangle`).

With a parametric delay and no report-date effect, epinowcast (at the
simulator's pin) takes the delay as log probabilities and adds the
non-parametric part to them (its `ref_as_p`):
$\log p_{g,t,d} = \log f_{g,t}(d) + \eta^{np}_{g,t,d}$. Otherwise the
parts are logit hazards, summed and turned into probabilities. Either way
each delay's probability moves to its report day, and the log expected
reports are $\log \lambda_{g,t} + \eta^{ref}_{g,t} + \log p_{g,t,d}$, with
$\eta^{ref}$ the reference weekday's effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

from ..components.hazards import log_hazard_probs
from .effects import (
    ReferenceEffects,
    ReferenceFrame,
    ReportEffects,
    report_days,
)
from .triangle import aggregate_to_report_days, combine_logit_hazards

__all__ = ["ReportingExpectation"]


class ReportingExpectation(RandomVariable):
    r"""A simulation's log expected reports (module docstring).

    The result is recorded as `"{name}::log_expected"`, `(G, T, D)`.

    Parameters
    ----------
    name : str
        Name of the component; its parts are `"{name}::reference"` and
        `"{name}::report"`.
    frame : ReferenceFrame
        The triangle's strata and dates.
    design, values : mapping
        The simulation's design and values.
    max_delay : int
        Delays in the window, $D$.
    """

    def __init__(
        self,
        name: str,
        frame: ReferenceFrame,
        design: Mapping[str, Any],
        values: Mapping[str, Any],
        max_delay: int,
    ) -> None:
        super().__init__(name=name)
        self.frame = frame
        self.design = design
        self.values = values
        self.max_delay = max_delay
        self.report = ReportEffects(
            f"{name}::report", frame, design, values, max_delay
        )
        parametric = design["delay_form"] != "non_parametric"
        self.as_log_probs = parametric and not self.report.has_report_effect
        self.reference = ReferenceEffects(
            f"{name}::reference",
            frame,
            design,
            values,
            max_delay,
            as_log_pmf=self.as_log_probs,
        )
        self.reports = report_days(frame, design, values, max_delay)

    def sample(self, log_lambda: ArrayLike, **kwargs: object) -> Array:
        """Return the log expected reports.

        Parameters
        ----------
        log_lambda : array_like
            Log expected reference events by stratum and day, `(G, T)`.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            `(G, T, D)`.
        """
        parametric, non_parametric = self.reference()
        report, by_reference = self.report()
        if self.as_log_probs:
            log_p = parametric + non_parametric
        else:
            G, T = len(self.frame.place), len(self.frame.dates)
            if report is None:
                report = jnp.zeros((G, T + self.max_delay - 1))
            lh = combine_logit_hazards(parametric, non_parametric, report)
            log_p = log_hazard_probs(lh)
        log_p = aggregate_to_report_days(log_p, self.reports)
        log_lam = jnp.asarray(log_lambda)
        if by_reference is not None:
            log_lam = log_lam + by_reference
        out = log_lam[..., None] + log_p
        numpyro.deterministic(f"{self.name}::log_expected", out)
        return out
