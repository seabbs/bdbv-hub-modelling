# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The reporting delay's pmf from a simulation's values.

The parametric delays are the shared double interval censored pmfs
(`components.pmfs`), as the simulator's pinned epinowcast discretises
them. With no parametric delay (`"none"`) the hazard is constant, a
geometric delay already on whole days.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np

from ..components.pmfs import Delay, censored_pmf

__all__ = ["FAMILIES", "report_delay_pmf"]

FAMILIES = ("exponential", "lognormal", "gamma")
"""The simulator grid's parametric reporting delays (`Delay.from_values`)."""


def report_delay_pmf(
    family: str, values: Mapping[str, Any], max_delay: int
) -> np.ndarray:
    r"""Return the reporting delay's pmf on 0, ..., max_delay - 1.

    Parameters
    ----------
    family : str
        One of `FAMILIES` (double censored, truncated at `max_delay`) or
        `"none"`, a geometric delay with hazard $1 / (1 + \text{delay\_mean})$
        truncated at `max_delay` and renormalised.
    values : mapping
        The values (`Delay.from_values`).
    max_delay : int
        Delays in the window.

    Returns
    -------
    numpy.ndarray
        The pmf, summing to 1.

    Raises
    ------
    ValueError
        If `family` is unknown.
    """
    if family == "none":
        hazard = 1 / (1 + values["delay_mean"])
        pmf = hazard * (1 - hazard) ** np.arange(max_delay)
        return np.asarray(pmf / pmf.sum(), dtype=float)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    with jax.enable_x64(True):
        delay = Delay.from_values(family, values)
        return np.asarray(censored_pmf(delay, max_delay), dtype=float)
