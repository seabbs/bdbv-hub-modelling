# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A simulation's reported counts from its expected onsets.

`reporting_model` is the reporting step as one numpyro model: the
expected reports (`expectation.ReportingExpectation`), then for each
observation draw the counts (`observation.ObservedReports`) and their
revisions (`revisions.Revisions`). `report_simulation` samples it forward
and returns the counts in the simulator's long format, with the values
the step adds.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax.typing import ArrayLike
from numpyro import handlers

from .effects import ReferenceFrame
from .expectation import ReportingExpectation
from .observation import ObservedReports, nb1_scale
from .revisions import Revisions

__all__ = [
    "COUNT_COLUMNS",
    "START",
    "report_simulation",
    "reporting_frame",
    "reporting_model",
    "tabulate",
]

COUNT_COLUMNS = ("draw", "location", "reference_date", "report_date", "count")
"""The counts' columns: one row per non-zero increment."""

START = dt.date(2026, 1, 3)
"""The reporting frame's first date before a simulation's offset."""


def reporting_frame(
    values: Mapping[str, Any],
    n_days: int,
    places: Sequence[str],
    ages: Sequence[str],
    days: int | None = None,
) -> ReferenceFrame:
    """Return a simulation's reporting frame: its strata and dates.

    Parameters
    ----------
    values : mapping
        The values (`offset`).
    n_days : int
        Days of the frame.
    places, ages : sequence of str
        Each stratum's place (`P01`, ...) and age group (`A1`, ...).
    days : int, optional
        Dates the frame's arrays run over, at least `n_days`; `n_days`
        if None. Later dates pad them to a shared shape.

    Returns
    -------
    ReferenceFrame
        From `START` plus the offset.
    """
    first = np.datetime64(START) + int(values["offset"])
    return ReferenceFrame(
        dates=first + np.arange(n_days if days is None else days),
        place=np.array([int(p[1:]) - 1 for p in places]),
        age=np.array([int(a[1:]) - 1 for a in ages]),
        n_days=n_days,
    )


def _delay_family(design: Mapping[str, Any]) -> str:
    parametric = design["delay_form"] != "non_parametric"
    return str(design["delay_family"]) if parametric else "none"


def reporting_model(
    frame: ReferenceFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    onsets: ArrayLike,
    draws: int = 1,
) -> None:
    """Run the reporting step as a numpyro model.

    Parameters
    ----------
    frame : ReferenceFrame
        The triangle's strata and dates.
    design, values : mapping
        The simulation's design and values.
    onsets : array_like
        Expected onsets over the triangle's days, by strata; floored at
        1e-8, and 0 on the frame's padded days.
    draws : int
        Observation draws: sites `"draw_{k}::reports::counts"` and, with
        revisions, `"draw_{k}::revisions::counts"`.
    """
    D = int(values["max_delay"])
    x = jnp.asarray(onsets, dtype=float)
    x = jnp.pad(x, ((0, len(frame.dates) - x.shape[0]), (0, 0)))
    log_lambda = jnp.log(jnp.maximum(x, 1e-8)).T
    expected = ReportingExpectation("expected", frame, design, values, D)
    log_expected = expected(log_lambda=log_lambda)
    observed = ObservedReports(
        "reports",
        family=str(np.atleast_1d(values["obs_family"])[0]),
        sqrt_phi=np.atleast_1d(values["obs_sqrt_phi"]),
        onsets=onsets,
        delay_family=_delay_family(design),
        values=values,
        max_delay=D,
    )
    for k in range(draws):
        with handlers.scope(prefix=f"draw_{k}", divider="::"):
            counts = observed(log_expected=log_expected)
            if design["revisions"] == "yes":
                Revisions(
                    "revisions",
                    values["revise_over"],
                    values["revise_move"],
                    values["revise_keep"],
                    frame.observed,
                )(counts=counts)
            else:
                numpyro.deterministic("reported", counts)


def _location(place: str, age: str, multi_place: bool, multi_age: bool) -> str:
    if multi_place and multi_age:
        return f"{place}-{age}"
    return age if multi_age else place


def report_simulation(
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    onsets: ArrayLike,
    places: Sequence[str],
    ages: Sequence[str],
    seed: int,
    draws: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Report one simulation's expected onsets.

    Parameters
    ----------
    design, values : mapping
        The simulation's design and values (with the latent step's
        additions).
    onsets : array_like
        Expected onsets, days of the reporting frame by strata, in the
        strata's order (places, then ages within each place).
    places, ages : sequence of str
        Each stratum's place (`P01`, ...) and age group (`A1`, ...).
    seed : int
        The step's seed for this simulation.
    draws : int
        Observation draws.

    Returns
    -------
    tuple
        The counts (`COUNT_COLUMNS`) and the values the step adds:
        `stratum_location`, `low_start_days`, `lambda_hat`, `sum_p2` and
        `sqrt_phi_obs`.
    """
    x = np.asarray(onsets, dtype=float)
    T = x.shape[0]
    frame = reporting_frame(values, T, places, ages)
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=int(seed)),
    ):
        reporting_model(frame, design, values, x, draws)
    return tabulate(tr, design, values, x, places, ages, draws)


def tabulate(
    trace: Mapping[str, Any],
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    onsets: np.ndarray,
    places: Sequence[str],
    ages: Sequence[str],
    draws: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a traced reporting model's counts and added values.

    Parameters
    ----------
    trace : mapping
        A trace of `reporting_model` (on its own or in a larger model).
    design, values : mapping
        The simulation's design and values.
    onsets : numpy.ndarray
        The expected onsets it reported, days by strata.
    places, ages : sequence of str
        Each stratum's place and age group.
    draws : int
        Observation draws.

    Returns
    -------
    tuple
        As `report_simulation`.
    """
    tr = trace
    x = np.asarray(onsets, dtype=float)
    T = x.shape[0]
    multi_place = int(design["n_places"]) > 1
    multi_age = int(design["n_ages"]) > 1
    location = [
        _location(p, a, multi_place, multi_age)
        for p, a in zip(places, ages, strict=True)
    ]
    dates = np.datetime64(START) + int(values["offset"]) + np.arange(T)
    D = int(values["max_delay"])
    frames = []
    for k in range(draws):
        if design["revisions"] == "yes":
            revised = tr[f"draw_{k}::revisions::counts"]["value"]
            dense = np.asarray(revised)[:, :T, :T]
            g, t, r = np.nonzero(np.round(dense))
            count = np.round(dense[g, t, r])
        else:
            tri = np.asarray(tr[f"draw_{k}::reported"]["value"])[:, :T]
            g, t, d = np.nonzero(tri)
            r = t + d
            keep = r <= T - 1
            g, t, r = g[keep], t[keep], r[keep]
            count = tri[g, t, d[keep]]
        frames.append(
            pd.DataFrame(
                {
                    "draw": k + 1,
                    "location": np.asarray(location)[g],
                    "reference_date": dates[t].astype(str),
                    "report_date": dates[r].astype(str),
                    "count": count.astype(int),
                }
            )
        )
    counts = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(COUNT_COLUMNS))
    )
    total = x.sum(axis=1)
    low = int(np.argmax(total >= 1)) if np.any(total >= 1) else T
    lam, s2, scale = nb1_scale(x, _delay_family(design), values, D)
    family = np.atleast_1d(values["obs_family"])
    sqrt_phi = np.atleast_1d(np.asarray(values["obs_sqrt_phi"], dtype=float))
    sqrt_phi_obs = np.where(family == "negbin1d", sqrt_phi * scale, sqrt_phi)
    additions = {
        "stratum_location": location,
        "low_start_days": low,
        "lambda_hat": lam,
        "sum_p2": s2,
        "sqrt_phi_obs": sqrt_phi_obs.tolist(),
    }
    return counts.loc[:, list(COUNT_COLUMNS)], additions
