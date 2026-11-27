"""Ascertainment that changes over time
(`reporting.ascertainment.AscertainmentChange`): a change shared,
related or independent across strata, a link to the stratum's counts of
either sign, and noise by week, stratum and both."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers
from scipy.special import expit, logit
from test_delay_change_strata import _design as _delay_design
from test_delay_change_strata import _frame, _values
from test_delay_counts import _log_lambda
from test_reporting_expectation import _report

from nowcastrenewal.reporting.ascertainment import ascertainment_change
from nowcastrenewal.reporting.expectation import ReportingExpectation

MAX_DELAY, N_DAYS, G = 8, 70, 3
LEVEL = [0.3, 0.5, 0.2]


def _asc_values(**kw: Any) -> dict[str, Any]:
    v = {
        "ascertainment": LEVEL,
        "asc_drift": 1.0,
        "asc_step": 0.8,
        "asc_step_at": 0.5,
        "asc_periodic": 0.3,
        "asc_periodic_weeks": 4.0,
        "asc_smooth_amp": 0.3,
        "asc_smooth_period": 40.0,
        "asc_smooth_phase": 0.0,
        "asc_walk": list(0.1 * np.arange(12.0)),
        "asc_walk_sd": 0.1,
        "asc_own_change": ["drift", "drift", "drift"],
        "asc_own_drift": [1.0, -1.0, 0.0],
        "asc_own_step": [0.8, 0.8, 0.8],
        "asc_own_step_at": [0.3, 0.5, 0.7],
        "asc_own_periodic": [0.3] * G,
        "asc_own_periodic_weeks": [4.0] * G,
        "asc_own_smooth_amp": [0.3] * G,
        "asc_own_smooth_period": [40.0] * G,
        "asc_own_smooth_phase": [0.0] * G,
        "asc_rho": 0.6,
        "asc_switch_at": 0.5,
        "asc_switch_days": 2.0,
        "asc_counts_sign": [-1, -1, -1],
        "asc_counts_slope": 0.3,
        "asc_counts_sd": 0.3,
        "asc_noise_sd": 0.3,
    }
    return {**_values(), **v, **kw}


def _asc_design(**kw: str) -> dict[str, str]:
    base = {
        "ascertainment_change": "static",
        "ascertainment_change_strata": "shared",
        "ascertainment_change_switch": "none",
        "ascertainment_counts": "none",
        "ascertainment_noise": "none",
    }
    return {**base, **kw}


def _ascertained(design, values=None, seed=1) -> tuple[np.ndarray, Any]:
    """Return each stratum's ascertainment by day, and the trace."""
    values = _asc_values() if values is None else values
    asc = ascertainment_change("asc", _frame(), design, values)
    assert asc is not None
    ll = _log_lambda()
    with handlers.seed(rng_seed=seed), handlers.trace() as tr:
        out = np.asarray(asc(log_lambda=ll))
    ratio = np.exp(out - ll)
    return np.asarray(values["ascertainment"])[:, None] * ratio, tr


def test_static_ascertainment_adds_nothing() -> None:
    assert ascertainment_change("a", _frame(), _asc_design(), {}) is None
    assert ascertainment_change("a", _frame(), {}, {}) is None


def test_a_shared_drift_moves_every_stratum_and_keeps_the_level() -> None:
    a, _ = _ascertained(_asc_design(ascertainment_change="drift"))
    assert np.all((a > 0) & (a < 1))
    # A weekly drift: up from week to week, flat within weeks.
    assert np.all(np.diff(a, axis=1) > -1e-6)
    assert np.all(a[:, -1] > a[:, 0])
    shift = logit(a) - logit(np.asarray(LEVEL))[:, None]
    # The same shift in every stratum, centred over the days.
    np.testing.assert_allclose(shift, shift[:1].repeat(G, 0), atol=1e-5)
    assert shift.mean() == pytest.approx(0, abs=1e-5)


def test_independent_strata_follow_their_own_change() -> None:
    v = _asc_values(
        asc_own_change=["step", "static", "drift"], asc_own_drift=[1, 1, -1]
    )
    design = _asc_design(
        ascertainment_change="drift", ascertainment_change_strata="independent"
    )
    a, _ = _ascertained(design, v)
    shift = logit(a) - logit(np.asarray(LEVEL))[:, None]
    assert np.ptp(shift[1]) == pytest.approx(0, abs=1e-5)
    jumps = np.abs(np.diff(shift[0])) > 1e-4
    assert jumps.sum() == 1
    assert np.all(np.diff(shift[2]) < 1e-6)
    assert shift[2, -1] < shift[2, 0]


def test_related_strata_share_part_of_their_change() -> None:
    design = _asc_design(
        ascertainment_change="drift", ascertainment_change_strata="related"
    )
    a, _ = _ascertained(design)
    shift = logit(a)
    slopes = np.polyfit(np.arange(N_DAYS), shift.T, 1)[0]
    # The shared drift up, the strata's own 1, -1 and 0: 0.6 + 0.4 x own.
    assert slopes[0] > slopes[2] > slopes[1] > 0


def test_strata_diverge_at_the_switch() -> None:
    design = _asc_design(
        ascertainment_change="static",
        ascertainment_change_strata="independent",
        ascertainment_change_switch="diverge",
    )
    a, _ = _ascertained(design)
    shift = logit(a) - logit(np.asarray(LEVEL))[:, None]
    gap = np.ptp(shift, axis=0)
    assert gap[:21].max() < 1e-3 < gap[-21:].min()


@pytest.mark.parametrize("sign", [1, -1])
def test_ascertainment_follows_the_counts_with_its_sign(sign: int) -> None:
    v = _asc_values(asc_counts_sign=[sign] * G)
    a, _ = _ascertained(_asc_design(ascertainment_counts="shared"), v)
    x = np.log1p(np.exp(_log_lambda()))
    for g in range(G):
        assert sign * np.corrcoef(logit(a[g]), x[g])[0, 1] > 0.99


def test_strata_counts_slopes_can_differ_in_sign() -> None:
    v = _asc_values(asc_counts_sign=[1, -1, 1])
    a, tr = _ascertained(_asc_design(ascertainment_counts="strata"), v)
    x = np.log1p(np.exp(_log_lambda()))
    assert np.corrcoef(logit(a[0]), x[0])[0, 1] > 0.99
    assert np.corrcoef(logit(a[1]), x[1])[0, 1] < -0.99
    assert "asc::counts::strata_slope" in tr


@pytest.mark.parametrize(
    ("level", "sites"),
    [
        ("time", {"asc::jitter_ascertainment"}),
        ("strata", {"asc::strata_ascertainment"}),
        ("strata_time", {"asc::strata_jitter_ascertainment"}),
        (
            "all",
            {
                "asc::jitter_ascertainment",
                "asc::strata_ascertainment",
                "asc::strata_jitter_ascertainment",
            },
        ),
    ],
)
def test_noise_layers_on_ascertainment(level: str, sites: set) -> None:
    a, tr = _ascertained(_asc_design(ascertainment_noise=level))
    assert {k for k in tr if "ascertainment" in k} == sites
    shift = logit(a) - logit(np.asarray(LEVEL))[:, None]
    week = np.asarray(_frame().week)
    for w in np.unique(week):
        assert np.allclose(np.ptp(shift[:, week == w], axis=1), 0, atol=1e-5)
    if level == "time":
        np.testing.assert_allclose(shift, shift[:1].repeat(G, 0), atol=1e-5)
    if level == "strata":
        assert np.allclose(np.ptp(shift, axis=1), 0, atol=1e-5)
        assert np.ptp(shift[:, 0]) > 0


def test_reproducible() -> None:
    design = _asc_design(
        ascertainment_change="walk",
        ascertainment_change_strata="independent",
        ascertainment_noise="all",
    )
    v = _asc_values(asc_own_change=["walk"] * G)
    a, _ = _ascertained(design, v, seed=3)
    b, _ = _ascertained(design, v, seed=3)
    c, _ = _ascertained(design, v, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.all(expit(logit(a)) < 1)


def _expected(design: dict[str, Any], values: dict[str, Any]) -> np.ndarray:
    exp = ReportingExpectation("e", _frame(), design, values, MAX_DELAY)
    with handlers.seed(rng_seed=2):
        return np.exp(np.asarray(exp(log_lambda=_log_lambda())))


def test_ascertainment_changes_the_totals_and_the_delay_does_not() -> None:
    base = {**_delay_design(delay_change="static"), **_report()}
    v = {**_asc_values(), "delay_counts_sign": [1] * G}
    v.update(delay_counts_slope=0.3, delay_counts_sd=0.3)
    lam = np.exp(_log_lambda())
    # The delay linked to the counts moves when cases are reported, not
    # how many: totals over the window's delays stay the expected counts
    # wherever the window is complete.
    by_delay = _expected({**base, "delay_counts": "shared"}, v)
    inside = slice(0, N_DAYS - MAX_DELAY)
    np.testing.assert_allclose(
        by_delay.sum(-1)[:, inside], lam[:, inside], rtol=1e-4
    )
    # Ascertainment linked to the counts changes the totals.
    asc = _expected({**base, **_asc_design(ascertainment_counts="shared")}, v)
    ratio = asc.sum(-1)[:, inside] / lam[:, inside]
    assert np.ptp(ratio) > 0.1
