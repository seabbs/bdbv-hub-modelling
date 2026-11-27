"""The delay change by stratum (`reporting.structure.StructuredChange`):
shared, related or independent strata, a switch between shared and own
changes, and two kinds of change combined."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers
from test_reporting_effects import VALUES

from nowcastrenewal.reporting.effects import (
    ReferenceEffects,
    ReferenceFrame,
    change_kinds,
    change_path,
)
from nowcastrenewal.reporting.structure import switch_weight

MAX_DELAY, N_DAYS, FIRST, G = 8, 70, "2026-01-05", 3


def _values(**kw: Any) -> dict[str, Any]:
    own = {
        "own_change": ["drift", "drift", "drift"],
        "own_drift": [0.6, -0.6, 0.0],
        "own_step": [0.5, 0.5, 0.5],
        "own_step_at": [0.3, 0.5, 0.7],
        "own_periodic": [0.2, 0.2, 0.2],
        "own_periodic_weeks": [3.0, 4.0, 5.0],
        "own_smooth_amp": [0.2, 0.2, 0.2],
        "own_smooth_period": [30.0, 30.0, 30.0],
        "own_smooth_phase": [0.0, 1.0, 2.0],
        "own_slow_at": [0.5, 0.5, 0.5],
        "own_slow_weeks": [1, 1, 1],
        "own_slow_mult": [2.0, 2.0, 2.0],
        "delay_change_rho": 0.6,
        "delay_switch_at": 0.5,
        "delay_switch_days": 7.0,
        "delay_change_second": "none",
        "walk_sd": 0.1,
    }
    return {**VALUES, "weeks": N_DAYS // 7, **own, **kw}


def _design(**kw: str) -> dict[str, str]:
    base = {
        "delay_form": "parametric",
        "delay_family": "lognormal",
        "delay_change": "drift",
        "shocks": "none",
        "delay_places": "shared",
        "delay_dow": "no",
        "delay_change_strata": "shared",
        "delay_change_switch": "none",
        "delay_change_combine": "single",
    }
    return {**base, **kw}


def _frame() -> ReferenceFrame:
    return ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.arange(G),
        age=np.zeros(G, dtype=int),
    )


def _mu(design: dict[str, str], values: dict[str, Any], seed: int = 1):
    frame = _frame()
    effects = ReferenceEffects("ref", frame, design, values, MAX_DELAY)
    with handlers.seed(rng_seed=seed), handlers.trace() as tr:
        effects()
    return np.asarray(tr["ref::location"]["value"]), tr


def _centred(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def test_shared_strata_change_together() -> None:
    mu, tr = _mu(_design(), _values())
    assert np.allclose(mu, mu[:1])
    assert np.ptp(mu[0]) > 0
    assert "ref::change::walk" not in tr


def test_independent_strata_each_follow_their_own_kind() -> None:
    # One stratum steps, one is static, one drifts.
    v = _values(own_change=["step", "static", "drift"])
    mu, _ = _mu(_design(delay_change_strata="independent"), v)
    frame = _frame()
    for g, kinds in enumerate(v["own_change"]):
        own = {k[4:]: v[k][g] for k in v if k.startswith("own_")}
        want = change_path(frame, [kinds], {**v, **own}, 1.0)
        got = mu[g] - mu[g].mean()
        # Centred jointly, so each stratum's own path up to a constant.
        np.testing.assert_allclose(got, _centred(want), atol=1e-6)
    assert np.ptp(mu[1]) == pytest.approx(0, abs=1e-6)
    step = mu[0][1:] - mu[0][:-1]
    assert np.count_nonzero(np.abs(step) > 1e-6) == 1


def test_related_strata_share_part_of_the_change() -> None:
    v = _values()
    mu, _ = _mu(_design(delay_change_strata="related"), v)
    frame = _frame()
    shared = change_path(frame, ["drift"], v, 1.0)
    rho = v["delay_change_rho"]
    for g in range(G):
        own = change_path(
            frame, ["drift"], {**v, "drift": v["own_drift"][g]}, 1
        )
        want = rho * shared + (1 - rho) * own
        np.testing.assert_allclose(
            mu[g] - mu[g].mean(), _centred(want), atol=1e-6
        )
    # Not identical: each stratum's drift is ordered as its own.
    assert not np.allclose(mu, mu[:1])
    slopes = np.polyfit(np.arange(N_DAYS), mu.T, 1)[0]
    assert slopes[0] > slopes[2] > slopes[1]


def test_related_strata_correlate_more_than_independent_ones() -> None:
    def spread(relation: str) -> float:
        v = _values(own_change=["walk"] * G, delay_change_rho=0.9)
        design = _design(delay_change="walk", delay_change_strata=relation)
        out = []
        for seed in range(20):
            mu, _ = _mu(design, v, seed)
            out.append(np.mean(np.std(mu, axis=0)))
        return float(np.mean(out))

    assert spread("related") < spread("independent")


@pytest.mark.parametrize("switch", ["diverge", "converge"])
def test_strata_switch_between_shared_and_own(switch: str) -> None:
    v = _values(delay_switch_days=2.0)
    design = _design(
        delay_change_strata="independent", delay_change_switch=switch
    )
    mu, _ = _mu(design, v)
    gap = np.ptp(mu, axis=0)
    early, late = gap[:21], gap[-21:]
    if switch == "diverge":
        assert early.max() < 1e-3 < late.min()
    else:
        assert late.max() < 1e-3 < early.min()


def test_switch_weight_is_centred_where_drawn() -> None:
    frame = _frame()
    w = switch_weight(frame, "diverge", 0.5, 14.0)
    assert w[0] < 0.01 < 0.99 < w[N_DAYS - 1]
    mid = 0.5 * (N_DAYS - 1)
    assert np.interp(0.5, w, np.arange(N_DAYS)) == pytest.approx(mid, abs=1)
    np.testing.assert_allclose(
        switch_weight(frame, "converge", 0.5, 14.0), 1 - w
    )
    assert np.all(switch_weight(frame, "none", 0.5, 14.0) == 1)


def test_two_kinds_combine() -> None:
    v = _values(delay_change_second="step")
    design = _design(delay_change_combine="combined")
    assert change_kinds(design, v) == ("drift", "step")
    assert change_kinds(_design(), v) == ("drift",)
    mu, _ = _mu(design, v)
    frame = _frame()
    want = change_path(frame, ["drift"], v, 1) + change_path(
        frame, ["step"], v, 1
    )
    np.testing.assert_allclose(mu[0] - mu[0].mean(), _centred(want), atol=1e-6)
    # Own changes combine too.
    v = _values(own_change=["drift+step", "static", "walk+periodic"])
    mu, tr = _mu(_design(delay_change_strata="independent"), v)
    assert tr["ref::change::walk"]["value"].shape[0] == G
    assert np.ptp(mu[1]) == pytest.approx(0, abs=1e-6)


def test_own_walks_are_reproducible_and_differ_by_stratum() -> None:
    v = _values(own_change=["walk"] * G)
    design = _design(delay_change="walk", delay_change_strata="independent")
    a, _ = _mu(design, v, seed=4)
    b, _ = _mu(design, v, seed=4)
    c, _ = _mu(design, v, seed=5)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(a[0], a[1])


def test_a_single_stratum_takes_the_shared_change() -> None:
    frame = ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.array([0]),
        age=np.array([0]),
    )
    v = _values()
    one = ReferenceEffects(
        "ref", frame, _design(delay_change_strata="independent"), v, MAX_DELAY
    )
    assert one.change is None
