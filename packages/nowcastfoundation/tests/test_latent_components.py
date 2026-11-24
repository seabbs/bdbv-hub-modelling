"""The simulator's renewal-step components, on closed-form cases.

The renewal without depletion and the reference events are also checked
live against epinowcast (`test_latent_epinowcast.py`), and the pmfs
against primarycensored (`test_pmfs.py`).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable

from nowcastrenewal.components.pmfs import (
    Delay,
    delay_pmf,
    generation_time_pmf,
)
from nowcastrenewal.components.reference import ReferenceEvents
from nowcastrenewal.components.renewal import run_renewal
from nowcastrenewal.latent.imports import Imports
from nowcastrenewal.latent.mixing import PlaceMixing


def gt_pmf(mean: float, family: str, cv: float) -> np.ndarray:
    """The grid's generation time, from the shared pmfs."""
    return generation_time_pmf(Delay.from_mean_cv(family, mean, cv))


def incubation_pmf(mean: float, cv: float = 0.5) -> np.ndarray:
    """The grid's infection-to-reference delay, from the shared pmfs."""
    return delay_pmf(Delay.from_mean_cv("gamma", mean, cv))


def matrix(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


def run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a component's sample in float64 under a seed."""
    with jax.enable_x64(True):
        return handlers.seed(fn, 0)(*args, **kwargs)


def test_an_established_stratum_opens_at_its_ascertained_level() -> None:
    # A stratum at a constant level before and during the span has
    # constant expected reference events from day 1.
    pmf = incubation_pmf(9.0)
    events = ReferenceEvents(
        "ref",
        delay_rvs=[DeterministicPMF("d", pmf)],
        ascertainment_rv=DeterministicVariable("asc", np.array([0.17])),
    )
    out = run(
        events.sample, np.full((30, 1), 1503.0), before=np.array([1503.0])
    )
    np.testing.assert_allclose(out, 1503.0 * 0.17, rtol=1e-12)


def test_mixing_shares_ramp_over_the_switch_days() -> None:
    # A time-varying share rises (or falls) linearly from day
    # round(switch_at * n) over `switch_days`; the others are constant.
    n, share, at, days = 60, 0.2, 0.3, 9.0
    ramp = np.clip((np.arange(1, n + 1) - round(at * n)) / days, 0, 1)
    for switch, want in (("on", share * ramp), ("off", share * (1 - ramp))):
        m = PlaceMixing(
            "mixing",
            "time_varying",
            share=share,
            xy=np.zeros((2, 2)),
            distance_range=1.0,
            switch=switch,
            switch_at=at,
            switch_days=days,
        )
        np.testing.assert_allclose(m.shares(n), want, rtol=1e-12)
    flat = PlaceMixing("mixing", "symmetric", share=share)
    np.testing.assert_allclose(flat.shares(n), share, rtol=1e-12)
    np.testing.assert_allclose(PlaceMixing("m", "none").shares(n), 0.0)


def test_mixing_kernels_receive_from_the_other_places_in_age() -> None:
    # Rows receive and sum to 1 over the other places, within an age
    # group; a quiet place neither sends nor receives; `asymmetric`
    # receives only from places earlier in `order`.
    place = np.array([0, 0, 1, 1, 2, 2])
    age = np.array([1, 2, 1, 2, 1, 2])
    sym, _ = run(
        PlaceMixing("m", "symmetric", share=0.2, quiet=[2]).sample,
        10,
        place=place,
        age=age,
    )
    want = np.zeros((6, 6))
    want[0, 2] = want[2, 0] = want[1, 3] = want[3, 1] = 1.0
    np.testing.assert_allclose(sym, want, rtol=1e-12)
    asym = PlaceMixing("m", "asymmetric", share=0.2, order=[2, 0, 1])
    kernel = asym.place_kernel(3)
    np.testing.assert_allclose(
        kernel, [[0, 0, 1], [0.5, 0, 0.5], [0, 0, 0]], rtol=1e-12
    )


def test_imports_fall_on_their_days() -> None:
    # Constant: `rate` times the unit each day. Pulses: `pulse_size`
    # times the unit from day round(at * n), spread over that day and the
    # next along `spread`, once per day however many land on it. A late
    # stratum gets its unit spread over `late_days`
    # from day round(late_at * n), and nothing else.
    n, unit = 50, np.array([2.0, 1.0, 0.0])
    late = np.array([False, True, False])
    const = np.asarray(
        run(Imports("i", kind="constant", rate=0.01).sample, n, unit=unit)
    )
    np.testing.assert_allclose(const, np.tile([0.02, 0.01, 0.0], (n, 1)))
    pulses = np.asarray(
        run(
            Imports(
                "i",
                kind="pulses",
                pulse_at=[0.2, 0.5, 0.5001],
                pulse_size=4.0,
                spread=[0.5, 0.3, 0.2],
                late=late,
                late_at=0.4,
                late_days=5,
            ).sample,
            n,
            unit=unit,
        )
    )
    want = np.zeros((n, 3))
    want[[9, 10, 11], 0] = want[[24, 25, 26], 0] = [4.0, 2.4, 1.6]
    want[19:24, 1] = 0.2
    np.testing.assert_allclose(pulses, want, rtol=1e-12)


def test_pulses_need_a_spread() -> None:
    with pytest.raises(ValueError, match="spread"):
        Imports("i", kind="pulses", pulse_at=[0.5], pulse_size=1.0)


def test_the_trickle_reaches_every_stratum_with_cases_but_late_ones() -> None:
    late = np.array([False, True, False])
    imports = Imports("imports", kind="none", late=late, trickle=0.3)
    out = np.asarray(run(imports.sample, 10, unit=np.array([1.0, 1.0, 0.0])))
    np.testing.assert_allclose(out[:, 0], 0.3)
    np.testing.assert_allclose(out[:, 2], 0.0)
    # The late stratum's own import is its unit over the late days.
    assert out[:, 1].sum() == pytest.approx(1.0)


def test_the_exhausted_renewal_never_infects_more_than_its_population() -> None:
    # R far above 1 from a large start: infections stop at the population
    # and the susceptibles never go below 0.
    gt = gt_pmf(5.0, "gamma", 0.5)
    start = np.zeros((len(gt), 3))
    start[-1] = [250.0, 100.0, 50.0]
    rt = np.exp(np.full((120, 3), 2.0))
    pop = np.array([1e4, 5e3, 2e4])
    with jax.enable_x64(True):
        out = run_renewal(start, gt, rt, pop)
    assert (np.asarray(out.infections).sum(axis=0) <= pop + 1e-6).all()
    assert (np.asarray(out.susceptible) >= 0).all()


def test_the_fitted_models_do_not_import_the_simulator_components() -> None:
    # The fit keys hash the forecasters' import closure, so the simulator's
    # components stay out of it and adding them refits nothing.
    code = (
        "import sys, nowcastrenewal.forecaster, nowcastrenewal.zones;"
        "mods = ('imports', 'mixing');"
        "print([m for m in mods"
        " if f'nowcastrenewal.latent.{m}' in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


def test_latent_onsets_use_the_drawn_incubation_family_and_cv() -> None:
    from nowcastfoundation.simulator import draws
    from nowcastrenewal.latent.simulate import simulate_latent

    grid = draws.read_grid(
        __import__("pathlib").Path(__file__).parents[3]
        / "packages"
        / "nowcastfoundation"
        / "config"
        / "finetune"
        / "grid.yaml"
    )
    rng = draws.generator(5)
    while True:
        d = draws.draw_design(grid, rng, weeks=8)
        if d["window"] == "early" and d["n_places"] * d["n_ages"] == 1:
            break
    d["incubation_family"] = "lognormal"
    v = draws.draw_values(grid, d, rng)
    v["incubation_mean"], v["incubation_cv"] = [6.0], 1.0
    gt = gt_pmf(v["gt_mean"], d["gt_family"], v["gt_cv"])
    strata = pd.DataFrame({"place": ["P01"], "age": ["A1"], ".group": [1]})
    run_ = run(simulate_latent, 70, gt, strata, d, v)
    pmf = delay_pmf(Delay.from_mean_cv("lognormal", 6.0, 1.0))
    inf = run_.infections[:, 0]
    want = v["ascertainment"][0] * np.convolve(inf, pmf)[: len(inf)]
    np.testing.assert_allclose(run_.onsets[:, 0], want, rtol=1e-6, atol=1e-9)
