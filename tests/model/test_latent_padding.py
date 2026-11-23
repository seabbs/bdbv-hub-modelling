"""The latent run padded to a bucket of days is the run without padding.

`simulation.outbreak` runs the latent arrays over the frame's days
rounded up to `BUCKET`, so that series of near lengths share compiled
shapes. Timings refer to the series' own length (`horizon`), the run is
causal and its draws are days first, so cropped to the series it is the
run without padding: each timed component bitwise, whole latent runs to
1e-12 (XLA may round a longer array's ops differently in the last bit)
and the counts of whole simulations exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator.draws import (
    draw_design,
    draw_values,
    generator,
    read_grid,
)
from nowcastrenewal.components.pmfs import Delay, generation_time_pmf
from nowcastrenewal.latent import imports as im
from nowcastrenewal.latent import mixing as mx
from nowcastrenewal.latent.rt import RELATIONS, PlaceRt, RtPath
from nowcastrenewal.latent.simulate import simulate_latent
from nowcastrenewal.simulation import outbreak

from .latent_r import load

GRID = read_grid(Path(__file__).parents[2] / "config/finetune/grid.yaml")
PAD = 37
RTOL = 1e-12


def run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a component's sample in float64 under a seed."""
    with jax.enable_x64(True):
        return np.asarray(handlers.seed(fn, 0)(*args, **kwargs))


def test_padded_paths_keep_their_days() -> None:
    f = load("paths")
    n = f["n"]
    for kind in f["paths"]:
        path = RtPath.from_values(kind, f["values"])
        np.testing.assert_array_equal(
            path.log_r(n + PAD, horizon=n)[:n], path.log_r(n), err_msg=kind
        )


@pytest.mark.parametrize("relation", RELATIONS)
def test_padded_r_keeps_its_days(relation: str) -> None:
    f = load("paths")
    n = f["n"]
    values = {**f["values"], "amplitude": 0.4, "peak": 0.3}

    def place_rt(horizon: int | None) -> PlaceRt:
        return PlaceRt(
            "rt",
            path=RtPath.from_values("rise_fall_sharp", values),
            relation=relation,
            values=values,
            noise_scale=5.0,
            place_sd=0.3,
            place_rho=0.5,
            place_lags=[0, 9, 20],
            drift=1.7,
            coupling=0.4,
            horizon=horizon,
        )

    place = np.array([0, 0, 1, 2])
    np.testing.assert_array_equal(
        run(place_rt(n).sample, n + PAD, place=place)[:n],
        run(place_rt(None).sample, n, place=place),
    )


@pytest.mark.parametrize("switch", ["on", "off"])
def test_padded_mixing_keeps_its_days(switch: str) -> None:
    n = 100
    kw = {"share": 0.3, "switch": switch, "switch_at": 0.4}
    kw["switch_days"] = 12.0
    for kind in mx.KINDS:
        if kind in ("distance", "time_varying"):
            kw["xy"] = np.array([[0.0, 0.0], [1.0, 2.0]])
        padded = mx.PlaceMixing("mixing", kind, horizon=n, **kw)
        own = mx.PlaceMixing("mixing", kind, **kw)
        np.testing.assert_array_equal(
            padded.shares(n + PAD)[:n], own.shares(n), err_msg=kind
        )


@pytest.mark.parametrize("kind", im.KINDS)
def test_padded_imports_keep_their_days(kind: str) -> None:
    n = 100
    kw = {
        "kind": kind,
        "rate": 0.05,
        "pulse_at": [0.2, 0.9, 1.0],
        "pulse_size": 3.0,
        "late": np.array([False, True, False]),
        "late_at": 0.95,
        "late_days": 14,
        "trickle": 0.1,
    }
    unit = np.array([2.0, 1.0, 0.0])
    padded = im.Imports("imports", horizon=n, **kw)
    own = im.Imports("imports", **kw)
    np.testing.assert_array_equal(
        run(padded.sample, n + PAD, unit=unit)[:n],
        run(own.sample, n, unit=unit),
    )


def _sims(seed: int, k: int) -> list[tuple[pd.DataFrame, dict, dict, int]]:
    """Grid simulations of up to six strata, as the batch draws them."""
    out = []
    for i in range(200):
        design = draw_design(GRID, generator(seed, i, 1), weeks=8)
        if int(design["n_places"]) * int(design["n_ages"]) > 6:
            continue
        values = draw_values(GRID, design, generator(seed, i, 2))
        strata = pd.DataFrame(sb._strata(design))
        out.append((strata, design, values, sb._n_days(values)))
        if len(out) == k:
            return out
    raise RuntimeError("too few simulations")


def _latent(sim: tuple, key: int, pad: int, lag_bucket: int = 1) -> Any:
    strata, design, values, n_days = sim
    gt = generation_time_pmf(
        Delay.from_mean_cv(
            design["gt_family"], values["gt_mean"], values["gt_cv"]
        )
    )
    n = n_days - len(gt)
    with jax.enable_x64(True):
        return handlers.seed(simulate_latent, jax.random.key(key))(
            n,
            gt,
            strata,
            design,
            values,
            kinds=tuple(GRID["factors"]["trajectory"]),
            days=n + pad,
            lag_bucket=lag_bucket,
        )


def test_padded_latent_runs_are_the_runs_without_padding() -> None:
    ran = 0
    for i, sim in enumerate(_sims(4242, 12)):
        try:
            own = _latent(sim, i, 0)
        except ValueError:
            continue
        padded = _latent(sim, i, PAD, lag_bucket=16)
        for name in ("infections", "onsets", "r", "reff"):
            np.testing.assert_allclose(
                getattr(padded, name),
                getattr(own, name),
                rtol=RTOL,
                err_msg=name,
            )
        pd.testing.assert_frame_equal(padded.path, own.path, rtol=RTOL)
        assert padded.attack == pytest.approx(own.attack, rel=RTOL)
        ran += 1
    assert ran >= 6


def test_days_shorter_than_the_series_raise() -> None:
    sim = _sims(4242, 1)[0]
    with pytest.raises(ValueError, match="days must be at least n"):
        _latent(sim, 0, -1)


def _traced(sim: tuple, key: int) -> tuple[Any, dict[str, Any]]:
    strata, design, values, n_days = sim
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=jax.random.key(key)),
    ):
        latent = outbreak.outbreak_model(n_days, strata, design, values)
    return latent, tr


def _unbucketed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outbreak, "BUCKET", 1)
    monkeypatch.setattr(outbreak, "LAG_BUCKET", 1)


def _revised(seed: int) -> tuple:
    """A grid simulation with revisions and a delay change that runs."""
    for sim in _sims(seed, 60):
        design = sim[1]
        if design["revisions"] != "yes" or design["delay_change"] == "static":
            continue
        try:
            _traced(sim, 0)
        except ValueError:
            continue
        return sim
    raise RuntimeError("no simulation with revisions ran")


def test_bucketed_outbreaks_expect_the_same_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sims = [*_sims(818, 3), _revised(818)]

    def runs() -> list[Any]:
        out = []
        for i, sim in enumerate(sims):
            try:
                out.append(_traced(sim, i))
            except ValueError as e:
                out.append(str(e))
        return out

    bucketed = runs()
    _unbucketed(monkeypatch)
    for sim, a, b in zip(sims, bucketed, runs(), strict=True):
        if isinstance(a, str):
            assert a == b
            continue
        n_days = sim[3]
        np.testing.assert_allclose(a[0].onsets, b[0].onsets, rtol=RTOL)
        site = "expected::log_expected"
        expected = np.asarray(b[1][site]["value"])
        assert expected.shape[1] == n_days
        np.testing.assert_allclose(
            np.asarray(a[1][site]["value"])[:, :n_days],
            expected,
            rtol=RTOL,
            atol=RTOL,  # centred terms near 0 are summed in another order
        )


def _totals(sim: tuple, keys: range) -> np.ndarray:
    """Each key's reported total."""
    strata, design, values, n_days = sim
    out = []
    for key in keys:
        _, counts, _ = outbreak.simulate_outbreak(
            key, n_days, strata, design, values
        )
        out.append(counts["count"].sum())
    return np.asarray(out, dtype=float)


def test_bucketed_counts_match_in_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Padding the triangle moves its count and revision draws, so the
    # counts are the same in distribution, not draw for draw: with the
    # same keys, the totals' mean and quantiles agree within their noise.
    sim = _revised(4242)
    keys = range(40)
    bucketed = _totals(sim, keys)
    _unbucketed(monkeypatch)
    own = _totals(sim, keys)
    diff = bucketed - own
    se = diff.std(ddof=1) / np.sqrt(len(keys))
    assert abs(diff.mean()) < 4 * se + 1e-9
    for q in (0.1, 0.5, 0.9):
        assert np.quantile(bucketed, q) == pytest.approx(
            np.quantile(own, q), rel=0.05
        )
