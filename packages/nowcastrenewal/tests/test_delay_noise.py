"""Noise layers on the delay terms (`reporting.structure.NoiseLayers`): by
week, by stratum and by stratum and week, the same structure on every
delay parameter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers
from test_delay_jitter import VALUES

from nowcastrenewal.reporting.effects import ReferenceEffects, ReferenceFrame
from nowcastrenewal.reporting.structure import NoiseLayers, noise_layers

MAX_DELAY, N_DAYS, FIRST = 8, 35, "2026-01-05"
NOISE: dict[str, Any] = {
    **VALUES,
    "delay_strata_sd": 0.3,
    "delay_strata_jitter_sd": 0.2,
}


def _frame(n_strata: int = 3) -> ReferenceFrame:
    return ReferenceFrame(
        dates=np.datetime64(FIRST) + np.arange(N_DAYS),
        place=np.arange(n_strata),
        age=np.zeros(n_strata, dtype=int),
    )


def _layer(seed: int = 1, **sds: float) -> np.ndarray:
    frame = _frame()
    with handlers.seed(rng_seed=seed):
        return np.asarray(NoiseLayers("n", frame, **sds)(term="x"))


def _blocks(x: np.ndarray, week: np.ndarray) -> np.ndarray:
    return np.stack([x[:, week == w][:, 0] for w in np.unique(week)], 1)


def test_time_layer_is_shared_by_strata_and_constant_within_weeks() -> None:
    x = _layer(time_sd=0.2)
    week = np.asarray(_frame().week)
    assert np.allclose(x, x[:1])
    for w in np.unique(week):
        assert np.ptp(x[:, week == w]) == pytest.approx(0)
    assert np.ptp(x) > 0


def test_strata_layer_is_static_and_differs_between_strata() -> None:
    x = _layer(strata_sd=0.3)
    assert np.allclose(np.ptp(x, axis=1), 0)
    assert np.ptp(x[:, 0]) > 0


def test_strata_time_layer_moves_each_stratum_week_by_week() -> None:
    x = _layer(strata_time_sd=0.2)
    week = np.asarray(_frame().week)
    for w in np.unique(week):
        assert np.allclose(np.ptp(x[:, week == w], axis=1), 0)
    weekly = _blocks(x, week)
    # Each stratum moves, and not together.
    assert np.all(np.ptp(weekly, axis=1) > 0)
    assert not np.allclose(weekly - weekly[:1], 0)


def test_layers_are_centred_and_reproducible() -> None:
    sds = {"time_sd": 0.2, "strata_sd": 0.3, "strata_time_sd": 0.2}
    x = _layer(**sds)
    assert x.mean() == pytest.approx(0, abs=1e-6)
    np.testing.assert_array_equal(x, _layer(**sds))
    assert not np.allclose(x, _layer(seed=2, **sds))


def test_no_layer_is_zero_with_no_sites() -> None:
    frame = _frame()
    noise = NoiseLayers("n", frame)
    assert not noise
    with handlers.seed(rng_seed=1), handlers.trace() as tr:
        x = noise(term="x")
    assert not tr
    assert np.allclose(np.asarray(x), 0)


@pytest.mark.parametrize(
    ("level", "want"),
    [
        ("none", (0.2, 0.0, 0.0)),
        ("strata", (0.2, 0.3, 0.0)),
        ("strata_time", (0.2, 0.0, 0.2)),
        ("both", (0.2, 0.3, 0.2)),
    ],
)
def test_design_sets_the_layers(level: str, want: tuple) -> None:
    design = {"delay_jitter": "yes", "delay_strata_noise": level}
    assert noise_layers(design, NOISE) == want
    # Designs before the stratum layers have time noise only.
    assert noise_layers({"delay_jitter": "no"}, NOISE) == (0.0, 0.0, 0.0)


def _effects(**design: str) -> tuple[ReferenceFrame, Any]:
    base = {
        "delay_form": "parametric",
        "delay_family": "lognormal",
        "delay_change": "static",
        "shocks": "none",
        "delay_places": "shared",
        "delay_dow": "no",
        "delay_jitter": "no",
        "delay_strata_noise": "none",
    }
    frame = _frame()
    effects = ReferenceEffects(
        "ref", frame, {**base, **design}, NOISE, MAX_DELAY
    )
    with handlers.seed(rng_seed=3), handlers.trace() as tr:
        effects()
    return frame, tr


def test_every_delay_parameter_takes_the_stratum_layers() -> None:
    _, tr = _effects(delay_strata_noise="strata")
    mu = np.asarray(tr["ref::location"]["value"])
    scale = np.asarray(tr["ref::scale"]["value"])
    for x in (mu, scale):
        assert np.allclose(np.ptp(x, axis=1), 0)
        assert np.ptp(x[:, 0]) > 0
    assert {"ref::strata_location", "ref::strata_scale"} <= set(tr)
    _, tr = _effects(delay_strata_noise="both", delay_form="non_parametric")
    assert {
        "ref::strata_hazard",
        "ref::strata_jitter_hazard",
        "ref::strata_shape",
        "ref::strata_jitter_shape",
    } <= set(tr)


def test_without_stratum_layers_strata_share_their_delay() -> None:
    _, tr = _effects(delay_jitter="yes")
    mu = np.asarray(tr["ref::location"]["value"])
    assert np.allclose(mu, mu[:1])
    assert not any("strata" in k for k in tr)
