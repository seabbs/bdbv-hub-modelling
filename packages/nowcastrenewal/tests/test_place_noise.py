"""Places' R deviations (`latent.rt`): their sd is the grid's `place_sd`."""

from __future__ import annotations

import jax
import numpy as np
from numpyro import handlers

from nowcastrenewal.latent.rt import WeeklyNoise, place_innovation_sd


def test_places_deviate_by_place_sd_in_the_long_run() -> None:
    # `place_sd` (times the generation time) is the deviations' sd, not
    # the weekly innovations', which the AR(1) would inflate 1.67 times.
    scale, place_sd = 12.0, 0.02
    sd = place_innovation_sd(scale, place_sd)
    assert np.isclose(sd, scale * place_sd * np.sqrt(1 - 0.8**2))
    noise = WeeklyNoise("w")
    draws = []
    for seed in range(40):
        with jax.enable_x64(True), handlers.seed(rng_seed=seed):
            e = np.asarray(noise(n=7 * 400, sd=sd, noise_name="p"))
        draws.append(e[7 * 50 :: 7])
    got = np.std(np.concatenate(draws))
    assert abs(got / (scale * place_sd) - 1) < 0.1
