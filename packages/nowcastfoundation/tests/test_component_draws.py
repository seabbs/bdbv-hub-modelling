"""Large draws on flat arrays of a few sizes (`components.draws`).

The count samplers drawn flat keep each cell's distribution, the site's
shape and its log density; only which draw lands in which cell changes.
Two shapes of one padded size share one compile. A simulation's expected
reports do not change, and its counts change in distribution only.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
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
from nowcastrenewal.components.draws import (
    MIN_SIZE,
    PaddedDraws,
    flat_size,
    padded_draws,
)
from nowcastrenewal.simulation import outbreak

GRID = read_grid(
    Path(__file__).parents[3]
    / "packages/nowcastfoundation/config/finetune/grid.yaml"
)
RATE = jnp.arange(1.0, 25.0).reshape(2, 3, 4)


def _families() -> list[dist.Distribution]:
    return [
        dist.Poisson(RATE),
        dist.NegativeBinomial2(
            mean=RATE, concentration=jnp.full((2, 1, 1), 3.0)
        ),
        dist.Binomial(total_count=jnp.full((2, 3, 4), 30), probs=RATE / 25),
        dist.Bernoulli(probs=RATE / 25),
    ]


@pytest.mark.parametrize("i", range(4))
def test_flat_draws_keep_each_cells_distribution(i: int) -> None:
    with jax.enable_x64(True):
        base = _families()[i]
        padded = PaddedDraws(base)
        keys = jax.random.split(jax.random.key(i), 400)
        x = np.stack([np.asarray(padded.sample(k)) for k in keys])
        assert x.shape == (400, 2, 3, 4)
        np.testing.assert_array_equal(
            padded.log_prob(jnp.asarray(x[0])), base.log_prob(jnp.asarray(x[0]))
        )
        mean = np.asarray(base.mean)
        se = np.sqrt(np.asarray(base.variance) / len(keys))
    assert (np.abs(x.mean(axis=0) - mean) < 4.5 * se).all()


def test_flat_sizes_are_few_and_mostly_cells() -> None:
    n = np.arange(1, 4_000_001)
    sizes = np.array([flat_size(int(k)) for k in n])
    assert (sizes >= n).all()
    big = n > MIN_SIZE
    assert ((sizes[big] - n[big]) / sizes[big]).max() < 0.25
    # Four lengths a doubling: 1024 up to 4M cells is 12 doublings.
    assert len(set(sizes.tolist())) <= 1 + 4 * 12


def test_an_empty_site_draws_nothing() -> None:
    with jax.enable_x64(True):
        x = PaddedDraws(dist.Poisson(jnp.ones((2, 0, 4)))).sample(
            jax.random.key(0)
        )
    assert x.shape == (2, 0, 4)


def test_only_large_unobserved_sites_are_drawn_flat() -> None:
    on = jnp.arange(24).reshape(2, 3, 4) % 2 == 0

    def model() -> None:
        numpyro.sample("big", dist.Poisson(RATE))
        numpyro.sample("masked", dist.Poisson(RATE).mask(on))
        numpyro.sample("small", dist.Poisson(jnp.ones((3, 4))))
        numpyro.sample("normal", dist.Normal(jnp.zeros((2, 3, 4)), 1.0))
        numpyro.sample("seen", dist.Poisson(RATE), obs=jnp.ones((2, 3, 4)))

    with jax.enable_x64(True):
        tr = handlers.trace(handlers.seed(padded_draws(model), 0)).get_trace()
    kinds = {name: type(site["fn"]).__name__ for name, site in tr.items()}
    assert kinds["big"] == "PaddedDraws"
    assert kinds["masked"] == "MaskedDistribution"
    assert isinstance(tr["masked"]["fn"].base_dist, PaddedDraws)
    assert kinds["small"] == "Poisson"
    assert kinds["normal"] == "Normal"
    assert kinds["seen"] == "Poisson"
    assert tr["big"]["value"].shape == (2, 3, 4)
    # The mask still leaves cells out of the log density.
    lp = tr["masked"]["fn"].log_prob(tr["masked"]["value"])
    assert (np.asarray(lp)[~np.asarray(on)] == 0).all()


def test_shapes_of_one_padded_size_share_a_compile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with jax.enable_x64(True), jax.log_compiles(True):
        first = dist.Poisson(jnp.full((3, 7, 11), 2.0))
        PaddedDraws(first).sample(jax.random.key(0)).block_until_ready()
        caplog.clear()
        # Another shape under MIN_SIZE cells, at another rate: the same
        # flat draw, so only the pad, slice and reshape are compiled.
        other = dist.Poisson(jnp.full((2, 9, 13), 5.0))
        assert MIN_SIZE > 2 * 9 * 13
        with caplog.at_level(logging.WARNING, logger="jax"):
            PaddedDraws(other).sample(jax.random.key(1)).block_until_ready()
    compiled = [r.getMessage() for r in caplog.records]
    assert compiled, "the new shape's pad and slice compile"
    assert not [m for m in compiled if "_poisson" in m]


def _sims(seed: int, k: int) -> list[tuple[pd.DataFrame, dict, dict, int]]:
    out = []
    for i in range(300):
        design = draw_design(GRID, generator(seed, i, 1), weeks=8)
        if int(design["n_places"]) * int(design["n_ages"]) > 6:
            continue
        values = draw_values(GRID, design, generator(seed, i, 2))
        strata = pd.DataFrame(sb._strata(design))
        sim = (strata, design, values, sb._n_days(values))
        try:
            _traced(sim, 0)
        except ValueError:
            continue
        out.append(sim)
        if len(out) == k:
            return out
    raise RuntimeError("too few simulations")


def _traced(sim: tuple, key: int) -> tuple[Any, dict[str, Any]]:
    strata, design, values, n_days = sim
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=jax.random.key(key)),
    ):
        latent = outbreak.outbreak_model(n_days, strata, design, values)
    return latent, tr


def _unpadded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        outbreak, "padded_draws", lambda **_: contextlib.nullcontext()
    )


def _revised(seed: int) -> tuple:
    for sim in _sims(seed, 40):
        if sim[1]["revisions"] == "yes":
            return sim
    raise RuntimeError("no simulation with revisions ran")


@pytest.mark.slow
def test_flat_draws_leave_the_expected_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sims = [*_sims(31, 3), _revised(31)]
    flat = [_traced(sim, i) for i, sim in enumerate(sims)]
    _unpadded(monkeypatch)
    for i, (sim, (a_latent, a)) in enumerate(zip(sims, flat, strict=True)):
        b_latent, b = _traced(sim, i)
        np.testing.assert_array_equal(a_latent.onsets, b_latent.onsets)
        site = "expected::log_expected"
        np.testing.assert_array_equal(
            np.asarray(a[site]["value"]), np.asarray(b[site]["value"])
        )
        assert set(a) == set(b)
        for name in a:
            assert np.shape(a[name]["value"]) == np.shape(b[name]["value"])


def _totals(sim: tuple, keys: range) -> np.ndarray:
    strata, design, values, n_days = sim
    return np.asarray(
        [
            outbreak.simulate_outbreak(key, n_days, strata, design, values)[1][
                "count"
            ].sum()
            for key in keys
        ],
        dtype=float,
    )


@pytest.mark.slow
def test_flat_draws_match_the_counts_in_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _revised(4242)
    keys = range(40)
    flat = _totals(sim, keys)
    _unpadded(monkeypatch)
    own = _totals(sim, keys)
    diff = flat - own
    se = diff.std(ddof=1) / np.sqrt(len(keys))
    assert abs(diff.mean()) < 4 * se + 1e-9
    for q in (0.1, 0.5, 0.9):
        assert np.quantile(flat, q) == pytest.approx(
            np.quantile(own, q), rel=0.05
        )
