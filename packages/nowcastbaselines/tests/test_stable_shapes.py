"""The random walk models' JAX programs see a few shapes, not one a date.

Each nowcast date brings a release and a day more, so the arrays the
walks, the draws and the delay sums see grow by date.
They are padded to a few sizes (`nowcastutils.shapes.bucket`), and a
compiled program serves every length that pads to its size. The draws
are those of the unpadded arrays: JAX draws each element from its flat
index, so the padding only adds draws after the real ones, and the
Poisson draws leave the padding out (`nowcastbaselines.draws.poisson`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from nowcastbaselines.baselinenowcast.delays import (
    Settings,
    _delay_counts,
    _pair_sums,
    _pairs_at,
    delay_sums,
)
from nowcastbaselines.baselinenowcast.triangle import (
    preprocess_negative_values,
)
from nowcastbaselines.baselinenowcast.uncertainty import (
    sample_count,
    sample_nb,
)
from nowcastbaselines.draws import poisson
from nowcastbaselines.walk import random_walk, random_walks
from nowcastutils.shapes import bucket
from nowcastutils.snapshots import SnapshotArrays, snapshot_arrays

_COMPILES = [0]


def _count(event: str, _duration: float, **_: object) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count)


@contextmanager
def compiles() -> Iterator[list[int]]:
    """Count the programs JAX compiles in the block."""
    out = [0]
    before = _COMPILES[0]
    try:
        yield out
    finally:
        out[0] = _COMPILES[0] - before


def test_bucket_pads_by_at_most_an_eighth() -> None:
    sizes = [bucket(n) for n in range(1, 5000)]
    n = np.arange(1, 5000)
    assert (np.asarray(sizes) >= n).all()
    assert (np.asarray(sizes) <= np.maximum(n * 9 / 8, 16)).all()
    # A length growing a day at a time takes a few sizes.
    assert len(set(sizes[100:400])) < 20


@pytest.mark.parametrize("width", [3000, 3500, 4096])
def test_padded_poisson_draws_are_jaxs_on_the_unpadded_rates(
    width: int,
) -> None:
    # JAX's rejection loop runs until every element is accepted and keeps
    # the latest accepted draw, so padding the rates changes its draws.
    # `draws.poisson` leaves the padding out of the loop, and pins this
    # copy of JAX's sampler to JAX's own.
    rng = np.random.default_rng(width)
    lam = rng.gamma(0.7, 30.0, 3000).astype(np.float32)
    lam[:20] = 0.0
    key = jax.random.key(width)
    want = np.asarray(jax.random.poisson(key, lam))
    padded = np.zeros(width, np.float32)
    padded[:3000] = lam
    got = jax.jit(poisson)(key, padded, np.arange(width) < 3000)
    np.testing.assert_array_equal(np.asarray(got)[:3000], want)
    assert (np.asarray(got)[3000:] == 0).all()


@partial(jax.jit, static_argnums=3)
def _reference_walks(
    keys: jax.Array,
    x0: jax.Array,
    sigma: jax.Array,
    h: int,
    mean: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Each place's `random_walk`, unpadded, compiled as one program."""
    if mean is None:
        return jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(
            keys, x0, sigma
        )
    return jax.vmap(lambda k, x, s, m: random_walk(k, x, s, h, m))(
        keys, x0, sigma, mean
    )


@pytest.mark.parametrize("h", [1, 5, 37, 40])
def test_random_walks_draw_each_places_own_walk(h: int) -> None:
    # The rates are those of the unpadded walk to float32 rounding (XLA
    # fuses a walk of one step differently), and the counts are equal.
    rng = np.random.default_rng(h)
    x0 = rng.gamma(2.0, 20.0, (3, 50)).astype(np.float32)
    keys = jax.random.split(jax.random.key(h), 3)
    sigma = np.array([0.1, 0.4, 0.0], np.float32)
    lam, y = random_walks(keys, x0, sigma, h)
    want_lam, want_y = _reference_walks(keys, x0, sigma, h)
    np.testing.assert_allclose(np.asarray(lam), want_lam, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(y), want_y)
    # Steps in days, each with its own mean and spread.
    mean = rng.normal(0, 0.1, (3, h)).astype(np.float32)
    spread = np.repeat(sigma[:, None], h, axis=1)
    lam, y = random_walks(keys, x0, spread, h, mean)
    want_lam, want_y = _reference_walks(keys, x0, spread, h, mean)
    np.testing.assert_allclose(np.asarray(lam), want_lam, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(y), want_y)


def test_random_walks_compile_once_for_horizons_of_one_size() -> None:
    keys = jax.random.split(jax.random.key(0), 2)
    x0 = np.full((2, 30), 10.0, np.float32)
    sigma = np.array([0.2, 0.3], np.float32)
    random_walks(keys, x0, sigma, 33)
    with compiles() as n:
        for h in (34, 35, 36):
            np.asarray(random_walks(keys, x0, sigma, h)[1])
    assert n[0] == 0


def test_sample_count_draws_as_unpadded() -> None:
    rng = np.random.default_rng(1)
    shape = (40, 3, 11)
    reported = rng.poisson(20, shape).astype(np.float32)
    remaining = rng.gamma(1.0, 5.0, shape).astype(np.float32)
    remaining[:, 0, :2] = 0.0
    size = rng.gamma(2.0, 3.0, shape).astype(np.float32)
    key = jax.random.key(4)
    got = np.asarray(sample_count(key, reported, remaining, size))
    want = np.asarray(jnp.asarray(reported) + sample_nb(key, remaining, size))
    np.testing.assert_array_equal(got, want)
    # Draws of each place and week, as of the arrays stacked by sample.
    got = sample_count(key, reported[0], remaining[0], size[0], 40)
    stacked = [np.broadcast_to(x[0], shape) for x in (remaining, size)]
    want = np.asarray(
        jnp.asarray(np.broadcast_to(reported[0], shape))
        + sample_nb(key, *stacked)
    )
    np.testing.assert_array_equal(got, want)


def test_sample_count_compiles_once_for_weeks_of_one_size() -> None:
    key = jax.random.key(0)

    def draw(w: int) -> np.ndarray:
        x = np.full((4, w), 5.0, np.float32)
        return sample_count(key, x, x, x, 1000)

    # 1000 draws of 4 places' 261 and 262 days pad to one size.
    draw(261)
    with compiles() as n:
        draw(262)
    assert n[0] == 0


def _snapshots(n_days: int) -> SnapshotArrays:
    """Daily releases of a country and two regions, one a day."""
    rng = np.random.default_rng(0)
    first = pd.Timestamp("2021-04-06")
    final = {
        loc: rng.poisson(lam, n_days) for loc, lam in (("A", 40), ("B", 25))
    }
    rows = []
    for r in range(20, n_days):
        vintage = (first + pd.Timedelta(days=r)).date().isoformat()
        for loc, counts in final.items():
            for t in range(r + 1):
                share = 1 - 0.6 ** (r - t + 1)
                rows.append(
                    {
                        "vintage": vintage,
                        "location": f"CD-{loc}",
                        "onset_date": (first + pd.Timedelta(days=t))
                        .date()
                        .isoformat(),
                        "observed": int(counts[t] * share),
                        "imputed": 0,
                        "cutoff": vintage,
                    }
                )
    daily = pd.DataFrame(rows)
    last = daily["vintage"].max()
    return snapshot_arrays(daily, last, ("CD-A", "CD-B"))


def _reference_delay_sums(
    snap: SnapshotArrays, r: int, settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    """The pair sums of `delay_sums`, unpadded."""
    pairs = _pairs_at(snap, r, settings.max_delay)
    c = np.asarray(snap.counts, dtype=float)
    n_loc, n_rel, n_day = c.shape
    obs = np.asarray(snap.observed).copy()
    obs[r + 1 :] = False
    prev = np.concatenate(
        [np.zeros_like(c[:, :1]), c[:, :-1] * obs[None, :-1]], axis=1
    )
    inc = np.where(obs[None], c - prev, 0.0)
    rows = inc.transpose(0, 2, 1).reshape(n_loc * n_day, n_rel)
    mask = np.tile(obs.T, (n_loc, 1))
    fixed = np.asarray(preprocess_negative_values(rows, mask, validate=False))
    fixed = fixed.reshape(n_loc, n_day, n_rel).transpose(0, 2, 1)[:, : r + 1]
    counts = c.copy()
    counts[:, : r + 1] = np.where(
        obs[None, : r + 1], np.cumsum(fixed, axis=1), c[:, : r + 1]
    )
    np.testing.assert_array_equal(_delay_counts(snap, r), counts)
    window = np.arange(n_day) > int(snap.cut[r]) - settings.history_delay
    num, den = _pair_sums(
        jnp.asarray(counts),
        jnp.asarray(snap.observed & window[None, :]),
        jnp.asarray(snap.cut),
        jnp.asarray(pairs[:, 0]),
        settings.max_delay,
    )
    return np.asarray(num), np.asarray(den)


def test_delay_sums_are_those_of_the_unpadded_snapshots() -> None:
    snap = _snapshots(90)
    settings = Settings()
    for r in (10, len(snap.releases) - 1):
        got = delay_sums(snap, r, settings)
        assert got is not None
        num, den = _reference_delay_sums(snap, r, settings)
        np.testing.assert_array_equal(got[1], num)
        np.testing.assert_array_equal(got[2], den)


def test_delay_sums_compile_once_for_snapshots_of_one_size() -> None:
    settings = Settings()
    snap = _snapshots(97)
    delay_sums(snap, len(snap.releases) - 1, settings)
    with compiles() as n:
        for days in (98, 99):
            snap = _snapshots(days)
            for r in (5, len(snap.releases) - 1):
                delay_sums(snap, r, settings)
    assert n[0] == 0
