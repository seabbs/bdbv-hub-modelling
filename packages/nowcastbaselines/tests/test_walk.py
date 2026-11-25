"""The shared random walk."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nowcastbaselines.walk import (
    DailySteps,
    random_walk,
    same_weekday_variance,
    walk_noise,
    walk_variance,
    weekday_means,
)


def test_steps_are_cumulative_within_a_sample() -> None:
    key = jax.random.key(0)
    lam, y = random_walk(key, np.full(4000, 99.0, np.float32), 0.3, 3)
    lam = np.asarray(lam)
    assert lam.shape == np.asarray(y).shape == (4000, 3)
    level = np.log1p(lam)
    steps = np.diff(
        np.column_stack([np.full(4000, np.log(100.0)), level]), axis=1
    )
    assert steps.std(axis=0) == pytest.approx([0.3] * 3, rel=0.1)
    # Sample variance grows with the horizon, as steps accumulate.
    sd = level.std(axis=0)
    assert sd[0] < sd[1] < sd[2]
    assert sd[2] == pytest.approx(0.3 * np.sqrt(3), rel=0.1)
    assert abs(np.corrcoef(steps[:, 0], steps[:, 1])[0, 1]) < 0.1


def test_the_level_is_floored_at_zero_on_the_log_scale() -> None:
    key = jax.random.key(1)
    lam, y = random_walk(key, np.zeros(500, np.float32), 1.0, 4)
    assert (np.asarray(lam) >= 0).all()
    assert (np.asarray(y) >= 0).all()
    # Starting at 0 the walk is floored at 0 about half the time.
    assert 0.35 < (np.asarray(lam)[:, 0] == 0).mean() < 0.65


def test_no_spread_gives_a_flat_rate() -> None:
    lam, _ = random_walk(
        jax.random.key(2), np.full(10, 7.0, np.float32), 0.0, 3
    )
    assert np.allclose(np.asarray(lam), 7.0)


def test_one_key_gives_one_draw() -> None:
    a = random_walk(jax.random.key(3), np.full(5, 3.0, np.float32), 0.2, 2)
    b = random_walk(jax.random.key(3), np.full(5, 3.0, np.float32), 0.2, 2)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_the_noise_is_the_delta_method_poisson_variance() -> None:
    y = np.array([0.0, 1.0, 4.0, 9.0])
    v = y / (1 + y) ** 2
    np.testing.assert_allclose(walk_noise(y), v[1:] + v[:-1])
    # It matches the simulated variance once the counts are not small.
    rng = np.random.default_rng(0)
    for lam in (30.0, 300.0):
        sim = np.log1p(rng.poisson(lam, 200_000)).var()
        assert walk_noise(np.array([lam, lam]))[0] / 2 == pytest.approx(
            sim, rel=0.05
        )


def test_poisson_noise_alone_leaves_little_or_no_walk_variance() -> None:
    rng = np.random.default_rng(1)
    for lam in (8.0, 20.0, 100.0):
        y = rng.poisson(lam, 5000).astype(float)
        d = np.diff(np.log1p(y))
        raw = d.var(ddof=1)
        net = walk_variance(d, walk_noise(y))
        # The delta method leaves a little at small counts, under a fifth
        # of the raw variance, and none at large ones.
        assert net < 0.2 * raw
    assert walk_variance(d, walk_noise(y)) <= 0.001


def test_a_random_walk_on_large_counts_is_recovered() -> None:
    rng = np.random.default_rng(2)
    level = np.log(500.0) + np.cumsum(rng.normal(0, 0.3, (300, 9)), axis=1)
    y = rng.poisson(np.expm1(level)).astype(float)
    d = np.concatenate([np.diff(np.log1p(row)) for row in y])
    noise = np.concatenate([walk_noise(row) for row in y])
    assert np.sqrt(walk_variance(d, noise)) == pytest.approx(0.3, rel=0.05)


def test_a_walk_on_small_counts_loses_most_of_the_poisson_noise() -> None:
    rng = np.random.default_rng(3)
    level = np.log(10.0) + np.cumsum(rng.normal(0, 0.3, (2000, 9)), axis=1)
    y = rng.poisson(np.expm1(np.clip(level, 0, None))).astype(float)
    d = np.concatenate([np.diff(np.log1p(row)) for row in y])
    noise = np.concatenate([walk_noise(row) for row in y])
    assert d.std() > 0.45
    assert np.sqrt(walk_variance(d, noise)) == pytest.approx(0.3, abs=0.08)


def test_fewer_than_two_changes_give_no_variance() -> None:
    assert np.isnan(walk_variance(np.array([0.7]), np.array([0.1])))


def _weekdays(t: int) -> np.ndarray:
    return np.arange(1, t) % 7


def test_a_walk_with_no_weekday_pattern_has_no_step_means() -> None:
    # A plain random walk: each weekday's mean step is 0 within 3 of its
    # standard errors.
    rng = np.random.default_rng(2)
    t = 7 * 60 + 1
    y = np.expm1(np.log1p(1e5) + np.cumsum(rng.normal(0.0, 0.1, t)))
    changes = np.diff(np.log1p(y))
    weekday = _weekdays(t)
    means = weekday_means(changes, weekday)
    se = 0.1 / np.sqrt(np.bincount(weekday, minlength=7))
    assert (np.abs(means) < 3 * se).all()
    assert np.sqrt(same_weekday_variance([y])) == pytest.approx(0.1, rel=0.2)


def test_a_weekday_pattern_is_recovered_and_left_out_of_the_spread() -> None:
    # A walk with a fixed weekday effect: the mean step into each weekday
    # is the change of the effect into it, and the effect cancels in the
    # same-weekday changes behind the spread.
    rng = np.random.default_rng(3)
    t = 7 * 60 + 1
    effect = np.array([0.4, 0.2, 0.0, -0.1, -0.2, -0.6, 0.3])
    day = np.arange(t) % 7
    level = np.log1p(1e5) + np.cumsum(rng.normal(0.0, 0.05, t))
    y = np.expm1(level + effect[day])
    weekday = _weekdays(t)
    means = weekday_means(np.diff(np.log1p(y)), weekday)
    want = effect - np.roll(effect, 1)
    se = 0.05 / np.sqrt(np.bincount(weekday, minlength=7))
    assert (np.abs(means - want) < 3 * se).all()
    assert np.sqrt(same_weekday_variance([y])) == pytest.approx(0.05, rel=0.2)


def test_daily_steps_repeat_the_weekly_means() -> None:
    steps = DailySteps(
        mean=np.array([[0.1, 0.2, -0.3, 0.0, 0.0, 0.1, -0.1]]),
        sigma=np.array([0.05]),
    )
    mean, spread = steps.at(np.array([5]), 9)
    np.testing.assert_allclose(
        mean[0], [0.1, -0.1, 0.1, 0.2, -0.3, 0.0, 0.0, 0.1, -0.1]
    )
    np.testing.assert_allclose(spread, 0.05)
