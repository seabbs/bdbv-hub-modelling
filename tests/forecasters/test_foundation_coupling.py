"""Chronos-2's trajectories joined by the fitted coupling."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhub.data.zones import ZoneData, hub_zones
from bdbvhub.models import get_model
from nowcasteval.backtest import BacktestConfig, make_tasks
from nowcastfoundation import forecaster as foundation
from nowcastutils import trajectories
from nowcastutils.base import ForecastTask

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


class _Pipeline:
    """Quantiles from each group's own level, spread by step; counts calls."""

    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(self) -> None:
        self.groups = 0

    def predict_quantiles(
        self,
        inputs,
        prediction_length,
        quantile_levels,
        batch_size=256,  # noqa: ARG002
    ):
        import torch

        self.groups += len(inputs)
        out = []
        for g in inputs:
            target = np.atleast_2d(g["target"])
            own = np.nan_to_num(target[:, -7:]).mean(axis=1) + 1.0
            step = 1.0 + 0.1 * np.arange(prediction_length)
            z = np.asarray(quantile_levels) - 0.5
            q = own[:, None, None] * step[None, :, None] * (1 + z)[None, None]
            out.append(torch.tensor(q, dtype=torch.float32))
        return out, None


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> ForecastTask:
    """The latest fixture release's task, so earlier releases have matured."""
    latest = target["vintage"].max()
    rows = target.loc[target["vintage"] == latest]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


@pytest.fixture
def couplings(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record each `couple` call's method and fit."""
    seen: list = []
    real = trajectories.couple

    def couple(quantiles, levels, method, fit, n_samples, seed):
        seen.append((method, fit))
        return real(quantiles, levels, method, fit, n_samples, seed)

    monkeypatch.setattr(foundation, "couple", couple)
    return seen


def test_the_models_couple_by_the_fitted_gaussian_copula():
    assert get_model("fm_chronos_daily_ragged_ft").coupling == "gaussian"
    assert set(trajectories.METHODS) >= {"independent", "comonotone"}


def test_the_sampler_draws_through_couple(
    monkeypatch: pytest.MonkeyPatch, couplings: list
):
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    group = {"target": np.full((2, 20), 5.0)}
    draws = foundation.chronos_sampler(
        [group], 6, 200, 1, coupling={"method": "comonotone"}
    )[0]
    assert couplings[-1][0] == "comonotone"
    rank = draws.argsort(axis=0).argsort(axis=0)
    # Comonotone: each trajectory keeps its rank across a series' steps.
    assert (rank[:, 0, 0] == rank[:, 0, -1]).all()
    foundation.chronos_sampler([group], 6, 200, 1)
    assert couplings[-1][0] == "independent"


def test_the_gaussian_fit_reads_the_model_s_earlier_forecasts(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    get_model("fm_chronos_daily_ragged").forecast(task)
    # The task's block and the same block at earlier releases.
    assert pipe.groups > 1
    method, fit = couplings[-1]
    assert method == "gaussian"
    assert {"rho", "s", "n_origins"} <= set(fit)
    assert fit["n_origins"] >= 1


@pytest.mark.parametrize("method", ["independent", "comonotone"])
def test_the_si_couplings_need_no_earlier_forecasts(
    monkeypatch: pytest.MonkeyPatch,
    couplings: list,
    task: ForecastTask,
    method: str,
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = dataclasses.replace(
        get_model("fm_chronos_daily_ragged"), coupling=method
    )
    model.forecast(task)
    assert pipe.groups == 1
    assert couplings[-1] == (method, {"method": method})


def test_prefetch_holds_the_earlier_forecasts_too(
    monkeypatch: pytest.MonkeyPatch, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = get_model("fm_chronos_daily_ragged")
    alone = model.forecast(task)
    ready = foundation.prefetch(model, [task])
    pipe.groups = 0
    pd.testing.assert_frame_equal(ready.forecast(task), alone)
    assert pipe.groups == 0


def test_an_unknown_coupling_is_an_error_when_the_model_is_made():
    with pytest.raises(ValueError, match="unknown coupling"):
        dataclasses.replace(
            get_model("fm_chronos_daily_ragged"), coupling="schaake"
        )


def test_with_no_earlier_release_the_fit_is_independence(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    monkeypatch.setattr(foundation, "coupling_releases", lambda *_: [])
    out = get_model("fm_chronos_daily_ragged").forecast(task)
    assert not out.empty
    assert pipe.groups == 1
    method, fit = couplings[-1]
    assert method == "gaussian"
    assert (fit["n_origins"], fit["rho"], fit["s"]) == (0, 0.0, 0.0)


def test_earlier_blocks_are_scored_on_what_is_final_now(
    monkeypatch: pytest.MonkeyPatch, task: ForecastTask
):
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    model = get_model("fm_chronos_daily_ragged")
    calls = []
    past_blocks = foundation.FoundationModel._past_blocks

    def spy(self, rows, places, tri, start, horizon):
        out = past_blocks(self, rows, places, tri, start, horizon)
        calls.append((tri, start, horizon, out))
        return out

    monkeypatch.setattr(foundation.FoundationModel, "_past_blocks", spy)
    model.forecast(task)
    ((tri, start, horizon, past),) = calls
    assert past
    period = model._period(tri)
    now = np.concatenate([tri.final, period.totals])
    final = []
    for block, seen in past:
        # Each block is the task's block as it stood earlier: less context.
        assert np.atleast_2d(block["target"]).shape[1] < start
        assert seen.shape == (now.shape[0], horizon)
        # Final steps first, then NaN from the first step not yet final.
        n = int(np.isfinite(seen[0]).sum())
        assert n > 0
        assert np.isfinite(seen[0, :n]).all()
        assert np.isnan(seen[:, n:]).all()
        # They are the task's own values up to its last final step.
        at = tri.n_context - n if n < horizon else None
        if at is not None:
            np.testing.assert_array_equal(
                seen[: len(tri.final), :n],
                now[: len(tri.final), at : tri.n_context],
            )
        final.append(n)
    # Later releases have fewer steps final at the task's release.
    assert final == sorted(final, reverse=True)


def test_the_fit_takes_one_earlier_release_a_week_for_eight_weeks():
    weekly = [f"2026-{m:02d}-{d:02d}" for m, d in [(8, 1), (8, 8)]]
    weekly += [
        str(pd.Timestamp("2026-08-15") + pd.Timedelta(weeks=k))[:10]
        for k in range(10)
    ]
    got = foundation.coupling_releases(weekly)
    assert got == weekly[-9:-1]
    daily = [
        str(pd.Timestamp("2026-08-01") + pd.Timedelta(days=k))[:10]
        for k in range(80)
    ]
    got = foundation.coupling_releases(daily)
    assert len(got) == foundation.COUPLING_WEEKS
    last = pd.Timestamp(daily[-1])
    assert [(last - pd.Timestamp(d)).days for d in got] == list(
        range(56, 0, -7)
    )
    assert foundation.coupling_releases(daily[:1]) == []
