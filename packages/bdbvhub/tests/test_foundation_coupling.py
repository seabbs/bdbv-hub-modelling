"""Chronos-2's trajectories joined by the fitted coupling."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.zones import ZoneData, hub_zones
from nowcasteval.backtest import BacktestConfig
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


def test_the_models_choose_their_coupling_online():
    assert get_model("fm_chronos_daily_ragged_ft").coupling == "online"
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


def test_the_coupling_is_chosen_from_the_model_s_earlier_forecasts(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    chosen = []
    real = foundation.choose_coupling

    def spy(quantiles, levels, observed, days, day, history, starts, **kw):
        chosen.append(
            (quantiles.shape, observed.shape, days, day, history, starts, kw)
        )
        return real(
            quantiles, levels, observed, days, day, history, starts, **kw
        )

    monkeypatch.setattr(foundation, "choose_coupling", spy)
    model = get_model("fm_chronos_daily_ragged")
    model.forecast(task)
    # The task's block and the same block at earlier releases.
    assert pipe.groups > 1
    ((q_shape, seen_shape, days, day, history, starts, kw),) = chosen
    # Each earlier release's day, before the task's own.
    assert (np.diff(days) > 0).all()
    assert days[-1] < day
    assert day - days[0] <= trajectories.COUPLING_DAYS
    assert q_shape[:3] == seen_shape
    assert seen_shape[0] == pipe.groups - 1
    # The history ends before the first step drawn; each earlier block
    # starts before that.
    assert history.shape[0] == seen_shape[1]
    assert (starts < history.shape[1]).all()
    assert kw["n_samples"] == task.n_samples
    method, fit = couplings[-1]
    assert method == fit["method"] in {"gaussian", "schaake"}
    record = model.copula_fit
    assert record["method"] == method
    assert record["chosen_by"] in {"energy", "fallback"}
    assert "windows" not in record
    assert set(record["energy"]) <= {
        trajectories._label(m, w) for m, w in trajectories.CANDIDATES
    }


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
            get_model("fm_chronos_daily_ragged"), coupling="nonesuch"
        )


def test_with_no_earlier_release_the_history_couples(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    monkeypatch.setattr(foundation, "coupling_releases", lambda *_: [])
    out = get_model("fm_chronos_daily_ragged").forecast(task)
    assert not out.empty
    assert pipe.groups == 1
    method, fit = couplings[-1]
    # Not independent draws: the Schaake shuffle on the series' history.
    assert (method, fit["chosen_by"], fit["n_eval"]) == (
        "schaake",
        "fallback",
        0,
    )


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
    for block, seen, _, _ in past:
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


def test_earlier_blocks_raise_errors_other_than_no_rows(
    monkeypatch: pytest.MonkeyPatch, task: ForecastTask
):
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    hub_triangle = foundation.hub_triangle
    calls = []

    def broken(rows, source, places, last_time, max_lag=None, **kwargs):
        calls.append(last_time)
        # The task's own triangle is built first; earlier ones break.
        if len(calls) > 1:
            raise ValueError("a malformed release")
        return hub_triangle(rows, source, places, last_time, max_lag, **kwargs)

    monkeypatch.setattr(foundation, "hub_triangle", broken)
    with pytest.raises(ValueError, match="malformed"):
        get_model("fm_chronos_daily_ragged").forecast(task)


def test_a_block_over_the_limit_is_coupled_as_one_block(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    model = get_model("fm_chronos_daily_ragged")
    per = 2 + model.lag_count(task)
    cut = dataclasses.replace(model, block_limit=3 * per)
    assert len(cut._views(*_places_and_triangle(cut, task))) > 1
    out = cut.forecast(task)
    assert not out.empty
    # The views' quantiles are put back together and coupled once, by the
    # whole block's fit: its Schaake windows span every place.
    ((method, fit),) = couplings
    assert (method, fit["chosen_by"]) == ("schaake", "fallback")


def _places_and_triangle(model, task):
    rows, places, weeks = model._rows(task)
    return places, model._triangle(rows, places, weeks[-1])
