"""A block over the limit is cut into views at inference, as in training."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from nowcastfoundation.forecaster import FoundationModel, block_views
from nowcastutils.base import NATIONAL

sys.path.insert(0, str(Path(__file__).parent))

from test_daily_walk import HORIZONS, NOWCAST, _triangle


def _owners(views):
    return [p for _, owned in views for p in owned]


def test_a_nested_block_is_its_top_and_each_province_with_its_zones():
    zones = [f"XX-AA-0{i}" for i in range(1, 6)] + ["XX-BB-01"]
    places = (NATIONAL, "XX-AA", "XX-BB", *zones)
    views = block_views(places, 3)
    assert views[0] == ((NATIONAL, "XX-AA", "XX-BB"),) * 2
    assert all(len(v) <= 3 for v, _ in views)
    # Every place is forecast by exactly one view.
    assert sorted(_owners(views)) == sorted(places)
    assert ("XX-AA", "XX-AA-01", "XX-AA-02") in [v for v, _ in views]
    assert {p: v for v, o in views for p in o}["XX-BB-01"] == (
        "XX-BB",
        "XX-BB-01",
    )


def test_a_flat_block_is_cut_into_chunks_under_the_country():
    places = (NATIONAL, "A", "B", "C", "D", "E")
    views = block_views(places, 3)
    assert [v for v, _ in views] == [
        (NATIONAL, "A", "B"),
        (NATIONAL, "C", "D"),
        (NATIONAL, "E"),
    ]
    assert sorted(_owners(views)) == sorted(places)
    assert block_views(places, 6) == [(places, places)]


def _own_sampler(seen):
    # Each series' draws are its own last context value: a sampler under
    # which cutting a block into views must change nothing.
    def sample(groups, horizon, n, _seed):
        out = []
        for g in groups:
            seen.append(g)
            target = np.atleast_2d(g["target"])
            last = target[:, -1][None, :, None]
            out.append(np.broadcast_to(last, (n, target.shape[0], horizon)))
        return out

    return sample


def test_cutting_a_block_changes_nothing_a_series_forecasts_alone():
    from bdbvhub.models import get_model
    from nowcastgermany.data import nowcast_tasks

    task = nowcast_tasks(_triangle(), NOWCAST, 4, 7, horizons=HORIZONS)[0]
    model: FoundationModel = get_model("fm_chronos_daily_ragged")
    whole_seen, cut_seen = [], []
    whole = dataclasses.replace(model, sample=_own_sampler(whole_seen))
    per = 2 + whole.lag_count(task)
    cut = dataclasses.replace(
        model, sample=_own_sampler(cut_seen), block_limit=2 * per
    )
    a = whole.forecast_both(task)
    b = cut.forecast_both(task)
    assert len(whole_seen) < len(cut_seen)
    for g in cut_seen:
        series = np.atleast_2d(g["target"]).shape[0]
        series += len(g.get("past_covariates", {}))
        assert series <= 2 * per
    for x, y in zip(a, b, strict=True):
        assert (
            x.sort_values(list(x.columns[:-1]))
            .reset_index(drop=True)
            .equals(y.sort_values(list(y.columns[:-1])).reset_index(drop=True))
        )
