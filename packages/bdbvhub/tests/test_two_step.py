"""Two-step Chronos-2: baselinenowcast's point nowcast, then Chronos-2."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

from test_daily_walk import HORIZONS, NOWCAST, _triangle


def _capture(store):
    def sample(groups, horizon, n, _seed):
        store.extend(groups)
        return [
            np.zeros((n, np.atleast_2d(g["target"]).shape[0], horizon))
            for g in groups
        ]

    return sample


@pytest.fixture(scope="module")
def task():
    from nowcastgermany.data import nowcast_tasks

    return nowcast_tasks(_triangle(), NOWCAST, 4, 7, horizons=HORIZONS)[0]


def test_both_two_step_variants_are_registered():
    from bdbvhub.models import get_model

    zero = get_model("foundation_zeroshot_twostep")
    assert (zero.layout, zero.source, zero.two_step) == (
        "complete",
        "data",
        True,
    )
    from nowcastutils.base import list_models

    assert "foundation_twostep" in {s.name for s in list_models()}


def test_the_context_is_the_point_nowcast_to_the_cut_off(task):
    # Chronos-2 sees each place's counts corrected by baselinenowcast's
    # point nowcast, up to and with the cut-off day, and forecasts from
    # the day after it.
    from bdbvhub.models import get_model
    from nowcastbaselines.baselinenowcast import baselinenowcast
    from nowcastbaselines.daily import daily_triangle
    from nowcastutils.dates import as_date

    store: list = []
    model = dataclasses.replace(
        get_model("foundation_zeroshot_twostep"), sample=_capture(store)
    )
    model.forecast_both(task)
    group = store[-1]
    cutoff = as_date(task.cutoff)
    places = list(task.locations)
    target = np.atleast_2d(group["target"])
    # Count rows are the places in the block's order: the country first.
    place = places[0]
    row = 1 + places.index(place)
    tri, mask = daily_triangle(task.daily_history, place, cutoff, 120, 40)
    point = np.asarray(baselinenowcast(tri, mask, output_type="point"))
    assert target[row, -1] == pytest.approx(point[-1], rel=1e-5)
    assert target[row, -10] == pytest.approx(point[-10], rel=1e-5)
    # The context's last day is the cut-off, at least what is reported.
    reported = task.daily_history.loc[
        (task.daily_history["vintage"] == task.cutoff)
        & (task.daily_history["location"] == place)
    ].set_index("onset_date")["observed"]
    assert target[row, -1] >= reported.get(task.cutoff, 0.0)
