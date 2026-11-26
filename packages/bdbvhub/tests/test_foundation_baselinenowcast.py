"""The baselinenowcast variants: its nowcast, then Chronos-2."""

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastgermany.data import ALL_AGES, GERMANY

NOWCAST = "2021-11-22"
HORIZONS = tuple(range(-28, 29))


def _triangle(
    days: int = 200, delay: int = 40, weekday: float = 0.0
) -> pd.DataFrame:
    # Two states and Germany their sum, each day's count reported over
    # `delay` days with a geometric delay, as cumulative report rows.
    rng = np.random.default_rng(1)
    last = dt.date.fromisoformat(NOWCAST)
    share = 0.85 ** np.arange(delay + 1)
    share /= share.sum()
    rows = []
    totals: dict[tuple[str, str], int] = {}
    for state, level in (("DE-BB", 40.0), ("DE-BE", 90.0)):
        for i in range(days):
            day = last - dt.timedelta(days=days - 1 - i)
            # `weekday`: weekends report that share fewer cases, as
            # German hospitalisations by report date do.
            dip = 1 - weekday * (day.weekday() >= 5)
            final = rng.poisson(level * dip * (1 + 0.2 * np.sin(i / 9)))
            seen = np.cumsum(rng.multinomial(final, share))
            for d in range(delay + 1):
                report = day + dt.timedelta(days=d)
                if report > last:
                    break
                key = (day.isoformat(), report.isoformat())
                totals[key] = totals.get(key, 0) + int(seen[d])
                rows.append((day.isoformat(), state, report, int(seen[d])))
    out = [
        {
            "reference_date": r,
            "location": s,
            "age_group": ALL_AGES,
            "report_date": rep.isoformat(),
            "confirm": c,
        }
        for r, s, rep, c in rows
    ]
    out += [
        {
            "reference_date": r,
            "location": GERMANY,
            "age_group": ALL_AGES,
            "report_date": rep,
            "confirm": c,
        }
        for (r, rep), c in totals.items()
    ]
    return pd.DataFrame(out)


pytest.importorskip("jax")


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


def test_both_baselinenowcast_variants_are_registered():
    from bdbvhub.models import get_model

    zero = get_model("foundation_zeroshot_baselinenowcast")
    assert (zero.layout, zero.source, zero.baselinenowcast) == (
        "complete",
        "data",
        True,
    )
    from nowcastutils.base import list_models

    assert "foundation_baselinenowcast" in {s.name for s in list_models()}


def test_the_context_is_the_point_nowcast_to_the_cut_off(task):
    # Chronos-2 sees each place's counts corrected by baselinenowcast's
    # point nowcast, up to and with the cut-off day, and forecasts from
    # the day after it.
    from bdbvhub.models import get_model
    from nowcastbaselines.baselinenowcast import baselinenowcast
    from nowcastbaselines.baselinenowcast.triangle import daily_triangle
    from nowcastutils.dates import as_date

    store: list = []
    model = dataclasses.replace(
        get_model("foundation_zeroshot_baselinenowcast"), sample=_capture(store)
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


@pytest.fixture(scope="module")
def many():
    from nowcastgermany.data import nowcast_tasks

    return nowcast_tasks(_triangle(), NOWCAST, 300, 7, horizons=HORIZONS)[0]


def test_the_nowcast_days_carry_baselinenowcast_draws(many):
    # Each sample's days to the cut-off are baselinenowcast's draws, so
    # they spread as its draws do.
    import jax

    from bdbvhub.models import get_model
    from nowcastbaselines.baselinenowcast import baselinenowcast
    from nowcastbaselines.baselinenowcast.triangle import daily_triangle
    from nowcastutils.dates import as_date

    model = dataclasses.replace(
        get_model("foundation_zeroshot_baselinenowcast"), sample=_capture([])
    )
    _, daily = model.forecast_both(many)
    cutoff = str(many.cutoff)[:10]
    place = next(iter(many.locations))
    tri, mask = daily_triangle(
        many.daily_history, place, as_date(cutoff), 120, 40
    )
    ref = np.asarray(
        baselinenowcast(tri, mask, key=jax.random.key(0), draws=4000)
    )[:, -1]
    day = daily.loc[
        (daily["location"] == place) & (daily["date"] == cutoff), "value"
    ].to_numpy()
    assert len(day) == many.n_samples
    assert day.std() > 0
    assert day.mean() == pytest.approx(ref.mean(), rel=0.15)
    assert day.std() == pytest.approx(ref.std(), rel=0.25)
    # Reproducible from the task.
    _, again = model.forecast_both(many)
    pd.testing.assert_frame_equal(daily, again)


def _sums_to_cut_off(model, task, labels, draws):
    # Each target to the cut-off is the walk nowcast's draw, rounded, and
    # the country the sum of its provinces by sample.
    from nowcastutils.base import NATIONAL

    weekly, _ = model.forecast_both(task)
    cutoff = str(task.cutoff)[:10]
    past = [w for w in task.target_end_dates if w <= cutoff]
    assert past
    want = np.maximum(np.round(draws), 0.0)
    for w in past:
        k = labels.index(w)
        for j, loc in enumerate([*task.locations, NATIONAL]):
            got = (
                weekly.loc[
                    (weekly["location"] == loc)
                    & (weekly["target_end_date"] == w)
                ]
                .sort_values("sample")["value"]
                .to_numpy()
            )
            if loc == NATIONAL:
                expect = want[:, :, k].sum(axis=1)
            else:
                expect = want[:, j, k]
            np.testing.assert_allclose(got, expect)
    k = labels.index(past[-1])
    assert want[:, :, k].std(axis=0).min() > 0


def test_the_7_day_sums_to_the_cut_off_are_the_walk_nowcast(many):
    # The 7-day sums to the cut-off are randomwalknowcast's 7-day nowcast
    # draws, and Chronos-2's context of them is their median.
    from bdbvhub.models import get_model
    from nowcastbaselines.randomwalk import RandomWalkNowcast

    store: list = []
    model = dataclasses.replace(
        get_model("foundation_zeroshot_baselinenowcast"),
        sample=_capture(store),
    )
    days, draws = RandomWalkNowcast().rolling_nowcast(many)
    _sums_to_cut_off(model, many, days, draws)
    assert days[-1] == str(many.cutoff)[:10]
    target = np.atleast_2d(store[-1]["target"])
    n = 1 + len(many.locations)
    for j in range(len(many.locations)):
        median = np.median(draws[:, j, -1])
        assert target[n + 1 + j, -1] == pytest.approx(median, rel=1e-5)
        assert target[n + 1 + j, -8] == pytest.approx(
            np.median(draws[:, j, -8]), rel=1e-5
        )


def test_hub_week_targets_take_the_walk_weekly_nowcast():
    # A task of hub weeks takes the walk's nowcast of hub weeks.
    from bdbvhub.models import get_model
    from nowcastbaselines.randomwalk import RandomWalkNowcast
    from nowcastgermany.data import nowcast_tasks

    task = nowcast_tasks(_triangle(), NOWCAST, 200, 7, horizons=(-9, -2, 5))[0]
    model = dataclasses.replace(
        get_model("foundation_zeroshot_baselinenowcast"), sample=_capture([])
    )
    weeks, draws = RandomWalkNowcast().weekly_nowcast(task)
    _sums_to_cut_off(model, task, weeks, draws)
