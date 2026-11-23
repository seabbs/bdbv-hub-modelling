"""The random walk nowcast on the daily triangle gives both German views."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from nowcastgermany.data import ALL_AGES, GERMANY, nowcast_tasks

LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
NOWCAST = "2021-11-22"
HORIZONS = tuple(range(-28, 29))


def _triangle(days: int = 200, delay: int = 40) -> pd.DataFrame:
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
            final = rng.poisson(level * (1 + 0.2 * np.sin(i / 9)))
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


@pytest.fixture(scope="module")
def task():
    tasks = nowcast_tasks(_triangle(), NOWCAST, 200, 7, horizons=HORIZONS)
    return tasks[0]


@pytest.fixture(scope="module")
def model():
    from bdbvhub.models import get_model

    return get_model("randomwalknowcast_daily")


def _reported_sums(task, k):
    # Each place's k-day sums of the counts reported by the cut-off.
    h = task.daily_history
    last = h.loc[h["vintage"] == task.cutoff]
    by = last.pivot_table(
        index="onset_date", columns="location", values="observed"
    ).fillna(0.0)
    return by.rolling(k).sum()


def test_the_7_day_view_gives_every_target_day_from_one_forecast(task, model):
    out = model.forecast(task)
    assert set(out["target_end_date"]) == set(task.target_end_dates)
    assert set(out["sample"]) == set(range(1, task.n_samples + 1))
    assert (out["value"] >= 0).all()
    # National is the sum of the places by sample.
    wide = out.pivot_table(
        index=["target_end_date", "sample"], columns="location", values="value"
    )
    places = [c for c in wide.columns if c in task.locations]
    national = [c for c in wide.columns if c not in task.locations]
    assert np.allclose(wide[national[0]], wide[places].sum(axis=1))


def test_daily_nowcasts_are_at_least_what_is_reported(task, model):
    # The daily view adds draws of the counts still to come to those
    # reported. (A 7-day window's partly reported delay cells come from
    # the draw alone, as R's rolling aggregator does, so 7-day draws are
    # not floored at the reported sum.)
    q = model.daily_quantiles(task, (0.025,)).set_index(["location", "date"])
    seen = _reported_sums(task, 1)
    for place in task.locations:
        for day in seen.index[-28:]:
            if day in task.target_end_dates:
                assert q.loc[(place, day), "value"] >= seen.loc[day, place]


def test_the_daily_view_gives_ordered_quantiles_of_every_day(task, model):
    q = model.daily_quantiles(task, LEVELS)
    assert set(q["date"]) == set(task.target_end_dates)
    assert set(q["quantile"]) == set(LEVELS)
    for _, g in q.groupby(["location", "date"]):
        v = g.sort_values("quantile")["value"].to_numpy()
        assert (np.diff(v) >= -1e-9).all()


def test_each_view_fits_its_errors_on_its_own_sums(task, model, monkeypatch):
    # As baselinenowcast's aggregator: the 7-day view's retrospective
    # errors on 7-day rolling sums, the daily view's on daily counts.
    from nowcastbaselines import daily

    seen = []
    real = daily.baselinenowcast

    def spy(*args, **kwargs):
        agg = kwargs.get("ref_time_aggregator")
        if kwargs.get("output_type", "samples") == "samples":
            seen.append(getattr(agg, "keywords", {}).get("k", 1))
        return real(*args, **kwargs)

    monkeypatch.setattr(daily, "baselinenowcast", spy)
    model.forecast(task)
    assert set(seen) == {7}
    seen.clear()
    model.daily_quantiles(task, LEVELS)
    assert set(seen) == {1}


def test_a_short_history_still_gives_every_target_day():
    # At the first development date the releases start 35 days back, as
    # the oldest target's 7-day sum does: the triangle must reach it even
    # though the lag window is capped by the history.
    from bdbvhub.models import get_model

    short = nowcast_tasks(_triangle(days=35), NOWCAST, 50, 7, horizons=HORIZONS)
    model = get_model("randomwalknowcast_daily")
    for task in short:
        out = model.forecast(task)
        assert set(out["target_end_date"]) == set(task.target_end_dates)
