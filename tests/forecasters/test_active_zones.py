"""Active zones in the joint block; the others from an all-zones forecast."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

from nowcastfoundation.forecaster import (
    FoundationModel,
    HubTriangle,
    active_places,
    active_views,
    active_zone_rule,
)

PLACES = ("CD", "CD-AA", "CD-BB", "CD-AA-01", "CD-AA-02", "CD-BB-01")
TOPS, ZONES = PLACES[:3], PLACES[3:]


def _owners(views):
    return [p for _, owned in views for p in owned]


def test_active_zones_join_the_top_and_the_rest_come_from_province_views():
    views = active_views(PLACES, {"CD-AA-01"}, None)
    assert views == [
        ((*TOPS, "CD-AA-01"),) * 2,
        (("CD-AA", "CD-AA-01", "CD-AA-02"), ("CD-AA-02",)),
        (("CD-BB", "CD-BB-01"), ("CD-BB-01",)),
    ]


def test_with_every_zone_active_there_is_one_block():
    assert active_views(PLACES, set(ZONES), None) == [(PLACES, PLACES)]


def test_blocks_over_the_limit_are_cut_and_each_place_has_one_owner():
    views = active_views(PLACES, {"CD-AA-01"}, 2)
    assert all(len(v) <= 2 for v, _ in views)
    assert sorted(_owners(views)) == sorted(PLACES)
    # Every zone, active ones too, is in the zones' forecast.
    rest = [v for v, o in views if not set(o) & {*TOPS, "CD-AA-01"}]
    assert {p for v in rest for p in v} >= set(ZONES) - {"CD-AA-01"}


def _tri(older, recent):
    # 70 days; each zone's daily count is `older` from day 14 to day 41
    # and `recent` over the last 28 days (the last 4 weeks).
    final = np.ones((len(PLACES), 70))
    final[3:] = 0.0
    final[3:, 14:42] = np.asarray(older, dtype=float)[:, None]
    final[3:, 42:] = np.asarray(recent, dtype=float)[:, None]
    partial = np.full((len(PLACES), 70, 3), 0.5)
    days = [
        str(np.datetime64("2026-09-01") + np.timedelta64(d, "D"))
        for d in range(70)
    ]
    return HubTriangle(days, final, partial, last=69)


def test_a_zone_is_active_with_enough_cases_and_one_lately():
    # 8 weeks: zone 1 has 28 + 28 * 0.5 = 42 cases with recent ones;
    # zone 2 has 28 cases, none in the last 4 weeks; zone 3 has 14.
    tri = _tri([1.0, 1.0, 0.0], [0.5, 0.0, 0.5])
    assert active_places(PLACES, tri, 20, 8, 4, "daily") == {"CD-AA-01"}
    # With no recent-case rule, zone 2 is active too.
    assert active_places(PLACES, tri, 20, 8, 0, "daily") == {
        "CD-AA-01",
        "CD-AA-02",
    }
    # At least the cases: 14 counts when the threshold is 14.
    assert "CD-BB-01" in active_places(PLACES, tri, 14, 8, 4, "daily")


def test_the_rule_is_read_from_the_finetune_config(tmp_path):
    path = tmp_path / "finetune.toml"
    path.write_text(
        'hash = "x"\nactive_zone_cases = 20\nactive_zone_weeks = 8\n'
        "active_zone_recent_weeks = 4\n"
    )
    assert active_zone_rule(path) == (20, 8, 4)
    path.write_text('hash = "x"\n')
    assert active_zone_rule(path) == (None, 8, 4)


def test_the_model_cuts_by_activity_when_configured():
    model = FoundationModel(
        name="m",
        source="daily",
        layout="ragged",
        active_zone_cases=20,
        active_zone_weeks=8,
        active_zone_recent_weeks=4,
    )
    tri = _tri([1.0, 1.0, 0.0], [0.5, 0.0, 0.5])
    assert model._views(PLACES, tri) == active_views(PLACES, {"CD-AA-01"}, None)
    off = dataclasses.replace(model, active_zone_cases=None)
    assert off._views(PLACES, tri) == [(PLACES, PLACES)]
