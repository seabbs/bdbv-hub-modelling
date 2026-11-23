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


def _tri(recent):
    # 70 days; each zone's daily count in the last 56 days is `recent`.
    final = np.ones((len(PLACES), 70))
    final[3:] = 0.0
    final[3:, 14:] = np.asarray(recent, dtype=float)[:, None]
    partial = np.full((len(PLACES), 70, 3), 0.5)
    days = [
        str(np.datetime64("2026-09-01") + np.timedelta64(d, "D"))
        for d in range(70)
    ]
    return HubTriangle(days, final, partial, last=69)


def test_a_zone_is_active_with_more_than_the_cases_in_the_window():
    # 56 days of 0.2 a day is 11.2 cases; 0.1 a day is 5.6.
    tri = _tri([0.2, 0.1, 0.0])
    assert active_places(PLACES, tri, 10, 8, "daily") == {"CD-AA-01"}
    assert active_places(PLACES, tri, 5, 8, "daily") == {
        "CD-AA-01",
        "CD-AA-02",
    }


def test_the_rule_is_read_from_the_finetune_config(tmp_path):
    path = tmp_path / "finetune.toml"
    path.write_text(
        'hash = "x"\nactive_zone_cases = 10\nactive_zone_weeks = 8\n'
    )
    assert active_zone_rule(path) == (10, 8)
    path.write_text('hash = "x"\n')
    assert active_zone_rule(path) == (None, 8)


def test_the_model_cuts_by_activity_when_configured():
    model = FoundationModel(
        name="m",
        source="daily",
        layout="ragged",
        active_zone_cases=10,
        active_zone_weeks=8,
    )
    tri = _tri([0.2, 0.1, 0.0])
    assert model._views(PLACES, tri) == active_views(PLACES, {"CD-AA-01"}, None)
    off = dataclasses.replace(model, active_zone_cases=None)
    assert off._views(PLACES, tri) == [(PLACES, PLACES)]
