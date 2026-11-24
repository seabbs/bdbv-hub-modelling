"""Reading score and forecast rows: numbers and a model's source."""

from flowlib.dashboard import rows


def test_numbers_read_as_floats_and_the_rest_as_none():
    assert rows.num("1.5") == 1.5
    assert rows.num(2) == 2.0
    assert rows.num("") is None
    assert rows.num(None) is None


def test_a_model_s_source_is_daily_or_weekly_else_both():
    assert rows.source("fm_chronos_daily_ragged") == "daily"
    assert rows.source("fm_chronos_weekly_ragged_ft") == "weekly"
    assert rows.source("randomwalknowcast") is None
    assert rows.sources("fm_chronos_weekly_ragged") == ["weekly"]
    assert rows.sources("randomwalknowcast") == ["daily", "weekly"]
