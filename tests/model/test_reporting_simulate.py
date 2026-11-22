"""One simulation's reporting step (`reporting.simulate`).

The expected reports are replayed against epinowcast elsewhere
(`test_reporting_expectation.py`); here the counts are checked against
those expected reports over many observation draws, and the long format
and added values against their definitions.
"""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
from numpyro import handlers

from nowcastrenewal.reporting.simulate import (
    COUNT_COLUMNS,
    report_simulation,
    reporting_model,
)

from .test_reporting_effects import _design
from .test_reporting_expectation import REPORT_VALUES, _report

T = 42
PLACES, AGES = ["P01", "P02"], ["A1", "A1"]
DESIGN = {
    **_design(delay_form="hybrid", delay_change="drift"),
    **_report(report_dow="yes"),
    "n_places": 2,
    "n_ages": 1,
    "revisions": "no",
}
VALS = {
    **REPORT_VALUES,
    "max_delay": 8,
    "offset": 3,
    "obs_family": ["negbin1d", "negbin1d"],
    "obs_sqrt_phi": [0.4, 0.6],
    "revise_over": [0.1, 0.1],
    "revise_move": [0.1, 0.1],
    "revise_keep": [4.0, 4.0],
}
ONSETS = np.stack([np.linspace(1, 30, T), np.linspace(5, 15, T)], axis=1)


def test_counts_average_to_the_expected_reports() -> None:
    from nowcastrenewal.reporting.simulate import reporting_frame

    draws = 300
    frame = reporting_frame(VALS, T, PLACES, AGES)
    with (
        jax.enable_x64(True),
        handlers.trace() as tr,
        handlers.seed(rng_seed=9),
    ):
        reporting_model(frame, DESIGN, VALS, ONSETS, draws)
    expected = np.exp(np.asarray(tr["expected::log_expected"]["value"]))
    counts = np.stack(
        [np.asarray(tr[f"draw_{k}::reported"]["value"]) for k in range(draws)]
    )
    t = np.arange(T)[:, None] + np.arange(8)[None, :]
    inside = t <= T - 1
    got = counts[:, :, inside].sum(axis=-1)  # draws by strata
    want = expected[:, inside].sum(axis=-1)
    se = got.std(axis=0) / np.sqrt(draws)
    assert np.all(np.abs(got.mean(axis=0) - want) < 5 * se)


def test_the_long_format_and_added_values() -> None:
    counts, added = report_simulation(
        DESIGN, VALS, ONSETS, PLACES, AGES, seed=3, draws=2
    )
    assert list(counts.columns) == list(COUNT_COLUMNS)
    assert set(counts["draw"]) == {1, 2}
    assert set(counts["location"]) == {"P01", "P02"}
    assert (counts["count"] > 0).all()
    first = pd.Timestamp("2026-01-03") + pd.Timedelta(days=3)
    ref = pd.to_datetime(counts["reference_date"])
    rep = pd.to_datetime(counts["report_date"])
    assert ref.min() >= first
    assert (rep >= ref).all()
    assert (rep <= first + pd.Timedelta(days=T - 1)).all()
    assert added["stratum_location"] == ["P01", "P02"]
    assert added["low_start_days"] == 0
    assert added["lambda_hat"] == np.median(ONSETS[ONSETS >= 1])
    np.testing.assert_allclose(
        added["sqrt_phi_obs"],
        np.array([0.4, 0.6]) * np.sqrt(added["lambda_hat"] * added["sum_p2"]),
    )


def test_revisions_keep_each_reference_days_total() -> None:
    plain, _ = report_simulation(
        DESIGN, VALS, ONSETS, PLACES, AGES, seed=5, draws=1
    )
    revised, _ = report_simulation(
        {**DESIGN, "revisions": "yes"}, VALS, ONSETS, PLACES, AGES, seed=5,
        draws=1,
    )  # fmt: skip
    key = ["location", "reference_date"]
    a = plain.groupby(key)["count"].sum()
    b = revised.groupby(key)["count"].sum()
    pd.testing.assert_series_equal(a, b.reindex(a.index).fillna(0).astype(int))
    assert len(revised) > len(plain)


def test_multi_place_and_age_locations() -> None:
    design = {**DESIGN, "n_ages": 2}
    places, ages = ["P01", "P01"], ["A1", "A2"]
    _, added = report_simulation(
        design, VALS, ONSETS, places, ages, seed=1, draws=1
    )
    assert added["stratum_location"] == ["P01-A1", "P01-A2"]
