"""Model inputs the task carries: the imputation delay and the places."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhub.data.delays import hub_delay_pmf
from bdbvhub.data.zones import zone_places
from nowcastrenewal.model import MAIN
from nowcastrenewal.model.inputs import (
    WeekTables,
    gravity_kernel,
    imputation_pmf,
)


def test_imputation_pmf_truncates_and_renormalises() -> None:
    full = hub_delay_pmf("2026-10-24", 60)
    np.testing.assert_allclose(
        imputation_pmf(full, 30), hub_delay_pmf("2026-10-24", 30)
    )
    padded = imputation_pmf(np.array([0.5, 0.5]), 3)
    np.testing.assert_allclose(padded, [0.5, 0.5, 0.0, 0.0])


def test_the_kernel_comes_from_the_places_table() -> None:
    places = pd.DataFrame(
        {
            "location": ["A", "B", "C"],
            "population": [1.0, 2.0, 4.0],
            "latitude": [0.0, 0.0, 0.0],
            "longitude": [0.0, 1.0, 3.0],
        }
    )
    k = gravity_kernel(("A", "B", "C"), places)
    np.testing.assert_allclose(k.sum(axis=0), 1.0)
    np.testing.assert_allclose(np.diag(k), 0.0)
    # From A, the larger and nearer B and C take more than they would
    # by distance alone.
    assert k[2, 0] > 0 and k[1, 0] > 0


def test_the_kernel_names_places_with_no_population() -> None:
    places = pd.DataFrame(
        {"location": ["A"], "population": [1.0], "latitude": [0.0]}
    ).assign(longitude=0.0)
    with pytest.raises(ValueError, match=r"population for \['B'\]"):
        gravity_kernel(("A", "B"), places)


def test_delay_counts_clip_delays_and_leave_out_masked_places() -> None:
    first = pd.Timestamp("2026-08-08").date()
    weeks = WeekTables(pd.DataFrame(), first, 13, ["2026-08-08", "2026-08-15"])
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "B", "C"],
            "onset_week": [
                "2026-08-08",
                "2026-08-15",
                "2026-08-08",
                "2026-08-08",
            ],
            "delay": [-2, 500, 3, 1],
        }
    )
    counts = weeks.delay_counts(delays, ["A", "B"], MAIN, ("B",), every=False)
    assert counts.shape == (2, 2, MAIN.dmax + 1)
    assert counts[0, 0, 0] == 1
    assert counts[1, 0, MAIN.dmax] == 1
    assert counts.sum() == 2


def test_zone_places_are_the_mapped_zones() -> None:
    out = zone_places()
    assert list(out.columns) == [
        "location",
        "population",
        "latitude",
        "longitude",
    ]
    assert out["location"].str.match(r"^CD-[A-Z]{2}-[0-9]{2}$").all()


def test_unmapped_zones_take_their_province_place() -> None:
    from nowcastrenewal.zones import _zone_places
    from nowcastutils.base import ForecastTask

    places = pd.DataFrame(
        {
            "location": ["DE-BY", "DE-BY-01", "DE-BY-02"],
            "population": [100.0, 10.0, 30.0],
            "latitude": [48.0, 47.0, 49.0],
            "longitude": [11.0, 10.0, 12.0],
        }
    )
    task = ForecastTask(
        vintage="2026-10-24",
        reference_date=pd.Timestamp("2026-10-24").date(),
        kind="real",
        cutoff="2026-10-24",
        target=pd.DataFrame(),
        delays=None,
        locations=("DE-BY-02", "DE-BY-09"),
        target_end_dates=(),
        places=places,
    )
    out = _zone_places(task).set_index("location")
    assert out.loc["DE-BY-02", "population"] == 30.0
    assert out.loc["DE-BY-09", "population"] == 20.0
    assert out.loc["DE-BY-09", "latitude"] == 48.0


def test_an_imputation_delay_with_no_mass_is_an_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="no mass"):
        imputation_pmf(np.array([0.0, 0.0, 1.0]), 1)
