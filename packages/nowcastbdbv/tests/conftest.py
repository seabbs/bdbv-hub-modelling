"""Fixtures of the renewal model fitted to the frozen BDBV fixture."""

import numpyro
import pandas as pd
import pytest

numpyro.set_host_device_count(2)

from bdbv_cases import MAIN_REPO_DATA, PLACES, VINTAGES, release  # noqa: E402

from nowcastbdbv.data.delays import hub_delay_pmf  # noqa: E402
from nowcastbdbv.data.tidy import PROVINCE_CENTRES, PROVINCE_POPULATIONS  # noqa: E402
from nowcastrenewal.model import MAIN  # noqa: E402
from nowcastrenewal.model.inputs import vintage_from_tables  # noqa: E402


@pytest.fixture(scope="session")
def table():
    if (MAIN_REPO_DATA / "locations.csv").exists():
        return pd.read_csv(MAIN_REPO_DATA / "locations.csv")
    return pd.DataFrame(
        {
            "location": list(PROVINCE_CENTRES),
            "population": [PROVINCE_POPULATIONS[p] for p in PROVINCE_CENTRES],
            "latitude": [c[0] for c in PROVINCE_CENTRES.values()],
            "longitude": [c[1] for c in PROVINCE_CENTRES.values()],
        }
    )


@pytest.fixture(scope="session")
def tables():
    return {v: release(v) for v in VINTAGES}


@pytest.fixture(scope="session")
def latest_vintage(tables, table):
    day = VINTAGES[-1]
    target, delays = tables[day]
    return vintage_from_tables(
        target,
        delays,
        day,
        PLACES,
        table,
        imputation_delay=hub_delay_pmf(day),
    )


@pytest.fixture(params=["main"])
def config(request):
    return {"main": MAIN}[request.param]
