import os
from pathlib import Path

import numpyro
import pandas as pd
import pytest

numpyro.set_host_device_count(2)

from nowcastbdbv.data.delays import hub_delay_pmf  # noqa: E402
from nowcastbdbv.data.tidy import PROVINCE_CENTRES, PROVINCE_POPULATIONS  # noqa: E402
from nowcastrenewal.model import MAIN  # noqa: E402
from nowcastrenewal.model.inputs import vintage_from_tables  # noqa: E402

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"
MAIN_REPO_DATA = Path(os.environ.get("BDBV_DATA", str(FIXTURE_DATA)))
# Releases of the frozen fixture with delay rows, oldest first.
VINTAGES = ("2026-10-09", "2026-10-24", "2026-11-08")
# Ituri first, a later start (Nord-Kivu), a late one (Tshopo) and two
# provinces with no case, modelled with importation only.
PLACES = ("CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-KE", "CD-KN")


def release(vintage):
    """Return the fixture's target and delay rows of one release."""
    target = pd.read_csv(FIXTURE_DATA / "target.csv")
    delays = pd.read_csv(FIXTURE_DATA / "delays.csv.gz")
    return (
        target.loc[target["vintage"] == vintage],
        delays.loc[delays["vintage"] == vintage],
    )


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
