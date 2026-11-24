"""The frozen fixture's releases and places the model tests fit to."""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DATA = ROOT / "tests" / "fixtures" / "data"
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
