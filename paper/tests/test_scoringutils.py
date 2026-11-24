"""Check the scores against R scoringutils on a fixed fixture, live.

`tests/scoringutils_reference.R` scores `tests/fixtures/scoring-fixture.csv`
with scoringutils' defaults each run; nothing R wrote is committed. The
test skips where Rscript or scoringutils is missing, and fails there in
CI's r-live job (`R_LIVE_REQUIRED`).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from paperfigs.schema import FORECAST_KEYS, validate_forecasts
from paperfigs.scores import score_forecasts

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
PACKAGES = ("scoringutils", "data.table", "purrr")
COLUMNS = ("score", "log_score", "bias", "coverage_50", "coverage_90")
QUANTILE_ONLY = ("coverage_95",)


def _skip(reason: str) -> None:
    if os.environ.get("R_LIVE_REQUIRED"):
        pytest.fail(reason)
    pytest.skip(reason)


def _scoringutils(out: Path) -> pd.DataFrame:
    """Score the fixture with R scoringutils into `out` and read it."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        _skip("needs Rscript")
    have = subprocess.run(
        [
            rscript,
            "-e",
            "cat(all(sapply(commandArgs(TRUE), requireNamespace, "
            "quietly = TRUE)))",
            *PACKAGES,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if have.stdout.strip() != "TRUE":
        _skip(f"needs the R packages {', '.join(PACKAGES)}")
    version = subprocess.run(
        [rscript, "-e", 'cat(packageVersion("scoringutils") >= "2.2.0")'],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.stdout.strip() != "TRUE":
        _skip("needs scoringutils 2.2 or later (CI's r-live installs 2.3.0)")
    subprocess.run(
        [rscript, str(HERE / "scoringutils_reference.R"), str(out)],
        cwd=HERE.parent,
        check=True,
    )
    return pd.read_csv(out, parse_dates=["origin_date", "target_date"])


@pytest.fixture(scope="module")
def compared(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    fc = validate_forecasts(
        pd.read_csv(
            FIXTURES / "scoring-fixture.csv", dtype={"sample_id": "Int64"}
        )
    )
    want = _scoringutils(tmp_path_factory.mktemp("su") / "expected.csv")
    # The fixture has no strata, so the validator sets them to "all".
    want["stratum"] = "all"
    # scoringutils' crps_sample is the nrg form.
    got = score_forecasts(
        fc, coverage=(0.5, 0.9, 0.95), crps_estimator="nrg", samples="crps"
    )
    keys = [*FORECAST_KEYS, "metric"]
    out = got.merge(want, on=keys, suffixes=("", "_su"), validate="1:1")
    assert len(out) == len(want) == len(got)
    return out


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("metric", ["wis", "crps"])
def test_matches_scoringutils(compared, column, metric) -> None:
    rows = compared.loc[compared["metric"] == metric]
    assert len(rows) == 12
    pd.testing.assert_series_equal(
        rows[column].reset_index(drop=True),
        rows[f"{column}_su"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("column", QUANTILE_ONLY)
def test_quantile_95_coverage_matches_scoringutils(compared, column) -> None:
    rows = compared.loc[compared["metric"] == "wis"]
    pd.testing.assert_series_equal(
        rows[column].reset_index(drop=True),
        rows[f"{column}_su"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )
