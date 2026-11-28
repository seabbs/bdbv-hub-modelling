"""The BDBV pairing sensitivity scores both pairings and keeps rho_p."""

from __future__ import annotations

from pathlib import Path

import pytest

from nowcastbdbv.data.tidy import load_target
from nowcasteval.backtest import real_vintages
from nowcasteval.pairing import RHO_COLUMNS, SCORE_COLUMNS

DATA = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "data"
CONFIG = (
    Path(__file__).resolve().parents[2]
    / "nowcastbdbv"
    / "config"
    / "backtest.toml"
)


@pytest.fixture(scope="module")
def sensitivity():
    from bdbvhub.pairing import bdbv_sensitivity

    real = load_target(str(DATA)).query("kind == 'real'")
    origins = real_vintages(real)[-3:-1]
    return bdbv_sensitivity(str(DATA), str(CONFIG), origins), origins


def test_the_sensitivity_gives_scores_and_each_task_s_rho_p(
    sensitivity,
) -> None:
    (scores, rho), origins = sensitivity
    assert list(scores.columns) == list(SCORE_COLUMNS)
    assert set(scores["variant"]) == {"off", "rolling"}
    assert list(rho.columns) == list(RHO_COLUMNS)
    assert set(rho["origin"]) <= set(origins)
    # The walks pair provinces, so rho_p is the provinces' level.
    assert set(rho["level"]) == {"provinces"}
    assert set(rho["model"]) <= {"randomwalk", "randomwalknowcast"}
