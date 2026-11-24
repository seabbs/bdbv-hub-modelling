"""German relative scores are pairwise relative skill."""

import numpy as np
import pandas as pd
import pytest

from nowcasteval.scoring import pairwise_skill, wis_quantile
from nowcastgermany import evaluation as germany
from nowcastgermany.evaluation import relative_scores

MATCH = ("target", "origin", "truth_vintage", "location", "target_end_date")


def _scores() -> pd.DataFrame:
    # The team forecasts the first two dates only, so the models share
    # different targets.
    rng = np.random.default_rng(11)
    rows = []
    for model in ("randomwalknowcast_daily", "fm", "team"):
        for day in range(4):
            if model == "team" and day > 1:
                continue
            for place in ("DE", "DE-BY"):
                rows.append(
                    {
                        "model": model,
                        "target": "7day",
                        "origin": f"2021-12-0{day + 1}",
                        "truth_vintage": "2022-01-31",
                        "location": place,
                        "target_end_date": f"2021-12-0{day + 1}",
                        "pool": "national" if place == "DE" else "states",
                        "wis": rng.gamma(2.0, 5.0),
                        "ae_median": rng.gamma(2.0, 5.0),
                        "coverage_50": 0.5,
                        "coverage_95": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_scores_are_pairwise_skill_on_each_models_own_forecasts():
    scores = _scores()
    base = "randomwalknowcast_daily"
    got = relative_scores(scores, ("pool",), base).set_index(["pool", "model"])
    want = pairwise_skill(
        scores,
        base,
        by=("pool",),
        metrics=("wis", "ae_median"),
        match_on=MATCH,
        rounds="origin",
    ).set_index(["pool", "model"])
    for key in want.index:
        assert got.loc[key, "rel_wis"] == pytest.approx(
            want.loc[key, "relative_wis"], rel=1e-12
        )
        assert got.loc[key, "rel_ae_median"] == pytest.approx(
            want.loc[key, "relative_ae_median"], rel=1e-12
        )
    team = scores.loc[
        (scores["model"] == "team") & (scores["pool"] == "states")
    ]
    assert got.loc[("states", "team"), "n"] == len(team)
    assert got.loc[("states", "team"), "rounds"] == 2
    assert got.loc[("states", "team"), "wis"] == pytest.approx(
        team["wis"].mean()
    )


def test_german_quantile_scores_stay_wis() -> None:
    # The German benchmark scores quantile forecasts, so it keeps WIS.
    assert "wis" in germany.SCORE_COLUMNS
    assert "crps" not in germany.SCORE_COLUMNS
    quantiles = pd.DataFrame(
        {
            "id": [0] * 3,
            "quantile": [0.25, 0.5, 0.75],
            "value": [4.0, 6.0, 9.0],
        }
    )
    assert "wis" in wis_quantile(
        np.array([5.0]),
        quantiles["value"].to_numpy()[None, :],
        quantiles["quantile"].to_numpy(),
    )
