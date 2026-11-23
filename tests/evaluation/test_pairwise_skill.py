"""Pairwise relative skill, checked live against R scoringutils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from model.r_live import run_r

from nowcasteval.scoring import pairwise_skill

_MODELS = ("a", "b", "c", "ref")


def _scores() -> pd.DataFrame:
    # Four models on 2 places by 6 dates by 2 horizons. Model c misses
    # half the dates, and b misses one place, so pairs share different
    # targets and each model has its own count of them.
    rng = np.random.default_rng(3)
    rows = []
    for model in _MODELS:
        for place in ("p1", "p2"):
            for day in range(6):
                for horizon in (0, 7):
                    if model == "c" and day % 2:
                        continue
                    if model == "b" and place == "p2":
                        continue
                    wis = rng.gamma(2.0, 3.0 + _MODELS.index(model))
                    rows.append(
                        {
                            "model": model,
                            "location": place,
                            "reference_date": f"2022-01-0{day + 1}",
                            "horizon": horizon,
                            "wis": wis,
                            "log_wis": np.log1p(wis) / 3,
                        }
                    )
    return pd.DataFrame(rows)


_R = """
library(data.table)
s <- as.data.table(input$scores)
out <- list()
for (m in input$metrics) {
  x <- s[, c("model", "location", "reference_date", "horizon", m),
    with = FALSE]
  x <- scoringutils::as_scores(x, metrics = m)
  for (by in list(NULL, "horizon")) {
    p <- scoringutils::get_pairwise_comparisons(
      x, by = c("model", by), metric = m, baseline = input$reference
    )
    keep <- c("model", by, paste0(m, "_relative_skill"),
      paste0(m, "_scaled_relative_skill"))
    p <- unique(p[, keep, with = FALSE])
    p[, metric := m]
    p[, by := if (is.null(by)) "" else by]
    setnames(p, paste0(m, c("_relative_skill", "_scaled_relative_skill")),
      c("theta", "scaled"))
    out[[length(out) + 1]] <- p
  }
}
rbindlist(out, fill = TRUE)
"""


@pytest.fixture(scope="module")
def r_skill() -> pd.DataFrame:
    scores = _scores()
    out = run_r(
        _R,
        {
            "scores": scores.to_dict(orient="list"),
            "metrics": ["wis", "log_wis"],
            "reference": "ref",
        },
        packages=("scoringutils", "data.table", "jsonlite"),
    )
    return pd.DataFrame(out)


@pytest.mark.parametrize("metric", ["wis", "log_wis"])
def test_pairwise_skill_matches_scoringutils(
    r_skill: pd.DataFrame, metric: str
) -> None:
    got = pairwise_skill(
        _scores(),
        reference="ref",
        metrics=(metric,),
        match_on=("location", "reference_date", "horizon"),
    ).set_index("model")
    want = r_skill.loc[
        (r_skill["metric"] == metric) & (r_skill["by"] == "")
    ].set_index("model")
    assert set(got.index) == set(_MODELS)
    for m in _MODELS:
        assert got.loc[m, f"skill_{metric}"] == pytest.approx(
            want.loc[m, "theta"], rel=1e-12
        )
        assert got.loc[m, f"relative_{metric}"] == pytest.approx(
            want.loc[m, "scaled"], rel=1e-12
        )
    assert got.loc["ref", f"relative_{metric}"] == 1


def test_pairwise_skill_by_horizon_matches_scoringutils(
    r_skill: pd.DataFrame,
) -> None:
    got = pairwise_skill(
        _scores(),
        reference="ref",
        by=("horizon",),
        match_on=("location", "reference_date"),
    ).set_index(["model", "horizon"])
    want = r_skill.loc[
        (r_skill["metric"] == "wis") & (r_skill["by"] == "horizon")
    ]
    for row in want.itertuples():
        key = (row.model, int(row.horizon))
        assert got.loc[key, "relative_wis"] == pytest.approx(
            row.scaled, rel=1e-12
        )


def test_each_model_reports_its_own_targets_and_rounds() -> None:
    got = pairwise_skill(
        _scores(),
        reference="ref",
        match_on=("location", "reference_date", "horizon"),
    ).set_index("model")
    assert got.loc["ref", "n"] == 24
    assert got.loc["b", "n"] == 12
    assert got.loc["c", "n"] == 12
    assert got.loc["c", "rounds"] == 3
    assert got.loc["a", "rounds"] == 6


def test_pairs_with_no_shared_targets_are_left_out() -> None:
    # Model d forecasts only dates no one else has, so it has no ratio
    # and no skill, and the others' skills are as without it.
    scores = _scores()
    extra = scores.loc[scores["model"] == "a"].assign(
        model="d", reference_date="2023-01-01"
    )
    got = pairwise_skill(
        pd.concat([scores, extra]),
        reference="ref",
        match_on=("location", "reference_date", "horizon"),
    ).set_index("model")
    alone = pairwise_skill(
        scores,
        reference="ref",
        match_on=("location", "reference_date", "horizon"),
    ).set_index("model")
    assert np.isnan(got.loc["d", "relative_wis"])
    for m in _MODELS:
        assert got.loc[m, "relative_wis"] == pytest.approx(
            alone.loc[m, "relative_wis"]
        )
