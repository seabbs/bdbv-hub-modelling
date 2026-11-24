"""The hub comparison's relative scores are pairwise relative skill."""

import numpy as np
import pandas as pd
import pytest

from bdbvhub.compare import summarise_comparison
from nowcasteval.scoring import pairwise_skill

BY = ("reference_date", "target")
MATCH = (*BY, "truth_vintage", "location", "target_end_date")


def _scores() -> pd.DataFrame:
    # A hub team that forecast only some weeks, beside the baseline and
    # one of our models.
    rng = np.random.default_rng(5)
    rows = []
    for model, source in (("flat", "ours"), ("ours", "ours"), ("team", "hub")):
        for week in range(4):
            if model == "team" and week % 2:
                continue
            for place in ("CD-IT", "CD-NK"):
                crps = rng.gamma(2.0, 4.0)
                rows.append(
                    {
                        "reference_date": "2026-10-31",
                        "target": "complete",
                        "model": model,
                        "source": source,
                        "truth_vintage": "2026-11-21",
                        "location": place,
                        "target_end_date": f"2026-10-{10 + 7 * week}",
                        "crps": crps,
                        "log_crps": np.log1p(crps) / 4,
                        "coverage_50": 0.5,
                        "coverage_90": 0.9,
                        "bias": 0.0,
                        "wis": crps,
                        "log_wis": np.log1p(crps) / 4,
                        "ae_median": crps,
                        "overprediction": 0.0,
                        "dispersion": crps,
                        "underprediction": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_crps_is_pairwise_skill_scaled_to_the_baseline():
    scores = _scores()
    got = summarise_comparison(scores, "flat").set_index("model")
    want = pairwise_skill(
        scores, "flat", by=BY, metrics=("crps", "log_crps"), match_on=MATCH
    ).set_index("model")
    for m in ("flat", "ours", "team"):
        assert got.loc[m, "rel_crps"] == pytest.approx(
            want.loc[m, "relative_crps"], rel=1e-12
        )
        assert got.loc[m, "rel_log_crps"] == pytest.approx(
            want.loc[m, "relative_log_crps"], rel=1e-12
        )
    assert got.loc["team", "n"] == 4
