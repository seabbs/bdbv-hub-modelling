"""Release backtest relative CRPS is pairwise relative skill."""

import numpy as np
import pandas as pd
import pytest

from nowcasteval.nowcast import relative_crps
from nowcasteval.scoring import pairwise_skill

MATCH = ("target", "origin", "truth_vintage", "location", "target_end_date")


def _scores() -> pd.DataFrame:
    # A stored model forecast only the first two origins, so the models
    # share different cells.
    rng = np.random.default_rng(2)
    rows = []
    for model in ("randomwalk", "a", "stored"):
        for k, origin in enumerate(("2026-10-17", "2026-10-24", "2026-10-31")):
            if model == "stored" and k == 2:
                continue
            for place in ("CD", "CD-IT"):
                crps = rng.gamma(2.0, 3.0)
                rows.append(
                    {
                        "model": model,
                        "target": "reported",
                        "origin": origin,
                        "truth_vintage": "2026-11-07",
                        "location": place,
                        "target_end_date": origin,
                        "crps": crps,
                        "log_crps": np.log1p(crps) / 3,
                        "coverage_50": 0.5,
                        "coverage_90": 1.0,
                        "bias": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_crps_is_pairwise_and_means_are_each_model_s_own():
    scores = _scores()
    got = relative_crps(scores).set_index("model")
    want = pairwise_skill(
        scores,
        "randomwalk",
        by=("target",),
        metrics=("crps", "log_crps"),
        match_on=MATCH,
        rounds="origin",
    ).set_index("model")
    for m in ("randomwalk", "a", "stored"):
        assert got.loc[m, "rel_crps"] == pytest.approx(
            want.loc[m, "relative_crps"], rel=1e-12
        )
        assert got.loc[m, "rel_log_crps"] == pytest.approx(
            want.loc[m, "relative_log_crps"], rel=1e-12
        )
    own = scores.loc[scores["model"] == "stored"]
    assert got.loc["stored", "n"] == len(own)
    assert got.loc["stored", "crps"] == pytest.approx(own["crps"].mean())
    base = scores.loc[scores["model"] == "randomwalk", "crps"].mean()
    assert got.loc["stored", "base_crps"] == pytest.approx(base)
