"""The coupling evidence: trajectories by each coupling, scored jointly."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from nowcasteval import coupling

LEVELS = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])


def _rows(n_origins=14, places=3, steps=6, rho=0.95, seed=0, lag=7):
    """Quantile rows of N(100, 10) cells; truths move together over steps."""
    rng = np.random.default_rng(seed)
    out = []
    first = pd.Timestamp("2021-11-01")
    for k in range(n_origins):
        origin = first + pd.Timedelta(days=7 * k)
        for p in range(places):
            z = rng.standard_normal()
            for h in range(-steps + 1, 1):
                z = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal()
                end = origin + pd.Timedelta(days=h)
                for lv in LEVELS:
                    out.append(
                        {
                            "model": "m",
                            "origin": str(origin.date()),
                            "location": f"P{p}",
                            "target_end_date": str(end.date()),
                            "horizon": h,
                            "quantile": lv,
                            "value": 100 + 10 * ndtri(lv),
                            "truth": 100 + 10 * z,
                            "truth_vintage": str(
                                (end + pd.Timedelta(days=lag)).date()
                            ),
                        }
                    )
    return pd.DataFrame(out)


@pytest.fixture(scope="module")
def scores():
    return coupling.coupling_scores(_rows(), n_samples=300, seed=2)


def test_every_origin_is_scored_by_every_coupling(scores):
    assert len(scores) == 14 * len(coupling.COUPLINGS)
    assert (scores["n_cells"] == 18).all()
    first = scores.loc[scores["origin"] == scores["origin"].min()]
    # The first origin has nothing earlier to fit from: independence.
    gauss = first.loc[first["coupling"] == "gaussian"].iloc[0]
    assert (gauss["rho"], gauss["s"]) == (0.0, 0.0)


def test_the_fitted_coupling_finds_steps_moving_together(scores):
    late = scores.loc[
        (scores["coupling"] == "gaussian")
        & (scores["origin"] > scores["origin"].min())
    ]
    assert late["rho"].iloc[-1] > 0.6
    table = coupling.summary(scores, n_boot=200)
    sums = table.loc[table["score"] == "sum_crps"].set_index("coupling")
    # Sums over steps that move together are wider than independent
    # draws give: both coupled draws score better there.
    assert sums.loc["independent", "ratio"] == pytest.approx(1.0)
    assert sums.loc["gaussian", "ratio"] < 0.95
    assert sums.loc["comonotone", "ratio"] < 0.9
    assert (sums["lower"] <= sums["ratio"]).all()
    assert (sums["ratio"] <= sums["upper"]).all()
    assert sums.loc["independent", "cover90"] < sums.loc["gaussian", "cover90"]


def test_the_fit_reads_only_truth_reported_by_the_origin():
    # Truth reported 100 days on: nothing has matured at any origin in
    # these 14 weeks, so every fit is independence.
    rows = _rows(lag=100)
    got = coupling.coupling_scores(rows, ["gaussian"], n_samples=50)
    assert (got["rho"] == 0.0).all()
    assert (got["s"] == 0.0).all()
    got = coupling.coupling_scores(rows, ["schaake"], n_samples=50)
    assert (got["n_windows"] == 0).all()


def test_schaake_orders_draws_as_matured_windows(scores):
    shuffled = scores.loc[scores["coupling"] == "schaake"]
    # The first origin has no matured window: independence. Later ones
    # take windows of the truth reported by then, up to one per draw.
    assert shuffled["n_windows"].iloc[0] == 0
    assert shuffled["n_windows"].iloc[-1] > 0
    assert (shuffled["n_windows"] <= 300).all()
    assert scores.loc[scores["coupling"] != "schaake", "n_windows"].isna().all()
    table = coupling.summary(scores, n_boot=200)
    sums = table.loc[table["score"] == "sum_crps"].set_index("coupling")
    # Past windows move together over steps, as the truth does.
    assert sums.loc["schaake", "ratio"] < 0.95


def test_the_command_writes_scores_and_summary(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    rows = _rows(n_origins=4)
    for o, g in rows.groupby("origin"):
        g.to_csv(unit / f"{o}.csv.gz", index=False, compression="gzip")
    coupling.main([str(unit), str(tmp_path / "s.csv"), str(tmp_path / "t.csv")])
    assert len(pd.read_csv(tmp_path / "s.csv")) == 4 * len(coupling.COUPLINGS)
    assert set(pd.read_csv(tmp_path / "t.csv")["coupling"]) == set(
        coupling.COUPLINGS
    )
