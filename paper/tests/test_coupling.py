"""Chronos-2's Ebola trajectories scored under each coupling."""

import pandas as pd
import pytest

from paperfigs.dummy import dummy_study
from paperfigs.truth import rolling_truth


@pytest.fixture(scope="module")
def ebola():
    fc, obs = dummy_study("ebola", seed=1, n_samples=40)
    delays = pd.Series(14, index=obs["location"].unique())
    return fc, obs, rolling_truth(obs, 14, first_after=True), delays


def test_the_rows_are_each_cell_s_quantiles_dated_by_the_round_close(
    ebola,
) -> None:
    from paperpipe.coupling import LEVELS, coupling_rows

    fc, obs, truth, delays = ebola
    rows = coupling_rows(fc, truth, obs, delays)
    assert set(rows["model"]) == {"foundation"}
    assert sorted(rows["quantile"].unique()) == pytest.approx(sorted(LEVELS))
    origins = pd.to_datetime(fc["origin_date"]).dt.strftime("%Y-%m-%d")
    closes = (
        pd.to_datetime(fc["origin_date"]) - pd.Timedelta(days=4)
    ).dt.strftime("%Y-%m-%d")
    assert set(rows["origin"]) <= set(closes)
    assert not set(rows["origin"]) & set(origins)
    # Each truth is the first release at least D days after its week.
    due = pd.to_datetime(rows["target_end_date"]) + pd.Timedelta(days=14)
    assert (pd.to_datetime(rows["truth_vintage"]) >= due).all()
    assert rows["truth"].notna().all()


def test_each_coupling_is_scored_against_the_copula(ebola) -> None:
    from paperpipe.coupling import coupling_rows, ebola_coupling

    fc, obs, truth, delays = ebola
    rows = coupling_rows(fc, truth, obs, delays)
    scores, summary = ebola_coupling(rows, n_samples=50)
    assert set(scores["coupling"]) == {"gaussian", "independent", "comonotone"}
    energy = summary.loc[summary["score"] == "log_energy"].set_index(
        "coupling"
    )["ratio"]
    assert energy["gaussian"] == pytest.approx(1.0)
    assert energy[["independent", "comonotone"]].notna().all()


def test_the_coupling_step_writes_its_scores_and_summary(
    ebola, tmp_path
) -> None:
    from paperpipe.__main__ import main

    fc, obs, truth, delays = ebola
    paths = {}
    for name, frame in (("forecasts", fc), ("observations", obs)):
        paths[name] = tmp_path / f"{name}.csv.gz"
        frame.to_csv(paths[name], index=False)
    paths["truth"] = tmp_path / "truth.csv.gz"
    truth.to_csv(paths["truth"], index=False)
    paths["locations"] = tmp_path / "locations.csv"
    pd.DataFrame(
        {"location": delays.index, "delay_days": delays.to_numpy()}
    ).to_csv(paths["locations"], index=False)
    main(
        [
            "coupling",
            *(f"--{k}={v}" for k, v in paths.items()),
            "--samples",
            "30",
            "--out-scores",
            str(tmp_path / "scores.csv.gz"),
            "--out",
            str(tmp_path / "coupling.csv"),
        ]
    )
    got = pd.read_csv(tmp_path / "coupling.csv")
    assert set(got["coupling"]) == {"gaussian", "independent", "comonotone"}
