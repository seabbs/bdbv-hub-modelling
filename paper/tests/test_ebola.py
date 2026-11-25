"""Tests of the Ebola hub loader and scoring."""

import gzip
import zipfile

import pandas as pd
import pytest

from paperdata import ebola


def test_hub_samples_keep_weekly_target_only(hub):
    s = ebola.hub_samples(hub)
    assert set(s["model"]) == {"team-a"}
    assert len(s) == 200
    assert s["origin_date"].iloc[0] == pd.Timestamp("2026-10-31")


def test_hub_samples_keep_latest_forecast_date(hub):
    path = hub / "model-output/team-a/2026-10-31-team-a.csv"
    first = pd.read_csv(path)
    later = first.assign(forecast_date="2026-10-28", value=first["value"] * 2)
    pd.concat([first, later]).to_csv(path, index=False)
    s = ebola.hub_samples(hub)
    assert len(s) == 200
    assert s["value"].max() == 200


def _release(folder, ref, models, vintage="2026-10-25"):
    """Write a round release's `hub.zip` with each model's samples."""
    out = folder / f"round-{ref}"
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "hub.zip", "w") as z:
        for model, value in models.items():
            rows = pd.DataFrame(
                {
                    # A model's file also holds the next round's rows.
                    "reference_date": [ref] * 100 + ["2026-11-07"] * 100,
                    "forecast_date": vintage,
                    "target": ebola.TARGET,
                    "target_end_date": "2026-11-07",
                    "location": "CD",
                    "output_type": "sample",
                    "output_type_id": list(range(1, 101)) * 2,
                    "value": value,
                }
            )
            z.writestr(
                f"hub/{vintage}/{model}-samples.csv.gz",
                gzip.compress(rows.to_csv(index=False).encode()),
            )
            z.writestr(f"hub/{vintage}/{model}-quantiles.csv.gz", b"")
    return out


def _ledger(path, rows):
    pd.DataFrame(
        rows, columns=["reference_date", "hub_id", "status", "model"]
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def releases(tmp_path):
    """Round releases of two rounds and the ledger of what each sent."""
    folder = tmp_path / "releases"
    _release(folder, "2026-10-31", {"renewal": 5, "randomwalknowcast": 7})
    # A round whose release has no `hub.zip` yet gives nothing.
    (folder / "round-2026-11-07").mkdir()
    ledger = _ledger(
        tmp_path / "submissions.csv",
        [
            [
                "2026-10-31",
                "seabbs-rwnowcast",
                "submitted",
                "randomwalknowcast",
            ],
            ["2026-10-31", "seabbs-renewal", "failed", "renewal"],
        ],
    )
    return folder, ledger


def test_round_releases_give_the_unsubmitted_models_of_each_round(releases):
    s = ebola.release_samples(*releases)
    # The submitted walk is the hub's copy; only its round's rows are kept.
    assert set(s["model"]) == {"renewal"}
    assert set(s["origin_date"]) == {pd.Timestamp("2026-10-31")}
    assert len(s) == 100
    assert (s["value"] == 5).all()


def test_the_round_releases_file_is_what_load_reads(releases, hub, tmp_path):
    path = tmp_path / "round-releases.csv.gz"
    assert ebola.write_release_samples(*releases, path) == 100
    pd.testing.assert_frame_equal(
        ebola.round_samples(path), ebola.release_samples(*releases)
    )
    table = ebola.load(hub, releases=path)
    assert set(table["model"]) == {"team-a", "renewal"}
    assert set(table.loc[table["model"] == "renewal", "value"]) == {5}


def test_round_close_is_tuesday_before():
    close = ebola.round_close(pd.Timestamp("2026-11-14"))
    assert close == pd.Timestamp("2026-11-10 22:59", tz="UTC")


def test_hub_samples_keep_trajectory_ids(hub):
    s = ebola.hub_samples(hub)
    assert sorted(s["sample_id"].unique()) == list(range(1, 101))


def test_load_gives_sample_schema(hub):
    table = ebola.load(hub)
    assert list(table.columns) == list(ebola.COLUMNS)
    assert "sample_id" in table
    assert table["quantile_level"].isna().all()
    assert set(table["study"]) == {"ebola"}
    assert set(table["horizon"]) == {-1, 1}
    assert len(table) == 200
    assert table["truth"].isna().all()


def test_hub_baseline_is_the_hub_model():
    assert ebola.HUB_BASELINE == "BDBVHub-baseline"


def test_release_observations_date_each_release_by_its_cutoff(tmp_path):
    pd.DataFrame(
        {
            "kind": ["real", "real", "simulated"],
            "vintage": ["2026-11-03", "2026-11-10", "2026-11-10"],
            "location": "CD",
            "location_name": "DRC",
            "target_end_date": "2026-10-24",
            "observed": 1,
            "imputed": 0,
            "total": [10, 12, 99],
            "cutoff": ["2026-11-01", "2026-11-08", "2026-11-08"],
        }
    ).to_csv(tmp_path / "target.csv", index=False)
    obs = ebola.release_observations(tmp_path)
    assert obs["as_of_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-11-01",
        "2026-11-08",
    ]
    assert obs["value"].tolist() == [10.0, 12.0]


def test_a_release_feeds_the_first_round_still_open_on_its_day() -> None:
    from paperdata.ebola import round_of

    # Round 2026-11-21 closes on Tuesday 2026-11-17.
    assert round_of(pd.Timestamp("2026-11-17")) == pd.Timestamp("2026-11-21")
    assert round_of(pd.Timestamp("2026-11-11")) == pd.Timestamp("2026-11-21")
    assert round_of(pd.Timestamp("2026-11-10")) == pd.Timestamp("2026-11-14")
    assert round_of(pd.Timestamp("2026-11-18")) == pd.Timestamp("2026-11-28")
