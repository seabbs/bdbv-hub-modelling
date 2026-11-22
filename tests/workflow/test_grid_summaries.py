"""Per-outbreak grid summaries (workflow/scripts/grid_summaries.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
import grid_summaries as gs


def _batch(path: Path) -> None:
    days = pd.date_range("2026-01-05", periods=120)
    t = np.arange(120)
    onsets = 50 * np.exp(-(((t - 60) / 15) ** 2))
    pd.DataFrame(
        {
            "day": t,
            "infections": onsets * 2,
            "onsets": onsets,
            "r": 1.0,
            "reff": 1.0,
            "date": days.strftime("%Y-%m-%d"),
            "location": "all",
            "outbreak": "1-0001",
        }
    ).to_csv(path / "latent-1.csv.gz", index=False)
    ref = np.repeat(days, 2)
    pd.DataFrame(
        {
            "outbreak": "1-0001-1",
            "timestep": "day",
            "location": "P01",
            "reference_date": ref.strftime("%Y-%m-%d"),
            "report_date": (
                ref + pd.to_timedelta(np.tile([0, 3], 120), "D")
            ).strftime("%Y-%m-%d"),
            "count": np.tile([3, 1], 120),
        }
    ).to_csv(path / "counts-1.csv.gz", index=False)
    pd.DataFrame(
        [{"outbreak": "1-0001", "window": "early", "gt_mean": 7.0, "weeks": 17}]
    ).to_csv(path / "settings-1.csv", index=False)


def test_one_wave_gives_one_peak_and_its_delays(tmp_path):
    _batch(tmp_path)
    out = gs.outbreaks(tmp_path)
    row = out.iloc[0]
    assert len(out) == 1
    assert row["pattern"] == "wave"
    assert row["n_peaks"] == 1
    assert row["reported"] == 480
    # A quarter of cases are reported 3 days late.
    assert np.isclose(row["mean_delay"], 0.75)
    assert row["reported_7d"] == 1.0
    table = gs.compare({"a": out, "b": out})
    assert set(table["source"]) == {"a", "b"}


def test_series_statistics_read_the_delays_weekdays_and_updates():
    days = pd.date_range("2026-01-05", periods=70)
    ref = np.repeat(days, 3)
    c = pd.DataFrame(
        {
            "reference_date": ref,
            "report_date": ref + pd.to_timedelta(np.tile([0, 3, 9], 70), "D"),
            # Mondays report twice as much on the day; one downward update.
            "count": np.tile([6, 3, 1], 70)
            * np.where(np.repeat(days.day_of_week, 3) == 0, 2, 1),
        }
    )
    c.loc[2, "count"] = -1
    s = gs.series_stats(c)
    assert s["delay_q50"] == 0
    assert s["delay_q90"] == 3
    assert s["delay_q99"] == 9
    assert 0.55 < s["day0_share"] < 0.65
    assert np.isclose(s["negative_share"], 1 / (3 * 70))
    # Monday's mean ratio over the others' is about 2.
    assert 1.7 < s["weekday_amplitude"] < 2.3
    assert s["weekly_growth_max"] < 0.2


def test_outbreak_rows_carry_the_series_statistics(tmp_path):
    _batch(tmp_path)
    row = gs.outbreaks(tmp_path).iloc[0]
    assert row["day0_share"] == 0.75
    assert row["delay_q99"] == 3
    assert row["peak_share"] > 0.4
