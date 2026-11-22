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
            "report_date": (ref + pd.to_timedelta(np.tile([0, 3], 120), "D"))
            .strftime("%Y-%m-%d"),
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
