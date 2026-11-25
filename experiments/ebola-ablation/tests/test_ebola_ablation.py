"""The Ebola ablation (workflow/scripts/ebola_ablation.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "ebola-ablation"))
import ebola_ablation as ea  # noqa: E402


def _reports() -> pd.DataFrame:
    """Two provinces' reports: 1,000 cases of one onset day each."""
    ref = pd.Timestamp("2026-09-01")
    return pd.DataFrame(
        {
            "province": ["CD-IT", "CD-IT", "CD-NK"],
            "ref": [ref, ref, ref],
            "rep": [ref + pd.Timedelta(days=d) for d in (2, 9, 30)],
            "count": [600, 400, 1000],
        }
    )


def test_the_delay_line_is_in_days_and_reported_by_the_vintage() -> None:
    rng = np.random.default_rng(1)
    line = ea.delay_line(_reports(), "2026-09-15", cap=500, rng=rng)
    assert set(line.columns) >= {"vintage", "location", "onset_week", "delay"}
    assert (line["vintage"] == "2026-09-15").all()
    # CD-NK's cases are reported after the vintage.
    assert set(line["location"]) == {"CD-IT"}
    assert set(line["delay"]) <= {2, 9}
    assert len(line) == 500
    assert (line["onset_week"] == "2026-09-05").all()
    share = (line["delay"] == 2).mean()
    assert 0.5 < share < 0.7


def test_the_delay_line_keeps_every_case_under_the_cap() -> None:
    rng = np.random.default_rng(1)
    line = ea.delay_line(_reports(), "2026-10-15", cap=5000, rng=rng)
    assert line.groupby("location").size().to_dict() == {
        "CD-IT": 1000,
        "CD-NK": 1000,
    }
    assert (line.loc[line["location"] == "CD-NK", "delay"] == 30).all()


def test_the_backtest_config_takes_the_draws_and_delay(tmp_path) -> None:
    import tomllib

    out = tmp_path / "backtest.toml"
    ea.backtest_config(
        ROOT / "packages/nowcastbdbv/config/backtest.toml", out, 200, 42
    )
    conf = tomllib.loads(out.read_text())
    assert conf["samples"]["backtest"] == 200
    assert conf["forecast"]["max_delay"] == 42
