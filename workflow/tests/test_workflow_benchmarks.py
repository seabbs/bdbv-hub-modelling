"""Backtest and scoring time, gathered from the workflow's records."""

from __future__ import annotations

import pandas as pd
from flowlib import benchmarks


def _bench(path, seconds, rss):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\t"
        f"mean_load\tcpu_time\n{seconds}\t0:00:01\t{rss}\t1\t1\t1\t0\t0\t1\t1\n"
    )


def test_jobs_and_model_dates_are_one_table(tmp_path):
    _bench(
        tmp_path / "benchmarks" / "score_germany" / "reference.tsv", 12.5, 900
    )
    unit = tmp_path / "germany-days" / "reference"
    unit.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["randomwalknowcast", "fm_chronos_daily_ragged"],
            "origin": ["2021-12-01", ""],
            "stage": ["forecast", "prefetch"],
            "seconds": [1.5, 30.0],
            "dates": [2, 2],
        }
    ).to_csv(unit / "timings.csv", index=False)
    got = benchmarks.summarise([tmp_path])
    jobs = got.query("kind == 'job'")
    assert jobs[["rule", "key", "seconds", "max_rss"]].to_numpy().tolist() == [
        ["score_germany", "reference", 12.5, 900.0]
    ]
    cells = got.query("kind == 'model_date'").sort_values("seconds")
    assert cells[
        ["key", "model", "origin", "stage", "seconds"]
    ].to_numpy().tolist() == [
        ["reference", "randomwalknowcast", "2021-12-01", "forecast", 1.5],
        ["reference", "fm_chronos_daily_ragged", "", "prefetch", 30.0],
    ]


def test_nothing_recorded_is_an_empty_table(tmp_path):
    got = benchmarks.summarise([tmp_path])
    assert got.empty
    assert list(got.columns) == list(benchmarks.COLUMNS)
