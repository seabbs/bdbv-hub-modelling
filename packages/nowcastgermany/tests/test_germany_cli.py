"""`nowcastgermany <command>`: the German benchmark's command line."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nowcastgermany.cli import main
from nowcastgermany.evaluation import load_germany_config

ROOT = Path(__file__).resolve().parents[3]


def _germany_data(tmp_path: Path) -> tuple[Path, Path]:
    """The fixture as the hub's triangle, and a config for it."""
    long = pd.read_csv(
        ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
    )
    keys = ["location", "age_group", "reference_date"]
    long = long.sort_values([*keys, "report_date"])
    long["change"] = (
        long.groupby(keys)["confirm"].diff().fillna(long["confirm"])
    )
    long["delay"] = (
        pd.to_datetime(long["report_date"])
        - pd.to_datetime(long["reference_date"])
    ).dt.days
    wide = long.pivot_table(
        index=keys, columns="delay", values="change"
    ).rename(columns=lambda d: f"value_{d}d")
    data = tmp_path / "germany"
    data.mkdir()
    wide.reset_index().rename(columns={"reference_date": "date"}).to_csv(
        data / "COVID-19_hospitalizations.csv", index=False
    )
    text = (ROOT / "config" / "germany.toml").read_text()
    text = text.replace('"2021-11-22"', '"2021-10-04"')
    text = text.replace('"2022-04-29"', '"2021-10-05"')
    text = text.replace("delay = 40", "delay = 10")
    text = text.replace('"2022-02-10"', '"2021-10-06"')
    config = tmp_path / "germany.toml"
    config.write_text(text)
    return data, config


@pytest.mark.slow
def test_backtest_scores_a_block_of_dates(tmp_path: Path) -> None:
    data, config = _germany_data(tmp_path)
    out = tmp_path / "results"
    assert main(["backtest", "--config", str(config), "--data",
                 str(data), "--models", "randomwalk", "--chunk", "1/2",
                 "--out", str(out)]) == 0  # fmt: skip
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["origin"]) == {"2021-10-05"}
    assert set(scores["model"]) == {"randomwalk"}
    forecasts = pd.read_csv(out / "forecasts.csv.gz")
    assert set(forecasts["origin"]) == {"2021-10-05"}


def test_teams_scores_the_hub_teams(tmp_path: Path) -> None:
    data, config = _germany_data(tmp_path)
    rows = [
        {
            "location": "DE",
            "age_group": "00+",
            "forecast_date": "2021-10-04",
            "target_end_date": "2021-10-04",
            "target": "0 day ahead inc hosp",
            "type": "quantile",
            "quantile": q,
            "value": 100.0 * (i + 1),
            "pathogen": "COVID-19",
        }
        for i, q in enumerate((0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975))
    ]
    names = load_germany_config(config).submissions
    models = ("KIT-simple_nowcast", "KIT-frozen_baseline")
    for name, model in zip(names, models, strict=True):
        pd.DataFrame(rows).assign(model=model).to_csv(
            data / Path(name).name, index=False
        )
    out = tmp_path / "results"
    assert main(["teams", "--config", str(config), "--data",
                 str(data), "--out", str(out)]) == 0  # fmt: skip
    teams = pd.read_csv(out / "teams.csv.gz")
    assert set(teams["model"]) == {"KIT", "KIT-frozen_baseline"}
    assert len(teams) == 2
    assert (out / "teams-forecasts.csv.gz").exists()
    assert main(["paper", "--config", str(config), "--results",
                 str(out)]) == 0  # fmt: skip
    paper = pd.read_csv(out / "paper-forecasts.csv.gz")
    assert set(paper["model"]) == {"KIT", "KIT-frozen_baseline"}
    meta = json.loads((out / "paper-forecasts.json").read_text())
    assert meta["split_date"] == "2021-10-06"
    assert meta["baseline"] == "KIT-frozen_baseline"


@pytest.mark.slow
def test_held_out_half_goes_to_its_own_directory(
    tmp_path: Path,
) -> None:
    data, config = _germany_data(tmp_path)
    config.write_text(
        config.read_text().replace('"2021-10-06"', '"2021-10-05"')
    )
    out = tmp_path / "results"
    assert main(["backtest", "--config", str(config), "--data",
                 str(data), "--models", "randomwalk", "--held-out",
                 "--out", str(out)]) == 0  # fmt: skip
    scores = pd.read_csv(out / "held-out" / "scores.csv.gz")
    assert set(scores["origin"]) == {"2021-10-05"}
    assert not (out / "scores.csv.gz").exists()


def test_asof_writes_the_data_each_date_saw(
    tmp_path: Path,
) -> None:
    data, config = _germany_data(tmp_path)
    out = tmp_path / "results"
    assert main(["asof", "--config", str(config), "--data",
                 str(data), "--out", str(out)]) == 0  # fmt: skip
    asof = pd.read_csv(out / "asof.csv.gz")
    assert set(asof["origin"]) == {"2021-10-04", "2021-10-05"}
