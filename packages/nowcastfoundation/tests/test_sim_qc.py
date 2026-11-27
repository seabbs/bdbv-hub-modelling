"""The deterministic sim-QC check of a round (`sim-qc`)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parents[3]
from sim_rounds import sim_dir  # noqa: E402

# The grid sets no sim QC thresholds: the round's QC records each
# simulation's summaries and excludes none.
GRID_FILE = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())[
    "simulate"
]["grid_file"]
GRID = yaml.safe_load((ROOT / GRID_FILE).read_text())


def test_the_grid_sets_no_thresholds():
    assert "qc" not in GRID


def test_the_round_s_check_is_recorded(tmp_path):
    sims = tmp_path / "r1" / "sims"
    sims.mkdir(parents=True)
    sim_dir(sims)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nowcastfoundation.finetune.qc",
            str(tmp_path / "r1"),
        ],
        cwd=ROOT,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    assert sorted(p.name for p in qc.iterdir()) == ["input.csv", "summary.txt"]
    table = pd.read_csv(qc / "input.csv")
    assert len(table) == 4
    assert (table["withdrawn_share"] >= 0).all()
    assert "summarised 4 simulations" in (qc / "summary.txt").read_text()


def test_the_qc_lists_every_structure_factor():
    from nowcastfoundation.finetune.qc import CELL

    for factor in (
        "delay_strata_noise",
        "delay_change_strata",
        "delay_change_switch",
        "delay_change_combine",
        "delay_counts",
        "bursts",
        "ascertainment_change",
        "ascertainment_change_strata",
        "ascertainment_change_switch",
        "ascertainment_counts",
        "ascertainment_noise",
    ):
        assert factor in CELL
        assert factor in GRID["factors"]
