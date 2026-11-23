"""The deterministic sim-QC check of a round (scripts/sim_qc.py)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts"))
from test_finetune_chronos import _sim_dir  # noqa: E402

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
    _sim_dir(sims)
    subprocess.run(
        [sys.executable, "scripts/sim_qc.py", str(tmp_path / "r1")],
        cwd=ROOT,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    assert sorted(p.name for p in qc.iterdir()) == ["input.csv", "summary.txt"]
    assert len(pd.read_csv(qc / "input.csv")) == 4
    assert "summarised 4 simulations" in (qc / "summary.txt").read_text()
