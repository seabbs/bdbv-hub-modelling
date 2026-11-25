"""Fine-tuning never computes or shows what the paper holds out.

The German benchmark's held-out half (nowcast dates from its split date)
and the grid's simulated test round are for the paper's workflow only:
the `finetune` target plans no job on them, and the dashboard reads
none of their outputs.
"""

import gzip
import re
import sys
from pathlib import Path

import pytest
import yaml
from flowlib.dashboard import context, nowcasts, runs, sims

ROOT = Path(__file__).parents[2]
CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
GERMANY = ROOT / "packages" / "nowcastgermany" / "config" / "germany.toml"
SPLIT = "2022-02-10"
TEST_ROUND = CONFIG["grid_rounds"]["test_round"]

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))


def test_the_split_date_is_the_german_config_s():
    from nowcastgermany.evaluation import load_germany_config

    assert load_germany_config(GERMANY).split_date == SPLIT


def test_finetune_plans_no_held_out_job(tmp_path):
    pytest.importorskip("snakemake")
    from test_workflow_dryrun import dry_run

    out = dry_run(tmp_path, "finetune", quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    text = out.stdout
    assert "germany_heldout" not in text
    # The German held-out half's flags; the trainer's own validation
    # round is --held-out-seeds.
    assert not re.search(r"--held-?out(?![-\w])", text)
    assert "/held-out/" not in text
    assert f"/r{TEST_ROUND}/" not in text
    assert f"sims-r{TEST_ROUND}" not in text


def test_the_checkpoints_are_scored_on_the_development_half_only():
    from nowcastgermany import evaluation as de

    dates = de.nowcast_dates(de.load_germany_config(GERMANY), held_out=False)
    assert dates
    assert max(dates) < SPLIT


def test_the_dashboard_knows_the_german_split_and_the_test_round(tmp_path):
    ctx = context.build(
        CONFIG, {"ROOT": str(tmp_path), "SCORE_CACHE": str(tmp_path)}
    )
    assert ctx.held_out_from == {"germany": SPLIT}
    import dashboard

    assert dashboard.hidden_rounds(CONFIG) == {TEST_ROUND}


def _file(path, origin_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("model,location,target_end_date,quantile,value,truth\n")
        for i, q in enumerate(nowcasts.SEVEN):
            f.write(
                f"randomwalknowcast,DE,2022-02-01,{q},{origin_value + i},5\n"
            )


def test_the_dashboard_reads_no_nowcast_date_of_the_held_out_half(tmp_path):
    _file(tmp_path / "reference" / "2022-02-09.csv.gz", 1)
    _file(tmp_path / "reference" / f"{SPLIT}.csv.gz", 100)
    _file(tmp_path / "reference" / "2022-03-01.csv.gz", 100)
    refs = runs.load_references(
        [
            {
                "key": "rw",
                "label": "rw",
                "colour": "#000",
                "model": "^randomwalknowcast$",
            }
        ],
        [],
    )
    built = nowcasts.build(tmp_path, [], refs, until=SPLIT)
    assert built["origins"] == ["2022-02-09"]
    assert set(built["series"]["DE"]["rw"]["7day"]["daily"]) == {"2022-02-09"}


def test_the_simulations_dashboard_never_lists_the_test_round(tmp_path):
    for r in (33, TEST_ROUND):
        folder = tmp_path / "rounds" / f"r{r}" / "sims"
        folder.mkdir(parents=True)
        (folder / "settings-1.csv").write_text("outbreak\n")
    assert list(sims.sources(tmp_path, hidden={TEST_ROUND})) == ["r33"]
    index = sims.Index(tmp_path, tmp_path / "cache", {}, hidden={TEST_ROUND})
    assert index.data(f"r{TEST_ROUND}") is None
