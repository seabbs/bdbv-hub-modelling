"""The workflow config is checked against its schema when it loads."""

import copy
from pathlib import Path

import pytest
import yaml

utils = pytest.importorskip("snakemake.utils")

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "workflow" / "schemas" / "config.schema.yaml"
CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
PAPER = yaml.safe_load(
    (ROOT / "paper" / "workflow" / "config.yaml").read_text()
)


def check(config):
    utils.validate(copy.deepcopy(config), str(SCHEMA))


def test_the_committed_config_is_valid_with_the_paper_s_beside_it():
    check({**CONFIG, **PAPER})


@pytest.mark.parametrize(
    ("where", "key", "value"),
    [
        ("finetune", "prefixess", [100]),
        ("finetune", "eval", "elsewhere"),
        ("finetune", "prefixes", "100"),
        ("simulate", "batch", 2.5),
        ("grid_rounds", "rounds", None),
        ("score", "mem_mb", "6 GB"),
    ],
)
def test_a_wrong_or_unknown_setting_is_an_error(where, key, value):
    config = copy.deepcopy(CONFIG)
    config[where][key] = value
    with pytest.raises(Exception, match=r"(?i)validat"):
        check(config)


def test_a_run_takes_finetune_s_settings_and_no_others():
    config = copy.deepcopy(CONFIG)
    config["runs"] = {"runb": {"cuts": 32, "prefixes": [100, 200]}}
    check(config)
    config["runs"]["runb"]["cutz"] = 32
    with pytest.raises(Exception, match=r"(?i)validat"):
        check(config)


def test_rounds_given_on_the_command_line_are_valid():
    # Snakemake reads `--config rounds=[4,5]` as text.
    check({**CONFIG, "rounds": ["4", "5"]})
    with pytest.raises(Exception, match=r"(?i)validat"):
        check({**CONFIG, "rounds": ["four"]})


def test_a_missing_section_is_an_error():
    config = copy.deepcopy(CONFIG)
    del config["grid_rounds"]
    with pytest.raises(Exception, match=r"(?i)validat"):
        check(config)


def test_the_workflow_validates_its_config_at_load():
    text = (ROOT / "workflow" / "Snakefile").read_text()
    assert 'validate(config, "schemas/config.schema.yaml")' in text
