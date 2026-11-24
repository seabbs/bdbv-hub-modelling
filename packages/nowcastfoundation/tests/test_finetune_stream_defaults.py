"""The trainer's command-line rates are the workflow's and the SI's."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]

from nowcastfoundation.finetune import stream as fs  # noqa: E402


def test_the_cli_rates_match_the_workflow_config():
    argv = "sims out --train-seeds 1 2 --held-out-seeds 3 4"
    args = fs.build_parser().parse_args(argv.split())
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    assert args.learning_rate == 1e-5
    assert args.learning_rate == config["finetune"]["learning_rate"]
    assert args.bias_lr == config["finetune"]["bias_lr"]


def test_the_loss_horizon_enters_the_settings():
    # Unset, only the prediction length is stated.
    full = fs.horizon_settings(None)
    assert "loss_horizon_days" not in full
    assert full["prediction_length"] == (
        f"each block's horizon in whole patches of {fs.ft.OUTPUT_PATCH} steps"
    )
    week = fs.horizon_settings(7)
    assert week["loss_horizon_days"] == 7
    assert "7 days after the cut-off" in week["prediction_length"]


def test_the_loss_horizon_defaults_to_the_full_horizon():
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    assert fs.build_parser().parse_args(argv).loss_horizon_days is None
    args = fs.build_parser().parse_args([*argv, "--loss-horizon-days", "7"])
    assert args.loss_horizon_days == 7


def test_dropping_quiet_cut_offs_is_off_until_set():
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    assert fs.build_parser().parse_args(argv).quiet_min_cases is None
    args = fs.build_parser().parse_args([*argv, "--quiet-min-cases", "10"])
    assert args.quiet_min_cases == 10
    assert fs.quiet_settings(None) == {}
    rule = fs.quiet_settings(10)["drop_quiet"]
    assert "fewer than 10 cases" in rule
    assert f"{fs.ft.QUIET_WINDOW_DAYS} days" in rule


def test_the_batch_and_checkpoint_defaults_are_the_workflow_config_s():
    # One source: the trainer reads the run's defaults from the config.
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    args = fs.build_parser().parse_args(argv)
    assert args.batch_size == config["finetune"]["batch_size"]
    assert args.checkpoint_above == config["finetune"]["checkpoint_above"]


def test_the_checkpoint_threshold_is_a_machine_setting_outside_the_hash():
    # Recomputing activations in backward changes memory, not the steps,
    # so a machine with less GPU memory may checkpoint more without a new
    # hash.
    from nowcastfoundation.finetune import verify as finetune_verify

    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    args = fs.build_parser().parse_args([*argv, "--checkpoint-above", "0"])
    assert args.checkpoint_above == 0
    assert "checkpoint_above" in finetune_verify.NOT_HASHED
    assert "batch_size" not in finetune_verify.NOT_HASHED
    assert not hasattr(args, "memory_gb")


def test_the_block_limit_is_the_workflow_config_s_and_hashed():
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    args = fs.build_parser().parse_args(argv)
    assert args.block_limit == config["finetune"]["block_limit"]
    args = fs.build_parser().parse_args([*argv, "--block-limit", "900"])
    assert args.block_limit == 900
    # The limit shapes the blocks, so it is in the settings.
    assert fs.view_settings(900) == {
        "block_limit": 900,
        "block_views": fs.ft.VIEW_RULE,
    }


def test_views_are_recorded_once_per_block(tmp_path):
    import pandas as pd

    view = {
        "kind": "subtree",
        "collapsed": [],
        "places_before": 12,
        "places_after": 3,
        "series_before": 2400,
        "series_after": 600,
    }
    top = {**view, "kind": "top", "collapsed": ["strata"], "places_after": 4}
    pairs = [
        ("1-0001", {"source": "daily", "cut": 3, "view": top}),
        ("1-0001", {"source": "daily", "cut": 3, "view": view}),
        ("1-0002", {"source": "daily", "cut": 3, "view": {"kind": "whole"}}),
    ]
    line = fs.record_views(tmp_path, pairs, "1-100")
    assert line.startswith("1 systems over the block limit cut into 2 views")
    # A resumed run cuts the same blocks again: no row is added twice.
    fs.record_views(tmp_path, pairs, "1-100")
    rows = pd.read_csv(tmp_path / fs.VIEWS_FILE)
    assert len(rows) == 2
    assert list(rows["kind"]) == ["top", "subtree"]
    assert list(rows["collapsed"].fillna("")) == ["strata", ""]
    assert fs.record_views(tmp_path, pairs[2:], "1-100") == (
        "every system within the block limit"
    )


def test_the_layout_is_cutoff_and_in_the_settings():
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    assert fs.build_parser().parse_args(argv).layout == "cutoff"
    args = fs.build_parser().parse_args([*argv, "--layout", "complete"])
    assert args.layout == "complete"
    assert fs.layout_settings("cutoff") == {"layout": "cutoff"}
    assert fs.layout_settings("complete") == {"layout": "complete"}


def test_a_block_has_its_nowcast_steps():
    block = {"n_targets": 2, "n_covariates": 0, "places": ("A", "B")}
    assert fs.nowcast_steps(block | {"nowcast": 3}) == 3
    cutoff = {"n_targets": 2, "n_covariates": 6, "places": ("A", "B")}
    assert fs.nowcast_steps(cutoff | {"nowcast": 0}) == 0


def test_the_trainer_does_not_publish():
    # Lead 01:05 (20): publishing is the workflow's `publish_checkpoint`
    # (with an offloaded run's losses); the trainer's own flag was a
    # second, half-wired way that refused offloaded runs.
    import pytest

    argv = "SIM OUT --train-seeds 1 2 --held-out-seeds 3 4 --publish"
    with pytest.raises(SystemExit):
        fs.build_parser().parse_args(argv.split())
