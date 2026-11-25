"""The trainer's cores are kept for it: jobs beside it never take them."""

import re
import sys
from pathlib import Path

import pytest
import yaml
from flowlib import localconfig, machine

ROOT = Path(__file__).parents[2]


def _resources(args):
    start = args.index("--resources") + 1
    end = next(
        (i for i, a in enumerate(args[start:], start) if a.startswith("--")),
        len(args),
    )
    return args[start:end]


def _defaults(args):
    start = args.index("--default-resources") + 1
    end = next(
        (i for i, a in enumerate(args[start:], start) if a.startswith("--")),
        len(args),
    )
    return args[start:end]


def test_jobs_beside_the_trainer_share_the_cores_it_leaves():
    args = machine.args({}, cores=12, total_mb=32768, path="")
    assert "side_cores=8" in _resources(args)
    defaults = _defaults(args)
    assert "side_cores=min(threads,8)" in defaults
    assert "mem_mb=2048" in defaults


def test_a_machine_sets_the_trainer_s_cores():
    cfg = {"TRAINER_CORES": "6"}
    args = machine.args(cfg, cores=12, total_mb=8192, path="")
    assert "side_cores=6" in _resources(args)


def test_each_gpu_keeps_a_trainer_s_cores():
    args = machine.args({"GPUS": "0,1"}, cores=12, total_mb=8192, path="")
    assert "side_cores=4" in _resources(args)


def test_no_cores_are_kept_where_nothing_trains_or_none_are_left():
    for cfg, cores in (({"TRAINER_CORES": "0"}, 12), ({}, 4), ({}, 2)):
        args = machine.args(cfg, cores=cores, total_mb=8192, path="")
        assert not any(r.startswith("side_cores") for r in _resources(args))
        assert "side_cores" not in " ".join(_defaults(args))
        assert "mem_mb=2048" in _defaults(args)


def test_the_kept_cores_default_to_the_trainer_s_threads():
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    default = dict(localconfig.DEFAULTS)["TRAINER_CORES"]
    assert int(default) == config["finetune"]["cut_workers"] + 1


def test_the_trainer_claims_none_of_the_side_cores(tmp_path):
    pytest.importorskip("snakemake")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_workflow_dryrun import FT, dry_run

    extra = machine.args({}, cores=12, total_mb=65536, path="")
    out = dry_run(tmp_path, "finetune", config=FT, quiet=False, extra=extra)
    assert out.returncode == 0, out.stderr + out.stdout
    claims = re.findall(
        r"^rule (\w+):\n(?:    .*\n)*?    resources: .*?side_cores=(\d+)",
        out.stdout,
        re.M,
    )
    got = {}
    for rule, side in claims:
        got.setdefault(rule, set()).add(int(side))
    assert got["train_segment"] == {0}
    assert got["score_run_germany"] == {2}
    beside = [v for r, v in got.items() if r != "train_segment"]
    assert all(min(v) >= 1 for v in beside)
