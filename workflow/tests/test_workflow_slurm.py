"""The Slurm profile: every job a Slurm job, sized by its rule."""

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from flowlib import compute

ROOT = Path(__file__).parents[2]
PROFILE = ROOT / "workflow" / "profiles" / "slurm" / "config.yaml"


def _profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


def _set(key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in _profile()[key]:
        rule, value = line.split(":", 1)
        name, got = value.split("=", 1)
        out.setdefault(rule, {})[name] = got
    return out


def _option(args, flag):
    start = args.index(flag) + 1
    end = next(
        (i for i, a in enumerate(args[start:], start) if a.startswith("--")),
        len(args),
    )
    return args[start:end]


def test_compute_slurm_runs_the_slurm_profile_with_its_defaults():
    args = compute.args({"COMPUTE": "slurm"})
    assert args[:2] == ["--workflow-profile", "workflow/profiles/slurm"]
    defaults = dict(r.split("=") for r in _option(args, "--default-resources"))
    assert int(defaults["mem_mb"]) > 0
    assert int(defaults["runtime"]) > 0
    assert "slurm_account" not in defaults
    assert "--slurm-partition-config" not in args


def test_the_site_s_account_and_partitions_come_from_the_local_config():
    cfg = {
        "COMPUTE": "slurm",
        "SLURM_ACCOUNT": "proj123",
        "SLURM_PARTITIONS": "/site/partitions.yaml",
    }
    args = compute.args(cfg)
    assert "slurm_account=proj123" in _option(args, "--default-resources")
    assert _option(args, "--slurm-partition-config") == [
        "/site/partitions.yaml"
    ]


def test_a_cluster_run_has_no_one_machine_gpu_or_core_caps():
    # Slurm queues GPU jobs across nodes, so several runs train at once;
    # the trainer's kept cores (side_cores) are one machine's.
    args = compute.args({"COMPUTE": "slurm", "TRAINER_CORES": "4"})
    flat = " ".join(args)
    assert "side_cores" not in flat
    assert "gpu=" not in flat
    caps = dict(r.split("=") for r in _profile()["resources"])
    assert "gpu" not in caps


def test_a_site_profile_replaces_the_one_compute_gives():
    for mode in ("single", "slurm"):
        cfg = {"COMPUTE": mode, "SNAKEMAKE_PROFILE": "/site/profile"}
        assert compute.args(cfg) == ["--workflow-profile", "/site/profile"]


def test_the_example_partitions_send_gpu_jobs_to_a_gpu_partition():
    partitions = pytest.importorskip(
        "snakemake_executor_plugin_slurm.partitions"
    )
    example = PROFILE.parent / "partitions.example.yaml"
    got = {p.name: p.limits for p in partitions.read_partition_file(example)}
    assert any(limits.max_gpu >= 1 for limits in got.values())
    assert any(limits.max_gpu == 0 for limits in got.values())


def test_batches_and_the_trainer_claim_what_they_use():
    # A batch claims no memory on one machine (it is run by count); a
    # Slurm job without a claim gets the whole node or the partition's
    # default. The trainer's GPU comes from its rule (`gpu=1`), its
    # memory from its last peak (flowlib.memory).
    sized = _set("set-resources")
    assert int(sized["simulate_batch"]["mem_mb"]) > 0
    assert "mem_mb" not in sized["train_segment"]
    for rule in ("simulate_batch", "train_segment"):
        assert int(sized[rule]["runtime"]) > 0, rule
    threads = dict(r.split("=") for r in _profile()["set-threads"])
    assert threads["simulate_batch"] == "1"
    assert int(threads["train_segment"]) > 1


def test_the_one_machine_caps_hold_on_the_cluster():
    caps = dict(r.split("=") for r in _profile()["resources"])
    assert {"sim_slots", "awaiting"} <= set(caps)
    assert caps["awaiting"] == "1"


def test_no_cluster_names_are_committed():
    # Account and partitions are each site's, in config/local.env.
    got = _profile()
    assert "default-resources" not in got
    for key in ("slurm_partition", "slurm_account"):
        assert not any(key in v for v in got["set-resources"]), key


def test_a_dry_run_sizes_each_batch_as_a_slurm_job(tmp_path):
    pytest.importorskip("snakemake")
    pytest.importorskip("snakemake_executor_plugin_slurm")
    snakemake = Path(sys.executable).parent / "snakemake"
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("rounds: [4]\n")
    out = subprocess.run(
        [
            str(snakemake),
            "simulate",
            "-s",
            "workflow/Snakefile",
            "-n",
            *compute.args({"COMPUTE": "slurm"}),
            "--configfile",
            str(overlay),
        ],
        cwd=ROOT,
        env={
            "PATH": f"{snakemake.parent}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    batches = out.stdout.split("rule simulate_batch:")[1:]
    assert len(batches) == 40
    assert all("mem_mb=4096" in b and "runtime=120" in b for b in batches)


def test_every_gpu_job_asks_slurm_for_its_gpu(tmp_path):
    # The trainer and, with forecast.device cuda, every forecasting job
    # but the German units (`score.device: cpu`) take `gpu=1`, which the
    # plugin submits as --gpus=1.
    pytest.importorskip("snakemake")
    utils = pytest.importorskip("snakemake_executor_plugin_slurm.utils")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_workflow_dryrun import FT, dry_run

    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nCOMPUTE=slurm\n"
        "DEVICE=cuda\nFORECAST_DEVICE=cuda\n"
    )
    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    out = dry_run(
        tmp_path,
        "finetune",
        config=config,
        quiet=False,
        extra=[*compute.args({"COMPUTE": "slurm"}), "--cores", "1"],
    )
    assert out.returncode == 0, out.stderr + out.stdout
    gpu = {}
    for block in out.stdout.split("\nrule ")[1:]:
        rule = block.split(":", 1)[0]
        res = re.search(r"resources: (.*)", block)
        if res:
            gpu.setdefault(rule, set()).add("gpu=1" in res[1])
    for rule in (
        "train_segment",
        "evaluate_base",
        "evaluate_checkpoint",
        "seven_day_check",
    ):
        assert gpu[rule] == {True}, rule
    assert gpu["score_run_germany"] == {False}
    job = types.SimpleNamespace(resources={"gpu": 1})
    assert "--gpus=1" in utils.set_gres_string(job)


def _commands(out: str) -> dict[str, str]:
    return {b.split(":", 1)[0]: b for b in out.split("\nrule ")[1:] if ":" in b}


def test_the_launcher_is_off_under_slurm_and_on_one_machine(tmp_path):
    # Slurm gives each job its cores and memory; on one machine the
    # launcher pins them and waits for the memory (flowlib.launch).
    pytest.importorskip("snakemake")
    pytest.importorskip("snakemake_executor_plugin_slurm")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_workflow_dryrun import FT, dry_run

    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    rules = ("train_segment", "evaluate_base", "score_run_germany")
    for compute_mode, on in (("single", True), ("slurm", False)):
        root = tmp_path / compute_mode
        root.mkdir()
        (root / "local.env").write_text(
            f"ROOT={root}\nHOST_NAME=test-host\nCOMPUTE={compute_mode}\n"
        )
        extra = (
            [*compute.args({"COMPUTE": "slurm"}), "--cores", "1"]
            if compute_mode == "slurm"
            else []
        )
        out = dry_run(
            root, "finetune", config=config, quiet=False, extra=["-p", *extra]
        )
        assert out.returncode == 0, out.stderr + out.stdout
        got = _commands(out.stdout)
        for rule in rules:
            assert ("flowlib/launch.py" in got[rule]) is on, (
                compute_mode,
                rule,
            )
