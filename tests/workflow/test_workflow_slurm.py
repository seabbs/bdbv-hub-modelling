"""The Slurm profile: every job a Slurm job, sized by its rule."""

import subprocess
import sys
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


def test_compute_slurm_runs_the_slurm_profile_alone():
    cfg = {"COMPUTE": "slurm"}
    assert compute.mode(cfg) == "slurm"
    assert compute.args(cfg) == [
        "--workflow-profile",
        "workflow/profiles/slurm",
    ]


def test_every_job_is_a_slurm_job_with_memory_and_a_runtime():
    got = _profile()
    assert got["executor"] == "slurm"
    defaults = dict(r.split("=") for r in got["default-resources"])
    assert int(defaults["mem_mb"]) > 0
    assert int(defaults["runtime"]) > 0


def test_batches_and_the_trainer_claim_what_they_use():
    # A batch claims no memory on one machine (it is run by count); a
    # Slurm job without a claim gets the whole node or the partition's
    # default. The trainer's GPU comes from its rule (`gpu=1`).
    sized = _set("set-resources")
    for rule in ("simulate_batch", "train_segment"):
        assert int(sized[rule]["mem_mb"]) > 0, rule
        assert int(sized[rule]["runtime"]) > 0, rule
    threads = dict(r.split("=") for r in _profile()["set-threads"])
    assert threads["simulate_batch"] == "1"
    assert int(threads["train_segment"]) > 1


def test_the_one_machine_caps_hold_on_the_cluster():
    caps = dict(r.split("=") for r in _profile()["resources"])
    assert {"sim_slots", "awaiting"} <= set(caps)
    assert caps["awaiting"] == "1"


def test_no_cluster_names_are_committed():
    # Partition and account come from SBATCH_* in the submit environment.
    text = PROFILE.read_text()
    for key in ("slurm_partition=", "slurm_account="):
        assert key not in text


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
            "--workflow-profile",
            "workflow/profiles/slurm",
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
