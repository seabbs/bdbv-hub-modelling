"""Several GPUs: one `gpu` slot each, and each GPU job on its own GPU."""

import fcntl
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from flowlib import gpus, machine

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "flowlib" / "gpus.py"


def _offload_dry_run(tmp_path, extra=()):
    """Return the dry run, with each job's command, of an offloaded run."""
    pytest.importorskip("snakemake")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_workflow_dryrun import FT, dry_run

    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    out = dry_run(tmp_path, "finetune", config=config, quiet=False, extra=extra)
    assert out.returncode == 0, out.stderr + out.stdout
    return out


def _nvidia_smi(tmp_path, count):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    lines = "".join(f"echo 'GPU {i}: Fake (UUID: x)'\n" for i in range(count))
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\n{lines}")
    (bin_dir / "nvidia-smi").chmod(0o755)
    return str(bin_dir)


def test_the_gpus_are_those_nvidia_smi_lists(tmp_path):
    path = _nvidia_smi(tmp_path, 2)
    assert machine.gpu_ids({}, path=path, environ={}) == ["0", "1"]
    assert machine.gpu_ids({}, path="", environ={}) == []


def test_the_local_config_or_cuda_visible_devices_names_them(tmp_path):
    path = _nvidia_smi(tmp_path, 4)
    env = {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert machine.gpu_ids({}, path=path, environ=env) == ["2", "3"]
    got = machine.gpu_ids({"GPUS": "1"}, path=path, environ=env)
    assert got == ["1"]


def test_the_gpu_resource_is_one_slot_per_gpu():
    for given, slots in (("", 1), ("0", 1), ("0,1", 2)):
        args = machine.args({"GPUS": given}, cores=8, total_mb=8192, path="")
        res = args[args.index("--resources") + 1 :]
        assert f"gpu={slots}" in res


def test_each_claim_takes_a_free_gpu(tmp_path):
    first = gpus.take(["0", "1"], tmp_path)
    second = gpus.take(["0", "1"], tmp_path)
    assert first is not None
    assert second is not None
    assert {first[0], second[0]} == {"0", "1"}
    assert gpus.take(["0", "1"], tmp_path) is None
    os.close(first[1])
    again = gpus.take(["0", "1"], tmp_path)
    assert again is not None
    assert again[0] == first[0]


def _run(tmp_path, local, command, **extra):
    env = {**os.environ, "LOCAL_CONFIG": str(local)}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("SLURM_JOB_ID", None)
    env["GPU_CLAIMS_DIR"] = str(tmp_path / "claims")
    env.update(extra)
    return subprocess.run(
        ["python3", str(SCRIPT), "SEEN=yes", "sh", "-c", command],
        env={**env, "PATH": f"{tmp_path / 'none'}:{env['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_a_job_runs_on_the_gpu_it_claims(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nGPUS=0,1\n")
    show = 'echo "$CUDA_VISIBLE_DEVICES $SEEN"'
    assert _run(tmp_path, local, show) == "0 yes\n"
    # GPU 0 held by another job: this one takes GPU 1.
    folder = tmp_path / "claims"
    fd = os.open(folder / "gpu-0.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert _run(tmp_path, local, show) == "1 yes\n"
    os.close(fd)


def test_the_claim_holds_while_the_job_runs(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nGPUS=0\n")
    # The job, still running, holds GPU 0's lock.
    check = (
        f'{sys.executable} -c "import fcntl, os; '
        f"fd = os.open('{tmp_path / 'claims' / 'gpu-0.lock'}', os.O_RDWR); "
        'fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)" '
        "2>/dev/null && echo free || echo held"
    )
    assert _run(tmp_path, local, check) == "held\n"


def test_without_cuda_a_job_runs_as_it_is(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\n")
    show = 'echo "[${CUDA_VISIBLE_DEVICES-unset}] $SEEN"'
    assert _run(tmp_path, local, show) == "[unset] yes\n"


def _gpu_commands(tmp_path, gpus_set):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nGPUS={gpus_set}\n"
        "FORECAST_DEVICE=cuda\n"
    )
    out = _offload_dry_run(tmp_path)
    return [
        line
        for line in out.stdout.splitlines()
        if "finetune-stream" in line
        or "evaluate_offloaded.py" in line
        or "nowcastgermany.checkpoint" in line
    ]


def test_every_gpu_job_claims_its_gpu_with_one_or_two_gpus(tmp_path):
    for count, given in ((1, "0"), (2, "0,1")):
        folder = tmp_path / f"gpus-{count}"
        folder.mkdir()
        lines = _gpu_commands(folder, given)
        assert any("finetune-stream" in line for line in lines)
        assert any("evaluate_offloaded.py" in line for line in lines)
        for line in lines:
            assert machine.CLAIM in line, line


def test_a_cpu_forecast_claims_no_gpu(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nFORECAST_DEVICE=cpu\n"
    )
    out = _offload_dry_run(tmp_path)
    lines = [
        line
        for line in out.stdout.splitlines()
        if "evaluate_offloaded.py" in line
    ]
    assert lines
    assert not any(machine.CLAIM in line for line in lines)


def test_a_slurm_job_runs_on_the_gpus_slurm_gave_it(tmp_path):
    # Slurm assigns the job its GPUs and sets CUDA_VISIBLE_DEVICES; ids
    # are the node's, so no lock is taken.
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nGPUS=0,1\n")
    show = 'echo "$CUDA_VISIBLE_DEVICES $SEEN"'
    got = _run(
        tmp_path, local, show, SLURM_JOB_ID="7", CUDA_VISIBLE_DEVICES="3"
    )
    assert got == "3 yes\n"
    assert not (tmp_path / "claims").exists()


def test_a_cluster_executor_assigns_the_gpus(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nGPUS=0,1\nCOMPUTE=slurm\n")
    show = 'echo "$CUDA_VISIBLE_DEVICES $SEEN"'
    assert _run(tmp_path, local, show, CUDA_VISIBLE_DEVICES="5") == "5 yes\n"
    assert not (tmp_path / "claims").exists()


def test_the_locks_are_this_node_s_not_the_shared_root(monkeypatch):
    monkeypatch.delenv("GPU_CLAIMS_DIR", raising=False)
    folder = gpus.claims_dir()
    assert folder.parent == Path("/tmp")
    assert str(os.getuid()) in folder.name


def _evaluate_workers(tmp_path, device):
    """Return the threads and EVALUATE_WORKERS of the one-pass evaluations."""
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nGPUS=0\n"
        f"FORECAST_DEVICE={device}\nFORECAST_THREADS=1\nCORES=8\n"
    )
    out = _offload_dry_run(tmp_path, ["--cores", "8"])
    got = set()
    for block in out.stdout.split("\nrule ")[1:]:
        if not block.startswith(("evaluate_base:", "evaluate_checkpoint:")):
            continue
        threads = re.search(r"threads: (\d+)", block)
        workers = re.search(r"EVALUATE_WORKERS=(\d+) ", block)
        assert workers, block
        got.add((int(threads[1]) if threads else 1, int(workers[1])))
    assert got
    return got


def test_the_held_out_pass_takes_a_worker_a_forecast_s_threads(tmp_path):
    # On the CPU a worker for each of the job's `forecast.threads` (one
    # here); on CUDA one process, which holds the GPU.
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cuda").mkdir()
    got = _evaluate_workers(tmp_path / "cpu", "cpu")
    assert max(threads for threads, _ in got) > 1
    for threads, workers in got:
        assert workers == threads
    for _, workers in _evaluate_workers(tmp_path / "cuda", "cuda"):
        assert workers == 1
