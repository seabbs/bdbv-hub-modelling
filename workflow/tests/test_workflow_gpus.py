"""Several GPUs: one `gpu` slot each, and each GPU job on its own GPU."""

import fcntl
import os
import subprocess
import sys
from pathlib import Path

from flowlib import gpus, machine

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "flowlib" / "gpus.py"
sys.path.insert(0, str(Path(__file__).parent))

from test_workflow_dryrun import FT, dry_run  # noqa: E402


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
    assert first is not None and second is not None
    assert {first[0], second[0]} == {"0", "1"}
    assert gpus.take(["0", "1"], tmp_path) is None
    os.close(first[1])
    again = gpus.take(["0", "1"], tmp_path)
    assert again is not None and again[0] == first[0]


def _run(tmp_path, local, command):
    env = {**os.environ, "LOCAL_CONFIG": str(local)}
    env.pop("CUDA_VISIBLE_DEVICES", None)
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
    folder = tmp_path / gpus.FOLDER
    fd = os.open(folder / "gpu-0.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert _run(tmp_path, local, show) == "1 yes\n"
    os.close(fd)


def test_the_claim_holds_while_the_job_runs(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nGPUS=0\n")
    # The job, still running, holds GPU 0's lock.
    check = (
        f"{sys.executable} -c \"import fcntl, os; "
        f"fd = os.open('{tmp_path / gpus.FOLDER / 'gpu-0.lock'}', os.O_RDWR); "
        "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\" "
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
    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    out = dry_run(tmp_path, "finetune", config=config, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
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
    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    out = dry_run(tmp_path, "finetune", config=config, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    lines = [
        line
        for line in out.stdout.splitlines()
        if "evaluate_offloaded.py" in line
    ]
    assert lines
    assert not any(machine.CLAIM in line for line in lines)
