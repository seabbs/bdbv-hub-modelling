"""Runs train at once, a GPU each, and take turns when GPUs are fewer.

Snakemake schedules two runs' first segments for real, with the
simulator and the trainer stood in for by a `uv` that writes their
outputs and logs when each trainer ran and on which GPU.
"""

import pytest

pytest.importorskip("snakemake")

from test_workflow_dryrun import FT, dry_run

pytestmark = pytest.mark.slow

UV = """#!/bin/bash
now() { python3 -c 'import time; print(time.time())'; }
case "$*" in
*finetune-stream*)
  set -- $(echo "$@" | sed -E 's/.*finetune-stream //')
  train=$2
  n=$(echo "$@" | sed -E 's/.*--until ([0-9]+).*/\\1/')
  start=$(now)
  sleep 3
  mkdir -p "$train/n$n/finetuned-ckpt"
  echo '{"hash": "x"}' > "$train/n$n/settings.json"
  touch "$train/n$n/finetuned-ckpt/adapter_model.safetensors"
  echo "$train $CUDA_VISIBLE_DEVICES $start $(now)" >> LOG
  ;;
*nowcastfoundation.simulator*)
  set -- $(echo "$@" | sed -E 's/.*simulator //')
  mkdir -p "$4"
  echo x | gzip > "$4/counts-$3.csv.gz"
  echo x | gzip > "$4/latent-$3.csv.gz"
  (echo outbreak; seq 1 "$2") > "$4/settings-$3.csv"
  ;;
esac
"""


def _train(tmp_path, gpus):
    """Return each trainer's run, GPU, start and end, trained on `gpus`."""
    log = tmp_path / "trainers.log"
    ids = ",".join(str(i) for i in range(gpus))
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nGPUS={ids}\n"
    )
    (tmp_path / "bin").mkdir()
    uv = tmp_path / "bin" / "uv"
    uv.write_text(UV.replace("LOG", str(log)))
    uv.chmod(0o755)
    config = {**FT, "runs": {"a": {}, "b": {}}, "train_runs": ["a", "b"]}
    targets = [
        f"{tmp_path}/runs/{run}/train/n250/settings.json" for run in "ab"
    ]
    out = dry_run(
        tmp_path,
        *targets,
        config=config,
        dry=False,
        extra=(
            "--cores",
            "16",
            "--resources",
            f"gpu={gpus}",
            "sim_slots=16",
            "awaiting=1",
            "--latency-wait",
            "1",
        ),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [line.split() for line in log.read_text().splitlines()]
    return {
        train.split("/")[-2]: (gpu, float(start), float(end))
        for train, gpu, start, end in rows
    }


def test_two_gpus_train_two_runs_at_once(tmp_path):
    got = _train(tmp_path, 2)
    assert set(got) == {"a", "b"}
    (ga, sa, ea), (gb, sb, eb) = got["a"], got["b"]
    assert {ga, gb} == {"0", "1"}
    assert sa < eb
    assert sb < ea


def test_one_gpu_trains_the_runs_in_turn(tmp_path):
    got = _train(tmp_path, 1)
    assert set(got) == {"a", "b"}
    (ga, sa, ea), (gb, sb, eb) = got["a"], got["b"]
    assert ga == gb == "0"
    assert ea <= sb or eb <= sa
