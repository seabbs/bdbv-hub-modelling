"""The trainer's memory is kept for it: jobs beside it never take it.

A segment's trainer becomes ready the moment the segment before is
saved. Snakemake gives freed memory to the jobs ready at the time, so
without a share of its own the next trainer waits for whatever scoring
took the memory while the segment before trained.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from flowlib import compute, machine, memory

ROOT = Path(__file__).parents[2]

HEADER = "s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\n"


def _benchmark(root, run, n, peak):
    path = Path(root, "runs", run, memory.TRAINER_BENCHMARK.format(n=n))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{HEADER}1\t0:00:01\t{peak}\t1\t1\t1\t0\t0\n")


def _option(args, name):
    start = args.index(name) + 1
    end = next(
        (i for i, a in enumerate(args[start:], start) if a.startswith("--")),
        len(args),
    )
    return args[start:end]


def _side_mem(args):
    got = [r for r in _option(args, "--resources") if r.startswith("side_mem")]
    return int(got[0].split("=")[1]) if got else None


def test_a_trainer_is_kept_its_rule_s_minimum_before_any_benchmark(tmp_path):
    assert memory.trainer_mb(tmp_path) == memory.TRAINER_MIN_MB
    assert memory.trainer_mb(None) == memory.TRAINER_MIN_MB


def test_a_trainer_is_kept_the_most_any_segment_took(tmp_path):
    # archie's n3200 peaked at 8486 MB: n4800 asked for 10752.
    _benchmark(tmp_path, "runa", 1600, 2000)
    _benchmark(tmp_path, "runa", 3200, 8485.83)
    assert memory.trainer_mb(tmp_path) == 10752


def test_jobs_beside_the_trainer_share_the_memory_it_leaves(tmp_path):
    _benchmark(tmp_path, "runa", 3200, 8485.83)
    cfg = {"ROOT": str(tmp_path)}
    args = machine.args(cfg, cores=12, total_mb=49861, path="")
    assert "mem_mb=49861" in _option(args, "--resources")
    assert _side_mem(args) == 49861 - 10752


def test_each_gpu_keeps_a_trainer_s_memory(tmp_path):
    cfg = {"ROOT": str(tmp_path), "GPUS": "0,1"}
    args = machine.args(cfg, cores=12, total_mb=32768, path="")
    assert _side_mem(args) == 32768 - 2 * memory.TRAINER_MIN_MB


def test_no_memory_is_kept_where_nothing_trains_or_none_is_left(tmp_path):
    for cfg, total in (
        ({"TRAINER_CORES": "0"}, 32768),
        ({}, memory.TRAINER_MIN_MB),
    ):
        args = machine.args(
            {"ROOT": str(tmp_path), **cfg}, cores=12, total_mb=total, path=""
        )
        assert _side_mem(args) is None


def test_ssh_keeps_a_trainer_s_memory_for_each_pooled_gpu(tmp_path):
    budgets = {
        "server": {"cores": 12, "mem_mb": 50000, "gpu": 1},
        "workstation": {"cores": 10, "mem_mb": 24000, "gpu": 1},
    }
    cfg = {
        "COMPUTE": "ssh",
        "ROOT": str(tmp_path),
        "WORKERS": "server=local workstation=mac:/r:/c",
    }
    got = compute.args(cfg, budgets=budgets)
    assert _side_mem(got) == 74000 - 2 * memory.TRAINER_MIN_MB


def test_a_cluster_run_keeps_no_memory():
    # Slurm gives each job its own memory: there is no pool to share.
    args = compute.args({"COMPUTE": "slurm"})
    assert "side_mem_mb" not in " ".join(args)


def _claims(stdout):
    """Return each rule's (mem_mb, side_mem_mb) claims in a dry run."""
    got = {}
    for rule, res in re.findall(
        r"^rule (\w+):\n(?:    .*\n)*?    resources: ([^\n]*)", stdout, re.M
    ):
        mem = re.search(r"(?<!\w)mem_mb=(\d+)", res)
        side = re.search(r"side_mem_mb=(\d+)", res)
        got.setdefault(rule, set()).add(
            (int(mem.group(1)), int(side.group(1)) if side else None)
        )
    return got


def test_every_job_but_the_trainer_claims_its_memory_of_the_side_share(
    tmp_path,
):
    pytest.importorskip("snakemake")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_workflow_dryrun import FT, dry_run

    extra = machine.args(
        {"ROOT": str(tmp_path)}, cores=12, total_mb=262144, path=""
    )
    out = dry_run(tmp_path, "finetune", config=FT, quiet=False, extra=extra)
    assert out.returncode == 0, out.stderr + out.stdout
    got = _claims(out.stdout)
    assert {side for _, side in got["train_segment"]} == {0}
    beside = {r: v for r, v in got.items() if r != "train_segment"}
    assert "score_run_germany" in beside
    for rule, claims in beside.items():
        assert all(side == mem for mem, side in claims), rule


# Two segments and three scoring jobs on 13312 MB. The first segment
# claims 3840 MB, the second (from a longer segment's peak) 5120 MB;
# each scoring job 4608 MB, and B and C outlast the first segment.
SNAKEFILE = """
import sys

sys.path.insert(0, "{workflow}")
from flowlib import machine, memory

BENCH = "{root}/runs/a/" + memory.TRAINER_BENCHMARK
SIDE = {{"A": (3, 1), "B": (2, 10), "C": (1, 10)}}


rule all:
    input:
        "t1.done",
        "t2.done",
        expand("s{{i}}.done", i=SIDE),


rule train:
    input:
        lambda wc: [] if wc.n == "1" else ["t1.done"],
    output:
        "t{{n}}.done",
    priority: 10**6
    resources:
        mem_mb=memory.from_benchmark(BENCH, memory.TRAINER_MIN_MB),
        side_mem_mb=0,
        gpu=1,
    shell:
        "echo {{wildcards.n}} start $(python3 -c 'import time; "
        "print(time.time())') >> trainers.log; sleep 2; "
        "echo {{wildcards.n}} end $(python3 -c 'import time; "
        "print(time.time())') >> trainers.log; touch {{output}}"


rule side:
    output:
        "s{{i}}.done",
    priority: lambda wc: SIDE[wc.i][0]
    resources:
        mem_mb=4608,
    params:
        secs=lambda wc: SIDE[wc.i][1],
    shell:
        "sleep {{params.secs}}; touch {{output}}"


machine.share_side_memory(workflow.rules)
"""


def _gap(tmp_path, keep):
    """Return the seconds from the first segment's end to the next's start."""
    snakemake = Path(sys.executable).parent / "snakemake"
    _benchmark(tmp_path, "a", 1, 3000)
    _benchmark(tmp_path, "a", 2, 4000)
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        SNAKEFILE.format(workflow=ROOT / "workflow", root=tmp_path)
    )
    args = machine.args(
        {"ROOT": str(tmp_path), "TRAINER_CORES": "1"},
        cores=8,
        total_mb=13312,
        path="",
    )
    if not keep:
        args = [a for a in args if not a.startswith("side_mem_mb")]
    out = subprocess.run(
        [str(snakemake), "-s", str(snakefile), "-d", str(tmp_path), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    rows = (tmp_path / "trainers.log").read_text().split("\n")
    times = {tuple(r.split()[:2]): float(r.split()[2]) for r in rows if r}
    return times[("2", "start")] - times[("1", "end")]


@pytest.mark.slow
def test_the_next_segment_starts_while_the_scoring_runs(tmp_path):
    pytest.importorskip("snakemake")
    assert _gap(tmp_path, keep=True) < 4


@pytest.mark.slow
def test_without_its_share_the_next_segment_waits_for_the_scoring(tmp_path):
    # The night of 2026-11-25: n4800 waited 52 minutes for scoring jobs
    # that took the memory while n3200 trained.
    pytest.importorskip("snakemake")
    assert _gap(tmp_path, keep=False) > 5
