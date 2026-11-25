"""Dry runs of the workflow's targets against a temporary machine."""

import gzip
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from flowlib import compute

pytest.importorskip("snakemake")

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "workflow" / "config.yaml"
SNAKEMAKE = Path(sys.executable).parent / "snakemake"


def dry(tmp_path, *targets, config=None, extra=()):
    """Return the jobs per rule of a dry run of `targets`.

    `config` is merged into workflow/config.yaml; `extra` are more
    Snakemake options.
    """
    out = dry_run(tmp_path, *targets, config=config, extra=extra)
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", out.stdout, re.M))
    return {k: int(v) for k, v in jobs.items() if k != "total"}


def dry_run(tmp_path, *targets, config=None, extra=(), quiet=True, dry=True):
    """Return the finished process of a dry run of `targets`.

    Programs in <tmp_path>/bin (a stand-in `gh`) come first on PATH. Not
    `quiet`, the output holds each job's shell command. Not `dry`, the
    jobs run.
    """
    local = tmp_path / "local.env"
    if not local.exists():
        local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    if not dry:
        # A test run serves no dashboard.
        config = {**(config or {}), "dashboard": {"serve": False}}
    overlay.write_text(yaml.safe_dump(config or {}))
    return subprocess.run(
        [
            str(SNAKEMAKE),
            *targets,
            "-s",
            "workflow/Snakefile",
            *(["-n"] if dry else []),
            *(["--quiet", "rules"] if quiet else []),
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
            *(["--configfile", str(overlay)] if config else []),
            *extra,
        ],
        cwd=ROOT,
        env={
            "PATH": f"{tmp_path}/bin:{SNAKEMAKE.parent}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )


def test_simulate_runs_every_batch_of_a_round(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["simulate_batch"] == 40
    assert jobs["batch_record"] == 40
    assert jobs["round_manifest"] == 1
    assert jobs["sim_qc"] == 1
    assert jobs["round_ready"] == 1


def _sim_commands(out):
    return [
        line
        for line in out.stdout.splitlines()
        if "nowcastfoundation.simulator" in line
    ]


def test_batches_run_by_count_not_a_cap(tmp_path):
    # SIM_SLOTS sets how many batches run at once: each claims its rule's
    # memory, so the slots fit the machine's budget, and runs under no
    # prefix.
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\n"
    )
    out = dry_run(tmp_path, "simulate", config={"rounds": [4]}, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    lines = _sim_commands(out)
    assert len(lines) == 40
    claims = re.findall(
        r"rule simulate_batch:.*?resources: ([^\n]*)", out.stdout, re.S
    )
    assert len(claims) == 40
    assert all("sim_slots=1" in c for c in claims)
    rule = (ROOT / "workflow" / "rules" / "simulate.smk").read_text()
    claim = re.search(r"rule simulate_batch:.*?mem_mb=(\d+)", rule, re.S)
    assert all(f"mem_mb={claim.group(1)}," in c for c in claims)
    bare = re.compile(r"eigen=false\s+workflow/scripts/timed\.sh")
    assert all(bare.search(x) for x in lines)


def test_simulate_never_releases(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert not any(k.startswith("release") for k in jobs)


def test_release_is_its_own_target(tmp_path):
    jobs = dry(tmp_path, "release", config={"rounds": [4]})
    assert jobs["release_batch"] == 40


def test_batches_already_simulated_are_kept(tmp_path):
    sims = tmp_path / "rounds" / "r4" / "sims"
    sims.mkdir(parents=True)
    for seed in (30400, 30401):
        with gzip.open(sims / f"counts-{seed}.csv.gz", "wt") as f:
            f.write("x\n")
        (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["simulate_batch"] == 38


SPLIT = {"rounds": [10]}


def _host(tmp_path, seeds):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nSIM_SEEDS={seeds}\n"
    )


def test_a_round_shared_with_another_host_waits_for_its_seeds(tmp_path):
    _host(tmp_path, '"31000-31004 31006 31008-31039"')
    jobs = dry(tmp_path, "simulate", config=SPLIT)
    assert jobs["simulate_batch"] == 38
    assert jobs["await_batch"] == 2
    assert jobs["batch_record"] == 40


def test_a_second_host_simulates_and_pushes_only_its_seeds(tmp_path):
    _host(tmp_path, '"31005 31007"')
    jobs = dry(tmp_path, "push", config=SPLIT)
    assert jobs["simulate_batch"] == 2
    assert jobs["push_batch"] == 2
    assert "batch_record" not in jobs
    assert "await_batch" not in jobs


def test_a_host_can_take_whole_rounds(tmp_path):
    _host(tmp_path, "r10")
    jobs = dry(tmp_path, "push", config={"rounds": [10, 11]})
    assert jobs["push_batch"] == 40


def test_only_generic_profiles_ship():
    profiles = (ROOT / "workflow" / "profiles").iterdir()
    # ssh and slurm name no machine: ssh's workers are in the local
    # config, slurm's partition and account in the submit environment.
    assert sorted(p.name for p in profiles) == ["ci", "default", "slurm", "ssh"]


FT = {
    "rounds": [1, 2],
    "finetune": {
        "prefixes": [250, 500],
        "outbreaks": 500,
        # The fixtures' own chunks and evaluation, whatever the run's.
        "chunk": 250,
        "eval": "local",
        "rounds": [2],
        "validation_round": 1,
        "sims_from": "local",
    },
}


def test_finetune_trains_one_segment_per_prefix(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 2
    # 500 outbreaks are 25 batches of 20, one spare, and the held-out
    # round's 40 batches.
    assert jobs["fetch_batch"] == 66
    # On one machine the batches are simulated here.
    assert jobs["simulate_batch"] == 66


def test_finetune_asks_for_its_scores(tmp_path):
    # Fine-tuning produces its own scoring through the
    # DAG: each checkpoint's German unit and the reference units, cached
    # when unchanged, with no separate TARGETS entry or driver.
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["score_run_germany"] == jobs["train_segment"] == 2
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )
    assert jobs["score_germany"] == len(config["score"]["reference"])


def _hub(tmp_path):
    """The hub data's head, which the BDBV check's forecasts depend on."""
    (tmp_path / "hub").mkdir(exist_ok=True)
    (tmp_path / "hub" / "head.txt").write_text("abc\n")


def test_each_checkpoint_has_its_own_ft_check_page(tmp_path):
    # A new checkpoint is a new page, so it is built without
    # forcing; the reference models' forecasts are shared.
    _hub(tmp_path)
    config = {"ft_check": {"checkpoint": "slots-abcdef"}}
    out = dry_run(tmp_path, "ft_check", config=config, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "ft-check/slots-abcdef/bdbv-ft-check-slots-abcdef.html" in out.stdout
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", out.stdout, re.M))
    assert jobs["ft_check_page"] == "1"
    assert jobs["ft_check_forecast"] == "3"


def test_ft_check_forecasts_are_keyed_by_their_code(tmp_path):
    # A forecast folder made by older code is never reused:
    # every model's folder, the fine-tuned one too, carries the code key.
    import sys

    sys.path.insert(0, str(ROOT / "workflow"))
    from flowlib import scoring

    _hub(tmp_path)
    old = tmp_path / "hub" / "ft-check" / "forecasts"
    old.mkdir(parents=True)
    for name in ("randomwalknowcast", "ft-slots-abcdef"):
        (old / f"{name}.done").write_text("")
    config = {"ft_check": {"checkpoint": "slots-abcdef"}}
    out = dry_run(tmp_path, "ft_check", config=config, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    key = scoring.ft_check_key(ROOT)
    assert f"forecasts/ft-slots-abcdef@{key}.done" in out.stdout
    assert f"forecasts/randomwalknowcast@{key}.done" in out.stdout
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", out.stdout, re.M))
    assert jobs["ft_check_forecast"] == "3"


def test_a_fresh_host_fetches_the_hub_s_data_commit(tmp_path):
    # Without hub_watch.sh, <ROOT>/hub/head.txt is missing and the BDBV
    # check cannot be planned, so the workflow fetches it when missing,
    # and hub_watch.sh keeps it current.
    config = {"ft_check": {"checkpoint": "slots-abcdef"}}
    jobs = dry(tmp_path, "ft_check", config=config)
    assert jobs["hub_head"] == 1
    assert jobs["hub_data"] == 1


def test_finetune_asks_for_ft_check_of_released_checkpoints(tmp_path):
    # Each released checkpoint (its published.json) gets its
    # BDBV check page.
    _hub(tmp_path)
    seg = tmp_path / "runs" / "runa" / "train" / "n250"
    seg.mkdir(parents=True)
    (seg / "published.json").write_text(
        '{"tag": "chronos-slots-abcdef", "hash": "abcdef"}'
    )
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["ft_check_page"] == 1


def _job_priorities(stdout):
    """Return each job's rule, wildcards and priority from a dry run."""
    found = []
    for block in re.split(r"\n(?=(?:local)?rule \w+:)", stdout):
        rule = re.match(r"(?:local)?rule (\w+):", block)
        if rule:
            wildcards = re.search(r"wildcards: (.*)", block)
            got = re.search(r"priority: ([\d.]+)", block)
            found.append(
                (
                    rule[1],
                    wildcards[1] if wildcards else "",
                    float(got[1]) if got else 0.0,
                )
            )
    return found


def test_jobs_rank_trainer_evaluation_references_scoring_then_batches(
    tmp_path,
):
    # Freed cores must not go to batches for later segments while the
    # German reference, checkpoint scoring or the trainer's batch copies
    # wait, or the GPU idles. One order (flowlib.compute):
    # the trainer, the quick hand-offs (its batches' copies, a
    # checkpoint's release for its BDBV check), the references, each
    # checkpoint's scoring (newest first), then simulation batches. The
    # held-out evaluation goes right after the hand-offs and the batches
    # the next segment reads, so the dashboard's validation loss never
    # waits hours behind the CPU jobs.
    _hub(tmp_path)
    seg = tmp_path / "runs" / "runa" / "train" / "n250"
    seg.mkdir(parents=True)
    (seg / "published.json").write_text(
        '{"tag": "chronos-slots-abcdef", "hash": "abcdef"}'
    )
    # A publishing machine releases each checkpoint for its BDBV check.
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nPUBLISH=1\n"
    )
    config = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}
    out = dry_run(tmp_path, "finetune", config=config, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    tiers = {
        "train": [],
        "fetch": [],
        "reference": [],
        "checkpoint": [],
        "batch": [],
        "evaluate": [],
    }
    newest = {}
    for rule, wildcards, priority in _job_priorities(out.stdout):
        if rule == "train_segment":
            tiers["train"].append(priority)
        elif rule in (
            "fetch_batch",
            "verify_checkpoint",
            "publish_checkpoint",
            # Quick jobs a released checkpoint's BDBV check waits on, which
            # at priority 0 would queue behind every CPU job.
            "hub_data",
            "ft_check_page",
        ):
            tiers["fetch"].append(priority)
        elif rule == "score_germany" or (
            rule == "ft_check_forecast" and "fmodel=ft-" not in wildcards
        ):
            tiers["reference"].append(priority)
        elif rule in ("evaluate_base", "evaluate_checkpoint"):
            tiers["evaluate"].append(priority)
        elif rule in ("score_run_germany", "ft_check_forecast"):
            tiers["checkpoint"].append(priority)
            if rule == "score_run_germany":
                n = int(re.search(r"n=(\d+)", wildcards)[1])
                newest[n] = priority
        elif rule == "simulate_batch":
            tiers["batch"].append(priority)
    assert all(tiers.values()), tiers
    seen = {rule for rule, _, _ in _job_priorities(out.stdout)}
    assert {"hub_data", "ft_check_page"} <= seen
    assert min(tiers["train"]) > max(tiers["fetch"])
    next_batches = [p for p in tiers["batch"] if p > max(tiers["evaluate"])]
    later = [p for p in tiers["batch"] if p < min(tiers["checkpoint"])]
    assert next_batches
    assert later
    assert len(next_batches) + len(later) == len(tiers["batch"])
    assert min(tiers["fetch"]) > max(next_batches)
    assert min(tiers["evaluate"]) > max(tiers["reference"])
    assert min(tiers["reference"]) > max(tiers["checkpoint"])
    assert newest[500] > newest[250]


def test_finetune_builds_each_trained_run_s_german_report(tmp_path):
    # The run's report page comes with it: its checkpoints against
    # zero-shot and randomwalknowcast on the German development half.
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["german_report"] == 1


def test_a_run_table_pulls_its_checkpoints_and_their_scores(tmp_path):
    # One graph: asking for a run's table (as the paper does) trains and
    # scores every checkpoint in reach, not only those already saved.
    table = tmp_path / "runs" / "runa" / "reports" / "checkpoint-scores.csv"
    jobs = dry(tmp_path, str(table), config=FT)
    assert jobs["train_segment"] == 2
    assert jobs["score_run_germany"] == 2


def test_the_trainer_claims_the_cores_it_uses(tmp_path):
    # The trainer and its block-cutting workers count against the cores,
    # so the scoring jobs beside it fit in what is left.
    out = dry_run(
        tmp_path, "finetune", config=FT, quiet=False, extra=("--cores", "16")
    )
    assert out.returncode == 0, out.stderr + out.stdout
    trains = re.findall(
        r"rule train_segment:.*?threads: (\d+).*?--cut-workers (\d+)",
        out.stdout,
        re.S,
    )
    assert trains
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    workers = config["finetune"]["cut_workers"]
    assert all(t == (str(workers + 1), str(workers)) for t in trains)


SMALL = {
    **FT,
    "finetune": {
        **FT["finetune"],
        "prefixes": [10, 20],
        "outbreaks": 20,
        "chunk": 10,
        "batch_size": 64,
        "checkpoint_above": 0,
        "block_limit": 700,
    },
}


def test_a_small_trial_is_a_config_override(tmp_path):
    assert dry(tmp_path, "finetune", config=SMALL)["train_segment"] == 2
    out = dry_run(tmp_path, "finetune", config=SMALL, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    trains = [
        line for line in out.stdout.splitlines() if "finetune-stream" in line
    ]
    assert trains
    for line in trains:
        assert "--batch-size 64" in line
        assert "--checkpoint-above 0" in line
        assert "--block-limit 700" in line


def test_a_machine_s_block_limit_overrides_the_config(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nBLOCK_LIMIT=4000\n"
    )
    out = dry_run(tmp_path, "finetune", config=SMALL, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "--block-limit 4000" in out.stdout


def test_a_machine_s_checkpoint_threshold_overrides_the_config(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nCHECKPOINT_ABOVE=5\n"
    )
    out = dry_run(tmp_path, "finetune", config=SMALL, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "--checkpoint-above 5" in out.stdout
    assert "--checkpoint-above 0" not in out.stdout


RELEASE = {**FT, "finetune": {**FT["finetune"], "sims_from": "release"}}


def _released(tmp_path, seeds):
    """Release `seeds`: a stand-in `gh` lists them as sims-r<N> assets."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    rounds = {}
    for s in seeds:
        rounds.setdefault((s - 30000) // 100, []).append(s)
    for r, got in rounds.items():
        names = [f"counts-{s}.csv.gz" for s in got]
        names += [f"settings-{s}.csv" for s in got]
        (tmp_path / f"sims-r{r}.json").write_text(
            json.dumps({"assets": [{"name": n} for n in names]})
        )
    gh = bin_ / "gh"
    # `gh release view sims-r<N> ...`: the round's assets, or not found.
    gh.write_text(
        "#!/bin/sh\n"
        f'f="{tmp_path}/$3.json"\n'
        '[ -f "$f" ] && exec cat "$f"\n'
        "echo 'release not found' >&2; exit 1\n"
    )
    gh.chmod(0o755)


def _stale_list(tmp_path, seeds):
    """A batch list an earlier run wrote."""
    where = tmp_path / "runs" / "runa"
    where.mkdir(parents=True, exist_ok=True)
    (where / "available-batches.txt").write_text(
        "".join(f"{s}\n" for s in seeds)
    )


def test_finetune_takes_batches_from_releases(tmp_path):
    _released(tmp_path, [*range(30100, 30140), *range(30200, 30240)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert jobs["fetch_batch"] == 66
    assert "simulate_batch" not in jobs


def test_from_releases_only_released_batches_are_trained_on(tmp_path):
    # The held-out round and 14 batches of round 2: 280 outbreaks, enough
    # for n250 and its spare batch but not n500.
    _released(tmp_path, [*range(30100, 30140), *range(30200, 30214)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert jobs["train_segment"] == 1
    # 13 batches hold 260 outbreaks, then one spare, then the held-out.
    assert jobs["fetch_batch"] == 13 + 1 + 40


def test_from_releases_nothing_trains_before_the_held_out_round_is_in(tmp_path):
    _released(tmp_path, [*range(30100, 30139), *range(30200, 30240)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert "train_segment" not in jobs


def test_from_releases_each_run_lists_them_as_it_plans(tmp_path):
    # No listing job: the run lists the releases itself, and records it.
    _released(tmp_path, [*range(30100, 30140)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert "available_batches" not in jobs
    record = tmp_path / "runs" / "runa" / "available-batches.txt"
    assert record.read_text().split() == [str(s) for s in range(30100, 30140)]


REMOTE = {**FT, "finetune": {**FT["finetune"], "sims_from": "remote"}}


def test_from_the_rounds_host_only_listed_batches_are_trained_on(tmp_path):
    # A machine without the rounds streams them from the rounds host's
    # manifests, as they are simulated, before any release.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30214)})
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 1
    assert jobs["fetch_batch"] == 13 + 1 + 40
    assert "simulate_batch" not in jobs


def _records(tmp_path, r, seeds):
    """Batch records of round `r` on the rounds host, before its manifest."""
    records = tmp_path / "host" / "rounds" / f"r{r}" / "records"
    records.mkdir(parents=True)
    for s in seeds:
        (records / f"record-{s}.csv").write_text(
            f"{r},{s},block,r7,c,1,20,a,b,,1,2.0,h,t\n"
        )
    return records.parent


def test_from_the_rounds_host_a_round_is_listed_once_its_qc_ran(tmp_path):
    # QC runs once the round is whole, after its manifest: a batch it
    # later excludes must never be trained on, mid-round or in between.
    _rounds_host(tmp_path, {1: range(30100, 30140)})
    r2 = _records(tmp_path, 2, range(30200, 30214))
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert "train_segment" not in jobs
    (r2 / "manifest.csv").write_text(
        "header\n"
        + "".join(
            f"2,{s},block,r7,c,1,20,a,b,,1,2.0,h,t\n"
            for s in range(30200, 30214)
        )
    )
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert "train_segment" not in jobs
    (r2 / "qc").mkdir()
    (r2 / "qc" / "summary.txt").write_text("sim QC summarised\n")
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 1
    assert jobs["fetch_batch"] == 13 + 1 + 40


OFFLOAD = {**FT, "finetune": {**FT["finetune"], "eval": "offload"}}


def test_offloaded_evaluation_is_planned_after_each_segment(tmp_path):
    jobs = dry(tmp_path, "finetune", config=OFFLOAD)
    assert jobs["train_segment"] == 2
    assert jobs["evaluate_checkpoint"] == 2
    # Only checkpoints are scored offloaded: the trainer's step snapshots
    # are left unscored.
    assert not any(k.startswith("evaluate_steps") for k in jobs)


def test_local_evaluation_plans_no_evaluation_rules(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert not any(k.startswith("evaluate") for k in jobs)


def test_an_offloaded_checkpoint_publishes_before_its_held_out_losses(
    tmp_path,
):
    # The held-out losses take hours on the CPU, so the
    # release and its BDBV check never wait for them; they join the
    # release once scored.
    for n in (250, 500):
        _checkpoint(tmp_path, n)
    where = tmp_path / "runs" / "runa" / "train" / "n250"
    jobs = dry(tmp_path, f"{where}/published.json", config=OFFLOAD)
    assert jobs == {"verify_checkpoint": 1, "publish_checkpoint": 1}
    jobs = dry(tmp_path, "publish", config=OFFLOAD)
    assert jobs["publish_checkpoint"] == 2
    assert jobs["evaluate_checkpoint"] == 2
    assert jobs["release_held_out"] == 2


def test_a_publishing_run_checks_each_saved_checkpoint_once_released(
    tmp_path,
):
    # A saved checkpoint's BDBV check page is planned in the pass that
    # releases it, with no Snakemake checkpoint, whose DAG update can hang the
    # run; its forecast waits for the release.
    _hub(tmp_path)
    _local(tmp_path, "PUBLISH=1")
    _checkpoint(tmp_path, 250)
    settings = tmp_path / "runs" / "runa" / "train" / "n250" / "settings.json"
    settings.write_text('{"hash": "abcdef"}')
    jobs, edges = _dag(tmp_path, "finetune", FT)
    rules = {i: label.split("\\n")[0] for i, label in jobs.items()}
    names = list(rules.values())
    assert names.count("publish_checkpoint") == 2
    assert names.count("ft_check_release") == 1
    publish = {i for i, rule in rules.items() if rule == "publish_checkpoint"}
    forecast = {i for i, rule in rules.items() if rule == "ft_check_forecast"}
    assert any(a in publish and b in forecast for a, b in edges)


def test_a_checkpoint_is_scored_only_after_step_0(tmp_path):
    # Step 0 (base Chronos-2's loss, the sidecar's held_out_loss_start) is
    # scored on its own, so a checkpoint's evaluation waits for no
    # segment's other steps.
    for n in (250, 500):
        _checkpoint(tmp_path, n)
    sidecar = "runs/runa/train/n500/held-out.json"
    jobs = dry(tmp_path, str(tmp_path / sidecar), config=OFFLOAD)
    assert jobs["evaluate_base"] == 1
    assert jobs["evaluate_checkpoint"] == 1
    assert "evaluate_steps" not in jobs


def _dag(tmp_path, target, config):
    """Return a dry run's jobs (id to label) and edges (from, to)."""
    out = dry_run(tmp_path, target, config=config, extra=["--dag"])
    assert out.returncode == 0, out.stderr
    jobs = dict(re.findall(r'^\s*(\d+)\[label = "([^"]+)"', out.stdout, re.M))
    edges = re.findall(r"^\s*(\d+) -> (\d+)$", out.stdout, re.M)
    return jobs, edges


def test_a_checkpoint_s_evaluation_needs_only_step_0_and_earlier_settings(
    tmp_path,
):
    # Not the segments' step evaluations, which queue behind the trainer:
    # a checkpoint is scored between segments.
    for n in (250, 500):
        _checkpoint(tmp_path, n)
    sidecar = tmp_path / "runs/runa/train/n500/held-out.json"
    jobs, edges = _dag(tmp_path, str(sidecar), OFFLOAD)
    assert "evaluate_steps" not in jobs.values()
    (check,) = [
        j for j, label in jobs.items() if label == "evaluate_checkpoint"
    ]
    (base,) = [j for j, label in jobs.items() if label == "evaluate_base"]
    (first,) = [
        j
        for j, label in jobs.items()
        if label.startswith("train_segment\\n") and "n: 250" in label
    ]
    assert (base, check) in edges
    # The first checkpoint's settings are an input of n500's evaluation.
    assert (first, check) in edges


def _field(block, key):
    m = re.search(rf"^\s+{key}: (.*)$", block, re.M)
    return m[1] if m else ""


def _jobs_of(out, rule):
    """Each planned job of `rule`: its wildcards, priority and resources."""
    blocks = re.split(r"^\[[^\]]+\]$", out.stdout, flags=re.M)
    return [
        {
            "wildcards": _field(b, "wildcards"),
            "priority": float(_field(b, "priority") or 0),
            "resources": _field(b, "resources"),
            "shell": b,
        }
        for b in blocks
        if re.search(rf"^(?:local)?rule {rule}:$", b, re.M)
    ]


def test_step_0_is_scored_before_any_checkpoint(tmp_path):
    # Every checkpoint's sidecar needs step 0's loss.
    out = dry_run(tmp_path, "finetune", config=OFFLOAD, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    (base,) = _jobs_of(out, "evaluate_base")
    assert base["priority"] > max(
        j["priority"] for j in _jobs_of(out, "evaluate_checkpoint")
    )


def test_finetune_never_publishes(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert not any(k.startswith("publish") for k in jobs)


def test_finished_checkpoints_are_kept(tmp_path):
    n250 = tmp_path / "runs" / "runa" / "train" / "n250"
    (n250 / "finetuned-ckpt").mkdir(parents=True)
    (n250 / "state").mkdir()
    (n250 / "settings.json").write_text("{}")
    (n250 / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (n250 / "state" / "state.json").write_text('{"done": 250}')
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 1


def _checkpoint(tmp_path, n):
    where = tmp_path / "runs" / "runa" / "train" / f"n{n}"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "state").mkdir()
    (where / "settings.json").write_text("{}")
    (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (where / "state" / "state.json").write_text(f'{{"done": {n}}}')


def test_the_live_run_switches_over_without_retraining(tmp_path):
    # The slots run: n250 to n750 saved without state of their
    # own, the trainer's state part way to n1000, and a seed list that
    # reaches 1,500 outbreaks so far.
    train = tmp_path / "runs" / "runa" / "train"
    for n in (250, 500, 750):
        where = train / f"n{n}"
        (where / "finetuned-ckpt").mkdir(parents=True)
        (where / "settings.json").write_text("{}")
        (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (train / "state").mkdir()
    (train / "state" / "state.json").write_text('{"done": 875}')
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("".join(f"{30100 + i}\n" for i in range(60)))
    config = {
        **FT,
        "finetune": {
            **FT["finetune"],
            "prefixes": [250, 500, 750, 1000, 2000],
            "outbreaks": 2000,
            "seed_file": str(seeds),
        },
    }
    jobs = dry(tmp_path, "finetune", config=config)
    # Only n1000: n2000 needs more seeds than the list has.
    assert jobs["train_segment"] == 1


def priorities(tmp_path, *targets, config=None):
    """Return each rule's job priorities in a dry run (0 when unset)."""
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
    out = subprocess.run(
        [str(SNAKEMAKE), *targets, "-s", "workflow/Snakefile", "-n",
         "--cores", "1", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    found = defaultdict(list)
    for block in re.split(r"\n(?=rule \w+:)", out.stdout):
        rule = re.match(r"rule (\w+):", block)
        if rule:
            got = re.search(r"priority: ([\d.]+)", block)
            found[rule[1]].append(float(got[1]) if got else 0)
    return found


def batch_priorities(tmp_path, *targets, config=None):
    """Return each simulate_batch job's priority by seed in a dry run."""
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
    out = subprocess.run(
        [str(SNAKEMAKE), *targets, "-s", "workflow/Snakefile", "-n",
         "--cores", "1", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    found = {}
    for block in re.split(r"\n(?=rule \w+:)", out.stdout):
        if block.startswith("rule simulate_batch:"):
            seed = int(re.search(r"seed=(\d+)", block)[1])
            found[seed] = float(re.search(r"priority: ([\d.]+)", block)[1])
    return found


def test_a_run_over_rounds_simulates_them_in_seed_order(tmp_path):
    # Training reads batches in seed order, so the first round must not
    # wait on batches of later rounds.
    got = batch_priorities(tmp_path, "simulate", config={"rounds": [4, 5]})
    assert len(got) == 80
    seeds = sorted(got)
    assert [got[s] for s in seeds] == sorted(got.values(), reverse=True)
    assert all(15 <= p < 16 for p in got.values())
    assert len(set(got.values())) == 80


def test_the_trainer_goes_first_then_simulation_then_scoring(tmp_path):
    # One pool per machine: when jobs compete for cores or the GPU, the
    # trainer (the run's critical path) goes first, then the simulations
    # its next segment trains on, then the rest.
    got = priorities(tmp_path, "finetune", config=FT)
    trainer = compute.TRAIN_PRIORITY
    assert all(trainer <= p < trainer + 1 for p in got["train_segment"])
    others = [p for r, ps in got.items() if r != "train_segment" for p in ps]
    assert max(others) < compute.HANDOFF_PRIORITY + 2
    # The first segment reads its 13 batches, a spare and the held-out
    # round's 40; the second segment's batches wait below them.
    batches = batch_priorities(tmp_path, "finetune", config=FT)
    first = {*range(30200, 30214), *range(30100, 30140)}
    nxt, later = compute.NEXT_BATCH_PRIORITY, compute.BATCH_PRIORITY
    assert all(nxt <= batches[s] < nxt + 1 for s in first)
    assert all(later <= batches[s] < later + 1 for s in set(batches) - first)
    assert set(batches) - first


def test_publish_verifies_then_releases_each_checkpoint(tmp_path):
    for n in (250, 500):
        _checkpoint(tmp_path, n)
    jobs = dry(tmp_path, "publish", config=FT)
    assert jobs["verify_checkpoint"] == 2
    assert jobs["publish_checkpoint"] == 2
    assert "train_segment" not in jobs


def test_verify_alone_never_releases(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(
        tmp_path,
        f"{tmp_path}/runs/runa/train/n250/verified.txt",
        config=FT,
    )
    assert jobs == {"verify_checkpoint": 1}


def test_score_computes_nothing_on_bdbv_held_out_for_the_paper(tmp_path):
    # BDBV is held out entirely: no BDBV unit,
    # zone unit or data refresh is a scoring job.
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert not {"score_bdbv", "score_zones", "bdbv_data"} & set(jobs)
    rules = (ROOT / "workflow" / "rules" / "score.smk").read_text()
    assert not re.search(
        r"^rule (zones|score_bdbv|score_zones|bdbv_data):", rules, re.M
    )


def test_score_runs_each_unit_for_a_checkpoint_and_the_reference(tmp_path):
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_germany"] == 3
    assert jobs["score_release"] == 2
    assert jobs["score_table"] == 1
    assert "land" not in jobs


def test_units_start_by_their_resources_alone(tmp_path):
    blocks = score_jobs(tmp_path, {"score": {"labels": ["aaa"]}})
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert units
    for block in units:
        for gone in ("wait_for_room", "nice -n"):
            assert gone not in block, gone


def test_units_run_the_shared_checkpoint_evaluation(tmp_path):
    blocks = score_jobs(tmp_path, {"score": {"labels": ["aaa"]}})
    units = [b for b in blocks if re.match(r"(local)?rule score_\w+:", b)]
    assert units
    for block in units:
        assert "scripts/checkpoint_scores.py" not in block
    shells = "".join(units)
    assert "-m nowcastgermany.checkpoint germany" in shells


def test_units_make_the_rows_folder_a_worker_brings_back(tmp_path):
    blocks = score_jobs(tmp_path, {"score": {"labels": ["aaa"]}})
    shells = "".join(blocks)
    # The folder the shared evaluation writes German rows to.
    from nowcastgermany.checkpoint import GERMANY_UNIT

    assert f"mkdir -p {tmp_path}/scores/{GERMANY_UNIT}/aaa/" in shells


def test_skipped_checkpoints_are_not_scored(tmp_path):
    config = {"score": {"labels": None, "skip": ["slots-aaa"]}}
    cache = tmp_path / "scores"
    cache.mkdir()
    at = "2026-11-20T12:00:00Z"
    (cache / "releases.json").write_text(
        f'[{{"tagName": "chronos-slots-aaa", "publishedAt": "{at}"}},'
        f' {{"tagName": "chronos-ft-bbb", "publishedAt": "{at}"}}]'
    )
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_release"] == 1
    assert jobs["score_germany"] == 2


def test_a_checkpoint_released_after_the_tables_is_scored(tmp_path):
    # With mtime triggers alone, a new label's units under tables already
    # built would be "missing, no update", and no tick would score them.
    _fetched(tmp_path)
    cache = tmp_path / "scores"
    for unit, labels in (
        ("germany-days", ["reference", "slots-aaa"]),
        ("heldout", ["slots-aaa"]),
    ):
        (cache / unit).mkdir(parents=True)
        for label in labels:
            (cache / unit / f"{label}.done").write_text("")
    at = "2026-11-20T12:00:00Z"
    (cache / "releases.json").write_text(
        f'[{{"tagName": "chronos-slots-aaa", "publishedAt": "{at}"}},'
        f' {{"tagName": "chronos-slots-bbb", "publishedAt": "{at}"}}]'
    )
    (cache / "tables").mkdir()
    for name in ("checkpoint-scores.csv", "checkpoints.csv"):
        (cache / "tables" / name).write_text("x\n")
    jobs = dry(tmp_path, "score")
    assert jobs["score_germany"] == 1
    assert jobs["score_release"] == 1


def test_score_takes_checkpoints_from_the_release_list(tmp_path):
    cache = tmp_path / "scores"
    cache.mkdir()
    (cache / "releases.json").write_text(
        '[{"tagName": "chronos-slots-aaa",'
        ' "publishedAt": "2026-11-20T12:00:00Z"}]'
    )
    jobs = dry(tmp_path, "score")
    assert jobs["score_release"] == 1
    assert "release_list" not in jobs


def _fetched(tmp_path):
    """Write the German triangle, fetched before any unit ran."""
    triangle = tmp_path / "germany" / "COVID-19_hospitalizations.csv"
    triangle.parent.mkdir(parents=True, exist_ok=True)
    triangle.write_text("x\n")


def test_finished_units_are_kept(tmp_path):
    _fetched(tmp_path)
    (tmp_path / "scores" / "germany-days").mkdir(parents=True)
    (tmp_path / "scores" / "germany-days" / "reference.done").write_text("")
    config = {"score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_germany"] == 1


def _local(tmp_path, *lines):
    (tmp_path / "local.env").write_text(
        "\n".join([f"ROOT={tmp_path}", "HOST_NAME=test-host", *lines]) + "\n"
    )


def test_publish_takes_only_checkpoints_already_saved(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "publish", config=FT)
    assert jobs["publish_checkpoint"] == 1
    assert "train_segment" not in jobs


def test_a_machine_can_publish_as_it_trains(tmp_path):
    _local(tmp_path, "PUBLISH=1")
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["publish_checkpoint"] == 2


def test_a_machine_runs_only_its_own_targets(tmp_path):
    _local(tmp_path, "TARGETS=score")
    config = {**FT, "score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "all", config=config)
    assert "train_segment" not in jobs
    assert "simulate_batch" not in jobs
    assert jobs["score_table"] == 1


def test_where_batches_come_from_is_the_machine_s(tmp_path):
    _local(tmp_path, "SIMS_FROM=release")
    jobs = dry(tmp_path, "finetune", config=FT)
    assert "simulate_batch" not in jobs


def test_at_most_two_simulations_run_at_once(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\n")
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            "simulate",
            "-n",
            "-p",
            "--cores",
            "8",
            "--workflow-profile",
            "workflow/profiles/default",
            "--config",
            "rounds=[4]",
        ],
        cwd=ROOT,
        env={
            "PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    assert "sim_slots=1" in out.stdout


def test_a_failed_run_posts_a_mailbox_line(tmp_path):
    # A real run of one local rule that fails: n800_check on settings
    # with no losses.
    box = tmp_path / "box.md"
    _local(tmp_path, "NOTIFY=1", f"MAILBOX={box}")
    n800 = tmp_path / "runs" / "runa" / "train" / "n800"
    n800.mkdir(parents=True)
    (n800 / "settings.json").write_text("{}")
    # Evaluated in the trainer, so the report needs only the checkpoint.
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("finetune:\n  eval: local\ndashboard:\n  serve: false\n")
    target = tmp_path / "runs" / "runa" / "reports" / "n800-check.txt"
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            str(target),
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
            "--configfile",
            str(overlay),
        ],
        cwd=ROOT,
        env={
            "PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(tmp_path / "local.env"),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    line = box.read_text()
    assert "FAILED" in line
    assert ".snakemake/log/" in line


def test_the_check_follows_the_configured_prefix(tmp_path):
    config = {
        **FT,
        "finetune": {**FT["finetune"], "check_at": 500},
    }
    jobs = dry(tmp_path, "finetune", config=config)
    assert jobs["checkpoint_check"] == 1


def test_the_dashboard_render_is_its_own_target(tmp_path):
    assert dry(tmp_path, "dashboard") == {"dashboard": 1}
    jobs = dry(tmp_path, "all", config={**FT, "score": {"labels": ["aaa"]}})
    assert "dashboard" not in jobs


def _scored(tmp_path):
    import os
    import time

    cache = tmp_path / "scores"
    for study in ("germany-days", "heldout"):
        (cache / study).mkdir(parents=True, exist_ok=True)
    stamps = [
        cache / "germany-days" / "reference.done",
        cache / "germany-days" / "aaa.done",
        cache / "heldout" / "aaa.done",
    ]
    for f in stamps:
        f.write_text("")
    (cache / "tables").mkdir()
    for f in ("checkpoint-scores.csv", "checkpoints.csv", "landed"):
        (cache / "tables" / f).write_text("")
    old = time.time() - 3600
    for f in stamps:
        os.utime(f, (old + 60, old + 60))
    return {
        "score": {"labels": ["aaa"]},
    }, stamps


def test_land_runs_only_when_the_tables_changed(tmp_path):
    config, stamps = _scored(tmp_path)
    assert dry(tmp_path, "land", config=config) == {}
    stamps[1].write_text("")  # a unit finished again: newer than the table
    jobs = dry(tmp_path, "land", config=config)
    assert jobs["score_table"] == 1
    assert jobs["land"] == 1


def test_the_rounds_host_s_manifests_and_qc_are_mirrored(tmp_path):
    # The rounds host is this machine here (REMOTE empty), its rounds in
    # REMOTE_ROOT: round 1 whole, round 2 so far one batch.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: [30200]})
    dry(tmp_path, "finetune", config=REMOTE)
    run = tmp_path / "runs" / "runa"
    listed = (run / "available-batches.txt").read_text().split()
    assert listed == [str(s) for s in [*range(30100, 30140), 30200]]
    assert (run / "rounds" / "r1" / "qc" / "summary.txt").exists()
    assert (run / "rounds" / "r2" / "manifest.csv").exists()


NOBIAS = {
    **FT,
    "runs": {"nobias": {"series_bias": False}},
    "train_runs": ["nobias"],
}


def test_a_no_bias_control_is_a_run_in_its_own_folder(tmp_path):
    jobs = dry(tmp_path, "finetune", config=NOBIAS)
    assert jobs["train_segment"] == 2
    out = dry_run(tmp_path, "finetune", config=NOBIAS, quiet=False)
    assert f"{tmp_path}/runs/nobias/train/n250/settings.json" in out.stdout
    assert "--series-bias" not in out.stdout


def test_a_run_trains_the_next_segment_beside_the_scoring(tmp_path):
    # n500 trains once n250 is saved, while n250 is scored, not a run
    # later.
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 2
    assert jobs["score_run_germany"] == 2


def test_a_run_after_n250_trains_n500(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 1


def _rounds_host(tmp_path, rounds):
    """A rounds host on this machine with `rounds` {round: seeds} listed."""
    host = tmp_path / "host"
    for r, seeds in rounds.items():
        where = host / "rounds" / f"r{r}"
        (where / "qc").mkdir(parents=True)
        rows = "".join(
            f"{r},{s},block,r7,c,1,20,a,b,,1,2.0,h,t\n" for s in seeds
        )
        (where / "manifest.csv").write_text("header\n" + rows)
        (where / "qc" / "summary.txt").write_text("sim QC summarised\n")
    local = tmp_path / "local.env"
    local.write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nREMOTE_ROOT={host}\n"
    )
    return local


def test_a_fresh_listing_keeps_the_checkpoints_saved(tmp_path):
    # Each run lists the batches anew; the newer list must not make a
    # saved (protected) checkpoint look outdated.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30240)})
    _checkpoint(tmp_path, 250)
    dry(tmp_path, "finetune", config=REMOTE)
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 1


def test_each_run_lists_the_batches_afresh(tmp_path):
    # A list written by an earlier run, when round 2 had 13 batches, must
    # not cap a later run once the rounds host has the whole round.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30240)})
    _stale_list(tmp_path, [*range(30100, 30140), *range(30200, 30213)])
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 2


def test_a_no_bias_config_still_simulates_in_the_default_folder(tmp_path):
    # The guard is the trainer's: other targets run as before.
    jobs = dry(tmp_path, "simulate", config={**NOBIAS, "rounds": [4]})
    assert jobs["simulate_batch"] == 40


def test_an_unreachable_rounds_host_is_said(tmp_path):
    # A run that cannot copy a round's manifest says so, not silence.
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\n"
        f"REMOTE_ROOT={tmp_path}/no-such-host\n"
    )
    out = dry_run(tmp_path, "finetune", config=REMOTE)
    assert "manifest.csv" in out.stderr
    assert "could not copy" in out.stderr


def score_jobs(tmp_path, config, local_extra="", target="score"):
    """Return each score rule's shell command, priority and resources."""
    local = tmp_path / "local.env"
    local.write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nDEVICE=cuda\n{local_extra}"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config))
    out = subprocess.run(
        [str(SNAKEMAKE), target, "-s", "workflow/Snakefile", "-n", "-p",
         "--cores", "8", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    return re.split(r"\n(?=(?:local)?rule \w+:)", out.stdout)


def test_score_units_run_torch_beside_the_trainer_on_the_cpu(tmp_path):
    config = {"score": {"labels": ["aaa", "bbb"]}}
    blocks = score_jobs(tmp_path, config)
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert len(units) == 3
    for block in units:
        assert "DEVICE=cpu" in block
        assert "OMP_NUM_THREADS=1 " not in block
        assert "CUDA_VISIBLE_DEVICES" not in block
        assert not re.search(r"resources: .*gpu=1", block)
        # A few worker processes, not the machine: simulations keep
        # their cores. Each worker takes one thread.
        workers = config_score_workers()
        assert re.search(rf"threads: {workers}\b", block)
        assert f"SCORE_WORKERS={workers} " in block
        assert f"OMP_NUM_THREADS={workers} " in block
        assert "--xla_cpu_multi_thread_eigen=false" in block


def config_score_workers() -> int:
    """Return the config's German unit workers (`score.workers`)."""
    return int(yaml.safe_load(CONFIG.read_text())["score"]["workers"])


def test_score_workers_are_a_machine_setting(tmp_path):
    # A unit's threads are its worker processes, which the config sets and
    # a machine may override; its memory is each worker's.
    config = {"score": {"labels": ["aaa"]}}
    blocks = score_jobs(tmp_path, config, "SCORE_WORKERS=3\n")
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert units
    for block in units:
        assert re.search(r"threads: 3\b", block)
        assert "SCORE_WORKERS=3 " in block
        mem = int(re.search(r"mem_mb=(\d+)", block)[1])
        assert mem >= 3 * 3072


def test_the_reference_then_newer_checkpoints_are_scored_first(tmp_path):
    # The reference every checkpoint is compared with comes first.
    config = {"score": {"labels": ["aaa", "bbb"]}}
    blocks = score_jobs(tmp_path, config)
    got = {}
    for block in blocks:
        rule = re.match(r"rule (score_germany):", block)
        if rule:
            label = re.search(r"wildcards: label=(\S+)", block)[1]
            got[label] = float(re.search(r"priority: ([\d.]+)", block)[1])
    assert got["reference"] > got["bbb"] > got["aaa"]


def test_one_unit_runs_before_the_release_list_exists(tmp_path):
    # A unit's priority reads the list if it is there, else takes its band's
    # base: it never waits on the release_list checkpoint.
    jobs = dry(tmp_path, str(tmp_path / "scores" / "germany-days" / "aaa.done"))
    assert jobs["score_germany"] == 1


# The grid's rounds, never an earlier version's (flowlib.grids).


def test_release_of_a_retired_round_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="round 9 is an earlier grid's"):
        dry(tmp_path, "release", config={"rounds": [9]})


def test_the_grid_simulates_its_held_out_round_33(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [33]})
    assert jobs["simulate_batch"] == 40


def test_the_sim_qc_runs_with_no_thresholds(tmp_path):
    blocks = score_jobs(tmp_path, {"rounds": [4]}, target="simulate")
    qc = next(b for b in blocks if re.match(r"(local)?rule sim_qc:", b))
    assert "uv run sim-qc" in qc
    assert "--max-" not in qc


def _done_round(root, r, batches=40):
    """A simulated, QC'd round on disk, its files' times in rule order."""
    import os

    d = root / "rounds" / f"r{r}"
    first = 30000 + 100 * r
    for sub in ("sims", "records", "qc"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    stages = [
        [
            d / "sims" / f"counts-{s}.csv.gz"
            for s in range(first, first + batches)
        ]
        + [
            d / "sims" / f"settings-{s}.csv"
            for s in range(first, first + batches)
        ],
        [
            d / "records" / f"record-{s}.csv"
            for s in range(first, first + batches)
        ],
        [d / "manifest.csv"],
        [d / "qc" / "summary.txt"],
        [d / "seeds.txt", d / "READY"],
    ]
    for k, files in enumerate(stages):
        for f in files:
            if f.name.startswith("settings-"):
                f.write_text("outbreak\n")
            else:
                f.write_text("x\n")
            os.utime(f, (1_000_000 + 10 * k, 1_000_000 + 10 * k))
    return 1_000_000 + 10 * len(stages)


def test_a_round_s_batch_runs_the_grid(tmp_path):
    blocks = score_jobs(tmp_path, {"rounds": [4]}, target="simulate")
    text = " ".join("".join(blocks).replace("\\\n", " ").split())
    shell = text[text.index("-m nowcastfoundation.simulator") - 400 :]
    assert "ENW_CELLS" not in shell
    assert (
        "nowcastfoundation.simulator "
        "packages/nowcastfoundation/config/finetune/grid.yaml 20 " in shell
    )


def test_grid_figures_summarise_each_round_then_compare_them(tmp_path):
    """The grid summaries and figures are workflow targets, per round."""
    blocks = score_jobs(tmp_path, {"rounds": [4, 5]}, target="grid_figures")
    text = " ".join("".join(blocks).replace("\\\n", " ").split())
    rounds = tmp_path / "rounds"
    for r in (4, 5):
        assert (
            f"grid_summaries.py outbreaks {rounds}/r{r}/sims "
            f"{rounds}/r{r}/summaries/outbreaks.csv"
        ) in text
    assert f"r4={rounds}/r4/summaries/outbreaks.csv" in text
    assert "grid_figures.py" in text
    # From the batch files: no manifest or records of the round.
    assert "rule round_manifest" not in "".join(blocks)
    assert "rule batch_record" not in "".join(blocks)


def test_a_fetched_batch_is_a_copy_not_a_link_to_its_source(tmp_path):
    # Snakemake touches a job's outputs one after another; through a hard
    # link that moves its inputs' times too, and a clock tick between the
    # two touches leaves the counts older than the settings input, which
    # fails the job.
    import os
    import time

    sims = tmp_path / "rounds" / "r33" / "sims"
    sims.mkdir(parents=True)
    counts = sims / "counts-33300.csv.gz"
    counts.write_bytes(gzip.compress(b"x\n"))
    settings = sims / "settings-33300.csv"
    settings.write_text("x\n")
    now = time.time_ns()
    os.utime(counts, ns=(now, now))
    os.utime(settings, ns=(now + 10**6, now + 10**6))
    run = tmp_path / "runs" / "runa"
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nSIMS_FROM=local\n"
    )
    out = dry_run(
        tmp_path,
        str(run / "sims" / "counts-33300.csv.gz"),
        config=FT,
        extra=("--allowed-rules", "fetch_batch"),
        dry=False,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    for name, source in (
        ("counts-33300.csv.gz", counts),
        ("settings-33300.csv", settings),
    ):
        fetched = run / "sims" / name
        assert fetched.read_bytes() == source.read_bytes()
        assert fetched.stat().st_ino != source.stat().st_ino


def test_sim_reports_render_each_round_and_the_rounds_together(tmp_path):
    """The simulation report is a workflow target, per round and over all."""
    blocks = score_jobs(tmp_path, {"rounds": [4, 5]}, target="sim_reports")
    text = " ".join("".join(blocks).replace("\\\n", " ").split())
    rounds = tmp_path / "rounds"
    assert f"nowcastgermany.cli fetch --out {tmp_path}/germany" in text
    triangle = tmp_path / "germany" / "COVID-19_hospitalizations.csv"
    assert (
        f"sim_report.py germany {triangle}"
        f" 2022-02-10 {rounds}/report/germany-dev.csv"
    ) in text
    for r in (4, 5):
        assert (
            f"sim_report.py render {rounds}/r{r}/report/sim-report.html"
        ) in text
        assert (
            f"r{r}={rounds}/r{r}/sims:{rounds}/r{r}/summaries/outbreaks.csv"
            in text
        )
    assert f"sim_report.py render {rounds}/report/sim-report.html" in text
    assert "'Simulated outbreaks, rounds 4, 5'" in text
    # From the batch files and the German triangle only: nothing of the
    # rounds' records, and no BDBV data.
    joined = "".join(blocks)
    assert "rule batch_record" not in joined
    assert "bdbvhub.cli data" not in joined


def test_the_comparison_scores_every_run_s_checkpoints(tmp_path):
    out = dry_run(
        tmp_path, f"{tmp_path}/runs/comparison.csv", config=FT, quiet=False
    )
    assert out.returncode == 0, out.stderr + out.stdout
    assert "rule comparison:" in out.stdout
    assert "rule run_table:" in out.stdout
    assert f"{tmp_path}/runs/runa/reports/checkpoint-scores.csv" in out.stdout


@pytest.mark.parametrize(
    ("table", "report"),
    [
        ("comparison.csv", "checkpoint-scores.csv"),
        ("compare-daily.csv", "compare-daily.csv"),
    ],
)
def test_the_comparisons_ask_only_for_the_paper_s_runs(tmp_path, table, report):
    # The paper's live run and the runs it is compared with, not every run
    # the workflow config names.
    # The overlay merges into the paper config's comparisons, so each names
    # the same run here.
    pairs = dict.fromkeys(("slots", "delays", "realdata"), "noslots")
    config = {**FT, "live_run": "runa", "comparisons": pairs}
    out = dry_run(
        tmp_path, f"{tmp_path}/runs/{table}", config=config, quiet=False
    )
    assert out.returncode == 0, out.stderr + out.stdout
    for run in ("runa", "noslots"):
        assert f"{tmp_path}/runs/{run}/reports/{report}" in out.stdout
    for run in ("nodelays", "realdata", "runb"):
        assert f"{tmp_path}/runs/{run}/reports/{report}" not in out.stdout


def test_a_run_s_block_combinations_come_from_its_final_checkpoint(tmp_path):
    target = f"{tmp_path}/runs/runa/train/block-combinations.csv"
    out = dry_run(tmp_path, target, config=FT, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "rule block_combinations:" in out.stdout
    assert f"{tmp_path}/runs/runa/train/n500/settings.json" in out.stdout


def test_the_daily_comparison_scores_every_run_s_checkpoints(tmp_path):
    # The paper's `compare_daily`: each run's checkpoints on both German
    # views, from the run's own German units beside the reference.
    out = dry_run(
        tmp_path, f"{tmp_path}/runs/compare-daily.csv", config=FT, quiet=False
    )
    assert out.returncode == 0, out.stderr + out.stdout
    assert "rule compare_daily:" in out.stdout
    assert "rule daily_table:" in out.stdout
    assert f"{tmp_path}/runs/runa/reports/compare-daily.csv" in out.stdout


def test_an_interim_checkpoint_verifies_then_releases(tmp_path):
    # The hub's fine-tuned model is an interim
    # checkpoint (`finetune.interim`), released through the same rules.
    where = tmp_path / "runs" / "runa" / "train" / "interim" / "s48100"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "settings.json").write_text("{}")
    (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    jobs = dry(tmp_path, f"{where}/published.json", config=FT)
    assert jobs == {"verify_checkpoint": 1, "publish_checkpoint": 1}


def test_rt_check_is_one_job_per_model_and_origin(tmp_path):
    out = tmp_path / "rt"
    jobs = dry(
        tmp_path,
        f"{out}/renewalbvd-2026-11-22.html",
        config={"rt_check": {"store": "s", "data": "d", "out_dir": str(out)}},
    )
    assert jobs == {"rt_check": 1}


def test_each_checkpoint_gets_its_7_day_check(tmp_path):
    # The 7-day check is a workflow rule per checkpoint,
    # on the held-out blocks, whichever way the run evaluates.
    where = tmp_path / "runs" / "runa" / "train" / "n250"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "settings.json").write_text("{}")
    (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    jobs = dry(tmp_path, f"{where}/seven-day.json", config=FT)
    assert jobs == {"seven_day_check": 1}


def test_a_new_machine_fetches_the_german_triangle_before_scoring(tmp_path):
    # The German units read the fetched triangle, so with none yet the
    # fetch runs first.
    config = {"score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["germany_triangle"] == 1
    assert jobs["score_germany"] >= 1


def test_publish_hf_is_a_dry_run_until_a_repository_is_set(tmp_path):
    # `publish_hf` is opt-in and uploads only with a
    # repository in packages/nowcastfoundation/config/finetune.toml and dry_run
    # off; the repository
    # is empty until one is set, so it writes the folder only.
    where = tmp_path / "runs" / "runa" / "train" / "interim" / "s48100"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "settings.json").write_text("{}")
    (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    out = dry_run(tmp_path, f"{where}/huggingface.json", config=FT, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "rule publish_hf:" in out.stdout
    assert "rule verify_checkpoint:" in out.stdout
    (command,) = [
        line for line in out.stdout.splitlines() if "publish-hf" in line
    ]
    assert "--dry-run" in command
    assert "--repo" not in command


def test_a_scored_step_0_is_kept(tmp_path):
    # runa's step 0 was scored once on the laptop and copied in: a later
    # run never scores it again, the evaluator's code key and the
    # held-out set being older than it.
    import os

    for n in (250, 500):
        _checkpoint(tmp_path, n)
    train = tmp_path / "runs" / "runa" / "train"
    (train / "held-out-blocks.json").write_text('{"outbreaks": []}')
    (train / "evals").mkdir(exist_ok=True)
    step0 = train / "evals" / "step-0000000.json"
    step0.write_text('{"step": 0, "held_out_loss": 0.5}')
    for path in (
        train / "held-out-blocks.json",
        *train.glob("n*/settings.json"),
    ):
        os.utime(path, (1, 1))
    jobs = dry(tmp_path, "finetune", config=OFFLOAD)
    assert "evaluate_base" not in jobs
    assert jobs["evaluate_checkpoint"] == 2


def test_a_job_asks_for_its_last_peak(tmp_path):
    # Its benchmark's max_rss with a margin, not a fixed figure.
    from flowlib import memory

    for n in (250, 500):
        _checkpoint(tmp_path, n)
    bench = tmp_path / "runs/runa/train/benchmarks/seven_day_check/n250.tsv"
    bench.parent.mkdir(parents=True)
    bench.write_text("s\th:m:s\tmax_rss\n1\t0:00:01\t9000\n")
    out = dry_run(tmp_path, "finetune", config=OFFLOAD, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = [j for j in _jobs_of(out, "seven_day_check") if "n=250" in str(j)]
    assert jobs
    assert f"mem_mb={memory.padded(9000)}" in str(jobs[0])
