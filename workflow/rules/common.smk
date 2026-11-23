# Shared settings and helper functions of the workflow's rules, kept
# apart from the rules themselves (rules/*.smk). Machine settings come
# from config/local.env (flowlib.localconfig).

import functools
import json
import tomllib

from flowlib import (
    compute,
    evaluations,
    grids,
    runs,
    hosts,
    localconfig,
    manifest,
    notify,
    releases,
    scoring,
    segments,
    simulator,
)

from nowcastfoundation.sidecar import HELD_OUT_BLOCKS, SIDECAR

LOCAL = localconfig.read()
HOST = LOCAL["HOST_NAME"]
ROUNDS_DIR = f"{LOCAL['ROOT']}/rounds"
TARGETS = LOCAL["TARGETS"].split()
# The tools the shell rules call (envs/tools.yaml).
TOOLS_ENV = "../envs/tools.yaml"


def say(message):
    """A mailbox line, when NOTIFY is set in config/local.env."""
    notify.post(message, LOCAL)


def job_prefix(mem_mb):
    """The machine's command prefix for a job capped at `mem_mb` MB."""
    return LOCAL["JOB_PREFIX"].replace("{mem}", f"{int(mem_mb)}M")


# From simulate.smk.

SIM = config["simulate"]
REMOTE = LOCAL["REMOTE"]
REMOTE_ROUNDS = f"{LOCAL['REMOTE_ROOT'] or LOCAL['ROOT']}/rounds"


def seeds_of(round_):
    """A round's batch seeds, its grid block."""
    return segments.round_seeds(int(round_), SIM["seed_base"], SIM["batches"])


MINE = hosts.seeds(LOCAL["SIM_SEEDS"], seeds_of)


def here(seed):
    """Whether this host simulates batch `seed`."""
    return hosts.simulates(int(seed), MINE)


# The grid the simulator runs (its factors and sampling distributions).
GRID_FILE = SIM["grid_file"]


def batch_priority(wc):
    """A batch's priority: 15, earlier seeds a fraction higher."""
    return segments.batch_priority(int(wc.seed), 15, SIM["seed_base"])


def batch_files(wc):
    """A batch's files: simulated here, or pushed by its host."""
    if here(wc.seed):
        return {
            "counts": rules.simulate_batch.output.counts.format(**wc),
            "settings": rules.simulate_batch.output.settings.format(**wc),
        }
    return {"arrived": rules.await_batch.output[0].format(**wc)}


def grid_id():
    """The grid file the simulator runs, as its manifest identifier."""
    return simulator.grid_id(GRID_FILE)


def round_records(wc):
    """A round's batch manifest rows."""
    return [
        f"{ROUNDS_DIR}/r{wc.round}/records/record-{s}.csv" for s in seeds_of(wc.round)
    ]


def round_sims(wc):
    """A round's batches' counts, settings and latent files."""
    kinds = (("counts", "csv.gz"), ("settings", "csv"), ("latent", "csv.gz"))
    return [
        f"{ROUNDS_DIR}/r{wc.round}/sims/{kind}-{seed}.{ext}"
        for seed in seeds_of(wc.round)
        for kind, ext in kinds
    ]


def release_owner(wc):
    """Which grid round `wc.round`'s release belongs to (flowlib.grids)."""
    return grids.owner(
        int(wc.round), config["grid_rounds"], config.get("retired_rounds")
    )


def release_check(wc):
    """Raise unless the rounds asked for may be released (flowlib.grids)."""
    retired = config.get("retired_rounds")
    return grids.check_release(config["rounds"], config["grid_rounds"], retired) or []


def round_sources(wc, input):
    """`r<N>=<file>` for each round in `rounds` and its file."""
    return " ".join(f"r{r}={path}" for r, path in zip(config["rounds"], input))


def summary_sources(rounds, summaries):
    """`r<N>=<round's sims>:<summary>` for each round and its summary."""
    return " ".join(
        f"r{r}={ROUNDS_DIR}/r{r}/sims:{path}" for r, path in zip(rounds, summaries)
    )


# From finetune.smk.

# The fine-tuning runs (`runs` in workflow/config.yaml, flowlib.runs):
# each run's rules take the {run} wildcard, and its files are in
# <ROOT>/runs/<run>/.
RUNS = runs.resolve(config, LOCAL)
RUNS_DIR = f"{LOCAL['ROOT']}/runs"
RUN = f"{RUNS_DIR}/{{run}}"
TRAIN = f"{RUN}/train"


wildcard_constraints:
    run="[^/]+",


def trained_runs():
    """The runs `finetune` trains."""
    return runs.trained(config, RUNS)


def run_of(wc):
    """A job's run."""
    return RUNS[wc.run]


def ft(key):
    """A param: the job's run's `finetune` setting `key`."""
    return lambda wc: RUNS[wc.run].ft[key]


def local_or_ft(key, setting):
    """A param: this machine's `key` in local.env if set, else the run's
    `finetune` setting."""
    return lambda wc: LOCAL.get(key) or RUNS[wc.run].ft[setting]


# Files publishing reads beside a segment's outputs, copied back with
# them from a worker elsewhere (flowlib.compute).
SEGMENT_COPY_BACK = (
    "outbreaks.txt",
    "consumed.txt",
    "finetuned-ckpt/adapter_config.json",
)


def segment_copy_back(wc):
    return [f"{train_dir(wc.run)}/n{wc.n}/{name}" for name in SEGMENT_COPY_BACK]


def train_dir(run):
    return RUNS[run].train


# From evaluate.smk.

EVALUATE_PY = "uv run --group chronos python workflow/scripts/evaluate_offloaded.py"
EVALUATE_PRIORITY = 15
TRAIN_PRIORITY = compute.TRAIN_PRIORITY


def evaluate_prev(wc):
    """The settings of the checkpoint before segment n, if any."""
    prev = run_of(wc).prev[int(wc.n)]
    return f"{train_dir(wc.run)}/n{prev}/settings.json" if prev else []


def first_prefix(run):
    """The run's first checkpoint: the one with no previous."""
    return next(n for n, p in RUNS[run].prev.items() if p is None)


def evaluate_prev_dir(input):
    """The previous checkpoint's folder, if it has one."""
    return Path(input.prev).parent if input.prev else None


def evaluate_copy_in(wc):
    """The run's step evaluations and held-out blocks, for a worker."""
    return [f"{train_dir(wc.run)}/evals/", f"{train_dir(wc.run)}/{HELD_OUT_BLOCKS}"]


def earlier_settings(wc):
    """The settings of the run's checkpoints before n."""
    train, n = train_dir(wc.run), int(wc.n)
    return [f"{train}/n{p}/settings.json" for p in run_of(wc).ft["prefixes"] if p < n]


def evaluate_inputs_from(wc):
    """Files a checkpoint's evaluation reads that no rule declares."""
    r, n, train = run_of(wc), int(wc.n), train_dir(wc.run)
    return [
        f"{train}/n{n}/finetuned-ckpt/",
        f"{train}/{HELD_OUT_BLOCKS}",
        *(f"{train}/n{p}/{SIDECAR}" for p in r.ft["prefixes"] if p < n),
    ]


def offloaded(run, n):
    """Checkpoint n's held-out sidecar and its segment's step evaluations,
    when the run offloads evaluation."""
    if RUNS[run].ft["eval"] != "offload":
        return []
    train = train_dir(run)
    return [f"{train}/n{n}/{SIDECAR}", f"{train}/evals/n{n}.done"]


def round_of(seed):
    return (int(seed) - SIM["seed_base"]) // 100


def heldout_curve_inputs(wc):
    """Each of the run's checkpoints, with its held-out losses."""
    train, ns = train_dir(wc.run), RUNS[wc.run].ft["prefixes"]
    return [f"{train}/n{n}/settings.json" for n in ns] + [
        f for n in ns for f in offloaded(wc.run, n)
    ]


def training_seeds(run):
    """The run's training seeds, in the order it trains on them."""
    f = RUNS[run].ft
    return segments.training_seeds(f["seed_file"], f["rounds"], seeds_of)


def held_seeds(run):
    return seeds_of(RUNS[run].ft["validation_round"])


def held_range(run):
    """The run's held-out seeds, as the trainer takes them: first last."""
    seeds = held_seeds(run)
    return f"{seeds[0]} {seeds[-1]}"


def available_seeds(run):
    """The training seeds whose batches can be had now, in order."""
    listed = listed_batches(run) if RUNS[run].elsewhere else None
    return segments.available_seeds(training_seeds(run), held_seeds(run), listed)


def outbreaks_or_full(run, seed):
    """A batch's outbreaks if simulated, else a full batch."""
    return segments.outbreaks_or_full(
        [
            f"{RUNS[run].dir}/sims/settings-{seed}.csv",
            f"{ROUNDS_DIR}/r{round_of(seed)}/sims/settings-{seed}.csv",
        ],
        SIM["batch"],
    )


def segment_seeds(run, n):
    """The seeds segment n needs: n outbreaks, one spare, the held-out."""
    return segments.segment_seeds(
        n,
        available_seeds(run),
        held_seeds(run),
        lambda s: outbreaks_or_full(run, s),
    )


def segment_inputs(wc):
    run, n = wc.run, int(wc.n)
    there, train = RUNS[run], train_dir(run)
    # A finished checkpoint is final: its batches are not gathered again,
    # so a machine without them never retrains it.
    if Path(f"{train}/n{n}/settings.json").exists():
        return {}
    out = {
        "sims": expand(
            f"{there.dir}/sims/settings-{{seed}}.csv", seed=segment_seeds(run, n)
        ),
        "seeds": f"{there.dir}/seeds.txt",
    }
    if there.prev[n] is not None:
        out["prev"] = f"{train}/n{there.prev[n]}/settings.json"
    return out


def fetch_source(wc):
    if RUNS[wc.run].ft["sims_from"] != "local":
        return []
    r = round_of(wc.seed)
    return {
        "counts": f"{ROUNDS_DIR}/r{r}/sims/counts-{wc.seed}.csv.gz",
        "settings": f"{ROUNDS_DIR}/r{r}/sims/settings-{wc.seed}.csv",
    }


def saved(run, n):
    """Whether checkpoint n of the run is saved."""
    return Path(f"{train_dir(run)}/n{n}/settings.json").exists()


def trainable(run):
    """The prefixes done or within reach of the seed list."""
    f = RUNS[run].ft
    return segments.trainable(
        f["prefixes"],
        f["outbreaks"],
        SIM["batch"],
        available_seeds(run),
        lambda n: saved(run, n),
        lambda s: outbreaks_or_full(run, s),
    )


def to_train(run):
    """Prefixes `finetune` asks for of the run."""
    return segments.to_train(
        trainable(run), lambda n: saved(run, n), config.get("next_segment_only")
    )


def finetune_inputs(wc):
    """Every trained run's checkpoints, held-out losses and check."""
    out = []
    for run in trained_runs():
        there, train, ns = RUNS[run], train_dir(run), to_train(run)
        out += [f"{train}/n{n}/settings.json" for n in ns]
        # An offloaded run's held-out losses, scored after each segment.
        out += [f for n in ns for f in offloaded(run, n)]
        # A machine that publishes as it trains (PUBLISH in local.env).
        if LOCAL["PUBLISH"]:
            out += [f"{train}/n{n}/published.json" for n in ns]
        # The check once its checkpoint is asked for, so one segment a run
        # does not train on to it.
        if there.check in ns:
            out.append(f"{there.dir}/reports/n{there.check}-check.txt")
    return out


def published_checkpoints(wc):
    """Every trained run's saved checkpoints' release records."""
    return [
        f"{train_dir(run)}/n{n}/published.json"
        for run in trained_runs()
        for n in RUNS[run].ft["prefixes"]
        if saved(run, n)
    ]


# From score.smk.

SC = config["score"]
CACHE = LOCAL["SCORE_CACHE"]


def checkpoint_labels(wc=None):
    """The checkpoints to score, less `score.skip`."""
    if SC["labels"] is not None:
        return list(SC["labels"])
    return releases.chosen(checkpoints.release_list.get().output[0], SC)


ROOM = (
    f"workflow/scripts/wait_for_room.sh {LOCAL['MIN_FREE_GB']} "
    f"{LOCAL['MIN_DISK_GB']} {CACHE} {LOCAL['MAX_LOAD']} && "
)


# The CPU threads of a job beside the trainer (EVAL_THREADS, default 2):
# a few, so simulations keep the rest of the machine's cores.
UNIT_THREADS = int(LOCAL.get("EVAL_THREADS") or 2)


def score_order():
    """The checkpoint labels, oldest published first (none before listed)."""
    return releases.oldest_first(f"{CACHE}/releases.json", SC)


def unit_priority(base):
    """A unit's priority: `base`, newer checkpoints a fraction higher."""
    return lambda wc: releases.priority(score_order(), wc.label, base)


EVAL_DEVICE = machine.side_device(LOCAL)
"""The device evaluation and scoring run torch on (EVAL_DEVICE in
local.env), the CPU by default: the GPU is the trainer's."""


def newest_first(base):
    """A job's priority beside the trainer: `base`, newer segments higher."""
    return lambda wc: compute.side_priority(base, int(wc.n))


def unit_env(wc, threads):
    """The unit's torch device, threads and Chronos-2 batch."""
    return machine.shell_env(machine.torch_env(LOCAL, threads, EVAL_DEVICE))


GERMANY_UNIT = "germany-days"
"""The German units' cache folder, as `evaluation.checkpoint.GERMANY_UNIT`."""

EVALUATE = scoring.EVALUATE


def unit_rows(study):
    """The unit's rows folder, which a worker brings back with its output."""
    return lambda wc: f"{CACHE}/{study}/{wc.label}/"


def unit_shell(study, segment=""):
    """A unit's shell command (flowlib.scoring.unit_shell)."""
    return scoring.unit_shell(study, ROOM, segment)


def run_scores_dir(run):
    """The run's score cache: its own checkpoints' units."""
    return f"{RUNS_DIR}/{run}/scores"


def saved_prefixes(run):
    """The run's saved checkpoints, by outbreaks."""
    return [n for n in RUNS[run].ft["prefixes"] if saved(run, n)]


def run_units(run):
    """The German units of the run's saved checkpoints."""
    return [
        f"{run_scores_dir(run)}/{GERMANY_UNIT}/n{n}.done" for n in saved_prefixes(run)
    ]


def references():
    """The reference labels' German units, shared by every run."""
    return expand(f"{CACHE}/{GERMANY_UNIT}/{{label}}.done", label=SC["reference"])


def report_names(wc):
    """The German report's LABEL=NAME arguments and reference folders."""
    return scoring.report_args(
        wc.run, saved_prefixes(wc.run), SC["reference"], f"{CACHE}/{GERMANY_UNIT}"
    )


def run_scores(wc=None):
    """Each trained run's German report, over its saved checkpoints."""
    return [f"{RUNS_DIR}/{run}/reports/german/index.html" for run in trained_runs()]


def table_inputs(wc):
    labels = checkpoint_labels()
    every = [*SC["reference"], *labels]
    return {
        "germany": expand(f"{CACHE}/{GERMANY_UNIT}/{{label}}.done", label=every),
        "release": expand(f"{CACHE}/heldout/{{label}}.done", label=labels),
    }


def unit_stamps(wc=None):
    """Every unit the tables read, as one list (`rule score` asks for them
    itself, so a new label's units run under tables already built)."""
    return [f for files in table_inputs(wc).values() for f in files]


# The longer run directives' steps.


def write_record(input, output, params, wildcards):
    """Write a batch's manifest row, kept if the manifest has it."""
    manifest.record(
        output[0], params.old, params.sims, round_=int(wildcards.round),
        seed=int(wildcards.seed), mine=here(wildcards.seed), **batch_fields(),
    )  # fmt: skip


def batch_fields():
    """The manifest row fields a batch takes from this host and grid."""
    return {
        "grid": grid_id(),
        "draws": SIM["draws"],
        "batch": SIM["batch"],
        "host": HOST,
    }


def mark_ready(input, output, params, wildcards):
    """Write a round's seed list and READY, and say so."""
    seeds = seeds_of(wildcards.round)
    manifest.mark_ready(seeds, output.seeds, output.ready)
    qc = Path(input[0]).read_text().strip()
    say(f"round {wildcards.round} ready: {len(seeds)} batches, one grid block; {qc}")


def push_record(input, output, params, wildcards):
    """Send a helping host's batch and its manifest row to the rounds host."""
    s = int(wildcards.seed)
    row = manifest.new_row(
        params.sims, round_=int(wildcards.round), seed=s, **batch_fields()
    )
    hosts.push(params.sims, s, params.there, row, REMOTE)


@functools.cache
def listed_batches(run):
    """The batches that can be had now, listed once per Snakemake run.

    Listed when the run plans, not kept in a file, so a later run sees
    rounds finished since: the rounds host's whole rounds, mirrored in
    the run's `rounds/` (flowlib.hosts.mirrored_seeds), else the
    `sims-r<N>` releases' batches. Written to
    <run>/available-batches.txt for the record.
    """
    this = RUNS[run]
    rounds = {this.ft["validation_round"], *map(round_of, training_seeds(run))}
    seeds = set()
    for r in sorted(rounds):
        if this.ft["sims_from"] == "remote":
            seeds |= hosts.mirrored_seeds(
                REMOTE, f"{REMOTE_ROUNDS}/r{r}", f"{this.mirror}/r{r}"
            )
        else:
            seeds |= releases.release_seeds(SIM["repo"], r)
    record = Path(f"{this.dir}/available-batches.txt")
    record.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_seeds(sorted(seeds), record)
    return seeds


def check_report(input, output, params, wildcards):
    """Report whether checkpoint `check_at` beats zero-shot, and say so."""
    text, line = evaluations.check_report(
        Path(input[0]).parent, wildcards.run, int(wildcards.n),
        RUNS[wildcards.run].ft["prefixes"],
    )  # fmt: skip
    Path(output[0]).write_text(f"{text}\n")
    say(line)


def land_tables(input, output, params, wildcards):
    """Commit the score tables on the evidence branch and push them."""
    head = notify.land(input, params.evidence, params.checkpoints, params.branch)
    say(f"checkpoint scores landed at {head[:9]} on origin/{params.branch}")
