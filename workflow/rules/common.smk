# Shared settings and helper functions of the workflow's rules, kept
# apart from the rules themselves (rules/*.smk). Machine settings come
# from config/local.env (flowlib.localconfig).

import functools
import json
import tomllib

from flowlib import (
    compute,
    evaluations,
    runs,
    hosts,
    localconfig,
    manifest,
    notify,
    releases,
    rounds,
    scoring,
    segments,
    simulator,
    training,
)

from nowcastfoundation.sidecar import HELD_OUT_BLOCKS, SIDECAR

LOCAL = localconfig.read()
TARGETS = LOCAL["TARGETS"].split()
# The tools the shell rules call (envs/tools.yaml).
TOOLS_ENV = "../envs/tools.yaml"


def say(message):
    """A mailbox line, when NOTIFY is set in config/local.env."""
    notify.post(message, LOCAL)


def job_prefix(mem_mb):
    """The machine's command prefix for a job capped at `mem_mb` MB."""
    return LOCAL["JOB_PREFIX"].replace("{mem}", f"{int(mem_mb)}M")


# From simulate.smk (flowlib.rounds).

ROUNDS = rounds.Rounds(config, LOCAL, rules)
SIM = ROUNDS.sim
ROUNDS_DIR = ROUNDS.dir
REMOTE = ROUNDS.remote
REMOTE_ROUNDS = ROUNDS.remote_dir
GRID_FILE = ROUNDS.grid_file
seeds_of = ROUNDS.seeds_of
round_of = ROUNDS.round_of
here = ROUNDS.here
batch_priority = ROUNDS.batch_priority
batch_files = ROUNDS.batch_files
round_records = ROUNDS.round_records
round_sims = ROUNDS.round_sims
release_owner = ROUNDS.release_owner
release_check = ROUNDS.release_check
round_sources = ROUNDS.round_sources
summary_sources = ROUNDS.summary_sources
write_record = ROUNDS.write_record
mark_ready = ROUNDS.mark_ready
push_record = ROUNDS.push_record


# From finetune.smk and evaluate.smk (flowlib.training): the fine-tuning
# runs (`runs` in workflow/config.yaml, flowlib.runs); each run's rules
# take the {run} wildcard, and its files are in <ROOT>/runs/<run>/.

TRAINING = training.Training(config, LOCAL, ROUNDS)
RUNS = TRAINING.runs
RUNS_DIR = TRAINING.dir
RUN = f"{RUNS_DIR}/{{run}}"
TRAIN = f"{RUN}/train"
EVALUATE_PY = "uv run --group chronos python workflow/scripts/evaluate_offloaded.py"
EVALUATE_PRIORITY = 15
TRAIN_PRIORITY = compute.TRAIN_PRIORITY
trained_runs = TRAINING.trained_runs
run_of = TRAINING.run_of
train_dir = TRAINING.train_dir
ft = TRAINING.ft
local_or_ft = TRAINING.local_or_ft
segment_copy_back = TRAINING.segment_copy_back
saved = TRAINING.saved
evaluate_prev = TRAINING.evaluate_prev
first_prefix = TRAINING.first_prefix
evaluate_prev_dir = TRAINING.evaluate_prev_dir
evaluate_copy_in = TRAINING.evaluate_copy_in
earlier_settings = TRAINING.earlier_settings
evaluate_inputs_from = TRAINING.evaluate_inputs_from
offloaded = TRAINING.offloaded
heldout_curve_inputs = TRAINING.heldout_curve_inputs
check_report = TRAINING.check_report
training_seeds = TRAINING.training_seeds
held_range = TRAINING.held_range
listed_batches = TRAINING.listed_batches
segment_inputs = TRAINING.segment_inputs
fetch_source = TRAINING.fetch_source
finetune_inputs = TRAINING.finetune_inputs
published_checkpoints = TRAINING.published_checkpoints


wildcard_constraints:
    run="[^/]+",


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


def land_tables(input, output, params, wildcards):
    """Commit the score tables on the evidence branch and push them."""
    head = notify.land(input, params.evidence, params.checkpoints, params.branch)
    say(f"checkpoint scores landed at {head[:9]} on origin/{params.branch}")
