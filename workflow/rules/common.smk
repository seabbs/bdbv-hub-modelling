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
    machine,
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

localconfig.check()
LOCAL = localconfig.read()
# The German triangle at config/germany.toml's commit (`germany_triangle`),
# which the German scoring units and the real-data blocks read.
GERMANY_TRIANGLE = f"{LOCAL['ROOT']}/germany/COVID-19_hospitalizations.csv"
TARGETS = LOCAL["TARGETS"].split()
# The tools the shell rules call (envs/tools.yaml).
TOOLS_ENV = "../envs/tools.yaml"
# A mailbox line, when NOTIFY is set.
say = functools.partial(notify.post, cfg=LOCAL)


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
validation_curve_inputs = TRAINING.validation_curve_inputs
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


# From score.smk (flowlib.scoring).

SCORING = scoring.Scoring(config, LOCAL, TRAINING, checkpoints)
SC = SCORING.sc
CACHE = SCORING.cache
UNIT_THREADS = SCORING.unit_threads
FORECAST = SCORING.forecast
GERMANY_UNIT = scoring.GERMANY_UNIT
EVALUATE = scoring.EVALUATE
newest_first = compute.newest_first
checkpoint_labels = SCORING.checkpoint_labels
unit_priority = SCORING.unit_priority
unit_env = SCORING.unit_env
unit_rows = SCORING.unit_rows
unit_shell = SCORING.unit_shell
run_scores_dir = SCORING.run_scores_dir
saved_prefixes = SCORING.saved_prefixes
run_units = SCORING.run_units
references = SCORING.references
report_names = SCORING.report_names
run_scores = SCORING.run_scores
finetune_scores = SCORING.finetune_scores
table_inputs = SCORING.table_inputs
unit_stamps = SCORING.unit_stamps
land_tables = SCORING.land_tables


# The BDBV foundation-model check (rules/report.smk `ft_check`); FTC and
# FTC_DIR are set there.
def ft_check_checkpoint():
    """The fine-tuned checkpoint the check runs: the config's or the pin."""
    import tomllib

    pin = tomllib.loads(Path("config/finetune.toml").read_text())["hash"]
    return FTC["checkpoint"] or pin


def ft_check_folders():
    """The BDBV check's forecast folders: each model's and the checkpoint's
    (rules/report.smk `ft_check`)."""
    return [f"{FTC_DIR}/forecasts/{m}" for m in FTC["models"]] + [
        f"{FTC_DIR}/forecasts/ft-{ft_check_checkpoint()}"
    ]
