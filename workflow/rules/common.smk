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
# The German triangle at packages/nowcastgermany/config/germany.toml's commit
# (`germany_triangle`),
# which the German scoring units and the real-data blocks read.
GERMANY_TRIANGLE = f"{LOCAL['ROOT']}/germany/COVID-19_hospitalizations.csv"
# The BDBV foundation-model check's folder (rules/report.smk).
FTC_DIR = f"{LOCAL['ROOT']}/hub/ft-check"


def ft_check_page(ckpt):
    """The BDBV check page of one fine-tuned checkpoint."""
    return f"{FTC_DIR}/{ckpt}/bdbv-ft-check-{ckpt}.html"


def ft_check_released(wc=None):
    """The BDBV check page of every released checkpoint of the trained
    runs: each segment or interim checkpoint with a published.json."""
    import glob

    pages = []
    for run in TRAINING.trained_runs():
        for path in sorted(
            glob.glob(f"{TRAINING.dir}/{run}/train/**/published.json", recursive=True)
        ):
            tag = json.loads(Path(path).read_text()).get("hash")
            if tag:
                pages.append(ft_check_page(f"slots-{tag}"))
    return pages


def released_ft_check_page(wc):
    """The BDBV check page of saved checkpoint `ck` of the run."""
    settings = Path(train_dir(wc.run), wc.ck, "settings.json")
    return ft_check_page(f"slots-{json.loads(settings.read_text())['hash']}")


def ft_check_record(wc):
    """The release record of a BDBV check forecast's checkpoint, when it
    is one of any run's saved checkpoints, so it is released before it is
    forecast."""
    import glob

    if not wc.fmodel.startswith("ft-slots-"):
        return []
    key = wc.fmodel.removeprefix("ft-slots-")
    for run in TRAINING.runs:
        train = train_dir(run)
        for settings in [
            *glob.glob(f"{train}/n*/settings.json"),
            *glob.glob(f"{train}/interim/*/settings.json"),
        ]:
            if json.loads(Path(settings).read_text()).get("hash") == key:
                return str(Path(settings).with_name("published.json"))
    return []


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
CHECKPOINT_PRIORITY = compute.CHECKPOINT_PRIORITY
EVALUATION_PRIORITY = compute.EVALUATION_PRIORITY
REFERENCE_PRIORITY = compute.REFERENCE_PRIORITY
TRAIN_PRIORITY = compute.TRAIN_PRIORITY
HANDOFF_PRIORITY = compute.HANDOFF_PRIORITY
trained_runs = TRAINING.trained_runs
run_of = TRAINING.run_of
train_dir = TRAINING.train_dir
ft = TRAINING.ft
local_or_ft = TRAINING.local_or_ft
segment_copy_back = TRAINING.segment_copy_back
saved = TRAINING.saved
first_prefix = TRAINING.first_prefix
earlier_settings = TRAINING.earlier_settings
evaluate_inputs_from = TRAINING.evaluate_inputs_from
offloaded = TRAINING.offloaded
held_out_key = TRAINING.held_out_key
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
EVALUATE_THREADS = SCORING.evaluate_threads
FORECAST = SCORING.forecast
CHECKPOINT_FORECAST = SCORING.checkpoint_forecast
GERMANY_UNIT = scoring.GERMANY_UNIT
EVALUATE = scoring.EVALUATE
newest_first = compute.newest_first
checkpoint_labels = SCORING.checkpoint_labels
unit_priority = SCORING.unit_priority
ft_check_priority = SCORING.ft_check_priority
unit_env = SCORING.unit_env
checkpoint_env = SCORING.checkpoint_env
evaluation_priority = SCORING.evaluation_priority
unit_rows = SCORING.unit_rows
unit_shell = SCORING.unit_shell
run_scores_dir = SCORING.run_scores_dir
run_prefixes = SCORING.run_prefixes
run_units = SCORING.run_units
references = SCORING.references
report_names = SCORING.report_names
run_scores = SCORING.run_scores
finetune_scores = SCORING.finetune_scores
table_inputs = SCORING.table_inputs
unit_stamps = SCORING.unit_stamps
land_tables = SCORING.land_tables


# The Hugging Face Hub upload (rules/publish.smk `publish_hf`).
def hf_repo():
    """The Hub repository `publish_hf` uploads to, from packages/nowcastfoundation/config/finetune.toml
    ([huggingface] repo); empty until one is chosen."""
    import tomllib

    conf = tomllib.loads(
        Path("packages/nowcastfoundation/config/finetune.toml").read_text()
    )
    return str(conf.get("huggingface", {}).get("repo", ""))


def hf_args():
    """`publish-hf`'s options: the repository, the scores, the dry run."""
    hf = config["huggingface"]
    repo = hf_repo()
    args = [f"--repo {repo}"] if repo else []
    if hf["scores"]:
        args.append(f"--scores {hf['scores']}")
    if hf["dry_run"] or not repo:
        args.append("--dry-run")
    return " ".join(args)


# The BDBV foundation-model check (report.smk's ft_check rules).
FTC = config["ft_check"]


def ft_check_checkpoint():
    """The fine-tuned checkpoint the check runs: the config's, else the one
    packages/nowcastfoundation/config/finetune.toml names, `latest` resolved to its release (read
    only when the check is asked for)."""
    from nowcastfoundation.forecaster import finetuned_label

    return FTC["checkpoint"] or finetuned_label()


MEMORY = config["memory"]


def ft_check_mem(wc):
    """A BDBV check forecast's memory: a Chronos-2 model's (the zero-shot
    and fine-tuned foundation models, `ft-<checkpoint>`) or another's."""
    chronos = wc.fmodel.startswith(("ft-", "foundation"))
    return MEMORY["ft_check_chronos" if chronos else "ft_check"]


def ft_check_key():
    """The code key every BDBV check forecast folder carries
    (`scoring.ft_check_key`), so a folder from other code is never used."""
    return scoring.ft_check_key()


def ft_check_folders(ckpt):
    """The forecast folders of the check's models and the checkpoint's,
    each `<model>@<code key>`."""
    key = ft_check_key()
    return [f"{FTC_DIR}/forecasts/{m}@{key}" for m in FTC["models"]] + [
        f"{FTC_DIR}/forecasts/ft-{ckpt}@{key}"
    ]
