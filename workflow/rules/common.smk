# Shared settings and helper functions of the workflow's rules, kept
# apart from the rules themselves (rules/*.smk). Machine settings come
# from config/local.env (flowlib.localconfig).

import functools
import json
import shlex
import tomllib

from flowlib import (
    evaluations,
    grids,
    runs,
    hosts,
    localconfig,
    manifest,
    notify,
    releases,
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


def git_commit():
    """The commit the workflow runs at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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


def train_dir(run):
    return RUNS[run].train


# From evaluate.smk.

EVALUATE_PY = "uv run --group chronos python workflow/scripts/evaluate_offloaded.py"
EVALUATE_PRIORITY = 15
# Above the trainer's (20): a checkpoint and step 0 are scored between
# segments, a few minutes of GPU each, not after the whole run.
EVALUATE_CHECKPOINT_PRIORITY = 25


def evaluate_prev(wc):
    """The settings of the checkpoint before segment n, if any."""
    prev = run_of(wc).prev[int(wc.n)]
    return f"{train_dir(wc.run)}/n{prev}/settings.json" if prev else []


def first_prefix(run):
    """The run's first checkpoint: the one with no previous."""
    return next(n for n, p in RUNS[run].prev.items() if p is None)


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


def training_seeds(run):
    """The run's training seeds, in the order it trains on them."""
    f = RUNS[run].ft
    if f["seed_file"]:
        return [int(s) for s in Path(f["seed_file"]).read_text().split()]
    return [s for r in f["rounds"] for s in seeds_of(r)]


def held_seeds(run):
    return seeds_of(RUNS[run].ft["validation_round"])


def held_range(run):
    """The run's held-out seeds, as the trainer takes them: first last."""
    seeds = held_seeds(run)
    return f"{seeds[0]} {seeds[-1]}"


def available_seeds(run):
    """The training seeds whose batches can be had now, in order.

    All of them from this machine's rounds, which the workflow simulates.
    From the rounds host or releases, those before the first batch not
    yet listed, and none until the validation round is all listed.
    """
    order = training_seeds(run)
    if not RUNS[run].elsewhere:
        return order
    listed = listed_batches(run)
    if not set(held_seeds(run)) <= listed:
        return []
    for i, seed in enumerate(order):
        if seed not in listed:
            return order[:i]
    return order


def outbreaks_or_full(run, seed):
    """A batch's outbreaks if simulated, else a full batch."""
    for settings in (
        Path(f"{RUNS[run].dir}/sims/settings-{seed}.csv"),
        Path(f"{ROUNDS_DIR}/r{round_of(seed)}/sims/settings-{seed}.csv"),
    ):
        if settings.exists():
            return segments.batch_outbreaks(settings)
    return SIM["batch"]


def segment_seeds(run, n):
    """The seeds segment n needs: n outbreaks, one spare, the held-out."""
    order = available_seeds(run)
    need = segments.seeds_for(n, order, lambda s: outbreaks_or_full(run, s))
    spare = order[len(need) : len(need) + 1]
    return need + spare + held_seeds(run)


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
    order = available_seeds(run)
    # A spare batch past each prefix but the run's last, which takes
    # every training simulation.
    return [
        n
        for n in f["prefixes"]
        if saved(run, n)
        or segments.reachable(
            n + SIM["batch"] * (n < f["outbreaks"]),
            order,
            lambda s: outbreaks_or_full(run, s),
        )
    ]


def to_train(run):
    """Prefixes `finetune` asks for of the run."""
    prefixes = trainable(run)
    if not config.get("next_segment_only"):
        return prefixes
    done = [n for n in prefixes if saved(run, n)]
    rest = [n for n in prefixes if n not in done]
    return done + rest[:1]


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


# From score.smk.

SC = config["score"]
CACHE = LOCAL["SCORE_CACHE"]


def checkpoint_labels(wc=None):
    """The checkpoints to score, less `score.skip`."""
    if SC["labels"] is not None:
        return list(SC["labels"])
    path = checkpoints.release_list.get().output[0]
    found = releases.labels(json.loads(Path(path).read_text()), SC["published_from"])
    skip = set(SC.get("skip") or ())
    return [lb for lb in found if lb not in skip]


ROOM = (
    f"workflow/scripts/wait_for_room.sh {LOCAL['MIN_FREE_GB']} "
    f"{LOCAL['MIN_DISK_GB']} {CACHE} {LOCAL['MAX_LOAD']} && "
)


# A unit's CPU threads (SCORE_THREADS, default 2): a few, so simulations
# keep the rest of the machine's cores.
UNIT_THREADS = int(LOCAL.get("SCORE_THREADS") or 2)


def score_order():
    """The checkpoint labels, oldest published first.

    Read from the release list if it is there, none otherwise: a job's
    priority cannot wait on the release_list checkpoint.
    """
    if SC["labels"] is not None:
        return list(SC["labels"])
    path = Path(f"{CACHE}/releases.json")
    if not path.exists():
        return []
    return releases.by_time(json.loads(path.read_text()), SC["published_from"])


def unit_priority(base):
    """A unit's priority: `base`, newer checkpoints a fraction higher."""
    return lambda wc: releases.priority(score_order(), wc.label, base)


SCORE_DEVICE = LOCAL.get("SCORE_DEVICE") or None
"""The device scoring units run torch on (SCORE_DEVICE in local.env), by
default the machine's: `cpu` keeps them off a GPU a trainer is using."""


def unit_env(wc, threads):
    """The unit's torch device, threads and Chronos-2 batch."""
    return machine.shell_env(machine.torch_env(LOCAL, threads, SCORE_DEVICE))


GERMANY_UNIT = "germany-days"
"""The German units' cache folder, as `evaluation.checkpoint.GERMANY_UNIT`."""

EVALUATE = "uv run --group chronos python -m nowcastfoundation.checkpoint"


def unit_rows(study):
    """The unit's rows folder, which a worker brings back with its output."""
    return lambda wc: f"{CACHE}/{study}/{wc.label}/"


def unit_shell(study, segment=""):
    """A unit of `study` caching under {params.cache} as {params.label};
    `segment`, if given, is the run's own checkpoint folder it loads."""
    return (
        f"mkdir -p {{params.copy_back}} && {ROOM}{{params.prefix}} nice -n 19 "
        f"env {{params.env}} {EVALUATE} {study} {{params.cache}} "
        f"{{params.label}} {segment}> {{log}} 2>&1 && touch {{output}}"
    )


def run_scores_dir(run):
    """The run's score cache: its own checkpoints' units."""
    return f"{RUNS_DIR}/{run}/scores"


def run_units(run):
    """The German units of the run's saved checkpoints."""
    return [
        f"{run_scores_dir(run)}/{GERMANY_UNIT}/n{n}.done"
        for n in RUNS[run].ft["prefixes"]
        if saved(run, n)
    ]


def references():
    """The reference labels' German units, shared by every run."""
    return expand(f"{CACHE}/{GERMANY_UNIT}/{{label}}.done", label=SC["reference"])


def report_names(wc):
    """The German report's LABEL=NAME arguments and reference folders."""
    names = {"reference": "Zero-shot", "randomwalknowcast": "Random walk nowcast"}
    names |= {
        f"n{n}": f"{wc.run} n{n}"
        for n in RUNS[wc.run].ft["prefixes"]
        if saved(wc.run, n)
    }
    args = [shlex.quote(f"{k}={v}") for k, v in names.items()]
    args += [
        f"--from {label}={CACHE}/{GERMANY_UNIT}/{label}" for label in SC["reference"]
    ]
    return " ".join(args)


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
    import datetime as dt

    r, s = int(wildcards.round), int(wildcards.seed)
    line = manifest.existing(params.old, r, s)
    if line is None and not here(s):
        raise ValueError(f"batch {s} arrived without its manifest row")
    line = line or manifest.row(
        params.sims,
        round_=r,
        seed=s,
        kind="block",
        grid=grid_id(),
        commit=git_commit(),
        draws=SIM["draws"],
        batch=SIM["batch"],
        host=HOST,
        finished=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    Path(output[0]).write_text(f"{line}\n")


def write_ready(seeds, seeds_file, ready_file):
    """Write a round's seed list, then READY with the time."""
    import datetime as dt

    Path(seeds_file).write_text("".join(f"{s}\n" for s in seeds))
    Path(ready_file).write_text(f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}\n")


def mark_ready(input, output, params, wildcards):
    """Write a round's seed list and READY, and say so."""
    seeds = seeds_of(wildcards.round)
    write_ready(seeds, output.seeds, output.ready)
    qc = Path(input[0]).read_text().strip()
    say(
        f"round {wildcards.round} ready: {len(seeds)} batches, one grid " f"block; {qc}"
    )


def push_record(input, output, params, wildcards):
    """Send a helping host's batch and its manifest row to the rounds host."""
    import datetime as dt

    r, s = int(wildcards.round), int(wildcards.seed)
    row = manifest.row(
        params.sims,
        round_=r,
        seed=s,
        kind="block",
        grid=grid_id(),
        commit=git_commit(),
        draws=SIM["draws"],
        batch=SIM["batch"],
        host=HOST,
        finished=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    hosts.push(params.sims, s, params.there, row, REMOTE)


@functools.cache
def listed_batches(run):
    """The batches that can be had now, listed once per Snakemake run.

    Listed when the run plans, not kept in a file, so a later run sees
    rounds finished since. From the rounds host the manifests and QC
    summaries of its rounds are mirrored in the run's `rounds/`, and a
    round counts once its QC has run; else the `sims-r<N>` releases'
    assets. Written to <run>/available-batches.txt for the record.
    """
    this = RUNS[run]
    rounds = sorted(
        {this.ft["validation_round"], *(round_of(s) for s in training_seeds(run))}
    )
    seeds = set()
    for r in rounds:
        if this.ft["sims_from"] == "remote":
            # The manifest and QC summary of a round there so far.
            here = Path(f"{this.mirror}/r{r}")
            (here / "qc").mkdir(parents=True, exist_ok=True)
            for name in ("manifest.csv", "qc/summary.txt"):
                there = f"{REMOTE_ROUNDS}/r{r}/{name}"
                got = subprocess.run(
                    ["rsync", "-a", hosts._at(REMOTE, there), str(here / name)],
                    capture_output=True,
                    text=True,
                )  # fmt: skip
                # A round with no QC yet has no summary; a round not
                # started has no manifest, and nor has an unreachable
                # host, so that is said (the mirror kept so far stays).
                if got.returncode and name == "manifest.csv":
                    print(
                        f"could not copy {there}: {got.stderr.strip()[:200]}",
                        file=sys.stderr,
                    )
            # QC runs once the round is whole: only whole rounds are
            # trained on.
            if (here / "qc" / "summary.txt").exists():
                seeds |= manifest.seeds(here / "manifest.csv")
            continue
        # A round not released yet has no release.
        got = subprocess.run(
            ["gh", "release", "view", f"sims-r{r}", "-R", SIM["repo"],
             "--json", "assets"],
            capture_output=True,
            text=True,
        )  # fmt: skip
        if got.returncode == 0:
            names = [a["name"] for a in json.loads(got.stdout)["assets"]]
            seeds |= releases.released_seeds(names)
    record = Path(f"{this.dir}/available-batches.txt")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("".join(f"{s}\n" for s in sorted(seeds)))
    return seeds


def check_report(input, output, params, wildcards):
    """Report whether checkpoint `check_at` beats zero-shot, and say so.

    On the held-out simulations only: BDBV is held out for the paper.
    """
    there, n = RUNS[wildcards.run], int(wildcards.n)
    s = evaluations.checkpoint_settings(Path(input[0]).parent)
    ok = s["held_out_loss"] < s["held_out_loss_start"]
    text = (
        f"{wildcards.run} n{n} {'beats' if ok else 'does NOT beat'} zero-shot: "
        f"held-out {s['held_out_loss']:.4f} vs "
        f"{s['held_out_loss_start']:.4f}"
    )
    Path(output[0]).write_text(f"{text}\n")
    after = [p for p in sorted(there.ft["prefixes"]) if p > n]
    nxt = f", check before n{after[0]}" if after else ""
    say(text if ok else f"ALERT {text}; training goes on{nxt}")


def land_tables(input, output, params, wildcards):
    """Commit the score tables on the evidence branch and push them."""
    import shutil

    def git(*args):
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    notify.check_branch(git("rev-parse", "--abbrev-ref", "HEAD"), params.branch)
    shutil.copy(input[0], params.evidence)
    shutil.copy(input[1], params.checkpoints)
    git("add", params.evidence, params.checkpoints)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-q", "-m", "evidence: checkpoint scores")
    git("push", "-q", "origin", params.branch)
    git("fetch", "-q", "origin", params.branch)
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", f"origin/{params.branch}"):
        raise SystemExit(f"{head} is not on origin/{params.branch}")
    say(f"checkpoint scores landed at {head[:9]} on origin/{params.branch}")
