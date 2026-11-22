# The paper workflow's settings and helper functions, kept apart from
# its rules (paper/workflow/Snakefile).

# This file is a module of the project workflow (workflow/Snakefile), which
# runs from the repository root; paper/ is two folders up from this file.
PAPER = Path(workflow.current_basedir).parents[1]


sys.path.insert(0, str(PAPER))


sys.path.append(str(PAPER.parent / "workflow"))


from flowlib import localconfig  # noqa: E402


from paperdata.synthetic_runner import (  # noqa: E402
    MODELS as SYNTHETIC_MODELS,
)


from paperpipe.config import MODES, run_settings, run_studies  # noqa: E402


from paperpipe.evidence import EVIDENCE as EVIDENCE_NAMES  # noqa: E402


from paperpipe.evidence import branch_head  # noqa: E402


from paperpipe.outputs import (  # noqa: E402
    JOINT_STUDIES,
    MAIN_TABLES,
    SI_TABLES,
    STUDY_SI_TABLES,
    figure_names,
)

R = config["results"]


BUILD = config["build"]


STUDIES = config["studies"]
# Run each step with the Python running the workflow.


# The paper's modules (paperpipe, paperfigs, ...) are in paper/, from the
# root the workflow runs in.
PYTHON = f"PYTHONPATH=paper {shlex.quote(sys.executable)}"


BDBV = shlex.quote(str(Path(sys.executable).parent / "bdbvhubrenewal"))


GERMANY = config["germany"]
# The tools the shell rules call, as the project workflow's.


TOOLS_ENV = "../../workflow/envs/tools.yaml"


# The last BDBV release the SI's real-data models trained on
# (config/finetune.toml); their BDBV comparison scores later rounds only.
with open(PAPER.parent / "config" / "finetune.toml", "rb") as f:
    REALDATA_AFTER = tomllib.load(f)["realdata_bdbv"]["release"]


EBOLA = config["ebola"]
# The fine-tuned checkpoint every `_ft` model the workflow runs loads, so


# each study uses the same one; recorded in results.yml.
# Set here, not read: so not an `envvars` requirement of the workflow.
if config.get("finetuned"):
    os.environ.update(BDBV_CHRONOS_FT=str(config["finetuned"]))


SYNTHETIC = config["synthetic"]
# Simulation folders may name machine settings, such as $ROOT.


SYNTHETIC["simulations"] = {
    mode: localconfig.expand(folder)
    for mode, folder in SYNTHETIC["simulations"].items()
}


TRIANGLE = f"{GERMANY['data']}/COVID-19_hospitalizations.csv"


# The benchmark's command line holds out the dates from its own split.
with open(GERMANY["config"], "rb") as f:
    _split = tomllib.load(f)["nowcast"]["split_date"]


if str(_split) != str(config["heldout"]["germany_from"]):
    raise WorkflowError(
        f"{GERMANY['config']} splits at {_split} but `heldout.germany_from` "
        f"is {config['heldout']['germany_from']}."
    )


# Rows from evidence branches, pinned by commit in `evidence` (config),
# or followed: an entry naming a `branch` reads its head, resolved here so
# a new commit there extracts the file anew.
EVIDENCE = {
    name: entry for name, entry in (config.get("evidence") or {}).items() if entry
}


for _entry in EVIDENCE.values():
    if "branch" in _entry:
        _entry["ref"] = branch_head(_entry["branch"], PAPER.parent)


def evidence_file(name):
    """The extracted file, named by its commit so a new pin extracts anew."""
    entry = EVIDENCE[name]
    suffix = "".join(Path(entry["path"]).suffixes)
    return f"{R}/evidence/{name}-{entry['ref'][:12]}{suffix}"


def provisional(study):
    """What is provisional in a study's further forecasts, if anything."""
    block = {"germany": GERMANY, "ebola": EBOLA}.get(study, {})
    notes = [e.get("provisional") for e in block.get("extra") or []]
    return " ".join(dict.fromkeys(n for n in notes if n))


def ebola_extra(mode):
    """Further Ebola samples, each kept to its models."""
    return [
        f"{R}/evidence/ebola-extra-{i}.csv.gz"
        for i in range(len(EBOLA.get("extra") or []))
    ]


def extra_args(entry):
    """The `germany-extra` options of a `germany.extra` entry."""
    args = []
    if entry.get("models"):
        args.append(f"--models {shlex.quote(entry['models'])}")
    if entry.get("suffix"):
        args.append(f"--suffix {shlex.quote(entry['suffix'])}")
    if entry.get("backtest"):
        args.append("--backtest")
    return " ".join(args)


wildcard_constraints:
    mode="|".join(MODES),
    study="|".join(STUDIES),


def settings(w):
    return run_settings(config, w.mode)


def germany_tables(mode):
    """The German benchmark's paper tables a run reads."""
    if not GERMANY["real"]:
        return []
    halves = [""] if mode == "development" else ["", "held-out/"]
    tables = [f"{GERMANY['results']}/{h}paper-forecasts.csv.gz" for h in halves]
    extra = [
        f"{R}/evidence/germany-extra-{i}.csv.gz"
        for i in range(len(GERMANY.get("extra") or []))
    ]
    return tables + extra


def sources(w):
    """A study's real data, as command line arguments."""
    if w.study == "synthetic":
        fc = SYNTHETIC["forecasts"][w.mode]
        return f"--synthetic-forecasts {shlex.quote(fc)}" if fc else ""
    if w.study == "germany" and GERMANY["real"]:
        tables = " ".join(germany_tables(w.mode))
        return f"--germany-forecasts {tables} --germany-triangle {TRIANGLE}"
    if w.study == "ebola" and EBOLA["hub"]:
        args = [
            f"--hub {shlex.quote(EBOLA['hub'])}",
            f"--ebola-data {shlex.quote(EBOLA['data'])}",
            f"--ebola-config {shlex.quote(EBOLA['config'])}",
            f"--ebola-backtest {shlex.quote(EBOLA['backtest'])}",
        ]
        if EBOLA.get("extra"):
            files = " ".join(ebola_extra(w.mode))
            args.append(f"--ebola-extra {files}")
        if EBOLA.get("n_samples"):
            models = " ".join(EBOLA["backtest_models"])
            args.append(f"--ebola-draws {EBOLA_DRAWS}")
            args.append(f"--ebola-draws-models {models}")
        if EBOLA["candidates"]:
            files = " ".join(shlex.quote(c) for c in EBOLA["candidates"])
            args.append(f"--candidates {files}")
        if EBOLA["store"]:
            args.append(f"--store {shlex.quote(EBOLA['store'])}")
            args.append(f"--store-ref {EBOLA['store_ref']}")
        return " ".join(args)
    return ""


def source_files(w):
    """Files a study's real data come from, built by the rules below."""
    if w.study == "germany":
        triangle = [TRIANGLE] if GERMANY["real"] else []
        return [*germany_tables(w.mode), *triangle]
    if w.study == "ebola" and EBOLA["hub"]:
        hub = f"{EBOLA['hub']}/target-data/time-series.csv"
        extra = ebola_extra(w.mode)
        target = f"{EBOLA['data']}/target.csv"
        draws = [EBOLA_DRAWS] if EBOLA.get("n_samples") else []
        return [hub, target, EBOLA["backtest"], *extra, *draws]
    if w.study == "synthetic" and synthetic_simulations(w.mode):
        return [SYNTHETIC["forecasts"][w.mode]]
    return []


def synthetic_simulations(mode):
    """The simulator's files of a run's seeds, or [] if any is missing."""
    folder = SYNTHETIC["simulations"][mode]
    seeds = run_settings(config, mode).synthetic_seeds
    files = [
        f"{folder}/{kind}-{seed}.{ext}"
        for seed in seeds
        for kind, ext in (("counts", "csv.gz"), ("settings", "csv"))
    ]
    return files if all(Path(f).exists() for f in files) else []


def limits(w):
    """Data limits of a study's loader, as command line arguments."""
    s = settings(w)
    if w.study == "synthetic":
        return "--seeds " + " ".join(str(x) for x in s.synthetic_seeds)
    if w.study == "germany" and s.germany_before is not None:
        return f"--before {s.germany_before:%Y-%m-%d}"
    if w.study == "ebola" and s.ebola_before is not None:
        return f"--before {s.ebola_before:%Y-%m-%d}"
    return study_split(w.mode, w.study)


def figures(mode, study):
    return [
        f"{R}/{mode}/figures/{part}/{study}-{name}.png"
        for part in ("main", "si")
        for name in figure_names(study, part)
    ]


def tables(mode, study):
    main = [*MAIN_TABLES, *(["grid"] if study == "synthetic" else [])]
    return [f"{R}/{mode}/tables/main/{study}-{n}.csv" for n in main] + [
        f"{R}/{mode}/tables/si/{study}-{n}.csv"
        for n in (*SI_TABLES, *STUDY_SI_TABLES.get(study, ()))
    ]


HELD_OUT_INCLUDES = (
    ("main", "scores"),
    ("main", "joint"),
    ("si", "scores-by-horizon"),
    ("si", "scores-by-location"),
)
"""The tables main.qmd and si.qmd include for every case study."""


def studies(mode):
    """The studies a run has: development leaves out the held-out ones."""
    return run_studies(STUDIES, mode)


def outputs(mode):
    design = [f"{R}/{mode}/figures/si/sim-examples.png"] + [
        f"{R}/{mode}/tables/si/{n}.md"
        for n in (
            "grid-epidemic",
            "grid-reporting",
            "grid-distributions",
            "epinowcast-model",
            "heldout-set",
        )
    ]
    evidence = [f"{R}/{mode}/evidence.yml"]
    joint = [
        f"{R}/{mode}/tables/main/{study}-joint.md"
        for study in studies(mode)
        if study in JOINT_STUDIES
    ]
    real = (
        [f"{R}/{mode}/tables/si/real-finetune.md"] if "germany" in studies(mode) else []
    )
    return (
        evidence
        + design
        + joint
        + real
        + [
            path
            for study in studies(mode)
            for path in [*figures(mode, study), *tables(mode, study)]
        ]
    )


# The release backtest of the models on main, from every past release. CI
# has run it by the paper step, so this rule then has nothing to do.
# The paper's own release backtest of `backtest_models` at its number of
# draws, from a copy of the backtest config with only `[samples] backtest`
# changed: the draws of the models the backtest runs itself. The stored
# fits keep `[forecast] n_samples`.
EBOLA_DRAWS = f"{R}/ebola-backtest/forecasts.csv.gz"


def study_split(mode, study):
    s = run_settings(config, mode)
    if study == "germany" and s.germany_split is not None:
        return f"--split {s.germany_split:%Y-%m-%d}"
    if study == "ebola" and s.ebola_split is not None:
        return f"--split {s.ebola_split:%Y-%m-%d}"
    return ""


def study_files(study):
    """A study's data, forecasts and scores, with `{mode}` left open."""
    return {
        "forecasts": f"{R}/{{mode}}/forecasts/{study}.csv.gz",
        "observations": f"{R}/{{mode}}/data/{study}-observations.csv.gz",
        "locations": f"{R}/{{mode}}/data/{study}-locations.csv",
        "meta": f"{R}/{{mode}}/data/{study}-meta.json",
        "scores": f"{R}/{{mode}}/scores/{study}.csv.gz",
        "crps_scores": f"{R}/{{mode}}/scores-crps/{study}.csv.gz",
        "truth": f"{R}/{{mode}}/truth/{study}.csv.gz",
    }


def grid_config(mode):
    """The `synthetic` block with the run's seeds, for the SI notes."""
    s = run_settings(config, "paper")
    d = run_settings(config, "development")
    return {
        **SYNTHETIC,
        "heldout_seeds": list(s.synthetic_seeds),
        "development_seeds": list(d.synthetic_seeds),
    }
