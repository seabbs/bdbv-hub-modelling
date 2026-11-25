# The Ebola ablation (#1123; workflow/scripts/ebola_ablation.py): held-out
# simulated outbreaks given one BDBV feature at a time and the hub data
# moved one feature at a time towards the simulations, each forecast by
# the ablation's models through the BDBV check (bdbvhub ft-check), scored
# against the reference and written as one page. Each variant is a folder
# <ROOT>/ablation/<side>/<variant> holding its hub data (data/), each
# model's forecasts (forecasts/<model>/) and its scores (summary.json).
# Target `ablation`; not part of `all`.

# The fine-tuned models among the ablation's (nowcasteval.labels.FINETUNED).
from nowcasteval.labels import FINETUNED as FINETUNED_MODELS

ABL = config["ablation"]
ABL_DIR = f"{LOCAL['ROOT']}/ablation"
ABL_SIDES = {"sim": ABL["sim_variants"], "bdbv": ABL["bdbv_variants"]}


def ablation_sims(wc=None):
    """The simulated outbreaks' counts the sim variants draw from."""
    base = ABL["sims"] or f"{ROUNDS_DIR}/r{ABL['round']}/sims"
    return [f"{base}/counts-{seed}.csv.gz" for seed in ABL["seeds"]]


def ablation_folders():
    """Every variant's folder, simulated variants first."""
    return [f"{ABL_DIR}/{side}/{v}" for side, vs in ABL_SIDES.items() for v in vs]


def ablation_checkpoint(model):
    """The checkpoint a fine-tuned model loads, else none."""
    return ABL["checkpoint"] if model in FINETUNED_MODELS else ""


wildcard_constraints:
    side="sim|bdbv",
    variant="[a-z_]+",


rule ablation_backtest_config:
    # The backtest config with the ablation's draws and longest delay.
    input:
        "packages/nowcastbdbv/config/backtest.toml",
    output:
        f"{ABL_DIR}/backtest.toml",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        samples=ABL["samples"],
        max_delay=ABL["max_delay"],
    shell:
        "uv run python workflow/scripts/ebola_ablation.py config "
        "--base {input} --samples {params.samples} "
        "--max-delay {params.max_delay} --out {output}"


rule ablation_data:
    input:
        hub=rules.hub_data.output,
        sims=lambda wc: ablation_sims() if wc.side == "sim" else [],
    output:
        f"{ABL_DIR}/{{side}}/{{variant}}/data/target.csv",
    log:
        f"{ABL_DIR}/{{side}}/{{variant}}/logs/data.log",
    conda:
        TOOLS_ENV
    params:
        src=lambda wc, input: Path(input.hub[0]).parent,
        out=lambda wc, output: Path(output[0]).parent,
        sims=lambda wc, input: (
            f"--sims {' '.join(input.sims)} --k {ABL['k']} --seed {ABL['seed']}"
            if wc.side == "sim"
            else ""
        ),
    shell:
        "uv run python workflow/scripts/ebola_ablation.py {wildcards.side} "
        "{wildcards.variant} {params.sims} --src {params.src} "
        "--out {params.out} > {log} 2>&1"


rule ablation_forecast:
    input:
        data=f"{ABL_DIR}/{{side}}/{{variant}}/data/target.csv",
        config=rules.ablation_backtest_config.output,
    output:
        f"{ABL_DIR}/{{side}}/{{variant}}/forecasts/{{model}}/release-samples.csv.gz",
    log:
        f"{ABL_DIR}/{{side}}/{{variant}}/logs/{{model}}.log",
    benchmark:
        f"{ABL_DIR}/{{side}}/{{variant}}/benchmarks/{{model}}.tsv"
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=FORECAST.gpu,
    params:
        env=unit_env,
        data=lambda wc, input: Path(input.data).parent,
        out=lambda wc, output: Path(output[0]).parent,
        checkpoint=lambda wc: ablation_checkpoint(wc.model),
        block_limit=ABL["block_limit"],
    shell:
        "env {params.env} CHRONOS_BLOCK_LIMIT={params.block_limit} "
        "BDBV_DATA_DIR={params.data} "
        "uv run --group chronos bdbvhub ft-check forecast "
        "--model {wildcards.model} --checkpoint '{params.checkpoint}' "
        "--config {input.config} --data {params.data} "
        "--out {params.out} > {log} 2>&1"


rule ablation_summary:
    input:
        lambda wc: expand(
            f"{ABL_DIR}/{wc.side}/{wc.variant}/forecasts/{{model}}/"
            "release-samples.csv.gz",
            model=ABL["models"],
        ),
    output:
        f"{ABL_DIR}/{{side}}/{{variant}}/summary.json",
    log:
        f"{ABL_DIR}/{{side}}/{{variant}}/logs/summary.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        folder=lambda wc, output: Path(output[0]).parent,
        reference=ABL["reference"],
    shell:
        "BDBV_DATA_DIR={params.folder}/data uv run python "
        "workflow/scripts/ebola_ablation.py summary {params.folder} "
        "--reference {params.reference} --out {output} > {log} 2>&1"


rule ablation_report:
    input:
        [f"{f}/summary.json" for f in ablation_folders()],
    output:
        f"{ABL_DIR}/ebola-ablation.html",
    log:
        f"{ABL_DIR}/logs/report.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        folders=" ".join(ablation_folders()),
        checkpoint=ABL["checkpoint"],
        reference=ABL["reference"],
        review=LOCAL["REVIEW_DIR"],
    shell:
        "(uv run python workflow/scripts/ebola_ablation.py report "
        "{params.folders} --checkpoint '{params.checkpoint}' "
        "--reference {params.reference} --out {output} "
        "&& if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "cp {output} '{params.review}/bdbv-ebola-ablation.html.tmp' && "
        "mv '{params.review}/bdbv-ebola-ablation.html.tmp' "
        "'{params.review}/bdbv-ebola-ablation.html'; fi) > {log} 2>&1"


rule ablation:
    input:
        rules.ablation_report.output,
