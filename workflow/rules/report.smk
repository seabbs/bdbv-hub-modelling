# Reports: what the steps made, read back for people. The simulation
# reports (each round's outbreaks against the German development half and
# the paper's reference values), a run's checkpoints on the German
# development half, and the jobs' benchmarks. Each rule reads the other
# steps' outputs and runs one script.


rule round_summaries:
    # One row per outbreak of a round (first draw, strata summed): design,
    # drawn values, size, dynamics, R, delays and the phase at the
    # training cut-offs (workflow/scripts/grid_summaries.py). From the
    # round's batch files, not its manifest, so a summary of a round
    # another run is simulating writes nothing into that round's records.
    input:
        round_sims,
    output:
        f"{ROUNDS_DIR}/r{{round}}/summaries/outbreaks.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/summaries/run.log",
    conda:
        TOOLS_ENV
    params:
        sims=lambda wc, input: Path(input[0]).parent,
    shell:
        "uv run python workflow/scripts/grid_summaries.py outbreaks "
        "{params.sims} {output} > {log} 2>&1"


rule grid_figures:
    # The run's rounds side by side: each summary's ECDF and the shares of
    # patterns and phases, with their quantiles and shares as a table.
    input:
        expand(
            f"{ROUNDS_DIR}/r{{round}}/summaries/outbreaks.csv",
            round=config["rounds"],
        ),
    output:
        ecdf=f"{ROUNDS_DIR}/figures/grid/compare-ecdf.png",
        shares=f"{ROUNDS_DIR}/figures/grid/compare-shares.png",
        table=f"{ROUNDS_DIR}/figures/grid/compare.csv",
    log:
        f"{ROUNDS_DIR}/figures/grid/run.log",
    conda:
        TOOLS_ENV
    params:
        out=lambda wc, output: Path(output.ecdf).parent,
        sources=round_sources,
    shell:
        """
        uv run python workflow/scripts/grid_summaries.py compare \
            {output.table} {params.sources} >{log} 2>&1
        uv run python workflow/scripts/grid_figures.py {params.out} \
            {params.sources} >>{log} 2>&1
        """


# The simulation report (workflow/scripts/sim_report.py): per round, and
# over every round in `rounds`, the science of the simulated outbreaks
# against the German development half and the paper's reference values.
# Target `sim_reports`. The German triangle is the hub's, at
# config/germany.toml's commit; no BDBV data.
GERMANY_SPLIT = tomllib.loads(Path("config/germany.toml").read_text())["nowcast"][
    "split_date"
]
REFERENCE_VALUES = "paper/data/reference-values.csv"


rule germany_triangle:
    output:
        f"{LOCAL['ROOT']}/germany/COVID-19_hospitalizations.csv",
    log:
        f"{LOCAL['ROOT']}/germany/fetch.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        out=lambda wc, output: Path(output[0]).parent,
    shell:
        "uv run python -m bdbvhub.cli germany-fetch --out {params.out} " "> {log} 2>&1"


rule germany_dev_stats:
    # The development half's series statistics: national, and medians
    # over states and over age groups.
    input:
        rules.germany_triangle.output,
    output:
        f"{ROUNDS_DIR}/report/germany-dev.csv",
    log:
        f"{ROUNDS_DIR}/report/germany-dev.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        split=GERMANY_SPLIT,
    shell:
        "uv run python workflow/scripts/sim_report.py germany {input} "
        "{params.split} {output} > {log} 2>&1"


rule sim_report:
    input:
        summaries=rules.round_summaries.output,
        germany=rules.germany_dev_stats.output,
        refs=REFERENCE_VALUES,
    output:
        f"{ROUNDS_DIR}/r{{round}}/report/sim-report.html",
    log:
        f"{ROUNDS_DIR}/r{{round}}/report/run.log",
    conda:
        TOOLS_ENV
    params:
        source=lambda wc, input: summary_sources([wc.round], [input.summaries]),
    shell:
        "uv run python workflow/scripts/sim_report.py render {output} "
        "'Simulated outbreaks, round {wildcards.round}' {input.refs} "
        "{input.germany} {params.source} > {log} 2>&1"


rule sim_report_run:
    # Every round in `rounds` together.
    input:
        summaries=expand(
            f"{ROUNDS_DIR}/r{{round}}/summaries/outbreaks.csv",
            round=config["rounds"],
        ),
        germany=rules.germany_dev_stats.output,
        refs=REFERENCE_VALUES,
    output:
        f"{ROUNDS_DIR}/report/sim-report.html",
    log:
        f"{ROUNDS_DIR}/report/run.log",
    conda:
        TOOLS_ENV
    params:
        title="Simulated outbreaks, rounds "
        + ", ".join(str(r) for r in config["rounds"]),
        sources=lambda wc, input: summary_sources(config["rounds"], input.summaries),
    shell:
        "uv run python workflow/scripts/sim_report.py render {output} "
        "'{params.title}' {input.refs} {input.germany} {params.sources} "
        "> {log} 2>&1"


rule sim_diseases:
    # The grid's training rounds' ranges against published values by
    # disease, for the SI's disease table (the paper's `evidence` rule).
    input:
        summaries=expand(
            f"{ROUNDS_DIR}/r{{round}}/summaries/outbreaks.csv",
            round=config["grid_rounds"]["rounds"],
        ),
        refs=REFERENCE_VALUES,
    output:
        f"{ROUNDS_DIR}/report/sim-diseases.csv",
    log:
        f"{ROUNDS_DIR}/report/sim-diseases.log",
    localrule: True
    conda:
        TOOLS_ENV
    shell:
        "uv run python workflow/scripts/sim_report.py diseases {output} "
        "{input.refs} {input.summaries} > {log} 2>&1"


rule sim_reports:
    input:
        expand(
            f"{ROUNDS_DIR}/r{{round}}/report/sim-report.html",
            round=config["rounds"],
        ),
        rules.sim_report_run.output,


rule german_report:
    # The run's checkpoints on the German development half against
    # zero-shot and randomwalknowcast (workflow/scripts/german_report.py).
    input:
        references(),
        lambda wc: run_units(wc.run),
    output:
        f"{RUN}/reports/german/index.html",
    log:
        f"{RUN}/logs/german-report.log",
    conda:
        TOOLS_ENV
    params:
        units=lambda wc: f"{run_scores_dir(wc.run)}/{GERMANY_UNIT}",
        out=lambda wc, output: Path(output[0]).parent,
        names=report_names,
    shell:
        "uv run python workflow/scripts/german_report.py {params.units} "
        "{params.out} {params.names} > {log} 2>&1"


rule benchmarks:
    # Opt-in: seconds and peak memory of every forecast and score job
    # (`benchmark:` files), and seconds by model and date (`timings.csv`),
    # in the score cache and the paper's results, as one CSV
    # (flowlib.benchmarks). Rerun with --forcerun benchmarks.
    output:
        f"{CACHE}/benchmarks.csv",
    log:
        f"{CACHE}/log/benchmarks.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        roots=f"{CACHE} {config['results']} {RUNS_DIR}",
    shell:
        "uv run python workflow/scripts/benchmarks.py {output} {params.roots} "
        "> {log} 2>&1"
