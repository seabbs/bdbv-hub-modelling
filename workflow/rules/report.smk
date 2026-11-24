# Reports: what the steps made, read back for people. The simulation
# reports (each round's outbreaks against the German development half and
# the paper's reference values), a run's checkpoints on the German
# development half, the jobs' benchmarks and our hub rounds against the
# hub teams. Each rule reads the other steps' outputs and runs one script.


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
        "uv run python -m nowcastgermany.cli fetch --out {params.out} " "> {log} 2>&1"


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


rule rt_check:
    # The Rt sense check of a stored renewal fit before its wk R_t is
    # submitted: Rt by country, province and zone over their cases, with
    # BVDOutbreakSize's Rt (bdbvhub.reporting.rtcheck). It reads the fit
    # the store holds for the current code (`fit-store unpack`), so the fit
    # that would be submitted, and copies the page to REVIEW_DIR
    # (config/local.env) as rt-check.html when it is set.
    output:
        f"{config['rt_check']['out_dir']}/{{model}}-{{origin}}.html",
    log:
        f"{config['rt_check']['out_dir']}/{{model}}-{{origin}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        store=config["rt_check"]["store"],
        data=config["rt_check"]["data"],
        review=LOCAL["REVIEW_DIR"],
    shell:
        """
        exec >{log} 2>&1
        fit=$(mktemp -d)
        trap 'rm -rf "$fit"' EXIT
        export BDBV_DATA_DIR={params.data}
        uv run bdbvhub fit-store unpack --store {params.store} \
            --data {params.data} --model {wildcards.model} \
            --origin {wildcards.origin} --out "$fit"
        uv run bdbvhub rt-check --fit "$fit" --data {params.data} \
            --out {output}
        if [ -n '{params.review}' ]; then
            mkdir -p '{params.review}'
            cp {output} '{params.review}/rt-check.html.tmp'
            mv '{params.review}/rt-check.html.tmp' '{params.review}/rt-check.html'
        fi
        """


# The hub rounds report: our hub submissions against every hub team on
# the hub's latest data (workflow/scripts/hub_rounds_report.py). It reads
# the hub's files as submitted and runs no model of ours. It reruns when
# <ROOT>/hub/head.txt changes: workflow/scripts/hub_watch.sh writes the
# hub's data commit there when it moves and then builds `hub_rounds`.
# Not part of `all`.

HUB = f"{LOCAL['ROOT']}/hub"


rule hub_data:
    # The hub's tidy data (`bdbvhub data`), including tasks.json.
    input:
        f"{HUB}/head.txt",
    output:
        f"{HUB}/data/target.csv",
    log:
        f"{HUB}/logs/data.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        out=lambda wc, output: Path(output[0]).parent,
    shell:
        "uv run bdbvhub data --raw {params.out}/raw --out {params.out} " "> {log} 2>&1"


rule hub_submissions_table:
    # Each round's submitted model per hub id, from the ledger branch
    # (submission_ci.submissions), so the report says which chain model
    # went out.
    input:
        f"{HUB}/head.txt",
    output:
        f"{HUB}/submissions.csv",
    log:
        f"{HUB}/logs/submissions.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        ledger=f"{HUB}/ledger",
    shell:
        "(git fetch -q origin submission-ledger && rm -rf {params.ledger} && "
        "mkdir -p {params.ledger} && git archive origin/submission-ledger "
        "| tar -x -C {params.ledger} && uv run --quiet --project ci "
        "submissions-table --state {params.ledger}/state.json "
        "--policy ci/policy.toml --hub-config config/hub.toml "
        "--out {output}) > {log} 2>&1"


rule hub_comparison:
    # Every hub team's files and ours scored on the same truth, beside
    # `hub_rounds.models` rerun on current code for each round
    # (`compare-hub --models`). The data folder is also set as
    # BDBV_DATA_DIR, which the hub's places are read from.
    input:
        target=rules.hub_data.output,
        table=rules.hub_submissions_table.output,
    output:
        scores=f"{HUB}/comparison/scores.csv",
        quantiles=f"{HUB}/comparison/quantiles.csv.gz",
        summary=f"{HUB}/comparison/summary.json",
    log:
        f"{HUB}/logs/comparison.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        data=lambda wc, input: Path(input.target[0]).parent,
        out=lambda wc, output: Path(output.scores).parent,
        models=" ".join(config["hub_rounds"]["models"]),
    shell:
        "BDBV_DATA_DIR={params.data} uv run bdbvhub compare-hub --fetch "
        "--data {params.data} --hub-dir {params.data}/hub-submissions "
        "--cache-dir {params.data}/raw/hub-submissions --out {params.out} "
        "--results {params.out}/results --models {params.models} --stored "
        "--submissions-table {input.table} > {log} 2>&1"


rule hub_rounds_report:
    input:
        script="workflow/scripts/hub_rounds_report.py",
        head=f"{HUB}/head.txt",
        target=rules.hub_data.output,
        table=rules.hub_submissions_table.output,
        scores=rules.hub_comparison.output.scores,
    output:
        f"{HUB}/hub-rounds.html",
    log:
        f"{HUB}/logs/report.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        comparison=lambda wc, input: Path(input.scores).parent,
        data=lambda wc, input: Path(input.target[0]).parent,
        rounds=config["hub_rounds"]["rounds"],
    shell:
        "uv run python {input.script} {output} "
        "{params.comparison} {params.data} {input.table} {input.head} "
        "{params.rounds} > {log} 2>&1"


rule hub_rounds:
    # The report, copied to REVIEW_DIR (config/local.env) when it is set.
    input:
        rules.hub_rounds_report.output,
    log:
        f"{HUB}/logs/publish.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        review=LOCAL["REVIEW_DIR"],
        data=lambda wc, input: Path(input[0]).with_name("hub-rounds-data"),
    shell:
        "if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "rm -rf '{params.review}/hub-rounds-data' && "
        "cp -r {params.data} '{params.review}/hub-rounds-data' && "
        "cp {input} '{params.review}/hub-rounds.html.tmp' && "
        "mv '{params.review}/hub-rounds.html.tmp' "
        "'{params.review}/hub-rounds.html'; fi > {log} 2>&1"


# The BDBV foundation-model check (bdbvhub.ftcheck): each model forecasts
# every hub release into its own folder, keyed by the code it depends on
# (scoring.ft_check_key), so the reference models are made once; the
# fine-tuned model's folder is named by its checkpoint, so a new one
# reruns only it. `ft_check` scores them together and copies the page to
# REVIEW_DIR as bdbv-ft-check.html. Not part of `all`.

FTC = config["ft_check"]
FTC_DIR = f"{HUB}/ft-check"


rule ft_check_forecast:
    input:
        rules.hub_data.output,
    output:
        touch(f"{FTC_DIR}/forecasts/{{fmodel}}.done"),
    log:
        f"{FTC_DIR}/logs/{{fmodel}}.log",
    benchmark:
        f"{FTC_DIR}/benchmarks/{{fmodel}}.tsv"
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=FORECAST.gpu,
    params:
        env=unit_env,
        code=lambda wc: scoring.ft_check_key(),
        data=lambda wc, input: Path(input[0]).parent,
        out=lambda wc: f"{FTC_DIR}/forecasts/{wc.fmodel}",
        model=lambda wc: (
            FTC["finetuned"] if wc.fmodel.startswith("ft-") else wc.fmodel
        ),
        checkpoint=lambda wc: wc.fmodel[3:] if wc.fmodel.startswith("ft-") else "",
    shell:
        "env {params.env} BDBV_DATA_DIR={params.data} "
        "uv run --group chronos bdbvhub ft-check forecast "
        "--model {params.model} --checkpoint '{params.checkpoint}' "
        "--data {params.data} --out {params.out} > {log} 2>&1"


rule ft_check:
    input:
        lambda wc: [f"{f}.done" for f in ft_check_folders()],
    output:
        f"{FTC_DIR}/bdbv-ft-check.html",
    log:
        f"{FTC_DIR}/logs/report.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        data=lambda wc: f"{HUB}/data",
        folders=lambda wc: " ".join(ft_check_folders()),
        checkpoint=lambda wc: ft_check_checkpoint(),
        reference=FTC["reference"],
        review=LOCAL["REVIEW_DIR"],
    shell:
        "(BDBV_DATA_DIR={params.data} uv run bdbvhub ft-check report "
        "--data {params.data} --reference {params.reference} "
        "--checkpoint {params.checkpoint} --out {output} {params.folders} "
        "&& if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "rm -rf '{params.review}/bdbv-ft-check-data' && "
        "cp -r $(dirname {output})/bdbv-ft-check-data '{params.review}/' && "
        "cp {output} '{params.review}/bdbv-ft-check.html.tmp' && "
        "mv '{params.review}/bdbv-ft-check.html.tmp' "
        "'{params.review}/bdbv-ft-check.html'; fi) > {log} 2>&1"


# The this-round page: every candidate's forecasts and nowcasts for the
# round the latest Submit run on main targeted, against the data as of
# the round, with the wk R_t quantiles, the gate's pick and the renewal
# fits' convergence (workflow/scripts/this_round_report.py). No scores.
# It reruns when <ROOT>/hub/round/run.txt (the run and its status) or
# <ROOT>/hub/round/store.txt (the fit store's commit) changes, which
# workflow/scripts/hub_watch.sh writes. Not part of `all`.


rule this_round_inputs:
    input:
        target=rules.hub_data.output,
        run=f"{HUB}/round/run.txt",
        store=f"{HUB}/round/store.txt",
    output:
        candidates=f"{HUB}/round/candidates.csv.gz",
        run=f"{HUB}/round/run/run.json",
    log:
        f"{HUB}/logs/this-round-inputs.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        out=lambda wc, output: Path(output.candidates).parent,
        data=lambda wc, input: Path(input.target[0]).parent,
    shell:
        "workflow/scripts/this_round_inputs.sh {params.out} {params.data} "
        "{input.run} > {log} 2>&1"


rule this_round_report:
    input:
        script="workflow/scripts/this_round_report.py",
        target=rules.hub_data.output,
        candidates=rules.this_round_inputs.output.candidates,
        run=rules.this_round_inputs.output.run,
    output:
        f"{HUB}/this-round.html",
    log:
        f"{HUB}/logs/this-round.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        store=lambda wc, input: Path(input.candidates).parent / "store",
        data=lambda wc, input: Path(input.target[0]).parent,
        run=lambda wc, input: Path(input.run).parent,
    shell:
        "uv run python {input.script} {output} {input.candidates} "
        "{params.store} {params.data} {params.run} > {log} 2>&1"


rule this_round:
    # The page, copied to REVIEW_DIR (config/local.env) when it is set.
    input:
        rules.this_round_report.output,
    log:
        f"{HUB}/logs/this-round-publish.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        review=LOCAL["REVIEW_DIR"],
        data=lambda wc, input: Path(input[0]).with_name("this-round-data"),
    shell:
        "if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "rm -rf '{params.review}/this-round-data' && "
        "cp -r {params.data} '{params.review}/this-round-data' && "
        "cp {input} '{params.review}/this-round.html.tmp' && "
        "mv '{params.review}/this-round.html.tmp' "
        "'{params.review}/this-round.html'; fi > {log} 2>&1"
