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
# packages/nowcastgermany/config/germany.toml's commit; the BDBV data are
# the hub's releases (`hub_data`), for the delays' structure only.
GERMANY_SPLIT = tomllib.loads(
    Path("packages/nowcastgermany/config/germany.toml").read_text()
)["nowcast"]["split_date"]
REFERENCE_VALUES = "paper/data/reference-values.csv"
# The German and BDBV delays' structure (`delay_reference`), from the
# BDBV releases the backtests use: a cumulative first release, more than
# the backtests' longest delay before the next, is left out.
DELAY_REFERENCE = f"{ROUNDS_DIR}/report/delay-reference.json"
BDBV_MAX_DELAY = tomllib.loads(
    Path("packages/nowcastbdbv/config/backtest.toml").read_text()
)["forecast"]["max_delay"]


rule germany_triangle:
    output:
        GERMANY_TRIANGLE,
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
        delays=DELAY_REFERENCE,
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
        "{input.germany} {input.delays} {params.source} > {log} 2>&1"


rule sim_report_run:
    # Every round in `rounds` together.
    input:
        summaries=expand(
            f"{ROUNDS_DIR}/r{{round}}/summaries/outbreaks.csv",
            round=config["rounds"],
        ),
        germany=rules.germany_dev_stats.output,
        delays=DELAY_REFERENCE,
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
        "'{params.title}' {input.refs} {input.germany} {input.delays} "
        "{params.sources} "
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


rule hub_head:
    # The hub's data commit when a host has none yet, so the BDBV check
    # can be planned on a host with no hub_watch.sh, which keeps it
    # current.
    output:
        f"{HUB}/head.txt",
    log:
        f"{HUB}/logs/head.log",
    priority: HANDOFF_PRIORITY
    localrule: True
    conda:
        TOOLS_ENV
    shell:
        "gh api 'repos/InsightNet-US/BDBV-Modeling-Hub/commits"
        "?path=target-data/time-series.csv&per_page=1' --jq '.[0].sha' "
        "> {output} 2> {log}"


rule hub_data:
    # The hub's tidy data (`bdbvhub data`), including tasks.json.
    input:
        f"{HUB}/head.txt",
    output:
        f"{HUB}/data/target.csv",
    log:
        f"{HUB}/logs/data.log",
    priority: HANDOFF_PRIORITY
    localrule: True
    conda:
        TOOLS_ENV
    params:
        out=lambda wc, output: Path(output[0]).parent,
    shell:
        # A token, as GitHub limits unauthenticated reads.
        'GITHUB_TOKEN="${{GITHUB_TOKEN:-$(gh auth token 2>/dev/null)}}" '
        "uv run bdbvhub data --raw {params.out}/raw --out {params.out} "
        "> {log} 2>&1"


rule delay_reference:
    # The structure of the delays (flowlib.delay_structure) in the German
    # development half and the BDBV hub's releases, to compare the
    # simulations' with: in the sim reports and the simulations dashboard.
    input:
        germany=rules.germany_triangle.output,
        hub=rules.hub_data.output,
    output:
        DELAY_REFERENCE,
    log:
        f"{ROUNDS_DIR}/report/delay-reference.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        split=GERMANY_SPLIT,
        daily=lambda wc, input: Path(input.hub[0]).parent / "daily.csv",
        max_delay=BDBV_MAX_DELAY,
    shell:
        "uv run python workflow/scripts/sim_report.py delays {input.germany} "
        "{params.split} {params.daily} {params.max_delay} {output} "
        "> {log} 2>&1"


rule hub_submissions_table:
    # Each round's submitted model per hub id, from the ledger branch
    # (bdbvsubmission.submissions), so the report says which chain model
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
        "| tar -x -C {params.ledger} && uv run --quiet --project packages/bdbvsubmission "
        "submissions-table --state {params.ledger}/state.json "
        "--policy packages/bdbvsubmission/policy.toml --hub-config packages/bdbvhub/config/hub.toml "
        "--out {output}) > {log} 2>&1"


rule hub_comparison:
    # Every hub team's files and ours scored on the same truth, beside
    # `hub_rounds.models` as the fit store holds them for each round, with
    # their zone runs (`fit-store load --models`, then `compare-hub`). The
    # data folder is also set as BDBV_DATA_DIR, which the hub's places are
    # read from.
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
        "(rm -rf {params.out}/store && "
        "uv run fit-store-artifact fetch {params.out}/store && "
        "BDBV_DATA_DIR={params.data} uv run bdbvhub fit-store load "
        "--data {params.data} --store {params.out}/store "
        "--results {params.out}/results --models {params.models} && "
        "BDBV_DATA_DIR={params.data} uv run bdbvhub compare-hub --fetch "
        "--data {params.data} --hub-dir {params.data}/hub-submissions "
        "--cache-dir {params.data}/raw/hub-submissions --out {params.out} "
        "--store {params.out}/store --results {params.out}/results --stored "
        "--submissions-table {input.table}) > {log} 2>&1"


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
# reruns only it. `ft_check_page` scores them together for one checkpoint,
# ft-check/<ckpt>/bdbv-ft-check-<ckpt>.html, copied to REVIEW_DIR under
# that name, so each checkpoint is its own target. `ft_check` is the page
# of ft_check.checkpoint (or packages/nowcastfoundation/config/finetune.toml's
# pin), copied to
# REVIEW_DIR as bdbv-ft-check.html too; `finetune` asks for the page of
# every released checkpoint (`ft_check_released`). Not part of `all`.


rule ft_check_forecast:
    # Each folder carries the code key (`ft_check_folders`): the models'
    # code, the release tasks, the backtest config and the active-zone
    # rule, so other code makes its own folder and none is reused.
    input:
        data=rules.hub_data.output,
        # A fine-tuned checkpoint's release, which the forecast fetches.
        released=ft_check_record,
    output:
        touch(f"{FTC_DIR}/forecasts/{{fmodel}}@{{key}}.done"),
    log:
        f"{FTC_DIR}/logs/{{fmodel}}@{{key}}.log",
    benchmark:
        f"{FTC_DIR}/benchmarks/{{fmodel}}@{{key}}.tsv"
    wildcard_constraints:
        key=r"[0-9a-f]{12}",
    priority: ft_check_priority(CHECKPOINT_PRIORITY, REFERENCE_PRIORITY)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        # Its model's peak under this code, else under another code key,
        # else the most any check forecast used.
        mem_mb=memory.from_benchmark(
            f"{FTC_DIR}/benchmarks/{{fmodel}}@{{key}}.tsv",
            2048,
            like=(("key",), ("fmodel", "key")),
        ),
        gpu=FORECAST.gpu,
    params:
        env=unit_env,
        data=lambda wc, input: Path(input.data[0]).parent,
        out=lambda wc: f"{FTC_DIR}/forecasts/{wc.fmodel}@{wc.key}",
        model=lambda wc: (
            FTC["finetuned"] if wc.fmodel.startswith("ft-") else wc.fmodel
        ),
        checkpoint=lambda wc: wc.fmodel[3:] if wc.fmodel.startswith("ft-") else "",
    shell:
        "env {params.env} BDBV_DATA_DIR={params.data} "
        "uv run --group chronos bdbvhub ft-check forecast "
        "--model {params.model} --checkpoint '{params.checkpoint}' "
        "--data {params.data} --out {params.out} > {log} 2>&1"


rule ft_check_page:
    input:
        lambda wc: [f"{f}.done" for f in ft_check_folders(wc.ckpt)],
    output:
        ft_check_page("{ckpt}"),
    log:
        f"{FTC_DIR}/logs/report-{{ckpt}}.log",
    wildcard_constraints:
        ckpt=r"slots-[0-9a-f]+|[0-9a-f]+",
    priority: HANDOFF_PRIORITY
    localrule: True
    conda:
        TOOLS_ENV
    params:
        data=lambda wc: f"{HUB}/data",
        folders=lambda wc: " ".join(ft_check_folders(wc.ckpt)),
        reference=FTC["reference"],
        review=LOCAL["REVIEW_DIR"],
        stem=lambda wc, output: Path(output[0]).stem,
    shell:
        "(BDBV_DATA_DIR={params.data} uv run bdbvhub ft-check report "
        "--data {params.data} --reference {params.reference} "
        "--checkpoint {wildcards.ckpt} --out {output} {params.folders} "
        "&& if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "rm -rf '{params.review}/{params.stem}-data' && "
        "cp -r $(dirname {output})/{params.stem}-data '{params.review}/' && "
        "cp {output} '{params.review}/{params.stem}.html.tmp' && "
        "mv '{params.review}/{params.stem}.html.tmp' "
        "'{params.review}/{params.stem}.html'; fi) > {log} 2>&1"


rule ft_check:
    # The configured checkpoint's page, also as REVIEW_DIR's latest,
    # bdbv-ft-check.html (its data folder is the checkpoint's own).
    input:
        lambda wc: ft_check_page(ft_check_checkpoint()),
    log:
        f"{FTC_DIR}/logs/latest.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        review=LOCAL["REVIEW_DIR"],
    shell:
        "if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "cp {input} '{params.review}/bdbv-ft-check.html.tmp' && "
        "mv '{params.review}/bdbv-ft-check.html.tmp' "
        "'{params.review}/bdbv-ft-check.html'; fi > {log} 2>&1"


# The this-round page: every candidate's forecasts and nowcasts for the
# round the latest Submit run on main targeted, against the data as of
# the round, with the wk R_t quantiles, the gate's pick and the renewal
# fits' convergence (workflow/scripts/this_round_report.py). No scores.
# It reruns when <ROOT>/hub/round/run.txt (the run and its status),
# <ROOT>/hub/round/store.txt (the fit store's version) or
# <ROOT>/hub/round/submissions.txt (the hub's newest model-output commit)
# changes, which workflow/scripts/hub_watch.sh writes. Not part of `all`.


rule this_round_inputs:
    input:
        target=rules.hub_data.output,
        run=f"{HUB}/round/run.txt",
        store=f"{HUB}/round/store.txt",
        submissions=f"{HUB}/round/submissions.txt",
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
