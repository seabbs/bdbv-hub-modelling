# Simulation rounds.
#
# Round r is the batches of `flowlib.segments.round_seeds`, each
# `batch` simulations of the grid (`simulate.grid_file`; each simulation
# the joint PyRenew model, nowcastrenewal.simulation) with exactly
# `draws` observation draws.
# Into
# <ROOT>/rounds/r<r>/sims/. Each batch gets a manifest row, the round's
# manifest joins them, the sim QC (scripts/sim_qc.py) checks the round
# against the grid's thresholds (its `qc`),
# and `round_ready` writes the seed lists and READY last. Every
# simulation is kept: QC exclusions are a list (qc/excluded.csv).
#
# Targets: `simulate` (the rounds in `rounds`), and the opt-in `release`,
# which uploads each batch to the release sims-r<r>.
#
# Two hosts can share the rounds (flowlib.hosts): each lists the seeds
# it takes in its local config (SIM_SEEDS; empty: all). A helping
# host's target is `push`, which sends each of its batches to the rounds
# host (REMOTE) with its manifest row; the rounds host's `simulate`
# simulates its own seeds and waits for the rest.


wildcard_constraints:
    round=r"\d+",
    seed=r"\d+",
    n=r"\d+",


rule simulate_batch:
    output:
        counts=f"{ROUNDS_DIR}/r{{round}}/sims/counts-{{seed}}.csv.gz",
        settings=f"{ROUNDS_DIR}/r{{round}}/sims/settings-{{seed}}.csv",
        latent=f"{ROUNDS_DIR}/r{{round}}/sims/latent-{{seed}}.csv.gz",
    log:
        f"{ROUNDS_DIR}/r{{round}}/sims/logs/{{seed}}.log",
    # In one pool, after the trainer (20) and before scoring (10 and
    # below): the trainer waits on these batches. Among themselves in
    # seed order, the order training reads them, so a run over several
    # rounds finishes each round before the next.
    priority: batch_priority
    retries: 1
    conda:
        TOOLS_ENV
    threads: 1
    resources:
        mem_mb=lambda wc, attempt: (
            SIM["mem_mb"] if attempt == 1 else SIM["retry_mem_mb"]
        ),
        # At most `--resources sim_slots=N` batches at once (2 by default).
        sim_slots=1,
    params:
        sims=lambda wc, output: Path(output.counts).parent,
        batch=SIM["batch"],
        draws=SIM["draws"],
        grid=GRID_FILE,
        engine=" ".join(simulator.ENGINE),
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        min_free=LOCAL["MIN_FREE_GB"],
        min_disk=LOCAL["MIN_DISK_GB"],
        max_load=LOCAL["MAX_LOAD"],
    shell:
        # The simulator (its code and the grid) must be the committed one,
        # as the manifest records the commit. A retry keeps the first
        # attempt's log. The simulator writes temporary files to a folder of
        # the batch's own, removed however the batch ends. JAX runs on one
        # thread, as the job has one.
        """
        if ! git diff --quiet HEAD -- {params.engine} {params.grid}; then
            echo "the simulator or its grid differs from HEAD" >{log}
            exit 1
        fi
        workflow/scripts/wait_for_room.sh {params.min_free} {params.min_disk} \
            {params.sims} {params.max_load}
        [ -s {log} ] && mv {log} {params.sims}/logs/{wildcards.seed}.first.log
        tmp={params.sims}/tmp/{wildcards.seed}
        rm -rf "$tmp" && mkdir -p "$tmp"
        trap 'rm -rf "$tmp"' EXIT
        uv=$(command -v uv)
        TMPDIR="$tmp" ENW_DRAWS={params.draws} JAX_PLATFORMS=cpu \
            XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
            {params.prefix} workflow/scripts/timed.sh \
            nice -n 19 "$uv" run --frozen python -m nowcastfoundation.simulator \
            {params.grid} {params.batch} {wildcards.seed} {params.sims} >{log} 2>&1
        gzip -t {output.counts}
        """


rule await_batch:
    # A batch another host simulates (not in SIM_SEEDS): done once that
    # host has pushed its files and then its row to this round's
    # manifest. One waits at a time (`--resources awaiting=1`).
    output:
        f"{ROUNDS_DIR}/r{{round}}/arrived/{{seed}}.done",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/await-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    resources:
        awaiting=1,
    params:
        manifest=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
        poll=60,
    shell:
        """
        exec >{log} 2>&1
        until grep -q '^{wildcards.round},{wildcards.seed},' \
            {params.manifest} 2>/dev/null; do
            sleep {params.poll}
        done
        touch {output}
        """


rule batch_record:
    input:
        unpack(batch_files),
    output:
        f"{ROUNDS_DIR}/r{{round}}/records/record-{{seed}}.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/record-{{seed}}.log",
    localrule: True
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        old=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    run:
        write_record(input, output, params, wildcards)


rule round_manifest:
    input:
        lambda wc: expand(
            f"{ROUNDS_DIR}/r{wc.round}/records/record-{{seed}}.csv",
            seed=seeds_of(wc.round),
        ),
    output:
        f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/manifest.log",
    localrule: True
    run:
        manifest.join(input, output[0])


rule sim_qc:
    input:
        rules.round_manifest.output,
    output:
        excluded=f"{ROUNDS_DIR}/r{{round}}/qc/excluded.csv",
        summary=f"{ROUNDS_DIR}/r{{round}}/qc/summary.txt",
    log:
        f"{ROUNDS_DIR}/r{{round}}/qc/run.log",
    conda:
        TOOLS_ENV
    params:
        round_dir=lambda wc, input: Path(input[0]).parent,
    shell:
        "uv run python scripts/sim_qc.py {params.round_dir} > {log} 2>&1"


rule round_ready:
    input:
        rules.sim_qc.output.summary,
    output:
        seeds=f"{ROUNDS_DIR}/r{{round}}/seeds.txt",
        ready=f"{ROUNDS_DIR}/r{{round}}/READY",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/ready.log",
    localrule: True
    run:
        mark_ready(input, output, params, wildcards)


# Refuses to upload into the release of an earlier grid's round
# (`retired_rounds`, flowlib.grids).
GRID_GUARD = """
        if [ "{params.owner}" = retired ]; then
            echo "{params.tag} holds an earlier grid's batches: not uploading"
            exit 1
        fi
"""


rule release_batch:
    # Opt-in (target `release`): uploads a batch to the release sims-r<r>,
    # so another machine can train on it before the round ends.
    input:
        unpack(batch_files),
        record=rules.batch_record.output,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/release-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        owner=lambda wc: grids.owner(
            int(wc.round), config["grid_rounds"], config.get("retired_rounds")
        ),
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    shell:
        """
        exec >{log} 2>&1
        """
        + GRID_GUARD
        + """
        gh release view {params.tag} -R {params.repo} >/dev/null 2>&1 \
            || gh release create {params.tag} -R {params.repo} --prerelease \
                --title "Simulations, round {wildcards.round}" \
                --notes "Fine-tuning simulations of round {wildcards.round}, one batch per asset, with their manifest rows." \
                >/dev/null 2>&1 \
            || gh release view {params.tag} -R {params.repo} >/dev/null
        gh release upload {params.tag} -R {params.repo} --clobber \
            {params.counts} {params.settings} {input.record}
        """


rule release_qc:
    # Opt-in (target `release`): the round's QC exclusions, so a machine
    # training from the releases skips them too.
    input:
        rules.sim_qc.output.excluded,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/qc.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/release-qc.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        named=f"{ROUNDS_DIR}/r{{round}}/qc/excluded-r{{round}}.csv",
        owner=lambda wc: grids.owner(
            int(wc.round), config["grid_rounds"], config.get("retired_rounds")
        ),
    shell:
        """
        exec >{log} 2>&1
        """
        + GRID_GUARD
        + """
        cp {input} {params.named}
        gh release upload {params.tag} -R {params.repo} --clobber {params.named}
        """


rule push_batch:
    # A helping host's batch, sent to the rounds host's round: its files,
    # then its manifest row (flowlib.hosts.push).
    input:
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/sent/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/push-{{seed}}.log",
    retries: 3
    localrule: True
    params:
        sims=lambda wc, input: str(Path(input.counts).parent),
        there=lambda wc: f"{REMOTE_ROUNDS}/r{wc.round}",
    run:
        push_record(input, output, params, wildcards)


rule push:
    # A helping host's target: its seeds (SIM_SEEDS) of the rounds in
    # `rounds`.
    input:
        [
            f"{ROUNDS_DIR}/r{r}/sent/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
            if here(s)
        ],


rule simulate:
    input:
        expand(f"{ROUNDS_DIR}/r{{round}}/READY", round=config["rounds"]),


rule round_summaries:
    # One row per outbreak of a round (first draw, strata summed): design,
    # drawn values, size, dynamics, R, delays and the phase at the
    # training cut-offs (workflow/scripts/grid_summaries.py). From the
    # round's batch files, not its manifest, so a summary of a round
    # another run is simulating writes nothing into that round's records.
    input:
        lambda wc: [
            f"{ROUNDS_DIR}/r{wc.round}/sims/{kind}-{seed}.{ext}"
            for seed in seeds_of(wc.round)
            for kind, ext in (
                ("counts", "csv.gz"),
                ("settings", "csv"),
                ("latent", "csv.gz"),
            )
        ],
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
        sources=lambda wc, input: " ".join(
            f"r{r}={path}" for r, path in zip(config["rounds"], input)
        ),
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
        source=lambda wc, input: (
            f"r{wc.round}={ROUNDS_DIR}/r{wc.round}/sims:{input.summaries}"
        ),
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
        sources=lambda wc, input: " ".join(
            f"r{r}={ROUNDS_DIR}/r{r}/sims:{path}"
            for r, path in zip(config["rounds"], input.summaries)
        ),
    shell:
        "uv run python workflow/scripts/sim_report.py render {output} "
        "'{params.title}' {input.refs} {input.germany} {params.sources} "
        "> {log} 2>&1"


rule sim_reports:
    input:
        expand(
            f"{ROUNDS_DIR}/r{{round}}/report/sim-report.html",
            round=config["rounds"],
        ),
        rules.sim_report_run.output,


rule release:
    input:
        # Never an earlier grid's round: its release holds that grid's
        # batches.
        lambda wc: grids.check_release(
            config["rounds"], config["grid_rounds"], config.get("retired_rounds")
        )
        or [],
        [
            f"{ROUNDS_DIR}/r{r}/released/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
        ],
        expand(f"{ROUNDS_DIR}/r{{r}}/released/qc.done", r=config["rounds"]),
