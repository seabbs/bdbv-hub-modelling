# Checkpoint scoring.
#
# Each checkpoint label gets the units of nowcastfoundation.checkpoint:
# `germany` (WIS on the German development half) and `release` (the
# release's settings, with the trainer's held-out loss).
# BDBV is held out for the paper, so no BDBV unit runs
# here; BDBV rows already in the tables stay.
# A unit caches one file per date or origin in <SCORE_CACHE>/<study>/
# <label>/ and resumes from it; its done stamp is written when it exits
# cleanly. `score_table` merges every label's rows into the evidence
# tables in <SCORE_CACHE>/tables/, newest checkpoint first. A unit runs
# torch on EVAL_DEVICE (the CPU by default; flowlib.machine.side_device)
# with EVAL_THREADS threads, beside the trainer, which keeps the GPU.
#
# Targets: `score`, or one unit, <SCORE_CACHE>/germany-days/<label>.done
# (`GERMANY_UNIT`); `germany` is the unit's study name.
# Opt-in, not in `score`: <SCORE_CACHE>/coupling/<label>-summary.csv, the
# SI's evidence on joining quantiles into trajectories
# (nowcasteval.coupling), from that unit's cached quantiles, on the CPU.
# `land` (opt-in) commits the tables to evidence/ and pushes.


wildcard_constraints:
    label=r"slots-[0-9a-f]+|[0-9a-f]+|reference[a-z-]*",


checkpoint release_list:
    # Lists the releases to score; `task nowcast-foundation:score` reruns it.
    output:
        f"{CACHE}/releases.json",
    log:
        f"{CACHE}/log/release-list.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
    shell:
        "gh release list -R {params.repo} --limit 500 "
        "--json tagName,publishedAt > {output} 2>{log}"


rule score_germany:
    output:
        f"{CACHE}/{GERMANY_UNIT}/{{label}}.done",
    log:
        f"{CACHE}/log/{GERMANY_UNIT}-{{label}}.log",
    benchmark:
        f"{CACHE}/benchmarks/score_germany/{{label}}.tsv"
    priority: unit_priority(5)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
        copy_back=unit_rows(GERMANY_UNIT),
        cache=lambda wc, output: Path(output[0]).parents[1],
        label=lambda wc: wc.label,
    shell:
        unit_shell("germany")


rule score_run_germany:
    # A run's own checkpoint n<N>, loaded from its segment folder with no
    # release, cached in the run's folder.
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{RUN}/scores/{GERMANY_UNIT}/n{{n}}.done",
    log:
        f"{RUN}/scores/log/{GERMANY_UNIT}-n{{n}}.log",
    benchmark:
        f"{RUN}/scores/benchmarks/score_run_germany/n{{n}}.tsv"
    priority: newest_first(5)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
        copy_back=lambda wc: f"{run_scores_dir(wc.run)}/{GERMANY_UNIT}/n{wc.n}/",
        cache=lambda wc, output: Path(output[0]).parents[1],
        label=lambda wc: f"n{wc.n}",
        segment=lambda wc, input: Path(input.settings).parent,
    shell:
        unit_shell("germany", "{params.segment} ")


rule run_scores:
    # Each trained run's saved checkpoints, scored and reported.
    input:
        run_scores,


rule run_table:
    # A run's saved checkpoints in the tables' rows (German relative
    # scores and held-out losses), under the run's name.
    input:
        references=references(),
        units=lambda wc: run_units(wc.run),
        offloaded=lambda wc: [
            f for n in saved_prefixes(wc.run) for f in offloaded(wc.run, n)
        ],
    output:
        f"{RUN}/reports/checkpoint-scores.csv",
    log:
        f"{RUN}/logs/run-table.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        evaluate=EVALUATE,
        train=lambda wc: train_dir(wc.run),
        scores=lambda wc: f"{run_scores_dir(wc.run)}/{GERMANY_UNIT}",
        reference=lambda wc, input: input.references[0].removesuffix(".done"),
    shell:
        "{params.evaluate} run {params.train} {params.scores} "
        "{params.reference} {wildcards.run} {output} > {log} 2>&1"


rule comparison:
    # Every run's checkpoints in one table, for the fine-tuning
    # comparisons (paperpipe.evidence `compare_*`).
    input:
        expand(f"{RUN}/reports/checkpoint-scores.csv", run=list(RUNS)),
    output:
        f"{RUNS_DIR}/comparison.csv",
    log:
        f"{RUNS_DIR}/logs/comparison.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        evaluate=EVALUATE,
    shell:
        "{params.evaluate} compare {output} {input} > {log} 2>&1"


rule coupling_germany:
    input:
        f"{CACHE}/{GERMANY_UNIT}/{{label}}.done",
    output:
        scores=f"{CACHE}/coupling/{{label}}-scores.csv",
        summary=f"{CACHE}/coupling/{{label}}-summary.csv",
    log:
        f"{CACHE}/log/coupling-{{label}}.log",
    benchmark:
        f"{CACHE}/benchmarks/coupling_germany/{{label}}.tsv"
    conda:
        TOOLS_ENV
    threads: 4
    params:
        unit=lambda wc: f"{CACHE}/{GERMANY_UNIT}/{wc.label}",
    shell:
        "uv run python -m nowcasteval.coupling {params.unit} "
        "{output.scores} {output.summary} > {log} 2>&1"


rule score_release:
    output:
        f"{CACHE}/heldout/{{label}}.done",
    log:
        f"{CACHE}/log/heldout-{{label}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        evaluate=EVALUATE,
        cache=lambda wc: CACHE,
    shell:
        "{params.evaluate} release {params.cache} {wildcards.label} > {log} 2>&1 "
        "&& touch {output}"


rule score_table:
    input:
        unpack(table_inputs),
    output:
        evidence=f"{CACHE}/tables/checkpoint-scores.csv",
        checkpoints=f"{CACHE}/tables/checkpoints.csv",
    log:
        f"{CACHE}/log/score-table.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        labels=lambda wc: " ".join(checkpoint_labels()),
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        cache=lambda wc: CACHE,
        evaluate=EVALUATE,
    shell:
        # Rows of other rules in the committed tables are kept.
        """
        exec >{log} 2>&1
        [ -f {params.evidence} ] && cp {params.evidence} {output.evidence}
        [ -f {params.checkpoints} ] && cp {params.checkpoints} {output.checkpoints}
        {params.evaluate} table {params.cache} \
            {output.evidence} {output.checkpoints} {params.labels}
        """


rule land:
    # Opt-in: commit the tables to evidence/ on the evidence branch only
    # (never main) and push, checking the commit reached origin.
    input:
        rules.score_table.output,
    output:
        touch(f"{CACHE}/tables/landed"),
    log:
        f"{CACHE}/log/land.log",
    localrule: True
    params:
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        branch=SC["branch"],
    run:
        land_tables(input, output, params, wildcards)


rule score:
    input:
        # The units are asked for themselves too: with mtime triggers alone
        # (the profiles'), a new label's units missing under tables already
        # built would be "missing, no update" and never scored.
        unit_stamps,
        rules.score_table.output,
        run_scores,
