# Fine-tuning.
#
# One continuous run of scripts/finetune_stream.py from base Chronos-2,
# cut into one segment per checkpoint prefix: segment N resumes from the
# state kept with the prefix before and stops once n<N> is saved
# (`--until N`). Its inputs are the batches that hold the first N
# outbreaks of the seed list, one spare, and the held-out batches,
# gathered in <ROOT>/runs/<run>/sims/ from this machine's rounds, the rounds host's
# (REMOTE) or the sims-r<N> releases (`fetch_batch`).
#
# Targets: `finetune` (every prefix of each run in `train_runs`), or one
# checkpoint,
# <ROOT>/runs/<run>/train/n<N>/settings.json. Publishing is the publish module's.
#
# A machine without the rounds takes its batches from the rounds host
# (`sims_from: remote`, REMOTE and REMOTE_ROOT in config/local.env), as
# each round's manifest lists them, or from the sims-r<N> releases
# (`release`). A segment is planned only once its batches can be had: the
# training seeds count up to the first batch not yet listed, and nothing
# trains before the whole validation round is (`listed_batches`, listed
# afresh each time a run plans). From the rounds host, its rounds'
# manifests and QC summaries are mirrored in <ROOT>/runs/<run>/rounds/.


rule fetch_batch:
    input:
        unpack(fetch_source),
    output:
        counts=f"{RUN}/sims/counts-{{seed}}.csv.gz",
        settings=f"{RUN}/sims/settings-{{seed}}.csv",
    log:
        f"{RUN}/logs/fetch-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        source=ft("sims_from"),
        repo=SIM["repo"],
        tag=lambda wc: f"sims-r{round_of(wc.seed)}",
        sims=lambda wc, output: Path(output.counts).parent,
        src=lambda wc, input: input.get("counts", ""),
        src_settings=lambda wc, input: input.get("settings", ""),
        there=lambda wc: hosts._at(REMOTE, f"{REMOTE_ROUNDS}/r{round_of(wc.seed)}/sims"),
    shell:
        # A copy where the rounds are on this machine (not a hard link:
        # Snakemake touches outputs one at a time, which through a link
        # moves the inputs' times too and can leave the counts older than
        # the settings input), a copy from the rounds host, else the
        # batch's assets of its round's release.
        """
        exec >{log} 2>&1
        if [ "{params.source}" = local ]; then
            cp {params.src} {output.counts}
            cp {params.src_settings} {output.settings}
        elif [ "{params.source}" = remote ]; then
            rsync -a {params.there}/counts-{wildcards.seed}.csv.gz \
                {params.there}/settings-{wildcards.seed}.csv {params.sims}/
        else
            gh release download {params.tag} -R {params.repo} --clobber \
                -D {params.sims} -p counts-{wildcards.seed}.csv.gz \
                -p settings-{wildcards.seed}.csv
        fi
        gzip -t {output.counts}
        """


rule seed_list:
    output:
        f"{RUN}/seeds.txt",
    log:
        f"{RUN}/logs/seed-list.log",
    localrule: True
    run:
        manifest.write_seeds(training_seeds(wildcards.run), output[0])


rule train_segment:
    input:
        unpack(segment_inputs),
    output:
        settings=protected(f"{TRAIN}/n{{n}}/settings.json"),
        adapter=protected(f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors"),
    log:
        f"{TRAIN}/n{{n}}.log",
    # The trainer, the run's critical path, goes before every other job.
    priority: TRAIN_PRIORITY
    conda:
        TOOLS_ENV
    resources:
        gpu=1,
    params:
        prev_args=lambda wc: runs.resume_options(run_of(wc), int(wc.n)),
        held=lambda wc: held_range(wc.run),
        prefixes=lambda wc: " ".join(str(p) for p in run_of(wc).ft["prefixes"]),
        stop_from=lambda wc: run_of(wc).ft["outbreaks"] + 1,
        blocks=lambda wc: run_of(wc).ft["cuts"] * run_of(wc).ft["draws"],
        sources=lambda wc: " ".join(run_of(wc).ft["sources"]),
        options=lambda wc: runs.trainer_options(run_of(wc)),
        device=machine.device(LOCAL),
        # The checkpoint threshold: this machine's, if set, else the
        # config's (memory only).
        checkpoint=local_or_ft("CHECKPOINT_ABOVE", "checkpoint_above"),
        batch_size=ft("batch_size"),
        # The block capacity: this machine's, if set, else the config's.
        block_limit=local_or_ft("BLOCK_LIMIT", "block_limit"),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        watermark=LOCAL.get("MPS_HIGH_WATERMARK", "0.85"),
        # The run's folders, as functions: they hold the outputs.
        train=lambda wc: train_dir(wc.run),
        run=lambda wc: run_of(wc).dir,
        grid_file=GRID_FILE,
        holdout=ft("holdout"),
        outbreaks=ft("outbreaks"),
        draws=ft("draws"),
        subset_seed=ft("subset_seed"),
        learning_rate=ft("learning_rate"),
        univariate_share=ft("univariate_share"),
        note=ft("note"),
        chunk=ft("chunk"),
        eval_every=ft("eval_every"),
        by_source_every=ft("by_source_every"),
        eval_mode=ft("eval"),
        copy_back=segment_copy_back,
    shell:
        # The run's own state resumes it; the state kept with the prefix
        # before replaces it if missing or behind (flowlib.segments).
        """
        uv run python workflow/scripts/prepare_segment.py {params.train} \
            {params.prev_args} >{log}
        DEVICE={params.device} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 \
            PYTORCH_MPS_HIGH_WATERMARK_RATIO={params.watermark} \
            PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 MPLCONFIGDIR="${{TMPDIR:-/tmp}}" \
            uv run --group chronos python scripts/finetune_stream.py \
            {params.run}/sims {params.train} --grid-file {params.grid_file} \
            --seed-list {input.seeds} --held-out-seeds {params.held} \
            --holdout {params.holdout} --outbreaks {params.outbreaks} \
            --prefixes {params.prefixes} \
            --stop-from {params.stop_from} \
            --blocks-per-simulation {params.blocks} \
            --draws-per-simulation {params.draws} \
            --subset-seed {params.subset_seed} \
            --learning-rate {params.learning_rate} \
            --sources {params.sources} \
            --univariate-share {params.univariate_share} \
            --note "{params.note}" --chunk {params.chunk} \
            --eval-every {params.eval_every} \
            --by-source-every {params.by_source_every} \
            --eval {params.eval_mode} \
            {params.options} \
            --checkpoint-above {params.checkpoint} \
            --batch-size {params.batch_size} \
            --block-limit {params.block_limit} \
            --until {wildcards.n} >>{log} 2>&1
        # An optional notification line; it never fails the rule.
        uv run python workflow/scripts/segment_saved.py {output.settings} \
            {wildcards.n} || true
        """


rule checkpoint_check:
    # A report, never a gate: does checkpoint `check_at` beat zero-shot on
    # the held-out simulations? An offloaded run's losses are in the
    # checkpoint's sidecar (rules/evaluate.smk).
    input:
        f"{TRAIN}/n{{n}}/settings.json",
        lambda wc: offloaded(wc.run, wc.n),
    output:
        f"{RUN}/reports/n{{n}}-check.txt",
    log:
        f"{RUN}/logs/n{{n}}-check.log",
    localrule: True
    run:
        check_report(input, output, params, wildcards)


rule finetune:
    input:
        finetune_inputs,


rule dashboard:
    # A static render of the dashboard for the configured runs, for a
    # finished run or the record; `task nowcast-foundation:dashboard` serves it live instead.
    output:
        f"{LOCAL['ROOT']}/reports/dashboard.html",
    log:
        f"{LOCAL['ROOT']}/logs/dashboard.log",
    localrule: True
    conda:
        TOOLS_ENV
    shell:
        "uv run --group paper python workflow/scripts/dashboard.py "
        "--once {output} > {log} 2>&1"
