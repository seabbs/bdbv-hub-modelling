# Fine-tuning on real data alone (`foundation_real`).
#
# A run with `data: real` (workflow/config.yaml `runs.realonly`) trains
# Chronos-2 from base on the German development half and the public BDBV
# releases, with the main run's recipe: the reporting triangle with its
# lag series, every level in one block with its 7-day rows, and the
# attention biases. Its blocks are cut once (`realonly_blocks_germany`,
# `realonly_blocks_bdbv`; the cut is packages/nowcastfoundation/config/
# finetune.toml `[realdata_germany]`, `[realdata_bdbv]`), and each
# checkpoint n<steps> is trained by `train_real` from the one before
# (`finetune-real`, nowcastfoundation.finetune.realdata). Its checkpoints
# are then scored, reported and published by the rules every run takes
# through the {run} wildcard (rules/score.smk, rules/report.smk,
# rules/publish.smk); a real checkpoint's release is
# `realdata-ft-<hash>`.
#
# Target: `finetune_realonly` (task workflow:finetune-realonly), which
# trains every checkpoint, scores them on the German development half,
# writes the run's tables and German report, and releases the final
# checkpoint. It writes a GitHub release, so it is never part of `all`.
#
# This is not the `realdata` run, which mixes German blocks into Run A's
# simulations (`finetune.real_training`, rule real_training_blocks).


rule realonly_blocks_germany:
    # The German triangle as available on the last day of the development
    # half, cut at every nowcast date with truth by then, from the daily
    # and weekly releases, each German task whole.
    wildcard_constraints:
        run=REAL_RUN,
    input:
        GERMANY_TRIANGLE,
    output:
        f"{RUN}/real/germany.pkl",
    log:
        f"{RUN}/logs/real-germany.log",
    conda:
        TOOLS_ENV
    threads: 4
    shell:
        "uv run python -m nowcastgermany.finetune_blocks {input} {output} "
        "--realdata --workers {threads} > {log} 2>&1"


rule realonly_blocks_bdbv:
    # Every public BDBV release up to the one in `[realdata_bdbv]`: daily
    # blocks of the country, provinces and zones at each daily release,
    # weekly blocks from the release's delay table, truth masked after
    # `truth_until`.
    wildcard_constraints:
        run=REAL_RUN,
    input:
        f"{HUB}/data/target.csv",
    output:
        f"{RUN}/real/bdbv.pkl",
    log:
        f"{RUN}/logs/real-bdbv.log",
    conda:
        TOOLS_ENV
    params:
        data=lambda wc, input: Path(input[0]).parent,
        block_limit=local_or_ft("BLOCK_LIMIT", "block_limit"),
    shell:
        "uv run python -m nowcastbdbv.finetune_blocks {params.data} {output} "
        "--block-limit {params.block_limit} > {log} 2>&1"


rule train_real:
    # Checkpoint n<steps> of a real run, resumed from the state kept with
    # the checkpoint before, on a GPU of its own as a segment is.
    wildcard_constraints:
        run=REAL_RUN,
    input:
        unpack(TRAINING.real_inputs),
    output:
        settings=protected(f"{TRAIN}/n{{n}}/settings.json"),
        adapter=protected(f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors"),
    log:
        f"{TRAIN}/logs/train-real-n{{n}}.log",
    priority: TRAINING.trainer_priority
    conda:
        TOOLS_ENV
    threads: 1
    resources:
        gpu=1,
        side_cores=0,
        side_mem_mb=0,
    params:
        train=lambda wc: train_dir(wc.run),
        prev=lambda wc, input: (
            f"--prev {Path(input.get('prev')).parent}/state"
            if input.get("prev")
            else ""
        ),
        options=lambda wc: runs.real_trainer_options(run_of(wc)),
        checkpoint=local_or_ft("CHECKPOINT_ABOVE", "checkpoint_above"),
        note=ft("note"),
        device=machine.device(LOCAL),
        claim=machine.CLAIM,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
    shell:
        "{params.claim} DEVICE={params.device} HF_HOME={params.hf_home} "
        "HF_HUB_OFFLINE=1 MPLCONFIGDIR=\"${{TMPDIR:-/tmp}}\" "
        "uv run --group chronos finetune-real {params.train} "
        "--blocks {input.germany} {input.bdbv} {params.options} "
        "--until {wildcards.n} {params.prev} "
        "--checkpoint-above {params.checkpoint} "
        "--note \"{params.note}\" > {log} 2>&1"


def realonly_inputs(wc=None):
    """What `finetune_realonly` asks of each real run: every checkpoint,
    its German unit, the run's tables and German report, and the final
    checkpoint's release."""
    out = []
    for name in TRAINING.real_runs():
        run = RUNS[name]
        ns = run.ft["prefixes"]
        out += [f"{run.train}/n{n}/settings.json" for n in ns]
        out += [f"{run.dir}/scores/{GERMANY_UNIT}/n{n}.done" for n in ns]
        out += [
            f"{run.dir}/reports/checkpoint-scores.csv",
            f"{run.dir}/reports/horizon-scores.csv",
            f"{run.dir}/reports/german/index.html",
            f"{run.train}/n{max(ns)}/published.json",
        ]
    return out


rule finetune_realonly:
    input:
        realonly_inputs,
