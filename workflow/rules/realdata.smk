# Fine-tuning on real data alone (`foundation_real`).
#
# A run with `data: real` (workflow/config.yaml `runs.realdata`) trains
# Chronos-2 from base on the German development half and the public BDBV
# releases, with the main run's recipe. Its blocks are the studies'
# forecast tasks laid out as the fine-tuned model forecasts them
# (nowcastfoundation.finetune.realtasks): the reporting triangle with its
# lag series at the data rule's lag count, every level in its blocks with
# their 7-day rows. It trains with the attention biases, in passes. Its
# blocks are cut once (`realdata_blocks_germany`,
# `realdata_blocks_bdbv`; the cut is packages/nowcastfoundation/config/
# finetune.toml `[realdata_germany]`, `[realdata_bdbv]`), and each
# checkpoint n<N> (thousandths of the training) is trained by `train_real` from the one before
# (`finetune-real`, nowcastfoundation.finetune.realdata). Its checkpoints
# are then scored, reported and published by the rules every run takes
# through the {run} wildcard (rules/score.smk, rules/report.smk,
# rules/publish.smk); a real checkpoint's release is
# `realdata-ft-<hash>`.
#
# Target: `finetune_realdata` (task workflow:finetune-realdata), which
# trains every checkpoint, scores them on the German development half,
# writes the run's tables and German report, and releases the final
# checkpoint. It writes a GitHub release, so it is never part of `all`.
#
# This is not `finetune.real_training`, which mixes German blocks into a
# run's simulations (rule real_training_blocks).


rule realdata_blocks_germany:
    # Every nowcast date with truth by the last day of the development
    # half: its German tasks (Germany with its states, with its ages),
    # from the triangle as available then, laid out as forecast.
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
    params:
        block_limit=local_or_ft("BLOCK_LIMIT", "block_limit"),
    shell:
        "uv run python -m nowcastgermany.finetune_blocks {input} {output} "
        "--realdata --block-limit {params.block_limit} > {log} 2>&1"


rule realdata_blocks_bdbv:
    # Every public BDBV release up to the one in `[realdata_bdbv]`: its
    # Ebola task laid out as forecast (the joint block of the country,
    # provinces and active zones, a block per province), truth masked
    # after `truth_until`.
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
        "uv run python -m bdbvhub.finetune_blocks {params.data} {output} "
        "--block-limit {params.block_limit} > {log} 2>&1"


rule train_real:
    # Checkpoint n<N> of a real run, N thousandths of its training, resumed from the state kept with
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


def realdata_inputs(wc=None):
    """What `finetune_realdata` asks of each real run: every checkpoint,
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


rule finetune_realdata:
    input:
        realdata_inputs,
