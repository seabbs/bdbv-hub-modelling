# Offloaded held-out evaluation (WP10): with `finetune.eval: offload`,
# score step 0 and each checkpoint, not the step snapshots between, in
# the `evaluate` group (WORKER_GROUPS places it), on the forecasting
# device (`forecast` in the config, flowlib.forecast) beside the trainer,
# newest segment first, below every other job (EVALUATION_PRIORITY): no
# release or check waits for it. Files and readers:
# workflow/flowlib/evaluations.py.


rule evaluate_base:
    # Step 0 (base Chronos-2) on every held-out block: the start loss
    # each checkpoint's sidecar needs.
    input:
        settings=lambda wc: (
            f"{train_dir(wc.run)}/n{first_prefix(wc.run)}/settings.json"
        ),
    output:
        f"{TRAIN}/evals/step-0000000.json",
    log:
        f"{TRAIN}/logs/evaluate-base.log",
    benchmark:
        f"{TRAIN}/benchmarks/evaluate_base.tsv"
    # Above every checkpoint's evaluation, which needs it.
    priority: EVALUATION_PRIORITY + 1
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=CHECKPOINT_FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=checkpoint_env,
        device=CHECKPOINT_FORECAST.device,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=lambda wc: [
            f"{train_dir(wc.run)}/evals/step-0000000/",
            f"{train_dir(wc.run)}/{HELD_OUT_BLOCKS}",
        ],
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} base {{params.where}} --device {{params.device}} "
        "> {log} 2>&1"


rule evaluate_checkpoint:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
        # Step 0 (base Chronos-2) only, for the start loss: not the
        # segments' step evaluations, which queue behind the trainer.
        base=f"{TRAIN}/evals/step-0000000.json",
        # Earlier checkpoints' settings: `evaluations.start_loss` reads their
        # held_out_loss_start when the run evaluated them in the trainer.
        earlier=earlier_settings,
    output:
        f"{TRAIN}/n{{n}}/{SIDECAR}",
    log:
        f"{TRAIN}/logs/evaluate-checkpoint-n{{n}}.log",
    benchmark:
        f"{TRAIN}/benchmarks/evaluate_checkpoint/n{{n}}.tsv"
    priority: newest_first(EVALUATION_PRIORITY)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=CHECKPOINT_FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=checkpoint_env,
        device=CHECKPOINT_FORECAST.device,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=evaluate_inputs_from,
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} checkpoint {{params.where}} --device {{params.device}} "
        "> {log} 2>&1"


rule seven_day_check:
    # Each checkpoint's 7-day row against the 7-day sums
    # of its daily rows (median ratio, sums in its 90% interval) and the
    # roughness of its median against the truth's, on the run's held-out
    # blocks (nowcastfoundation.finetune.evaluate.seven_day_check), on the
    # forecasting device beside the trainer, however the run evaluates.
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{TRAIN}/n{{n}}/seven-day.json",
    log:
        f"{TRAIN}/logs/seven-day-n{{n}}.log",
    benchmark:
        f"{TRAIN}/benchmarks/seven_day_check/n{{n}}.tsv"
    priority: newest_first(CHECKPOINT_PRIORITY)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=CHECKPOINT_FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=checkpoint_env,
        device=CHECKPOINT_FORECAST.device,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=lambda wc: [f"{train_dir(wc.run)}/{HELD_OUT_BLOCKS}"],
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} seven-day {{params.where}} --device {{params.device}} "
        "> {log} 2>&1"


rule validation_curve:
    # The run's validation loss by level at the start and at each
    # checkpoint, against simulations trained: the paper's fine-tune
    # curve (flowlib.evaluations.validation_curve).
    input:
        validation_curve_inputs,
    output:
        f"{RUN}/validation-curve.csv",
    log:
        f"{RUN}/logs/validation-curve.log",
    localrule: True
    params:
        train=lambda wc: train_dir(wc.run),
        prefixes=lambda wc: RUNS[wc.run].ft["prefixes"],
    run:
        evaluations.write_validation_curve(
            params.train, wildcards.run, params.prefixes, output[0]
        )
