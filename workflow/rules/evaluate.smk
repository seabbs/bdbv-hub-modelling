# Offloaded held-out evaluation (WP10): with `finetune.eval: offload`,
# score step 0 and each checkpoint, in
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
    threads: EVALUATE_THREADS
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
    threads: EVALUATE_THREADS
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
    threads: EVALUATE_THREADS
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


rule quick_check:
    # A few minutes' check of each checkpoint on a fixed sample of the
    # held-out blocks (nowcastfoundation.finetune.evaluate.quick_check):
    # its loss and coverage, for the dashboard's subset line and its
    # summary. On a whole GPU if the machine has one, above the trainer
    # (QUICK_PRIORITY), so it runs as the segment ends, before the next.
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{TRAIN}/n{{n}}/{evaluations.QUICK_CHECK}",
    log:
        f"{TRAIN}/logs/quick-check-n{{n}}.log",
    benchmark:
        f"{TRAIN}/benchmarks/quick_check/n{{n}}.tsv"
    priority: QUICK_PRIORITY
    conda:
        TOOLS_ENV
    threads: QUICK_THREADS
    resources:
        gpu=QUICK_FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=quick_env,
        device=QUICK_FORECAST.device,
        blocks=config["quick_check"]["blocks"],
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=lambda wc: [f"{train_dir(wc.run)}/{HELD_OUT_BLOCKS}"],
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} quick {{params.where}} --blocks {{params.blocks}} "
        "--device {params.device} > {log} 2>&1"


rule quick_check_base:
    # Base Chronos-2's quick check, once a run (step 0's snapshot), for
    # the reference line.
    input:
        settings=lambda wc: (
            f"{train_dir(wc.run)}/n{first_prefix(wc.run)}/settings.json"
        ),
    output:
        f"{TRAIN}/{evaluations.QUICK_BASE}",
    log:
        f"{TRAIN}/logs/quick-check-base.log",
    benchmark:
        f"{TRAIN}/benchmarks/quick_check_base.tsv"
    priority: QUICK_PRIORITY
    conda:
        TOOLS_ENV
    threads: QUICK_THREADS
    resources:
        gpu=QUICK_FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=quick_env,
        device=QUICK_FORECAST.device,
        blocks=config["quick_check"]["blocks"],
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=lambda wc: [
            f"{train_dir(wc.run)}/evals/step-0000000/",
            f"{train_dir(wc.run)}/{HELD_OUT_BLOCKS}",
        ],
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} quick-base {{params.where}} "
        "--blocks {params.blocks} --device {params.device} > {log} 2>&1"


rule validation_forecasts:
    # Each checkpoint's forecasts of a fixed sample of the run's held-out
    # (validation) blocks, with their history and truth, which the
    # dashboard sets against the truth
    # (nowcastfoundation.finetune.evaluate.validation_forecasts), on the
    # checkpoint scoring device, below every other job: nothing waits for
    # them.
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{TRAIN}/n{{n}}/{evaluations.VALIDATION_FILE}",
    log:
        f"{TRAIN}/logs/validation-forecasts-n{{n}}.log",
    benchmark:
        f"{TRAIN}/benchmarks/validation_forecasts/n{{n}}.tsv"
    priority: newest_first(EVALUATION_PRIORITY)
    conda:
        TOOLS_ENV
    threads: EVALUATE_THREADS
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
        f"{EVALUATE_PY} validation {{params.where}} --device {{params.device}} "
        "> {log} 2>&1"


rule validation_forecasts_base:
    # Base Chronos-2's forecasts of the same blocks, once a run, for the
    # dashboard's zero-shot line beside the checkpoints'.
    input:
        settings=lambda wc: (
            f"{train_dir(wc.run)}/n{first_prefix(wc.run)}/settings.json"
        ),
    output:
        f"{TRAIN}/{evaluations.VALIDATION_BASE}",
    log:
        f"{TRAIN}/logs/validation-forecasts-base.log",
    benchmark:
        f"{TRAIN}/benchmarks/validation_forecasts_base.tsv"
    priority: EVALUATION_PRIORITY
    conda:
        TOOLS_ENV
    threads: EVALUATE_THREADS
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
        f"{EVALUATE_PY} validation-base {{params.where}} "
        "--device {params.device} > {log} 2>&1"


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
