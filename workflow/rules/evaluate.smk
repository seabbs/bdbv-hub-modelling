# Offloaded held-out evaluation (WP10): with `finetune.eval: offload`,
# score step 0, each segment's step snapshots and its checkpoint, in
# the `evaluate` group (WORKER_GROUPS places it), on the forecasting
# device (`forecast` in the config, flowlib.forecast) beside the trainer, newest segment first. Files and
# readers: workflow/flowlib/evaluations.py.


rule evaluate_steps:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        # Its steps start after the previous checkpoint's.
        prev=evaluate_prev,
        # Step 0, scored on its own.
        base=f"{TRAIN}/evals/step-0000000.json",
    output:
        touch(f"{TRAIN}/evals/n{{n}}.done"),
    log:
        f"{TRAIN}/logs/evaluate-steps-n{{n}}.log",
    benchmark:
        f"{TRAIN}/benchmarks/evaluate_steps/n{{n}}.tsv"
    priority: newest_first(EVALUATE_PRIORITY)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        prev=lambda wc, input: runs.option("--prev", evaluate_prev_dir(input)),
        env=unit_env,
        device=FORECAST.device,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=evaluate_copy_in,
        copy_back=lambda wc: [f"{train_dir(wc.run)}/evals/"],
        eval_every=ft("eval_every"),
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} steps {{params.where}} {{params.prev}} "
        "--eval-every {params.eval_every} --device {params.device} "
        "> {log} 2>&1"


rule evaluate_base:
    # Step 0 (base Chronos-2) on the held-out blocks, before any other
    # evaluation: the start loss each checkpoint's sidecar needs.
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
    priority: EVALUATE_PRIORITY + 1
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=unit_env,
        device=FORECAST.device,
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
    priority: newest_first(EVALUATE_PRIORITY)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        gpu=FORECAST.gpu,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=unit_env,
        device=FORECAST.device,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=evaluate_inputs_from,
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} checkpoint {{params.where}} --device {{params.device}} "
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
