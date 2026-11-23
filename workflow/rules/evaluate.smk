# Offloaded held-out evaluation (WP10): with `finetune.eval: offload`,
# score each segment's step snapshots and its checkpoint after it ends, in
# the `evaluate` group (WORKER_GROUPS places it), on EVAL_DEVICE (the
# CPU by default) beside the trainer, newest segment first. Files and
# readers: workflow/flowlib/evaluations.py.


rule evaluate_steps:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        # Its steps start after the previous checkpoint's.
        prev=evaluate_prev,
        # After the previous segment's, so step 0 is scored first.
        before=evaluate_before,
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
    params:
        where=lambda wc, input: Path(input.settings).parent,
        prev=lambda wc, input: runs.option("--prev", evaluate_prev_dir(input)),
        env=unit_env,
        device=EVAL_DEVICE,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=evaluate_copy_in,
        copy_back=lambda wc: [f"{train_dir(wc.run)}/evals/"],
        eval_every=ft("eval_every"),
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} steps {{params.where}} {{params.prev}} "
        "--eval-every {params.eval_every} --device {params.device} "
        "> {log} 2>&1"


rule evaluate_checkpoint:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
        # Its segment's steps, chained back to step 0 (base Chronos-2).
        steps=f"{TRAIN}/evals/n{{n}}.done",
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
    params:
        where=lambda wc, input: Path(input.settings).parent,
        env=unit_env,
        device=EVAL_DEVICE,
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=evaluate_inputs_from,
    shell:
        "{params.env} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} checkpoint {{params.where}} --device {{params.device}} "
        "> {log} 2>&1"
