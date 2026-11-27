# Publishing checkpoints.
#
# A checkpoint is a run's `n<N>` folder or an interim one,
# `interim/s<step>` (`finetune.interim`), the `{ck}` wildcard.
#
# `verify_checkpoint` checks that a checkpoint's hash recomputes from its
# commit (`finetune-verify`); `publish_checkpoint` releases a
# verified one (workflow/scripts/publish_checkpoint.py), without waiting
# for its held-out losses, which `release_held_out` adds to the release
# once scored (rules/evaluate.smk); `publish_hf`
# uploads a verified one to the Hugging Face Hub (opt-in, named by its
# output only). Publishing writes
# GitHub releases, so it is the opt-in target `publish`, never part of
# `finetune` or `all`.


wildcard_constraints:
    ck=r"n\d+|interim/s\d+",


rule verify_checkpoint:
    input:
        settings=f"{TRAIN}/{{ck}}/settings.json",
        adapter=f"{TRAIN}/{{ck}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{TRAIN}/{{ck}}/verified.txt",
    log:
        f"{TRAIN}/{{ck}}/verify.log",
    priority: HANDOFF_PRIORITY
    localrule: True
    conda:
        TOOLS_ENV
    params:
        where=lambda wc, input: Path(input.settings).parent,
    shell:
        "uv run finetune-verify {params.where} > {output} 2>{log}"


rule publish_checkpoint:
    input:
        rules.verify_checkpoint.output,
    output:
        f"{TRAIN}/{{ck}}/published.json",
    log:
        f"{TRAIN}/{{ck}}/publish.log",
    priority: HANDOFF_PRIORITY
    localrule: True
    conda:
        TOOLS_ENV
    params:
        where=lambda wc, output: Path(output[0]).parent,
    shell:
        """
        exec >{log} 2>&1
        uv run python workflow/scripts/publish_checkpoint.py {params.where} {output}
        # An optional notification line; it never fails the rule.
        uv run python workflow/scripts/notify.py "checkpoint {wildcards.ck} published:" \
            "$(cat {output})" || true
        """


rule ft_check_release:
    # A saved checkpoint's BDBV check page (rules/report.smk), named by the
    # hash in its settings; its forecasts wait for its release
    # (`ft_check_record`).
    input:
        released_ft_check_page,
    output:
        touch(f"{TRAIN}/{{ck}}/ft-check.done"),
    priority: HANDOFF_PRIORITY
    localrule: True


rule release_held_out:
    # An offloaded checkpoint's held-out losses, added to its release once
    # scored, so the release never waits for them.
    input:
        published=f"{TRAIN}/{{ck}}/published.json",
        held_out=f"{TRAIN}/{{ck}}/{SIDECAR}",
    output:
        f"{TRAIN}/{{ck}}/held-out-released.txt",
    log:
        f"{TRAIN}/{{ck}}/held-out-released.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        where=lambda wc, output: Path(output[0]).parent,
    shell:
        "uv run python workflow/scripts/publish_checkpoint.py --held-out "
        "{params.where} {output} > {log} 2>&1"


rule publish_hf:
    # A verified checkpoint's Hub repository folder: its adapter, settings,
    # sha256sums.txt and model card (nowcastfoundation.hubpublish), in
    # <ck>/huggingface/, uploaded as the revision tagged with its spec to
    # [huggingface] repo in packages/nowcastfoundation/config/finetune.toml. A
    # dry run
    # (`huggingface.dry_run`, or no repo) writes the folder only. The
    # token is HF_TOKEN from the environment, never logged. Opt-in: no
    # target asks for it; name its output to run it.
    input:
        rules.verify_checkpoint.output,
    output:
        f"{TRAIN}/{{ck}}/huggingface.json",
    log:
        f"{TRAIN}/{{ck}}/huggingface.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        where=lambda wc, output: Path(output[0]).parent,
        args=lambda wc: hf_args(),
    shell:
        "uv run publish-hf {params.where} {output} {params.args} > {log} 2>&1"


rule publish:
    # The checkpoints saved so far, so it never schedules training and can
    # run beside a training run.
    input:
        published_checkpoints,
