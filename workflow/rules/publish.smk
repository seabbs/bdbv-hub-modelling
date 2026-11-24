# Publishing checkpoints.
#
# A checkpoint is a run's `n<N>` folder or an interim one,
# `interim/s<step>` (`finetune.interim`), the `{ck}` wildcard.
#
# `verify_checkpoint` checks that a checkpoint's hash recomputes from its
# commit (`finetune-verify`); `publish_checkpoint` releases a
# verified one (workflow/scripts/publish_checkpoint.py), and
# `publish_germany_unit` attaches a released checkpoint's German unit to
# its release for the Pages build. Publishing writes
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
        # An offloaded checkpoint is released with its held-out losses.
        lambda wc: offloaded(wc.run, wc.ck[1:]) if wc.ck[0] == "n" else [],
    output:
        f"{TRAIN}/{{ck}}/published.json",
    log:
        f"{TRAIN}/{{ck}}/publish.log",
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


rule publish_germany_unit:
    # A released checkpoint's German unit (score_germany), packed as the
    # stored forecasts the Pages build loads and attached to its own
    # release (`nowcastgermany unit-publish`, nowcastgermany.store).
    # Opt-in, as it writes to a release.
    input:
        f"{CACHE}/{GERMANY_UNIT}/{{label}}.done",
    output:
        touch(f"{CACHE}/{GERMANY_UNIT}/{{label}}.published"),
    log:
        f"{CACHE}/log/{GERMANY_UNIT}-{{label}}-publish.log",
    wildcard_constraints:
        label=r"slots-[0-9a-f]+",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        cache=lambda wc, input: Path(input[0]).parents[1],
    shell:
        "uv run nowcastgermany unit-publish {params.cache} {wildcards.label} "
        "> {log} 2>&1"


rule publish:
    # The checkpoints saved so far, so it never schedules training and can
    # run beside a training run.
    input:
        published_checkpoints,
