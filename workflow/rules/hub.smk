# The hub rounds report: our hub submissions against every hub team on
# the hub's latest data (workflow/scripts/hub_rounds_report.py). It reads
# the hub's files as submitted and runs no model of ours. It reruns when
# <ROOT>/hub/head.txt changes: workflow/scripts/hub_watch.sh writes the
# hub's data commit there when it moves and then builds `hub_rounds`.
# Not part of `all`.

HUB = f"{LOCAL['ROOT']}/hub"


rule hub_data:
    # The hub's tidy data (`bdbvhub data`), including tasks.json.
    input:
        f"{HUB}/head.txt",
    output:
        f"{HUB}/data/target.csv",
    log:
        f"{HUB}/logs/data.log",
    localrule: True
    params:
        out=lambda wc, output: Path(output[0]).parent,
    shell:
        "uv run bdbvhub data --raw {params.out}/raw --out {params.out} " "> {log} 2>&1"


rule hub_submissions_table:
    # Each round's submitted model per hub id, from the ledger branch
    # (submission_ci.submissions), so the report says which chain model
    # went out.
    input:
        f"{HUB}/head.txt",
    output:
        f"{HUB}/submissions.csv",
    log:
        f"{HUB}/logs/submissions.log",
    localrule: True
    params:
        ledger=f"{HUB}/ledger",
    shell:
        "(git fetch -q origin submission-ledger && rm -rf {params.ledger} && "
        "mkdir -p {params.ledger} && git archive origin/submission-ledger "
        "| tar -x -C {params.ledger} && uv run --quiet --project ci "
        "submissions-table --state {params.ledger}/state.json "
        "--policy ci/policy.toml --hub-config config/hub.toml "
        "--out {output}) > {log} 2>&1"


rule hub_comparison:
    # Every hub team's files and ours scored on the same truth
    # (`compare-hub` with no live model). The data folder is also set as
    # BDBV_DATA_DIR, which the hub's places are read from.
    input:
        target=rules.hub_data.output,
        table=rules.hub_submissions_table.output,
    output:
        scores=f"{HUB}/comparison/scores.csv",
        quantiles=f"{HUB}/comparison/quantiles.csv.gz",
        summary=f"{HUB}/comparison/summary.json",
    log:
        f"{HUB}/logs/comparison.log",
    localrule: True
    params:
        data=lambda wc, input: Path(input.target[0]).parent,
        out=lambda wc, output: Path(output.scores).parent,
    shell:
        "BDBV_DATA_DIR={params.data} uv run bdbvhub compare-hub --fetch "
        "--data {params.data} --hub-dir {params.data}/hub-submissions "
        "--cache-dir {params.data}/raw/hub-submissions --out {params.out} "
        "--results {params.out}/results --models --stored "
        "--submissions-table {input.table} > {log} 2>&1"


rule hub_rounds_report:
    input:
        script="workflow/scripts/hub_rounds_report.py",
        head=f"{HUB}/head.txt",
        target=rules.hub_data.output,
        table=rules.hub_submissions_table.output,
        scores=rules.hub_comparison.output.scores,
    output:
        f"{HUB}/hub-rounds.html",
    log:
        f"{HUB}/logs/report.log",
    localrule: True
    params:
        comparison=lambda wc, input: Path(input.scores).parent,
        data=lambda wc, input: Path(input.target[0]).parent,
        rounds=config["hub_rounds"]["rounds"],
    shell:
        "uv run python {input.script} {output} "
        "{params.comparison} {params.data} {input.table} {input.head} "
        "{params.rounds} > {log} 2>&1"


rule hub_rounds:
    # The report, copied to REVIEW_DIR (config/local.env) when it is set.
    input:
        rules.hub_rounds_report.output,
    localrule: True
    params:
        review=LOCAL["REVIEW_DIR"],
    shell:
        "if [ -n '{params.review}' ]; then mkdir -p '{params.review}' && "
        "cp {input} '{params.review}/hub-rounds.html.tmp' && "
        "mv '{params.review}/hub-rounds.html.tmp' "
        "'{params.review}/hub-rounds.html'; fi"
