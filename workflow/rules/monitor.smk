# The monitor (flowlib.monitor, flowlib.monitor_server): each MONITOR_DAG
# target's rule graph and job DAG (flowlib.dag), built for the monitor's
# /dag/ page in <ROOT>/monitor/dag/, with MONITOR_DAG_OPTIONS. Target
# `monitor`, opt-in; the monitor's server builds them again on its own
# timer. A graph is built without a lock and reads the run's files only.
from flowlib import dag

MONITOR_TARGETS = dag.targets(LOCAL)
MONITOR_DAG_DIR = dag.folder(LOCAL)


wildcard_constraints:
    name=r"[\w.-]+",


rule monitor_dag:
    output:
        f"{MONITOR_DAG_DIR}/{{name}}.json",
    log:
        f"{MONITOR_DAG_DIR}/logs/{{name}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        target=lambda wc: MONITOR_TARGETS[wc.name],
        options=" ".join(dag.options(LOCAL)),
    shell:
        "uv run --group paper python workflow/flowlib/dag.py "
        "{wildcards.name}={params.target} {params.options} > {log} 2>&1"


rule monitor:
    input:
        expand(f"{MONITOR_DAG_DIR}/{{name}}.json", name=list(MONITOR_TARGETS)),
