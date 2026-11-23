"""The workflow's rule graph and job DAG, coloured by each job's state."""

import json

from flowlib import dag

# `snakemake --dag` of `all` <- b{n} <- a{n}, n = 1, 2, with a1 made.
JOBS = r"""digraph snakemake_dag {
    graph[bgcolor=white, margin=0];
	0[label = "all", color = "0.22 0.6 0.85", style="rounded"];
	1[label = "b", color = "0.44 0.6 0.85", style="rounded"];
	2[label = "a\nn: 1", color = "0.00 0.6 0.85", style="rounded,dashed"];
	3[label = "b", color = "0.44 0.6 0.85", style="rounded"];
	4[label = "a\nn: 2", color = "0.00 0.6 0.85", style="rounded"];
	1 -> 0
	3 -> 0
	2 -> 1
	4 -> 3
}
"""
RULES = r"""digraph snakemake_dag {
	0[label = "all", color = "0.22 0.6 0.85", style="rounded"];
	1[label = "b", color = "0.44 0.6 0.85", style="rounded"];
	2[label = "a", color = "0.00 0.6 0.85", style="rounded"];
	1 -> 0
	2 -> 1
}
"""


def built():
    return {
        "name": "toy",
        "target": "all",
        "built": "2026-11-22T22:00:00",
        "jobs": dag.parse(JOBS),
        "rules": dag.parse(RULES),
    }


def test_a_dag_gives_each_job_s_rule_wildcards_and_whether_it_is_made():
    got = dag.parse(JOBS)
    assert got["nodes"]["2"] == {
        "rule": "a",
        "wildcards": {"n": "1"},
        "done": True,
    }
    assert got["nodes"]["1"]["wildcards"] == {}
    assert ["2", "1"] in got["edges"]


def test_a_job_the_monitor_saw_takes_its_state():
    seen = [{"rule": "b", "wildcards": {"n": "1"}, "state": "running"}]
    got = dag.states(dag.parse(JOBS), seen)
    assert got == {
        "0": "pending",
        "1": "running",
        "2": "done",
        "3": "pending",
        "4": "pending",
    }


def test_the_latest_state_of_a_job_wins():
    seen = [
        {"rule": "a", "wildcards": {"n": "2"}, "state": "running"},
        {"rule": "a", "wildcards": {"n": "2"}, "state": "failed"},
    ]
    assert dag.states(dag.parse(JOBS), seen)["4"] == "failed"


def test_a_rule_is_running_while_any_job_is_and_done_when_all_are():
    assert dag.rule_state({"done": 1, "running": 1, "pending": 1}) == (
        "running"
    )
    assert dag.rule_state({"done": 2, "pending": 0}) == "done"
    assert dag.rule_state({"done": 1, "pending": 1}) == "pending"
    assert dag.rule_state({"failed": 1, "running": 1}) == "failed"


def test_the_views_colour_and_count_the_jobs():
    seen = [{"rule": "b", "wildcards": {"n": "1"}, "state": "running"}]
    got = dag.views(built(), seen)
    assert got["totals"] == {
        "pending": 3,
        "running": 1,
        "done": 1,
        "failed": 0,
    }
    assert f'fillcolor="{dag.COLOURS["running"]}"' in got["rules"]
    assert 'label="a\\n1 of 2"' in got["rules"]
    assert 'label="a\\nn: 1"' in got["jobs"]


def test_the_monitor_s_rows_become_jobs():
    rows = [
        ("a", "{'n': '1'}", "Done"),
        ("b", "{'n': 2}", "Running"),
        ("c", None, "Error"),
        ("d", "{}", "Queued"),
    ]
    assert dag.monitor_jobs(rows) == [
        {"rule": "a", "wildcards": {"n": "1"}, "state": "done"},
        {"rule": "b", "wildcards": {"n": "2"}, "state": "running"},
        {"rule": "c", "wildcards": {}, "state": "failed"},
    ]


def test_targets_are_named_or_named_after_themselves():
    cfg = {"MONITOR_DAG": "runa=run_scores results/x.csv"}
    assert dag.targets(cfg) == {
        "runa": "run_scores",
        "results-x.csv": "results/x.csv",
    }
    assert dag.targets({}) == {}


def test_a_build_keeps_both_graphs(tmp_path):
    fake = tmp_path / "snakemake"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$*" in *--rulegraph*) cat "$(dirname "$0")/rules";;'
        ' *) cat "$(dirname "$0")/jobs";; esac\n'
    )
    fake.chmod(0o755)
    (tmp_path / "jobs").write_text(JOBS)
    (tmp_path / "rules").write_text(RULES)
    path = dag.build("toy", "all", tmp_path / "dag", snakemake=[str(fake)])
    got = json.loads(path.read_text())
    assert got["target"] == "all"
    assert len(got["jobs"]["nodes"]) == 5
    assert len(got["rules"]["nodes"]) == 3


def test_the_page_loads_viz_js_s_browser_bundle():
    # viz-js 3.29 ships the global `Viz` as dist/viz-global.js; the
    # lib/viz-standalone.js of older releases is a 404, so no graph drew.
    from flowlib import monitor_server as ms

    assert ms.VIZ.endswith("@viz-js/viz@3.29.0/dist/viz-global.js")
    assert ms.VIZ in ms.target_page(None, [], "x")


def test_the_page_says_when_graphviz_does_not_load():
    from flowlib import monitor_server as ms

    page = ms.target_page(
        {
            "target": "t",
            "built": "now",
            "jobs": {"nodes": {}, "edges": []},
            "rules": {"nodes": {}, "edges": []},
        },
        [],
        "x",
    )
    assert "onerror=" in page
    assert "Graphviz did not load" in page
