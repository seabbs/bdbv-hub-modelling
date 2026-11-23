"""The workflow's rule graph and job DAG, coloured by each job's state.

`build` asks Snakemake for a target's job DAG (`--dag`) and rule graph
(`--rulegraph`) once and keeps them as JSON in `<ROOT>/monitor/dag/`;
`dot` colours them from the jobs the monitor (Panoptes) has seen, so a
page can redraw them as the run goes without building the DAG again. A
job Snakemake would not run is done; one the monitor saw running,
finished or failed is that; the rest are pending. Standard library
only.

    python3 workflow/flowlib/dag.py NAME=TARGET [SNAKEMAKE OPTION ...]

builds one target's graphs from this checkout.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

STATES = ("pending", "running", "done", "failed")
"""A job's states, from first to last."""

COLOURS = {
    "pending": "#d9d9d9",
    "running": "#f2b134",
    "done": "#5aae61",
    "failed": "#d6604d",
}
"""The fill of a node in each state."""

_NODE = re.compile(r'^\s*(\d+)\[label = "([^"]*)".*?style="([^"]*)"\];$')
_EDGE = re.compile(r"^\s*(\d+) -> (\d+)$")


def parse(text: str) -> dict:
    """Return the nodes and edges of a Snakemake `--dag`/`--rulegraph`.

    Each node has its rule, the wildcards its label shows (Snakemake
    shows only those its dependencies do not have) and whether Snakemake
    would leave it (dashed: its output is up to date). Edges run from a
    dependency to the job that needs it.
    """
    nodes: dict[str, dict] = {}
    edges: list[list[str]] = []
    for line in text.splitlines():
        if m := _NODE.match(line):
            rule, *pairs = m[2].split("\\n")
            nodes[m[1]] = {
                "rule": rule,
                "wildcards": dict(p.split(": ", 1) for p in pairs),
                "done": "dashed" in m[3],
            }
        elif m := _EDGE.match(line):
            edges.append([m[1], m[2]])
    return {"nodes": nodes, "edges": edges}


def _upstream(graph: Mapping) -> dict[str, set[tuple[str, str]]]:
    """Return each node's wildcards, with those of all its dependencies."""
    deps: dict[str, list[str]] = {n: [] for n in graph["nodes"]}
    for u, v in graph["edges"]:
        deps[v].append(u)
    seen: dict[str, set[tuple[str, str]]] = {}

    def walk(node: str) -> set[tuple[str, str]]:
        if node not in seen:
            seen[node] = set(graph["nodes"][node]["wildcards"].items())
            for dep in deps[node]:
                seen[node] |= walk(dep)
        return seen[node]

    sys.setrecursionlimit(max(10_000, 2 * len(deps)))
    for node in deps:
        walk(node)
    return seen


def states(graph: Mapping, seen: Iterable[Mapping]) -> dict[str, str]:
    """Return each job's state, from the jobs the monitor has seen.

    `seen` holds a rule, its wildcards and a state per job, latest last.
    A job matches when its rule is the same and the wildcards its label
    and dependencies show agree with the job's.
    """
    got = {
        n: "done" if node["done"] else "pending"
        for n, node in graph["nodes"].items()
    }
    upstream = _upstream(graph)
    by_rule: dict[str, list[str]] = {}
    for n, node in graph["nodes"].items():
        by_rule.setdefault(node["rule"], []).append(n)
    for job in seen:
        wildcards = set(job["wildcards"].items())
        for n in by_rule.get(job["rule"], []):
            own = set(graph["nodes"][n]["wildcards"].items())
            if own <= wildcards and wildcards <= upstream[n]:
                got[n] = job["state"]
    return got


def rule_states(jobs: Mapping, job_states: Mapping[str, str]) -> dict:
    """Return each rule's jobs by state, from its jobs' states."""
    got: dict[str, dict[str, int]] = {}
    for n, node in jobs["nodes"].items():
        counts = got.setdefault(node["rule"], dict.fromkeys(STATES, 0))
        counts[job_states[n]] += 1
    return got


def rule_state(counts: Mapping[str, int]) -> str:
    """Return a rule's state from its jobs' counts.

    Failed or running if any job is, else done when all are, else pending.
    """
    for state in ("failed", "running"):
        if counts.get(state):
            return state
    if counts.get("done") and not counts.get("pending"):
        return "done"
    return "pending"


def group_wildcards(graph: Mapping) -> dict[str, dict[str, str]]:
    """Return the wildcards each job is grouped and labelled by.

    Its own (those its label shows) and each of its dependencies' that
    has one value among them, so a job that gathers many (`all`) shows
    none and one that reads one input of each (segment n<N>) shows it.
    """
    got = {}
    for n, wildcards in _upstream(graph).items():
        keys = [k for k, _ in wildcards]
        own = dict(graph["nodes"][n]["wildcards"])
        single = {k: v for k, v in wildcards if keys.count(k) == 1}
        got[n] = dict(sorted({**single, **own}.items()))
    return got


def navigable(built: Mapping, seen: Iterable[Mapping]) -> dict:
    """Return the page's data: the rule graph and each rule's jobs.

    Each rule has its jobs' count by state and its state
    (`rule_state`); each job its rule, wildcards (`group_wildcards`)
    and state; the edges join rules, from the one made first.
    """
    jobs, rules = built["jobs"], built["rules"]
    job_states = states(jobs, seen)
    counts = rule_states(jobs, job_states)
    wildcards = group_wildcards(jobs)
    totals = dict.fromkeys(STATES, 0)
    for state in job_states.values():
        totals[state] += 1
    return {
        "rules": [
            {
                "id": n,
                "rule": node["rule"],
                "counts": counts.get(node["rule"], dict.fromkeys(STATES, 0)),
                "state": rule_state(counts.get(node["rule"], {})),
            }
            for n, node in rules["nodes"].items()
        ],
        "edges": rules["edges"],
        "jobs": [
            {
                "rule": node["rule"],
                "wildcards": wildcards[n],
                "state": job_states[n],
            }
            for n, node in jobs["nodes"].items()
        ],
        "totals": totals,
    }


def monitor_jobs(rows: Iterable[Sequence]) -> list[dict]:
    """Return the monitor's jobs from rows of rule, wildcards and status.

    Panoptes keeps wildcards as a Python dict's text and a status of
    Running, Done or Error.
    """
    state = {"Running": "running", "Done": "done", "Error": "failed"}
    got = []
    for rule, wildcards, status in rows:
        if status not in state:
            continue
        try:
            parsed = ast.literal_eval(wildcards or "{}")
        except (ValueError, SyntaxError):
            parsed = {}
        got.append(
            {
                "rule": rule,
                "wildcards": {str(k): str(v) for k, v in dict(parsed).items()},
                "state": state[status],
            }
        )
    return got


def targets(cfg: Mapping[str, str]) -> dict[str, str]:
    """Return the targets `MONITOR_DAG` names, NAME=TARGET or TARGET."""
    got = {}
    for item in shlex.split(cfg.get("MONITOR_DAG", "")):
        name, _, target = item.rpartition("=")
        got[re.sub(r"[^\w.-]+", "-", name or target).strip("-")] = target
    return got


def folder(cfg: Mapping[str, str]) -> Path:
    """Return where built graphs are kept."""
    return Path(cfg["ROOT"]).expanduser() / "monitor" / "dag"


def build(
    name: str,
    target: str,
    where: Path,
    options: Sequence[str] = (),
    snakemake: Sequence[str] = (sys.executable, "-m", "snakemake"),
    cwd: Path | str | None = None,
) -> Path:
    """Build a target's job DAG and rule graph; return the JSON file.

    Snakemake builds them without a lock and judges a job done by its
    files' times alone, so it reads the run's outputs and never waits on
    it.
    """
    base = [
        *snakemake,
        target,
        "--nolock",
        "--rerun-triggers",
        "mtime",
        "--quiet",
        *options,
    ]
    got = {}
    for key, flag in (("jobs", "--dag"), ("rules", "--rulegraph")):
        out = subprocess.run(
            [*base, flag], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout
        got[key] = parse(out)
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "name": name,
                "target": target,
                "built": datetime.now().isoformat(timespec="seconds"),
                **got,
            }
        )
    )
    tmp.replace(path)
    return path


def options(cfg: Mapping[str, str]) -> list[str]:
    """Return the Snakemake options the graphs are built with.

    `MONITOR_DAG_OPTIONS`, for example a run's --configfile.
    """
    return shlex.split(cfg.get("MONITOR_DAG_OPTIONS", ""))


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flowlib import localconfig

    config = localconfig.read()
    spec, *extra = sys.argv[1:]
    for label, goal in targets({"MONITOR_DAG": spec}).items():
        print(build(label, goal, folder(config), [*options(config), *extra]))
