"""The monitor's server: Panoptes, with the workflow's DAG beside it.

Panoptes (`panoptes-ui`) shows each run's jobs; `/dag/` adds the rule
graph and job DAG of the targets `MONITOR_DAG` names (flowlib.dag),
coloured by the jobs Panoptes has seen and redrawn on each visit. A
thread builds the graphs again every `MONITOR_DAG_EVERY` seconds, so
new jobs and finished files show without Snakemake being asked on each
visit.

`task nowcast-workflow:monitor` serves it on `MONITOR_HOST` and
`MONITOR_PORT` (config/local.env), its database at `PANOPTES_DB_URL`.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowlib import dag, localconfig

CHECKOUT = Path(__file__).resolve().parents[2]
"""The checkout whose workflow the graphs are built from."""

VIZ = "https://cdn.jsdelivr.net/npm/@viz-js/viz@3.29.0/dist/viz-global.js"
"""Graphviz in the browser, which draws the graphs: viz-js's bundle that
defines the global `Viz` (from 3.x it is in dist/, not lib/)."""

MAX_JOBS = 1500
"""Jobs above which the page draws the rule graph only."""

LINK = (
    '<a class="nav-link" href="/dag/">'
    '<i class="nav-icon icon-share"></i>DAG</a>'
)
"""The link Panoptes' pages gain to the DAG."""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Workflow DAG</title>
<style>
:root {{ --fg: #222; --bg: #fff; --muted: #666; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg: #eee; --bg: #1b1b1b; --muted: #aaa; }}
}}
body {{ font-family: sans-serif; color: var(--fg); background: var(--bg);
  margin: 0 16px; }}
.graph {{ overflow: auto; max-width: 100%; }}
.muted {{ color: var(--muted); }}
.key span {{ display: inline-block; padding: 0 6px; margin-right: 4px;
  color: #222; border-radius: 3px; }}
</style></head><body>
<p><a href="/">Panoptes</a> · <a href="/dag/">DAG</a></p>
{body}
<p id="noviz" class="muted" hidden>Graphviz did not load from
<a href="{viz}">{viz}</a>, so the graphs are not drawn.</p>
<script src="{viz}"
  onerror="document.getElementById('noviz').hidden = false"></script>
<script>
if (window.Viz) Viz.instance().then(viz => {{
  for (const el of document.querySelectorAll("script.dot")) {{
    const svg = viz.renderSVGElement(JSON.parse(el.textContent));
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.maxWidth = "100%";
    el.parentElement.appendChild(svg);
  }}
}});
</script></body></html>
"""


def key() -> str:
    """Return the colour key."""
    spans = "".join(
        f'<span style="background:{dag.COLOURS[s]}">{s}</span>'
        for s in dag.STATES
    )
    return f'<p class="key">{spans}</p>'


def graph(text: str) -> str:
    """Return a graph's place on the page, its dot held as JSON."""
    data = json.dumps(text).replace("</", "<\\/")
    return (
        '<div class="graph"><script type="application/json" class="dot">'
        f"{data}</script></div>"
    )


def target_page(built: Mapping | None, seen: list[dict], name: str) -> str:
    """Return a target's page: its rule graph and, if not too big, its DAG."""
    if built is None:
        body = f"<h1>{html.escape(name)}</h1><p>Not built yet.</p>"
        return _PAGE.format(body=body, viz=VIZ)
    got = dag.views(built, seen)
    totals = ", ".join(f"{v} {k}" for k, v in got["totals"].items())
    n = sum(got["totals"].values())
    parts = [
        f"<h1>{html.escape(name)}</h1>",
        f'<p class="muted">Target {html.escape(built["target"])};'
        f" graph built {html.escape(built['built'])}; jobs: {totals}."
        " States from the monitor, redrawn every minute.</p>",
        key(),
        "<h2>Rules</h2>",
        graph(got["rules"]),
        "<h2>Jobs</h2>",
    ]
    if n > MAX_JOBS:
        parts.append(
            f'<p class="muted">{n} jobs, more than the {MAX_JOBS} drawn.</p>'
        )
    else:
        parts.append(graph(got["jobs"]))
    return _PAGE.format(body="".join(parts), viz=VIZ)


def index_page(names: list[str]) -> str:
    """Return the list of targets."""
    items = "".join(
        f'<li><a href="/dag/{html.escape(n)}">{html.escape(n)}</a></li>'
        for n in names
    )
    body = (
        f"<h1>Workflow DAG</h1><ul>{items}</ul>"
        if names
        else "<h1>Workflow DAG</h1><p>No targets: set MONITOR_DAG in"
        " config/local.env.</p>"
    )
    return _PAGE.format(body=body, viz=VIZ)


def add_dag(app, cfg: Mapping[str, str], seen) -> None:
    """Add the DAG pages to Panoptes' app, and a link to them on its pages.

    `seen` returns the jobs the monitor has seen.
    """
    names = dag.targets(cfg)
    where = dag.folder(cfg)

    def index():
        return index_page(list(names))

    def target(name: str):
        if name not in names:
            return "No such target", 404
        path = where / f"{name}.json"
        built = json.loads(path.read_text()) if path.exists() else None
        return target_page(built, seen(), name)

    app.add_url_rule("/dag/", "dag_index", index)
    app.add_url_rule("/dag/<name>", "dag_target", target)

    @app.after_request
    def link(response):
        if response.mimetype == "text/html" and not response.direct_passthrough:
            text = response.get_data(as_text=True)
            anchor = '<a class="nav-link" href="/workflows">'
            if anchor in text and LINK not in text:
                response.set_data(text.replace(anchor, LINK + anchor, 1))
        return response


def panoptes_jobs() -> list[dict]:
    """Return the jobs of Panoptes' running workflows, oldest first."""
    from panoptes.database import db_session
    from panoptes.models import WorkflowJobs, Workflows

    rows = (
        db_session.query(
            WorkflowJobs.name, WorkflowJobs.wildcards, WorkflowJobs.status
        )
        .join(Workflows, Workflows.id == WorkflowJobs.wf_id)
        .filter(Workflows.status == "Running")
        .order_by(WorkflowJobs.id)
        .all()
    )
    db_session.remove()
    return dag.monitor_jobs(rows)


def rebuild(cfg: Mapping[str, str]) -> None:
    """Build every target's graphs again, every `MONITOR_DAG_EVERY` s."""
    every = int(cfg.get("MONITOR_DAG_EVERY") or 900)
    while True:
        for name, target in dag.targets(cfg).items():
            try:
                dag.build(
                    name,
                    target,
                    dag.folder(cfg),
                    dag.options(cfg),
                    cwd=CHECKOUT,
                )
            except subprocess.CalledProcessError as err:
                sys.stderr.write(f"dag {name}: {err.stderr[-2000:]}\n")
        time.sleep(every)


def main() -> None:
    """Serve Panoptes with the DAG pages."""
    cfg = localconfig.read()
    from panoptes.app import app

    add_dag(app, cfg, panoptes_jobs)
    threading.Thread(target=rebuild, args=(cfg,), daemon=True).start()
    app.run(
        host=cfg["MONITOR_HOST"], port=int(cfg["MONITOR_PORT"]), threaded=True
    )


if __name__ == "__main__":
    main()
