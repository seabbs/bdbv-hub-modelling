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
import urllib.parse
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

PAN_ZOOM = (
    "https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/dist/svg-pan-zoom.min.js"
)
"""Pan and zoom for the drawn graph."""

MAX_GROUPS = 60
"""Job groups an expanded rule shows before grouping by one wildcard."""

LINK = (
    '<a class="nav-link" href="/dag/">'
    '<i class="nav-icon icon-share"></i>DAG</a>'
)
"""The link Panoptes' pages gain to the DAG."""

_STYLE = """<style>
:root { --fg: #222; --bg: #fff; --muted: #666; --line: #ccc; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #eee; --bg: #1b1b1b; --muted: #aaa; --line: #555; }
}
body { font-family: sans-serif; color: var(--fg); background: var(--bg);
  margin: 0 16px; }
.muted { color: var(--muted); }
.key span { display: inline-block; padding: 0 6px; margin-right: 4px;
  color: #222; border-radius: 3px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px;
  align-items: flex-start; margin: 8px 0; }
.controls fieldset { border: 1px solid var(--line); padding: 4px 8px; }
#graph { border: 1px solid var(--line); background: #fff; height: 75vh;
  overflow: hidden; }
#graph svg { width: 100%; height: 100%; }
#graph g.node, #graph g.cluster { cursor: pointer; }
</style>"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workflow DAG</title>
__STYLE__</head><body>
<p><a href="/">Panoptes</a> · <a href="/dag/">DAG</a></p>
__BODY__
</body></html>
"""

_TARGET = """<h1>__NAME__</h1>
<p class="muted">Target __TARGET__; graph built __BUILT__.
<span id="totals"></span> States from the monitor, refreshed every
minute. Click a rule to show its jobs, grouped by wildcard; click the
group's frame to fold it again. Drag to pan, scroll to zoom.</p>
__KEY__
<div class="controls">
<fieldset><legend>State</legend>__STATES__</fieldset>
<fieldset><legend>Rules</legend>
<select id="rules" multiple size="5">__RULES__</select><br>
<button id="all-rules" type="button">All rules</button></fieldset>
<fieldset><legend>View</legend>
<label><input type="checkbox" id="running-only"> running only</label><br>
<label><input type="checkbox" id="highlight" checked>
outline running</label><br>
<button id="fit" type="button">Fit to screen</button>
<button id="expand-all" type="button">Expand all</button>
<button id="collapse-all" type="button">Collapse all</button>
</fieldset>
</div>
<div id="graph"></div>
<p id="noviz" class="muted" hidden>Graphviz did not load from __VIZ__
(or svg-pan-zoom from __PAN_ZOOM__), so the graph is not drawn.</p>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA_URL = "/dag/__NAME_URL__.json";
const STATES = __STATE_LIST__;
const COLOURS = __COLOURS__;
const MAX_GROUPS = __MAX_GROUPS__;
let data = JSON.parse(document.getElementById("data").textContent);
const expanded = new Set();
let viz = null;
let pz = null;

function noviz() { document.getElementById("noviz").hidden = false; }
function esc(t) {
  return String(t).replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"')
    .replace(/\\n/g, "\\\\n");
}
function zero() { return Object.fromEntries(STATES.map(s => [s, 0])); }
function countsText(c) {
  const parts = STATES.filter(s => c[s]).map(s => c[s] + " " + s);
  return parts.join(", ") || "no jobs";
}
function stateOf(c) {
  if (c.failed) return "failed";
  if (c.running) return "running";
  if (c.done && !c.pending) return "done";
  return "pending";
}
function chosenStates() {
  return new Set([...document.querySelectorAll("input[name=state]:checked")]
    .map(e => e.value));
}
function chosenRules() {
  const v = [...document.getElementById("rules").selectedOptions]
    .map(o => o.value);
  return v.length ? new Set(v) : null;
}
function visible(c) {
  if (document.getElementById("running-only").checked) return c.running > 0;
  const chosen = chosenStates();
  return STATES.some(s => chosen.has(s) && c[s] > 0);
}
function bucket(jobs, key) {
  const m = new Map();
  for (const j of jobs) {
    const k = key(j);
    if (!m.has(k)) m.set(k, zero());
    m.get(k)[j.state] += 1;
  }
  return m;
}
function groups(rule) {
  const jobs = data.jobs.filter(j => j.rule === rule);
  let g = bucket(jobs, j => Object.entries(j.wildcards)
    .map(([k, v]) => k + ": " + v).join("\\n"));
  if (g.size > MAX_GROUPS) {
    const first = jobs.map(j => Object.keys(j.wildcards)[0]).find(k => k);
    g = bucket(jobs, j =>
      first in j.wildcards ? first + ": " + j.wildcards[first] : "");
  }
  return [...g.entries()].sort((a, b) =>
    a[0].localeCompare(b[0], undefined, {numeric: true}));
}
function node(id, label, c, extra) {
  const ring = document.getElementById("highlight").checked && c.running
    ? ', penwidth=3, color="#7f3b08"' : "";
  return '  "' + id + '"[label="' + esc(label) + '", fillcolor="'
    + COLOURS[stateOf(c)] + '", tooltip="' + esc(countsText(c)) + '"'
    + ring + (extra || "") + "];";
}
function toDot() {
  const rules = chosenRules();
  const lines = [
    "digraph dag {",
    "  graph[bgcolor=white, rankdir=LR, compound=true];",
    '  node[shape=box, style="rounded,filled", fontname=Arial, fontsize=10];',
    "  edge[color=grey];",
  ];
  const anchor = {};
  const frame = {};
  for (const r of data.rules) {
    if ((rules && !rules.has(r.rule)) || !visible(r.counts)) continue;
    if (!expanded.has(r.id)) {
      lines.push(node(r.id, r.rule + "\\n" + countsText(r.counts), r.counts,
        ', id="r' + r.id + '"'));
      anchor[r.id] = r.id;
      continue;
    }
    lines.push('  subgraph "cluster_' + r.id + '" {', '    id="c' + r.id
      + '"; label="' + esc(r.rule + " (" + countsText(r.counts) + ")")
      + '"; style="rounded,dashed"; color=grey; fontname=Arial; fontsize=10;'
      + ' tooltip="click to fold";');
    let i = 0;
    for (const [label, c] of groups(r.rule)) {
      if (!visible(c)) continue;
      const id = r.id + "_" + i++;
      anchor[r.id] = anchor[r.id] || id;
      lines.push("  " + node(id, label || r.rule, c));
    }
    if (!anchor[r.id]) {
      anchor[r.id] = r.id + "_none";
      lines.push('    "' + anchor[r.id]
        + '"[label="no jobs shown", style=dashed];');
    }
    lines.push("  }");
    frame[r.id] = "cluster_" + r.id;
  }
  for (const [u, v] of data.edges) {
    if (!(u in anchor && v in anchor)) continue;
    const at = [];
    if (frame[u]) at.push('ltail="' + frame[u] + '"');
    if (frame[v]) at.push('lhead="' + frame[v] + '"');
    lines.push('  "' + anchor[u] + '" -> "' + anchor[v] + '"'
      + (at.length ? "[" + at.join(", ") + "]" : "") + ";");
  }
  lines.push("}");
  return lines.join("\\n");
}
function totals() {
  const t = data.totals;
  document.getElementById("totals").textContent =
    "Jobs: " + STATES.map(s => t[s] + " " + s).join(", ") + ".";
}
function draw(keep) {
  if (!viz) return;
  const view = keep && pz ? {zoom: pz.getZoom(), pan: pz.getPan()} : null;
  const svg = viz.renderSVGElement(toDot());
  svg.removeAttribute("width");
  svg.removeAttribute("height");
  const box = document.getElementById("graph");
  if (pz) pz.destroy();
  box.replaceChildren(svg);
  pz = svgPanZoom(svg, {controlIconsEnabled: true, fit: true, center: true,
    minZoom: 0.05, maxZoom: 40, zoomScaleSensitivity: 0.3});
  if (view) { pz.zoom(view.zoom); pz.pan(view.pan); }
}
let down = null;
document.getElementById("graph").addEventListener("pointerdown", e => {
  down = [e.clientX, e.clientY];
});
document.getElementById("graph").addEventListener("click", e => {
  if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const g = e.target.closest("g.node, g.cluster");
  if (!g || !g.id) return;
  if (g.id[0] === "r") expanded.add(g.id.slice(1));
  else if (g.id[0] === "c") expanded.delete(g.id.slice(1));
  else return;
  draw(true);
});
const FILTERS = "input[name=state], #rules, #running-only, #highlight";
for (const el of document.querySelectorAll(FILTERS)) {
  el.addEventListener("change", () => draw(false));
}
document.getElementById("all-rules").addEventListener("click", () => {
  for (const o of document.getElementById("rules").options) o.selected = false;
  draw(false);
});
document.getElementById("fit").addEventListener("click", () => {
  if (pz) { pz.resize(); pz.fit(); pz.center(); }
});
document.getElementById("expand-all").addEventListener("click", () => {
  for (const r of data.rules) expanded.add(r.id);
  draw(false);
});
document.getElementById("collapse-all").addEventListener("click", () => {
  expanded.clear();
  draw(false);
});
totals();
if (!window.Viz || !window.svgPanZoom) noviz();
else Viz.instance().then(v => { viz = v; draw(false); });
setInterval(async () => {
  const r = await fetch(DATA_URL);
  if (r.ok) { data = await r.json(); totals(); draw(true); }
}, 60000);
</script>"""


def key() -> str:
    """Return the colour key."""
    spans = "".join(
        f'<span style="background:{dag.COLOURS[s]}">{s}</span>'
        for s in dag.STATES
    )
    return f'<p class="key">{spans}</p>'


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Return the template with each `__KEY__` replaced by its value."""
    for k, v in values.items():
        template = template.replace(f"__{k}__", v)
    return template


def _page(body: str, scripts: str = "") -> str:
    return _fill(_PAGE, {"STYLE": _STYLE, "BODY": body + scripts})


def target_page(built: Mapping | None, seen: list[dict], name: str) -> str:
    """Return a target's page: its rule graph, to expand rule by rule.

    The page draws the graph in the browser from `dag.navigable`'s data,
    which it fetches again from `/dag/<name>.json` every minute.
    """
    if built is None:
        return _page(f"<h1>{html.escape(name)}</h1><p>Not built yet.</p>")
    data = dag.navigable(built, seen)
    states = "".join(
        f'<label><input type="checkbox" name="state" value="{s}" checked>'
        f" {s}</label><br>"
        for s in dag.STATES
    )
    rules = "".join(
        f'<option value="{html.escape(r)}">{html.escape(r)}</option>'
        for r in sorted({r["rule"] for r in data["rules"]})
    )
    body = _fill(
        _TARGET,
        {
            "NAME": html.escape(name),
            "NAME_URL": urllib.parse.quote(name),
            "TARGET": html.escape(built["target"]),
            "BUILT": html.escape(built["built"]),
            "KEY": key(),
            "STATES": states,
            "RULES": rules,
            "VIZ": html.escape(VIZ),
            "PAN_ZOOM": html.escape(PAN_ZOOM),
            "STATE_LIST": json.dumps(list(dag.STATES)),
            "COLOURS": json.dumps(dag.COLOURS),
            "MAX_GROUPS": str(MAX_GROUPS),
            "DATA": json.dumps(data).replace("</", "<\\/"),
        },
    )
    scripts = (
        f'<script src="{VIZ}" onerror="noviz()"></script>\n'
        f'<script src="{PAN_ZOOM}" onerror="noviz()"></script>\n'
    )
    return _page(body, scripts)


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
    return _page(body)


def add_dag(app, cfg: Mapping[str, str], seen) -> None:
    """Add the DAG pages to Panoptes' app, and a link to them on its pages.

    `seen` returns the jobs the monitor has seen.
    """
    names = dag.targets(cfg)
    where = dag.folder(cfg)

    def index():
        return index_page(list(names))

    def load(name: str) -> Mapping | None:
        path = where / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def target(name: str):
        if name not in names:
            return "No such target", 404
        return target_page(load(name), seen(), name)

    def target_data(name: str):
        built = load(name) if name in names else None
        if built is None:
            return {"error": f"no graph for {name}"}, 404
        return dag.navigable(built, seen())

    app.add_url_rule("/dag/", "dag_index", index)
    app.add_url_rule("/dag/<name>.json", "dag_data", target_data)
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
