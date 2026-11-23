"""The monitor's server: Panoptes, with the DAG pages beside it."""

import json

import pytest
from flowlib import dag, monitor_server

# A target of two jobs, `all` <- `a` with n: 1, `a` not yet made.
JOBS = r"""digraph snakemake_dag {
	0[label = "all", color = "0.22 0.6 0.85", style="rounded"];
	1[label = "a\nn: 1", color = "0.00 0.6 0.85", style="rounded"];
	1 -> 0
}
"""
RULES = r"""digraph snakemake_dag {
	0[label = "all", color = "0.22 0.6 0.85", style="rounded"];
	1[label = "a", color = "0.00 0.6 0.85", style="rounded"];
	1 -> 0
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


class FakeApp:
    """The parts of a Flask app the DAG pages use."""

    def __init__(self):
        self.views = {}
        self.after = []

    def add_url_rule(self, rule, _endpoint, view):
        self.views[rule] = view

    def after_request(self, f):
        self.after.append(f)
        return f


def test_the_monitor_serves_each_target_s_page(tmp_path):
    cfg = {"ROOT": str(tmp_path), "MONITOR_DAG": "toy=all"}
    where = dag.folder(cfg)
    where.mkdir(parents=True)
    (where / "toy.json").write_text(json.dumps(built()))
    app = FakeApp()
    monitor_server.add_dag(app, cfg, lambda: [])
    assert "/dag/toy" in app.views["/dag/"]()
    page = app.views["/dag/<name>"]("toy")
    assert "viz-standalone.js" in page
    assert "a\\\\nn: 1" in page
    assert app.views["/dag/<name>"]("other")[1] == 404


def test_panoptes_pages_link_to_the_dag():
    flask = pytest.importorskip("flask")
    app = flask.Flask("panoptes")

    @app.route("/")
    def index():
        return '<a class="nav-link" href="/workflows">Workflows</a>'

    monitor_server.add_dag(app, {"ROOT": "/nowhere"}, lambda: [])
    page = app.test_client().get("/").get_data(as_text=True)
    assert page.index(monitor_server.LINK) < page.index("/workflows")
