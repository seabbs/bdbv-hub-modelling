"""One entry per host: the monitor, with the fine-tuning dashboard under it."""

import http.server
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from flowlib.dashboard import context, serve

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "workflow" / "flowlib" / "dashboard" / "page"


def test_the_dashboard_fetches_relative_urls():
    # So it works under the monitor's /finetune/ as well as on its own.
    for name in ("live.js", "nowcasts.js"):
        assert 'fetch("/' not in (PAGE / name).read_text()


def _runs(tmp_path, *names):
    for i, name in enumerate(names):
        status = tmp_path / "runs" / name / "train" / "stream-status.json"
        status.parent.mkdir(parents=True)
        status.write_text("{}")
        os.utime(status, (1000 + i, 1000 + i))


def test_the_dashboard_shows_every_run_under_root_newest_first(tmp_path):
    config = {"runs": {"a": {}, "b": {}, "c": {}}}
    _runs(tmp_path, "a", "b")
    got = context.workflow_runs(config, {"ROOT": str(tmp_path)}, ())
    assert [r.key for r in got] == ["b", "a"]
    assert got[0].live


def test_with_no_run_folders_it_shows_the_config_s_runs(tmp_path):
    config = {"runs": {"a": {}, "b": {}}}
    got = context.workflow_runs(config, {"ROOT": str(tmp_path)}, ())
    assert [r.key for r in got] == ["a", "b"]


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"upstream {self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_the_monitor_serves_the_dashboard_under_finetune():
    flask = pytest.importorskip("flask")
    from flowlib import monitor_server

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        app = flask.Flask("test")
        cfg = {
            "DASHBOARD_HOST": "127.0.0.1",
            "DASHBOARD_PORT": str(up.server_port),
        }
        monitor_server.add_finetune(app, cfg)
        client = app.test_client()
        got = client.get("/finetune/sections.json?have=x")
        assert got.status_code == 200
        assert got.get_data(as_text=True) == "upstream /sections.json?have=x"
        assert client.get("/finetune").status_code in (301, 302, 308)
    finally:
        up.shutdown()
    assert 'href="/finetune/"' in monitor_server.LINK


def test_the_workflow_keeps_one_monitor_serving(tmp_path):
    seen = tmp_path / "env.txt"
    script = tmp_path / "fake_monitor.py"
    script.write_text(
        "import os, sys, time\n"
        f"open({str(seen)!r}, 'w').write("
        "os.environ['PANOPTES_DB_URL'] + ' ' + ' '.join(sys.argv[1:]))\n"
        "time.sleep(30)\n"
    )
    local = {"ROOT": str(tmp_path)}
    first = serve.ensure_monitor(local, script=script)
    try:
        for _ in range(100):
            if seen.exists():
                break
            time.sleep(0.1)
        assert serve.ensure_monitor(local, script=script) == first
        db = f"sqlite:///{tmp_path}/monitor/panoptes.db"
        # Long-lived: it ends with no run.
        assert seen.read_text().strip() == db
    finally:
        subprocess.run(["kill", str(first)], check=False)


def test_the_workflow_starts_the_monitor_with_a_run():
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    assert "serve.ensure_monitor(LOCAL)" in snakefile
    assert sys.executable
