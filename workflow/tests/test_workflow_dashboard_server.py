"""The dashboard server answers the page and one location's data."""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import dashboard
from test_workflow_dashboard_page import _machine


@pytest.fixture
def server(tmp_path):
    ctx = _machine(tmp_path)
    ctx.once = False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read().decode()


def test_the_page_is_served(server):
    status, body = get(server + "/")
    assert status == 200
    assert "<h2>Nowcasts</h2>" in body


def test_one_location_s_data_is_served(server):
    status, body = get(server + "/data/germany/DE.json")
    assert status == 200
    data = json.loads(body)
    assert set(data) == {
        "series",
        "score",
        "parts",
        "split",
        "ranges",
        "byday",
        "stats",
        "truth",
        "latest",
    }


def test_the_data_holds_only_the_models_asked_for(server):
    _, whole = get(server + "/data/germany/DE.json")
    _, none = get(server + "/data/germany/DE.json?models=")
    assert json.loads(whole)["series"]
    assert json.loads(none)["series"] == {}
    assert json.loads(none)["truth"] == json.loads(whole)["truth"]


def test_answers_are_gzipped_for_a_client_that_takes_gzip(server):
    import gzip

    for path in ("/", "/data/germany/DE.json", "/sections.json?have="):
        plain = urllib.request.urlopen(server + path, timeout=30)
        assert plain.headers.get("Content-Encoding") is None
        req = urllib.request.Request(
            server + path, headers={"Accept-Encoding": "gzip, deflate"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.headers["Content-Encoding"] == "gzip"
            assert gzip.decompress(r.read()) == plain.read()


def status_of(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as err:
        return err.code


def test_unknown_locations_and_paths_are_not_found(server):
    assert status_of(server + "/data/germany/XX.json") == 404
    assert status_of(server + "/favicon.ico") == 404


def test_a_failing_data_request_is_a_server_error(server, monkeypatch):
    def broken(*_args):
        raise ValueError("bad cache")

    monkeypatch.setattr(dashboard.nowcasts, "location", broken)
    assert status_of(server + "/data/germany/DE.json") == 500


def test_once_writes_the_page(tmp_path):
    ctx = _machine(tmp_path)
    out = tmp_path / "page" / "index.html"
    dashboard.write_once(ctx, out)
    assert "Fine-tuning dashboard" in out.read_text()


def test_sections_send_only_what_changed(server):
    status, body = get(server + "/sections.json")
    assert status == 200
    first = json.loads(body)
    assert {"bar", "status", "loss", "nowcasts"} <= set(first)
    assert all(v["html"] for v in first.values())
    have = ",".join(f"{k}:{v['hash']}" for k, v in first.items())
    second = json.loads(get(server + "/sections.json?have=" + have)[1])
    assert set(second) == set(first)
    # Unchanged parts come back as their hash alone; the clock moves on.
    for key in ("loss", "nowcasts", "tables"):
        assert second[key] == {"hash": first[key]["hash"]}


def test_the_page_never_waits_on_a_build(tmp_path):
    # A rebuild (the nowcasts take ~450 s) must not hold up the page. A
    # request serves the last page
    # built, while another build holds the lock.
    ctx = _machine(tmp_path)
    ctx.once = False
    pages = dashboard.Pages(ctx)
    pages.refresh()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx, pages))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with dashboard.LOCK:
            status, body = get(url + "/")
            assert status == 200
            assert "<h2>Nowcasts</h2>" in body
            status, body = get(url + "/data/germany/DE.json")
            assert status == 200
    finally:
        srv.shutdown()


def test_before_the_first_build_the_page_says_it_is_building(tmp_path):
    ctx = _machine(tmp_path)
    ctx.once = False
    pages = dashboard.Pages(ctx)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx, pages))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, body = get(url + "/")
        assert status == 200
        assert "building" in body.lower()
        assert 'http-equiv="refresh"' in body
    finally:
        srv.shutdown()


def test_a_client_that_leaves_is_dropped_quietly(tmp_path, capsys):
    # A browser that gives up mid-answer is not a server error.
    class Gone:
        def write(self, _b):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    cls = dashboard.handler(_machine(tmp_path))
    h = cls.__new__(cls)
    h.path, h.command, h.request_version = "/favicon.ico", "GET", "HTTP/1.1"
    h.requestline, h.client_address = "GET /favicon.ico HTTP/1.1", ("x", 0)
    h.wfile = Gone()
    h.do_GET()
    assert "Traceback" not in capsys.readouterr().err


def _bare(cls, path):
    h = cls.__new__(cls)
    h.path, h.command, h.request_version = path, "GET", "HTTP/1.1"
    h.requestline, h.client_address = f"GET {path} HTTP/1.1", ("x", 0)
    return h


@pytest.mark.parametrize(
    "gone", [ConnectionAbortedError, ConnectionResetError, TimeoutError]
)
def test_every_kind_of_client_leaving_is_dropped_quietly(
    tmp_path, capsys, gone
):
    # The server must not need a respawn loop. A client that
    # leaves in any way is not a server error.
    class Gone:
        def write(self, _b):
            raise gone("gone")

        def flush(self):
            pass

    h = _bare(dashboard.handler(_machine(tmp_path)), "/")
    h.wfile = Gone()
    h.do_GET()
    assert "Traceback" not in capsys.readouterr().err


def test_a_client_gone_while_an_error_is_sent_is_dropped(tmp_path, monkeypatch):
    class Gone:
        def write(self, _b):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    cls = dashboard.handler(_machine(tmp_path))
    monkeypatch.setattr(
        cls, "_data", lambda *_: (_ for _ in ()).throw(RuntimeError("x"))
    )
    h = _bare(cls, "/data/germany/DE.json")
    h.wfile = Gone()
    h.do_GET()  # raises nothing


def test_the_server_listens_where_the_machine_settings_say(tmp_path):
    # The host and port are machine settings (outside the
    # repo), so no proxy forwards to a loopback-only server.
    from flowlib import localconfig

    env = tmp_path / "local.env"
    env.write_text("DASHBOARD_HOST=localhost\nDASHBOARD_PORT=8790\n")
    assert dashboard.address(localconfig.read(env, {})) == ("localhost", 8790)
    assert dashboard.address(localconfig.read(tmp_path / "none", {})) == (
        "127.0.0.1",
        8765,
    )
    localconfig.check(env, {})
