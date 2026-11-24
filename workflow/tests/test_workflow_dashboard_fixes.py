"""Dashboard config values keep their commas, and failures are reported."""

import io
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from flowlib.dashboard import rows, runs


def test_a_ref_regex_may_hold_commas():
    ref = runs.parse_reference(
        "multi=^foundation(_zeroshot){0,1}$,checkpoint=zero-shot,role=start"
    )
    assert ref.model == "^foundation(_zeroshot){0,1}$"
    assert ref.checkpoint == "zero-shot"
    assert ref.role == "start"


def test_a_ref_spec_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown"):
        runs.parse_reference("a=b,shade=3")


def test_config_and_spec_references_share_their_defaults():
    from_config = runs.load_references([{"key": "r", "model": "x"}], [])[0]
    from_spec = runs.parse_reference("r=x")
    assert from_config == from_spec


def test_shared_row_helpers():
    assert rows.num("0.5") == 0.5
    assert rows.num("") is None
    assert rows.source("foundation") == "daily"
    assert rows.source("randomwalknowcast") is None
    assert rows.sources("randomwalknowcast") == ["daily", "weekly"]


pytest.importorskip("matplotlib")

from flowlib.dashboard import page  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import dashboard  # noqa: E402
from test_workflow_dashboard_page import _machine  # noqa: E402


def test_a_page_with_no_runs_says_so(tmp_path):
    ctx = _machine(tmp_path)
    ctx.runs = []
    with pytest.raises(ValueError, match="no runs"):
        page.build(ctx)


def test_a_server_error_is_logged_not_sent(tmp_path, monkeypatch):
    # A failed build is logged with its detail; the page served (the last
    # one built, or the building notice) never carries it.
    ctx = _machine(tmp_path)

    def boom(_ctx):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(dashboard.page, "render", boom)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode()
    finally:
        srv.shutdown()
    assert "secret detail" not in body
    assert "building" in body.lower()
    assert "secret detail" in err.getvalue()


def test_the_random_walk_reference_is_randomwalknowcast():
    # Germany's random walk reference is randomwalknowcast, a 7-day sum
    # for every date, and not the forecast-only randomwalk.
    import re
    from pathlib import Path

    import yaml

    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / "workflow" / "config.yaml").read_text())
    rw = next(r for r in config["dashboard"]["references"] if r["key"] == "rw")
    pattern = re.compile(rw["model"])
    assert pattern.search("randomwalknowcast")
    assert not pattern.search("randomwalk")
