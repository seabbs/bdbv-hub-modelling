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
        "multi=^fm_chronos_(daily|weekly){1,2}$,checkpoint=zero-shot,role=start"
    )
    assert ref.model == "^fm_chronos_(daily|weekly){1,2}$"
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
    assert rows.source("fm_chronos_daily_ragged_ft") == "daily"
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
    ctx = _machine(tmp_path)

    def boom(_ctx):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(dashboard.page, "build", boom)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as got:
            urllib.request.urlopen(
                f"http://127.0.0.1:{srv.server_address[1]}/", timeout=30
            )
        body = got.value.read().decode()
    finally:
        srv.shutdown()
    assert got.value.code == 500
    assert "secret detail" not in body
    assert "secret detail" in err.getvalue()
