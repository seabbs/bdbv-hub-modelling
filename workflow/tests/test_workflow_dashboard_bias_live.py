"""The attention bias section a server builds: markup, script and data.

A server that kept its code while its checkout moved on sent the new
bias.js with the old markup, and the script failed on a missing element:
the page's assets are read once, with the code, and the section's
script finds every element it starts from.
"""

import json
import re
import shutil
import sys
from pathlib import Path

from flowlib.dashboard import page

sys.path.insert(0, str(Path(__file__).parent))
from test_workflow_dashboard_page import _machine

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard"
PAGE = PAGE / "page"
OPTIONAL = {"lrslot", "bm", "lrcard"}
"""Elements bias.js creates itself, and the learning rate's card, which a
run with no learning-rate groups has none of (bias.js checks)."""


def test_assets_are_read_once_with_the_code(tmp_path, monkeypatch):
    copy = tmp_path / "page"
    shutil.copytree(PAGE, copy)
    monkeypatch.setattr(page, "ASSETS", copy)
    page._asset.cache_clear()
    try:
        first = page._asset("bias.js")
        (copy / "bias.js").write_text("// a newer checkout's script")
        assert page._asset("bias.js") == first
    finally:
        page._asset.cache_clear()


def _matrix(v):
    return [[v + 0.01 * (i + j) for j in range(12)] for i in range(12)]


def _history(run):
    # As the trainer writes it: each relation's mean, range and values,
    # with the bias in the checkpoint's settings.
    path = run / "train" / "n250" / "settings.json"
    settings = json.loads(path.read_text())
    path.write_text(json.dumps({**settings, "series_bias": {"init": "zero"}}))
    rows = []
    for step, v in ((0, 0.0), (100, 0.2), (200, 0.4)):
        bias = {
            rel: {
                "mean": v * k,
                "min": v * k - 0.1,
                "max": v * k + 0.1,
                "values": _matrix(v * k),
            }
            for k, rel in enumerate(("same", "up", "down"), start=1)
        }
        rows.append({"step": step, "loss": 0.5, "bias": bias})
    (run / "train" / "history.json").write_text(json.dumps(rows))


def test_the_bias_section_holds_every_relation_s_panels(tmp_path):
    ctx = _machine(tmp_path)
    _history(tmp_path / "run")
    out = page.sections(ctx)["bias"]
    markup = re.sub(r"<script>.*?</script>", "", out, flags=re.S)
    ids = set(re.findall(r"id=([\w-]+)", markup))
    js = page._asset("bias.js")
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', js)) - OPTIONAL
    assert wanted <= ids, wanted - ids
    assert {"biasgrid", "biashead", "biasrefs", "bdata"} <= ids
    data = json.loads(re.search(r"id=bdata>(.*?)</script>", out).group(1))
    run = data["runs"][0]
    assert set(run["pts"]) == {"same", "up", "down"}
    assert all(len(v) == 3 for v in run["pts"].values())
    for rel in ("same", "up", "down"):
        assert len(run["latest"][rel]) == 12
        assert len(run["latest"][rel][0]) == 12
    assert [p["key"] for p in data["pairs"]] == ["same", "up", "down"]
