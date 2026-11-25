"""The attention bias section: equal panels in an even grid."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from flowlib.dashboard import page
from test_workflow_dashboard_page import _machine

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard"
PAGE = PAGE / "page"
PAIRS = [{"key": k, "label": k} for k in ("same", "up", "down")]


def _layout(pairs):
    script = (
        "const {biasLayout}=require(process.argv[1]);"
        "console.log(JSON.stringify(biasLayout(JSON.parse(process.argv[2]))))"
    )
    out = subprocess.run(
        ["node", "-e", script, str(PAGE / "bias.js"), json.dumps(pairs)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_panels_go_by_relation_over_training_then_the_latest_matrices():
    got = _layout(PAIRS)
    assert got["cols"] == 4
    assert got["panels"] == [
        ["line", "same"],
        ["line", "up"],
        ["line", "down"],
        ["lr", None],
        ["heat", "same"],
        ["heat", "up"],
        ["heat", "down"],
        ["layers", None],
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_one_relation_still_fills_an_even_grid():
    got = _layout(PAIRS[:1])
    assert got["cols"] == 2
    assert len(got["panels"]) == 4


def test_every_panel_and_the_learning_rate_share_one_size():
    js = (PAGE / "bias.js").read_text()
    width, height = page.BIAS_PANEL
    assert f"PW={width},PH={height}" in js
    card = page.lr_card(
        {
            "groups": {"lora": 1e-4},
            "lr_outbreaks": 0,
            "logged": {},
            "current": {},
        },
        10,
        "run",
    )
    assert f'viewBox="0 0 {width} {height}"' in card


def test_the_section_holds_the_learning_rate_inside_the_grid(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        page, "lr_card", lambda *_a, **_k: "<div class=card id=lrcard></div>"
    )
    out = page.sections(_machine(tmp_path))["bias"]
    grid = re.search(r"<div class=biasgrid id=biasgrid>(.*?)</div><script", out)
    assert grid
    assert "id=lrcard" in grid.group(1)
    assert "id=biashead" in out
