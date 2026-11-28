"""A run trained without the series attention bias shows no bias panels.

Its checkpoints' settings carry no `series_bias` entry (the trainer
writes one only with the bias), so the bias row skips it and names it
with a one-line note, whatever adapter files its folder holds.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from flowlib.dashboard import bias, runs
from test_workflow_dashboard_bias import NAMES, _safetensors

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard"
PAGE = PAGE / "page"
BIAS = {"series_bias": {"relations": ["same", "up", "down"]}}


def _run(folder, settings, history=True):
    """A run with one checkpoint and a live state holding bias tensors."""
    d = folder / "train" / "n100"
    _safetensors(
        d / "finetuned-ckpt" / "adapter_model.safetensors",
        {NAMES["same"]: [[1.0, 1.0], [1.0, 1.0]]},
    )
    (d / "settings.json").write_text(
        json.dumps({"steps": 100, "n_outbreaks": 100, **settings})
    )
    _safetensors(
        folder / "train" / "state" / "adapter_model.safetensors",
        {NAMES["same"]: [[2.0, 2.0], [2.0, 2.0]]},
    )
    if history:
        (folder / "train" / "history.json").write_text(
            json.dumps(
                [{"step": 50, "bias": {"same": {"values": [[0.5, 0.5]]}}}]
            )
        )


def test_settings_say_whether_a_run_has_the_series_bias(tmp_path):
    _run(tmp_path / "a", BIAS)
    _run(tmp_path / "b", {})
    a = runs.make_run("a", tmp_path / "a")
    b = runs.make_run("b", tmp_path / "b")
    assert bias.has_series_bias(a) is True
    assert bias.has_series_bias(b) is False
    # No checkpoint yet: unknown, so the run's own files decide.
    (tmp_path / "c").mkdir()
    assert bias.has_series_bias(runs.make_run("c", tmp_path / "c")) is None
    assert bias.has_series_bias(runs.make_run("x", None)) is None


def test_a_run_without_the_series_bias_is_named_not_drawn(tmp_path):
    _run(tmp_path / "finetuned", BIAS)
    _run(tmp_path / "nobiases", {})
    rs = [
        runs.make_run(
            "nobiases", tmp_path / "nobiases", label="No bias", live=True
        ),
        runs.make_run("finetuned", tmp_path / "finetuned", label="With bias"),
    ]
    got = bias.payload(rs, live_step=150)
    assert [r["key"] for r in got["runs"]] == ["finetuned"]
    assert got["nobias"] == [{"key": "nobiases", "label": "No bias"}]
    assert got["missing"] == []
    # The run with the bias is as it was alone.
    alone = bias.payload(rs[1:], live_step=None)
    assert got["runs"] == alone["runs"]
    assert got["pairs"] == alone["pairs"]


def test_only_runs_without_the_series_bias_give_no_panels(tmp_path):
    _run(tmp_path / "nobiases", {}, history=False)
    rs = [runs.make_run("nobiases", tmp_path / "nobiases", live=True)]
    got = bias.payload(rs, live_step=150)
    assert got["runs"] == []
    assert got["pairs"] == []
    assert [r["key"] for r in got["nobias"]] == ["nobiases"]


def _plan(data, on, lr):
    script = (
        "const {biasPlan}=require(process.argv[1]);"
        "const a=JSON.parse(process.argv[2]);"
        "console.log(JSON.stringify(biasPlan(a[0],a[1],a[2])))"
    )
    out = subprocess.run(
        [
            "node",
            "-e",
            script,
            str(PAGE / "bias.js"),
            json.dumps([data, on, lr]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


DATA = {
    "runs": [{"key": "finetuned", "label": "With bias"}],
    "missing": [{"key": "fresh", "label": "Fresh"}],
    "nobias": [{"key": "nobiases", "label": "No bias"}],
}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_selected_run_without_the_bias_gets_a_one_line_note():
    got = _plan(DATA, ["nobiases", "finetuned"], "nobiases")
    assert [r["key"] for r in got["runs"]] == ["finetuned"]
    assert got["notes"] == ["No bias: no attention bias in this run"]
    # The learning rate is the no-bias run's, so it is not drawn here.
    assert got["lr"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_only_runs_without_the_bias_collapse_to_the_note():
    got = _plan(DATA, ["nobiases"], "nobiases")
    assert got["runs"] == []
    assert got["notes"] == ["No bias: no attention bias in this run"]
    assert got["lr"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_run_with_the_bias_keeps_its_learning_rate():
    got = _plan(DATA, ["finetuned", "fresh"], "finetuned")
    assert [r["key"] for r in got["runs"]] == ["finetuned"]
    assert got["notes"] == ["Fresh: no attention bias logged or saved yet"]
    assert got["lr"] is True
    assert _plan(DATA, ["nobiases"], "finetuned")["lr"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_hidden_learning_rate_leaves_its_cell_empty():
    script = (
        "const {biasLayout}=require(process.argv[1]);"
        "console.log(JSON.stringify(biasLayout(["
        "{key:'same'},{key:'up'}],false)))"
    )
    out = subprocess.run(
        ["node", "-e", script, str(PAGE / "bias.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(out.stdout)
    assert got["cols"] == 3
    assert got["panels"] == [
        ["line", "same"],
        ["line", "up"],
        ["gap", None],
        ["heat", "same"],
        ["heat", "up"],
        ["layers", None],
    ]
