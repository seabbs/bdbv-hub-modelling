"""The attention bias over training: one trajectory per layer."""

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


def _run(tmp_path):
    """A run with a history entry, a checkpoint and its live state."""
    folder = tmp_path / "run"
    d = folder / "train" / "n250"
    _safetensors(
        d / "finetuned-ckpt" / "adapter_model.safetensors",
        {NAMES["same"]: [[1.0, 3.0], [5.0, 7.0]]},
    )
    (d / "settings.json").write_text(
        json.dumps({"steps": 250, "n_outbreaks": 250})
    )
    _safetensors(
        folder / "train" / "state" / "adapter_model.safetensors",
        {NAMES["same"]: [[2.0, 2.0], [8.0, 8.0]]},
    )
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {
                    "step": 100,
                    "bias": {
                        "same": {
                            "mean": 0.25,
                            "min": 0.0,
                            "max": 0.5,
                            "values": [[0.0, 0.0], [0.5, 0.5]],
                        }
                    },
                },
                # An entry with only the summary has no layers.
                {
                    "step": 200,
                    "bias": {"same": {"mean": 0.3, "min": 0.1, "max": 0.5}},
                },
            ]
        )
    )
    return runs.make_run("r", folder, live=True)


def test_each_layer_s_mean_over_heads_from_history_checkpoints_and_state(
    tmp_path,
):
    got = bias.payload([_run(tmp_path)], live_step=300)
    layers = got["runs"][0]["layers"]["same"]
    assert layers == [
        [100, [0.0, 0.5]],
        [250, [2.0, 6.0]],
        [300, [2.0, 8.0]],
    ]


def test_logged_steps_without_values_keep_the_run_s_mean(tmp_path):
    got = bias.payload([_run(tmp_path)], live_step=300)
    pts = got["runs"][0]["pts"]["same"]
    # The summary-only entry still gives the run's mean at its step.
    assert [p[0] for p in pts] == [100, 200, 250, 300]
    assert pts[1][1] == 0.3


def test_layer_means_average_each_row():
    assert bias.layer_means([[1.0, 3.0], [5.0, 7.0]]) == [2.0, 6.0]


def _shade(n):
    script = (
        "const {layerShade}=require(process.argv[1]);"
        "const n=+process.argv[2];"
        "console.log(JSON.stringify([...Array(n).keys()]"
        ".map(i=>layerShade('#123456',i,n))))"
    )
    out = subprocess.run(
        ["node", "-e", script, str(PAGE / "bias.js"), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_layers_shade_from_light_to_dark_in_the_relation_s_hue():
    got = _shade(12)
    assert len(set(got)) == 12
    assert all("#123456" in c for c in got)
    # Shallow layers are mixed into the card, deep ones with ink.
    assert "var(--card)" in got[0]
    assert "var(--fg)" in got[-1]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_one_layer_takes_the_relation_s_own_colour():
    assert _shade(1) == ["#123456"]


def test_the_panel_draws_layers_not_a_band():
    js = (PAGE / "bias.js").read_text()
    assert "<polygon" not in js
    assert "min to max" not in js
    assert "each layer, mean over heads" in js
