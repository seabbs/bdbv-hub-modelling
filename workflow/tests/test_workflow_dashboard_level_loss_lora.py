"""The size of the LoRA update."""

import json
import math
import struct

import pytest
from flowlib.dashboard import level_loss_lora, runs

HISTORY = [
    {
        "step": 0,
        "held_out_loss": 0.4,
        "by_level": {"leaf": 0.2, "national": 0.4},
        "by_source": {"daily": 0.2, "weekly": 0.8},
    },
    {"step": 10, "loss": 0.5},
    {"step": 100, "eval_loss": 0.39},
    {
        "step": 200,
        "eval_loss": 0.38,
        "by_source": {"daily": 0.1, "weekly": 0.6},
    },
]


def _safetensors(path, tensors):
    header, data = {}, b""
    for name, rows in tensors.items():
        flat = [x for r in rows for x in r]
        raw = struct.pack(f"<{len(flat)}f", *flat)
        header[name] = {
            "dtype": "F32",
            "shape": [len(rows), len(rows[0])],
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    head = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(head).to_bytes(8, "little") + head + data)


def _lora(layer, part, module):
    stem = (
        f"base_model.model.encoder.block.{layer}.layer.{part}."
        f"self_attention.{module}"
    )
    return f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"


def test_lora_update_size_by_layer(tmp_path):
    ckpt = tmp_path / "finetuned-ckpt"
    a0, b0 = _lora(0, 0, "q")
    a1, b1 = _lora(0, 1, "v")
    a2, b2 = _lora(1, 0, "q")
    _safetensors(
        ckpt / "adapter_model.safetensors",
        {
            # B @ A = [[1, 0], [0, 0]]: Frobenius norm 1.
            a0: [[1.0, 0.0]],
            b0: [[1.0], [0.0]],
            # B @ A = [[0, 2], [0, 0]]: norm 2.
            a1: [[0.0, 1.0]],
            b1: [[2.0], [0.0]],
            # Zero update.
            a2: [[0.0, 0.0]],
            b2: [[1.0], [1.0]],
        },
    )
    (ckpt / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}')
    # Scaled by alpha / r = 2; a layer's modules add in quadrature.
    got = level_loss_lora.lora_size(ckpt)
    assert got[0] == (0, pytest.approx(2 * math.sqrt(5)))
    assert got[1] == (1, 0.0)


def test_lora_size_of_a_run_by_checkpoint(tmp_path):
    folder = tmp_path / "run"
    a, b = _lora(0, 0, "q")
    for name in ("n250", "n500"):
        d = folder / "train" / name
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {a: [[1.0, 0.0]], b: [[1.0], [0.0]]},
        )
        (d / "finetuned-ckpt" / "adapter_config.json").write_text(
            '{"r": 1, "lora_alpha": 1}'
        )
        (d / "settings.json").write_text('{"steps": 10}')
    run = runs.Run("r5", "r5", folder=folder)
    assert level_loss_lora.run_lora(run) == {
        "n250": [(0, 1.0)],
        "n500": [(0, 1.0)],
    }


def test_the_section_shows_only_the_lora_update(tmp_path):
    # Validation loss by level is the loss section's; this one is LoRA's.
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(json.dumps(HISTORY))
    run = runs.Run("r5", "r5", folder=folder, live=True)
    out = level_loss_lora.section([run])
    assert out.count("<div class=card>") == 2
    assert "by encoder layer" in out
    assert "loss" not in out.lower()


def test_lora_update_size_by_kind(tmp_path):
    folder = tmp_path / "run"
    t_a, t_b = _lora(0, 0, "q")
    g_a, g_b = _lora(0, 1, "q")
    out = "base_model.model.output_patch_embedding.output_layer"
    for name, k in (("n250", 1.0), ("n500", 2.0)):
        d = folder / "train" / name
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {
                t_a: [[k, 0.0]],
                t_b: [[1.0], [0.0]],
                g_a: [[0.0, 3.0]],
                g_b: [[1.0], [0.0]],
                f"{out}.lora_A.weight": [[4.0, 0.0]],
                f"{out}.lora_B.weight": [[1.0], [0.0]],
            },
        )
        (d / "finetuned-ckpt" / "adapter_config.json").write_text(
            '{"r": 1, "lora_alpha": 1}'
        )
        (d / "settings.json").write_text(
            json.dumps({"steps": 10, "n_outbreaks": int(name[1:])})
        )
    run = runs.Run("r5", "r5", folder=folder)
    assert level_loss_lora.run_kinds(run) == {
        "time attention": [(250, 1.0), (500, 2.0)],
        "group attention": [(250, 3.0), (500, 3.0)],
        "output layer": [(250, 4.0), (500, 4.0)],
    }


def test_each_panel_has_its_explainer(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(json.dumps(HISTORY))
    run = runs.Run("r5", "r5", folder=folder, live=True)
    out = level_loss_lora.section(
        [run], tip=lambda key: f"<i data-key={key}></i>"
    )
    for key in level_loss_lora.EXPLAINERS:
        assert f"<i data-key={key}></i>" in out


def test_every_explainer_has_a_note():
    from flowlib.dashboard import page

    for key in level_loss_lora.EXPLAINERS:
        assert page.EXPLAIN.get(key)


def test_every_run_is_in_the_lora_panels(tmp_path):
    t_a, t_b = _lora(0, 0, "q")
    folders = {}
    for key, v in (("r5", 1.0), ("r6", 2.0)):
        folder = tmp_path / key
        d = folder / "train" / "n250"
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {t_a: [[v, 0.0]], t_b: [[1.0], [0.0]]},
        )
        (d / "finetuned-ckpt" / "adapter_config.json").write_text(
            '{"r": 1, "lora_alpha": 1}'
        )
        (d / "settings.json").write_text(
            json.dumps({"steps": 1500, "n_outbreaks": 250})
        )
        folders[key] = folder
    rs = [
        runs.make_run("r5", folders["r5"]),
        runs.make_run("r6", folders["r6"], live=True),
    ]
    out = level_loss_lora.section(rs)
    assert "time attention (r5)" in out
    assert "time attention (r6)" in out
    assert "simulations trained" in out
    # One legend: kinds by colour, runs by line style.
    assert out.count(">time attention</span>") == 1


def test_the_gram_matrices_are_the_products_of_the_lora_factors():
    from array import array

    import numpy as np
    from flowlib.dashboard import level_loss_lora as lll

    m = np.arange(6, dtype=np.float32).reshape(2, 3)
    values = array("f", m.ravel().tolist())
    np.testing.assert_allclose(lll._gram([2, 3], values, rows=True), m @ m.T)
    np.testing.assert_allclose(lll._gram([2, 3], values, rows=False), m.T @ m)
