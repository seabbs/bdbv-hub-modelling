"""The LoRA update: its size, direction and change, from the adapters."""

import json
import math
import shutil
import struct
import subprocess

import pytest
from flowlib.dashboard import level_loss_lora, runs


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


def _adapter(folder, name, tensors, n, r=1, alpha=1):
    d = folder / "train" / name
    _safetensors(d / "finetuned-ckpt" / "adapter_model.safetensors", tensors)
    (d / "finetuned-ckpt" / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha})
    )
    (d / "settings.json").write_text(json.dumps({"n_outbreaks": n}))
    return d / "finetuned-ckpt"


OUT = "base_model.model.output_patch_embedding.output_layer"


def test_each_module_s_update_size_factors_and_directions(tmp_path):
    a0, b0 = _lora(0, 0, "q")
    a1, b1 = _lora(0, 1, "v")
    ckpt = _adapter(
        tmp_path,
        "n250",
        {
            # B @ A = [[1, 0], [0, 0]]: norm 1, one direction.
            a0: [[1.0, 0.0], [0.0, 0.0]],
            b0: [[1.0, 0.0], [0.0, 0.0]],
            # B @ A = 3 I: norm 3 sqrt(2), two equal directions.
            a1: [[3.0, 0.0], [0.0, 3.0]],
            b1: [[1.0, 0.0], [0.0, 1.0]],
        },
        250,
        r=2,
        alpha=4,
    )
    got = level_loss_lora.modules(ckpt)
    q = got["base_model.model.encoder.block.0.layer.0.self_attention.q"]
    v = got["base_model.model.encoder.block.0.layer.1.self_attention.v"]
    # Scaled by alpha / r = 2.
    assert (q.kind, q.layer, q.proj) == ("time attention", 0, "q")
    assert (v.kind, v.layer, v.proj) == ("group attention", 0, "v")
    assert q.size == pytest.approx(2.0)
    assert v.size == pytest.approx(2 * 3 * math.sqrt(2))
    assert (q.a, q.b) == (pytest.approx(1.0), pytest.approx(1.0))
    assert v.a == pytest.approx(3 * math.sqrt(2))
    assert q.rank == pytest.approx(1.0)
    assert v.rank == pytest.approx(2.0)


def test_effective_rank_runs_from_one_direction_to_all():
    assert level_loss_lora.effective_rank([5.0, 0.0, 0.0]) == pytest.approx(1)
    assert level_loss_lora.effective_rank([2.0] * 8) == pytest.approx(8)
    assert 1 < level_loss_lora.effective_rank([2.0, 1.0]) < 2
    assert level_loss_lora.effective_rank([0.0, 0.0]) == 0.0


def test_the_change_between_checkpoints_is_the_update_s_difference(tmp_path):
    a, b = _lora(0, 0, "q")
    old = _adapter(tmp_path, "n250", {a: [[1.0, 0.0]], b: [[1.0], [0.0]]}, 250)
    new = _adapter(tmp_path, "n500", {a: [[1.0, 0.0]], b: [[1.0], [2.0]]}, 500)
    got = level_loss_lora.changes(old, new)
    # [[1, 0], [2, 0]] - [[1, 0], [0, 0]]: norm 2.
    name = a.removesuffix(".lora_A.weight")
    assert got[name] == pytest.approx(4.0)
    assert level_loss_lora.changes(new, new)[name] == pytest.approx(0.0)


def test_the_payload_has_each_view_of_each_run(tmp_path):
    folder = tmp_path / "run"
    tq_a, tq_b = _lora(0, 0, "q")
    tk_a, tk_b = _lora(0, 0, "k")
    g_a, g_b = _lora(1, 1, "q")
    for name, k in (("n500", 2.0), ("n250", 1.0)):
        _adapter(
            folder,
            name,
            {
                tq_a: [[k, 0.0]],
                tq_b: [[1.0], [0.0]],
                tk_a: [[0.0, 1.0]],
                tk_b: [[1.0], [0.0]],
                g_a: [[3.0, 0.0]],
                g_b: [[1.0], [0.0]],
                f"{OUT}.lora_A.weight": [[4.0, 0.0]],
                f"{OUT}.lora_B.weight": [[1.0], [0.0]],
            },
            int(name[1:]),
        )
    empty = tmp_path / "empty"
    (empty / "train").mkdir(parents=True)
    rs = [
        runs.Run("r5", "Run 5", "#123456", folder=folder),
        runs.Run("r6", "Run 6", "#654321", folder=empty),
    ]
    got = level_loss_lora.payload(rs)
    assert got["missing"] == [{"key": "r6", "label": "Run 6"}]
    (run,) = got["runs"]
    assert (run["key"], run["label"], run["color"]) == (
        "r5",
        "Run 5",
        "#123456",
    )
    assert run["n"] == [250, 500]
    assert run["layers"] == 2
    # Each layer's modules in quadrature; a layer with none is 0.
    time = run["size"]["time attention"]
    assert time[0] == pytest.approx([math.sqrt(2), 0.0])
    assert time[1] == pytest.approx([math.sqrt(5), 0.0])
    assert run["size"]["group attention"][1] == pytest.approx([0.0, 3.0])
    assert run["output"] == pytest.approx([4.0, 4.0])
    # Each projection's root mean square over the attention sublayers.
    assert run["proj"]["q"][1] == pytest.approx(math.sqrt((4 + 9) / 2))
    assert run["proj"]["k"][1] == pytest.approx(1.0)
    # The change since the previous checkpoint over the layer's size now.
    change = run["change"]["time attention"]
    assert change[0] is None
    assert change[1] == pytest.approx([1 / math.sqrt(5), None])
    assert run["change"]["group attention"][1] == pytest.approx([None, 0.0])
    # The latest checkpoint's effective rank and factor norms by layer.
    assert run["rank"]["time attention"] == pytest.approx([1.0, None])
    assert run["a"]["time attention"] == pytest.approx([math.sqrt(5), None])
    assert run["b"]["group attention"] == pytest.approx([None, 1.0])


def test_the_section_is_one_grid_with_its_data_and_explainers(tmp_path):
    folder = tmp_path / "run"
    a, b = _lora(0, 0, "q")
    _adapter(folder, "n250", {a: [[1.0, 0.0]], b: [[1.0], [0.0]]}, 250)
    run = runs.Run("r5", "r5", folder=folder, live=True)
    out = level_loss_lora.section(
        [run], tip=lambda key: f"<i data-key={key}></i>"
    )
    assert "id=loragrid" in out
    data = out.split("id=lodata>")[1].split("</script>")[0]
    assert json.loads(data)["runs"][0]["n"] == [250]
    assert "loss" not in out.lower().replace("lora", "")
    for key in level_loss_lora.EXPLAINERS:
        assert f"<i data-key={key}></i>" in out


def test_every_explainer_has_a_note():
    from flowlib.dashboard import page

    for key in level_loss_lora.EXPLAINERS:
        assert page.EXPLAIN.get(key)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_every_panel_is_laid_out_once():

    js = level_loss_lora.ASSET.read_text()
    out = subprocess.run(
        ["node", "-e", js + "\nconsole.log(JSON.stringify(loraLayout()))"],
        capture_output=True,
        text=True,
        check=True,
    )
    panels = json.loads(out.stdout)
    assert len(panels) == len(set(panels)) == 8
    assert {p.split(":")[0] for p in panels} == {"row1", "row2"}
