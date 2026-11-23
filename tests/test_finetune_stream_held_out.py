"""A restarted trainer reuses its held-out blocks when nothing changed."""

import numpy as np

from nowcastfoundation.finetune import stream as fs


def _blocks(tag):
    return [
        {
            "context": np.zeros((1, 2)),
            "n_targets": 1,
            "prediction_length": 1,
            "levels": ("national",),
            "tag": tag,
        }
    ]


def test_held_out_blocks_are_cut_once_per_key(tmp_path):
    cuts = []

    def cut(tag):
        def go():
            cuts.append(tag)
            return _blocks(tag)

        return go

    key = {"outbreaks": ["a", "b"], "seed": 1}
    first = fs.held_out_blocks(tmp_path, key, cut("first"))
    # A restart with the same outbreaks and settings cuts nothing.
    again = fs.held_out_blocks(tmp_path, key, cut("again"))
    assert cuts == ["first"]
    assert first[0]["tag"] == again[0]["tag"] == "first"
    # Any change to what the cut depends on cuts afresh.
    other = fs.held_out_blocks(tmp_path, {**key, "seed": 2}, cut("other"))
    assert cuts == ["first", "other"]
    assert other[0]["tag"] == "other"


def test_the_held_out_key_hashes_the_cutting_code_not_the_trainer():
    # A trainer edit outside the cut must not re-key the saved held-out
    # blocks (Run A re-cut them for 20 minutes after #753 and #754).
    import hashlib
    import inspect

    from nowcastfoundation.finetune import stream as fs

    code = fs.cut_code_sha256()
    assert fs.ft.TRAINER not in code
    want = hashlib.sha256(inspect.getsource(fs.cut).encode()).hexdigest()
    assert code["stream.cut"] == want
    assert "stream.snapshots" in code
    assert "foundation.layout_block" in code
    assert set(fs.ft.CODE) <= set(code)


def test_each_trainer_constant_has_its_own_docstring():
    # Review on #784: NEXT_CHUNK_* sat between HELD_OUT_* and their
    # docstrings. Each module-level constant is followed by its string.
    import ast
    import inspect

    from nowcastfoundation.finetune import stream as fs

    body = ast.parse(inspect.getsource(fs)).body
    names = (
        "HELD_OUT_BLOCKS",
        "HELD_OUT_KEY",
        "NEXT_CHUNK_BLOCKS",
        "NEXT_CHUNK_KEY",
    )
    for i, node in enumerate(body):
        if isinstance(node, ast.Assign) and node.targets[0].id in names:
            doc = body[i + 1]
            assert isinstance(doc, ast.Expr), node.targets[0].id
            assert isinstance(doc.value, ast.Constant), node.targets[0].id
