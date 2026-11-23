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
