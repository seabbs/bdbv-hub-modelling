"""The next chunk's cut blocks are saved, so a restart loads them."""

import numpy as np

from nowcastfoundation.finetune import stream as fs


def _pairs():
    block = {"context": np.ones((2, 3), dtype=np.float32), "n_targets": 1}
    return [("33400-0001-1", block)], ["daily cut 0: P01"]


def test_a_saved_chunk_loads_under_its_key_only(tmp_path):
    key = {"outbreaks": ["33400-0001"], "seed": 1, "code_sha256": {"a": "x"}}
    assert fs.load_chunk(tmp_path, key) is None
    pairs, empty = _pairs()
    fs.save_chunk(tmp_path, key, pairs, empty)
    got = fs.load_chunk(tmp_path, key)
    assert got is not None
    ((outbreak, block),) = got[0]
    assert outbreak == "33400-0001-1"
    np.testing.assert_array_equal(block["context"], pairs[0][1]["context"])
    assert got[1] == empty
    # Another chunk, or other cut code, cuts again.
    assert fs.load_chunk(tmp_path, key | {"outbreaks": ["33400-0002"]}) is None
    assert fs.load_chunk(tmp_path, key | {"code_sha256": {"a": "y"}}) is None


def test_saving_the_next_chunk_replaces_the_last(tmp_path):
    pairs, empty = _pairs()
    fs.save_chunk(tmp_path, {"outbreaks": ["a"]}, pairs, empty)
    fs.save_chunk(tmp_path, {"outbreaks": ["b"]}, pairs, empty)
    assert fs.load_chunk(tmp_path, {"outbreaks": ["a"]}) is None
    assert fs.load_chunk(tmp_path, {"outbreaks": ["b"]}) is not None
    assert len(list(tmp_path.glob("next-chunk-blocks*.pkl"))) == 1
