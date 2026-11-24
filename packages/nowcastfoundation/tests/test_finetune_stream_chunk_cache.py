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


def test_the_chunk_in_training_stays_saved_beside_the_next(tmp_path):
    # Lead 11:25: Run B's driver restart lost the chunk it was training
    # (801-900): only the next chunk (901-1000) was saved, so 801-900 was
    # cut again with the GPU idle. The last two chunks saved stay.
    pairs, empty = _pairs()
    for name in ("a", "b", "c"):
        fs.save_chunk(tmp_path, {"outbreaks": [name]}, pairs, empty)
    assert fs.load_chunk(tmp_path, {"outbreaks": ["a"]}) is None
    assert fs.load_chunk(tmp_path, {"outbreaks": ["b"]}) is not None
    assert fs.load_chunk(tmp_path, {"outbreaks": ["c"]}) is not None
    assert len(list(tmp_path.glob("chunk-blocks-*.pkl"))) == fs.CHUNK_SLOTS == 2


def test_saving_a_chunk_again_keeps_the_other(tmp_path):
    pairs, empty = _pairs()
    fs.save_chunk(tmp_path, {"outbreaks": ["a"]}, pairs, empty)
    fs.save_chunk(tmp_path, {"outbreaks": ["b"]}, pairs, empty)
    fs.save_chunk(tmp_path, {"outbreaks": ["b"]}, pairs, empty)
    assert fs.load_chunk(tmp_path, {"outbreaks": ["a"]}) is not None
    assert fs.load_chunk(tmp_path, {"outbreaks": ["b"]}) is not None
