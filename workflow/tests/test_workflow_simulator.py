"""The simulator's code files and its grid file's identifier."""

import hashlib

from flowlib import simulator


def test_every_engine_path_exists():
    for path in simulator.ENGINE:
        assert (simulator.ROOT / path).is_dir(), path


def test_the_grid_id_is_the_file_s_short_sha():
    grid = "packages/nowcastfoundation/config/finetune/grid.yaml"
    digest = hashlib.sha256((simulator.ROOT / grid).read_bytes()).hexdigest()
    assert simulator.grid_id(grid) == digest[:12]
