"""The simulator benchmark (`nowcastfoundation.simulator.bench`)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from nowcastfoundation.simulator import bench

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"


@pytest.mark.slow
def test_a_benchmark_times_a_seeded_batch_twice(tmp_path):
    out = tmp_path / "bench.json"
    bench.main([str(GRID_FILE), "2", "9191", str(out), "--weeks", "8"])
    got = json.loads(out.read_text())
    assert got["n"] == 2
    assert got["backend"] == "cpu"
    # A cold pass compiles; the warm pass reuses what it can.
    assert got["cold"]["compiles"] > 0
    assert got["cold"]["seconds"] >= got["cold"]["compile_seconds"] >= 0
    assert got["warm"]["compiles"] <= got["cold"]["compiles"]
    for p in ("cold", "warm"):
        assert got[p]["sims_per_second"] > 0
    assert got["peak_rss_mb"] > 0
    # Both passes draw the same counts: the digest is the equality check
    # each optimisation is held to.
    assert got["cold"]["counts_sha256"] == got["warm"]["counts_sha256"]
    assert len(got["cold"]["counts_sha256"]) == 64


def test_the_digest_ignores_the_gzip_header(tmp_path):
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    for f, mtime in ((a, 1), (b, 2)):
        with gzip.GzipFile(f, "wb", mtime=mtime) as g:
            g.write(b"x,y\n1,2\n")
    assert a.read_bytes() != b.read_bytes()
    assert bench.digest(a) == bench.digest(b)
