"""Each job's memory from its own benchmarks (`flowlib.memory`)."""

from types import SimpleNamespace

from flowlib import memory


def _bench(path, rss):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"s\th:m:s\tmax_rss\tmax_vms\n10.0\t0:00:10\t{rss}\t9999\n")


def test_a_job_asks_for_its_last_peak_with_a_margin(tmp_path):
    pattern = str(tmp_path / "bench" / "{model}@{key}.tsv")
    mem = memory.from_benchmark(pattern, 1024, like=("key",))
    wc = SimpleNamespace(model="zero", key="k1")
    # Never run: the rule's minimum.
    assert mem(wc, 1) == 1024
    _bench(tmp_path / "bench" / "zero@k0.tsv", 8000)
    # A new key takes the same model's peak under another key.
    assert mem(wc, 1) == memory.padded(8000)
    _bench(tmp_path / "bench" / "zero@k1.tsv", 3000)
    # Its own peak once it has one.
    assert mem(wc, 1) == memory.padded(3000)
    assert memory.padded(3000) >= 3000 * memory.MARGIN
    # Each retry after running out doubles it.
    assert mem(wc, 2) == 2 * memory.padded(3000)
    assert mem(wc, 3) == 4 * memory.padded(3000)


def test_the_minimum_stands_under_a_small_peak(tmp_path):
    pattern = str(tmp_path / "{n}.tsv")
    _bench(tmp_path / "1.tsv", 10)
    assert (
        memory.from_benchmark(pattern, 2048)(SimpleNamespace(n="1"), 1) == 2048
    )


def test_a_benchmark_without_a_peak_counts_as_none(tmp_path):
    path = tmp_path / "1.tsv"
    path.write_text("s\th:m:s\tmax_rss\n1\t0:00:01\tNA\n")
    assert memory.peak_mb(path) is None
    assert memory.peak_mb(tmp_path / "missing.tsv") is None
