"""Two hosts share a round: the seed split and the push of a batch."""

import hashlib
import subprocess
from pathlib import Path

import pytest
from flowlib import hosts, manifest

ROOT = Path(__file__).parents[2]
TIMED = ROOT / "workflow" / "scripts" / "timed.sh"


def _round(r):
    return range(30000 + 100 * r, 30000 + 100 * r + 3)


def test_seed_specs_are_seeds_ranges_and_rounds():
    assert hosts.seeds("31005 31900-31902 r2", _round) == {
        31005,
        31900,
        31901,
        31902,
        30200,
        30201,
        30202,
    }


def test_a_host_with_seeds_simulates_only_those():
    mine = hosts.seeds("31005 31900-31939", _round)
    assert hosts.simulates(31005, mine)
    assert hosts.simulates(31939, mine)
    assert not hosts.simulates(31006, mine)


def test_a_host_with_no_seeds_simulates_every_seed():
    assert hosts.simulates(31006, hosts.seeds("", _round))


def test_a_bad_seed_spec_is_an_error():
    with pytest.raises(ValueError, match="x12"):
        hosts.seeds("31005 x12", _round)


def _batch(sims: Path, seed: int) -> None:
    (sims / "logs").mkdir(parents=True)
    (sims / f"counts-{seed}.csv.gz").write_bytes(b"counts")
    (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    (sims / "logs" / f"{seed}.log").write_text("peak_rss_kb 10 wall_s 2\n")


def test_a_pushed_batch_lands_before_its_manifest_row(tmp_path):
    sims = tmp_path / "here" / "r10" / "sims"
    _batch(sims, 31005)
    there = tmp_path / "there" / "r10"
    row = "10,31005,block,r5,abc,1,21,x,y,,10,2,helper,now"
    hosts.push(sims, 31005, there, row, remote="")
    assert (there / "sims" / "counts-31005.csv.gz").read_bytes() == b"counts"
    assert (there / "sims" / "settings-31005.csv").exists()
    assert (there / "sims" / "logs" / "31005.log").exists()
    lines = (there / "manifest.csv").read_text().splitlines()
    assert lines == [manifest.HEADER, row]
    # Pushed again, the row is not repeated.
    hosts.push(sims, 31005, there, row, remote="")
    assert (there / "manifest.csv").read_text().count("31005") == 1


def test_the_row_of_a_pushed_batch_hashes_its_files(tmp_path):
    sims = tmp_path / "sims"
    _batch(sims, 31005)
    row = manifest.row(
        sims,
        round_=10,
        seed=31005,
        kind="block",
        grid="r5",
        commit="abc",
        draws=1,
        batch=21,
        host="helper",
        finished="now",
    )
    assert row.split(",")[7] == hashlib.sha256(b"counts").hexdigest()


def test_timing_writes_the_gnu_time_line_on_any_os():
    out = subprocess.run(
        ["bash", str(TIMED), "bash", "-c", "echo err >&2; echo out"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 0
    assert out.stdout == "out\n"
    assert "err" in out.stderr
    last = out.stderr.strip().splitlines()[-1].split()
    assert last[0] == "peak_rss_kb"
    assert int(last[1]) > 0
    assert last[2] == "wall_s"
    float(last[3])


def test_timing_without_a_time_program_gives_the_wall_time(tmp_path):
    # A cluster node may have neither GNU nor BSD time: the batch still
    # runs, with its wall time and no peak memory.
    got = subprocess.run(
        ["bash", str(TIMED), "bash", "-c", "echo hi >&2; exit 3"],
        env={"PATH": "/usr/bin:/bin", "TIMED_TIME": str(tmp_path / "none")},
        capture_output=True,
        text=True,
    )
    assert got.returncode == 3
    assert "hi" in got.stderr
    assert "peak_rss_kb" not in got.stderr
    last = got.stderr.strip().splitlines()[-1].split()
    assert last[0] == "wall_s"
    float(last[1])


def test_a_mirrored_round_counts_once_its_qc_ran(tmp_path):
    there, here = tmp_path / "there", tmp_path / "here"
    there.mkdir()
    (there / "manifest.csv").write_text("round,seed\n9,901\n9,902\n")
    assert hosts.mirrored_seeds("", there, here) == set()
    assert (here / "manifest.csv").exists()
    (there / "qc").mkdir()
    (there / "qc" / "summary.txt").write_text("ok\n")
    assert hosts.mirrored_seeds("", there, here) == {901, 902}


def test_an_unreachable_round_is_said_and_has_no_batches(tmp_path, capsys):
    got = hosts.mirrored_seeds("", tmp_path / "none", tmp_path / "here")
    assert got == set()
    assert "could not copy" in capsys.readouterr().err
