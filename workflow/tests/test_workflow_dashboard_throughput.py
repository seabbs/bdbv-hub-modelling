"""Training throughput from the trainer's chunk lines."""

import pytest
from flowlib.dashboard import logs, page

PLAN = (
    "2026-11-24 {t} [train] outbreaks {a}-{b}: 4000 blocks, "
    "1000000 series, {n} steps"
)
TIME = (
    "2026-11-24 {t} [train] chunk time: wait {w} s, cut {c} s, "
    "train {tr} s, eval {e} s"
)


def _segment(path, chunks):
    lines = []
    for a, n, w, c, tr, e in chunks:
        lines.append(PLAN.format(t="01:00:00", a=a, b=a + 99, n=n))
        lines.append(TIME.format(t="02:00:00", w=w, c=c, tr=tr, e=e))
    path.write_text("\n".join(lines) + "\n")


def test_each_chunk_s_time_is_tied_to_its_planned_chunk(tmp_path):
    log = tmp_path / "n200.log"
    _segment(log, [(101, 2400, 0, 600, 3000, 0)])
    got = logs.Log(log).chunk_times()
    assert got == [
        {
            "t": got[0]["t"],
            "first": 101,
            "sims": 100,
            "steps": 2400,
            "before": 0,
            "wait": 0.0,
            "cut": 600.0,
            "train": 3000.0,
            "eval": 0.0,
        }
    ]


def test_a_resumed_chunk_keeps_its_newest_time(tmp_path):
    (tmp_path / "train").mkdir()
    _segment(tmp_path / "train" / "n200.log", [(101, 2400, 0, 600, 3000, 0)])
    _segment(tmp_path / "train" / "n400.log", [(101, 2400, 0, 10, 2000, 0)])
    got = logs.RunLog(tmp_path).chunk_times()
    assert len(got) == 1
    assert got[0]["train"] == 2000.0
    assert got[0]["segment"] == "n400"


def test_throughput_is_over_the_chunks_own_time():
    rows = [
        {
            "sims": 100,
            "steps": 2400,
            "wait": 0,
            "cut": 600,
            "train": 2800,
            "eval": 200,
        },
        {
            "sims": 100,
            "steps": 1200,
            "wait": 0,
            "cut": 0,
            "train": 1600,
            "eval": 0,
        },
    ]
    got = logs.throughput(rows)
    # 3,600 steps and 200 simulations in 5,200 s.
    assert got["steps_per_min"] == pytest.approx(3600 / 5200 * 60)
    assert got["sims_per_hour"] == pytest.approx(200 / 5200 * 3600)
    assert got["gpu_busy"] == pytest.approx(4600 / 5200)
    assert got["share"]["cut"] == pytest.approx(600 / 5200)
    assert got["chunks"] == 2
    assert logs.throughput([]) is None


def test_the_status_has_one_throughput_line(tmp_path):
    (tmp_path / "train").mkdir()
    _segment(tmp_path / "train" / "n200.log", [(1, 2400, 0, 0, 3600, 0)])
    _segment(
        tmp_path / "train" / "n400.log",
        [(201, 2400, 0, 1800, 1800, 0)],
    )
    out = page.throughput_line(tmp_path, "n400")
    assert out.count("<p class=tput>") == 1
    assert "this segment (n400, 1 chunk)" in out
    assert "whole run (2 chunks)" in out
    # n400: 2,400 steps in 3,600 s; GPU busy half the time.
    assert "40.0 steps/min" in out
    assert "100 simulations/h" in out
    assert "GPU busy 50%" in out
    assert "cut 50%" in out
    assert page.throughput_line(tmp_path / "none", "n400") == ""


def test_the_line_sits_under_the_status_tiles():
    import inspect

    src = inspect.getsource(page)
    tiles = src[src.index('f"<div class=tiles>{tiles}</div>') :]
    assert "{tput}" in tiles[:200]


def test_a_chunk_resumed_part_way_counts_only_the_steps_it_timed(tmp_path):
    # The trainer resumed 918 steps into a 1,796-step chunk: its time is
    # for the other 878 steps, and for that share of its simulations.
    log = tmp_path / "n3200.log"
    log.write_text(
        "2026-11-25 22:23:22 [train] resumed at step 54400, "
        "outbreaks 3101-, batch 918\n"
        + PLAN.format(t="22:23:24", a=3101, b=3200, n=1796)
        + "\n"
        + TIME.format(t="22:39:08", w=0, c=1, tr=944, e=0)
        + "\n"
    )
    got = logs.Log(log).chunk_times()
    assert got[0]["steps"] == 1796
    assert got[0]["before"] == 918
    tput = logs.throughput(got)
    assert tput["steps_per_min"] == pytest.approx(878 / 945 * 60)
    assert tput["sims_per_hour"] == pytest.approx(100 * 878 / 1796 / 945 * 3600)
