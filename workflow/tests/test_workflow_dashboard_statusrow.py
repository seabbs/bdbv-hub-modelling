"""The dashboard's status: one condensed row, what is training now and
how long each checkpoint took."""

import inspect
import json
import re
from pathlib import Path

from flowlib.dashboard import page

PAGE = Path(page.__file__).parent / "page"
PLAN = (
    "2026-11-24 {t} [train] outbreaks {a}-{b}: 4000 blocks, "
    "1000000 series, {n} steps"
)
TIME = (
    "2026-11-24 {t} [train] chunk time: wait {w} s, cut {c} s, "
    "train {tr} s, eval {e} s"
)


def _run(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "n200.log").write_text(
        "\n".join(
            [
                PLAN.format(t="01:00:00", a=1, b=100, n=2400),
                TIME.format(t="02:00:00", w=0, c=60, tr=3300, e=240),
                PLAN.format(t="02:00:01", a=101, b=200, n=2400),
                TIME.format(t="03:00:00", w=0, c=60, tr=3000, e=1200),
            ]
        )
        + "\n"
    )
    (train / "n400.log").write_text(
        "\n".join(
            [
                PLAN.format(t="03:00:01", a=201, b=300, n=2400),
                TIME.format(t="04:00:00", w=0, c=600, tr=3000, e=0),
                PLAN.format(t="04:00:01", a=301, b=400, n=2416),
            ]
        )
        + "\n"
    )
    return tmp_path


def test_the_status_tiles_are_one_compact_row():
    # Eight tall tiles took two rows and 330 px above the plots.
    css = (PAGE / "style.css").read_text()
    rule = re.search(r"\.tiles\{([^}]*)\}", css)
    assert rule
    width = re.search(r"minmax\((\d+)px", rule[1])
    assert width
    assert int(width[1]) <= 150
    assert re.search(r"\.tile \.v\{[^}]*font-size:15px", css)


def test_training_now_names_the_chunk_and_its_progress(tmp_path):
    folder = _run(tmp_path)
    status = {"stage": "training", "step": 8720, "segment": "n400"}
    out = page.training_now(folder, status, 0.5)
    assert out.startswith("<p class=tnow>")
    assert "training" in out
    assert "simulations 301-400 of n400" in out
    # 7,200 steps in the three finished chunks, so 1,520 into this one.
    assert "1,520 of 2,416 steps" in out
    # 896 steps left at 0.5 a second.
    assert "30 min left" in out


def test_training_now_says_only_the_stage_when_the_chunk_is_unknown(tmp_path):
    status = {"stage": "cutting blocks of outbreaks 1-100", "step": 0}
    out = page.training_now(tmp_path, status, None)
    shown = out.rsplit("</span>", 1)[1]
    assert "cutting blocks of outbreaks 1-100" in shown
    assert "steps" not in shown
    assert page.training_now(tmp_path, {}, None) == ""


RESUME = (
    "2026-11-24 {t} [train] resumed at step {s}, outbreaks {a}-, batch {b}"
)
SAVED = "2026-11-24 {t} [train] checkpoint {n}: hash abc, CUDA 0.5 GB"


def _saved(train, name, sims, steps, published=None):
    (train / name).mkdir()
    (train / name / "settings.json").write_text(
        json.dumps({"sims_done": sims, "steps": steps, "chunk": 100})
    )
    if published:
        (train / name / "publish.log").write_text(
            f"2026-11-24 {published} [train] published chronos-slots-x: "
            "https://example.org\n"
        )


def _restarted(tmp_path):
    """A run whose n400 log lost its first chunk to a restart.

    n200 trained both its chunks in one go. n400 was restarted part way
    through its second chunk, 800 steps in: its log, written again by
    the restart, has only the rest of that chunk. n800 is training.
    """
    train = tmp_path / "train"
    train.mkdir()
    _saved(train, "n200", 200, 4800, published="03:05:00")
    _saved(train, "n400", 400, 9616)
    (train / "n200.log").write_text(
        "\n".join(
            [
                RESUME.format(t="00:59:59", s=0, a=1, b=0),
                PLAN.format(t="01:00:00", a=1, b=100, n=2400),
                TIME.format(t="02:00:00", w=0, c=60, tr=3300, e=240),
                PLAN.format(t="02:00:01", a=101, b=200, n=2400),
                SAVED.format(t="03:00:00", n="n200"),
                TIME.format(t="03:00:00", w=0, c=60, tr=3000, e=1200),
            ]
        )
        + "\n"
    )
    (train / "n400.log").write_text(
        "\n".join(
            [
                RESUME.format(t="04:33:19", s=8000, a=301, b=800),
                PLAN.format(t="04:33:20", a=301, b=400, n=2416),
                SAVED.format(t="05:00:00", n="n400"),
                TIME.format(t="05:00:00", w=0, c=0, tr=1600, e=0),
            ]
        )
        + "\n"
    )
    (train / "n800.log").write_text(
        "\n".join(
            [
                RESUME.format(t="05:10:00", s=9616, a=401, b=0),
                PLAN.format(t="05:10:01", a=401, b=500, n=2000),
                TIME.format(t="05:26:41", w=0, c=0, tr=1000, e=0),
                PLAN.format(t="05:26:42", a=501, b=600, n=2000),
            ]
        )
        + "\n"
    )
    return tmp_path


def _cells(out):
    rows = re.findall(r"<tr><td>(n\d+)</td>(.*?)</tr>", out)
    return {
        name: re.findall(r"<td[^>]*>(.*?)</td>", rest) for name, rest in rows
    }


def test_each_checkpoint_s_time_is_in_a_table(tmp_path):
    status = {"segment": "n800", "step": 12000}
    out = page.checkpoint_timings(_restarted(tmp_path), status)
    rows = _cells(out)
    assert list(rows) == ["n200", "n400", "n800"]
    # n200: simulations 1-200, 4,800 steps all timed: 7,860 s of chunk
    # time, 200 simulations in it, saved at 03:00 and released at 03:05.
    assert rows["n200"] == [
        "1-200",
        "4,800",
        "100%",
        "2 h 11 min",
        "92",
        "n/a",
        "24 Nov 03:00",
        "24 Nov 03:05",
    ]


def test_a_restarted_segment_counts_its_checkpoint_s_steps(tmp_path):
    status = {"segment": "n800", "step": 12000}
    rows = _cells(page.checkpoint_timings(_restarted(tmp_path), status))
    n400 = rows["n400"]
    # The checkpoints' steps, 9,616 less 4,800, not the one logged
    # chunk's 2,416; of them the log timed 1,616 (2,416 less the 800
    # done before the restart) in 1,600 s.
    assert n400[0] == "201-400"
    assert n400[1] == "4,816"
    assert n400[2] == "34%"
    # So about 4,768 s at the timed rate; 60.5 simulations of the chunk
    # in 1,600 s are 150 an hour.
    assert "1 h 19 min" in n400[3]
    assert "≈" in n400[3]
    assert n400[4] == "150"
    # Two hours of wall time from n200's save to n400's.
    assert n400[5] == "2 h 00 min"
    assert n400[6] == "24 Nov 05:00"
    assert n400[7] == "not yet"


def test_the_training_segment_is_marked_not_saved(tmp_path):
    status = {"segment": "n800", "step": 12000}
    rows = _cells(page.checkpoint_timings(_restarted(tmp_path), status))
    n800 = rows["n800"]
    # 2,384 steps so far, 2,000 of them in the finished chunk.
    assert n800[:3] == ["401-800", "2,384", "84%"]
    assert n800[6] == "training"
    assert n800[7] == ""


def test_the_table_has_no_rows_without_a_log(tmp_path):
    assert page.checkpoint_timings(tmp_path / "none", {}) == ""


def test_training_now_counts_from_the_resumed_step(tmp_path):
    # The log of n400 lost the chunks before the restart: the chunk's
    # steps run from its first step (8,000 less the 800 done), not from
    # the logged chunks' steps.
    folder = _restarted(tmp_path)
    (folder / "train" / "n800.log").unlink()
    (folder / "train" / "n400.log").write_text(
        RESUME.format(t="04:33:19", s=8000, a=301, b=800)
        + "\n"
        + PLAN.format(t="04:33:20", a=301, b=400, n=2416)
        + "\n"
    )
    status = {"stage": "training", "step": 8500, "segment": "n400"}
    out = page.training_now(folder, status, None)
    assert "simulations 301-400 of n400" in out
    assert "1,300 of 2,416 steps" in out


def test_the_status_section_shows_now_throughput_and_timings():
    src = inspect.getsource(page)
    tiles = src[src.index('f"<div class=tiles>{tiles}</div>') :][:300]
    for part in ("{now}", "{tput}", "{timings}"):
        assert part in tiles
