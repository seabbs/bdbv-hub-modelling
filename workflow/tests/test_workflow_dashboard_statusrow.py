"""The dashboard's status: one condensed row, what is training now and
how long each checkpoint took."""

import inspect
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


def test_each_checkpoint_s_time_is_in_a_table(tmp_path):
    out = page.checkpoint_timings(_run(tmp_path))
    rows = re.findall(r"<tr><td>(n\d+)</td>(.*?)</tr>", out)
    assert [r[0] for r in rows] == ["n200", "n400"]
    n200 = rows[0][1]
    # Two chunks, 4,800 steps; 6,300 s training, 120 s cutting, 1,440 s
    # evaluating, 7,860 s in all.
    assert "<td>2</td><td>4,800</td>" in n200
    assert "1 h 45 min" in n200
    assert "24 min" in n200
    assert "2 h 11 min" in n200
    assert page.checkpoint_timings(tmp_path / "none") == ""


def test_the_status_section_shows_now_throughput_and_timings():
    src = inspect.getsource(page)
    tiles = src[src.index('f"<div class=tiles>{tiles}</div>') :][:300]
    for part in ("{now}", "{tput}", "{timings}"):
        assert part in tiles
