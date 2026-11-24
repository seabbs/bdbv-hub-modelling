"""The dashboard's plots read well at the size they are shown."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from flowlib.dashboard import level_loss_lora, page, runs, svg

PAGE = Path(page.__file__).parent / "page"


def _node(script: str) -> object:
    out = subprocess.run(
        ["node", "-e", script, str(PAGE / "axes.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_the_level_and_lora_charts_are_drawn_at_their_card_width(tmp_path):
    # Drawn 620 wide and shrunk into a 430 px card, their text was 7 px.
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    history = [{"step": 0, "held_out_loss": 0.4, "by_level": {"leaf": 0.2}}]
    (folder / "train" / "history.json").write_text(json.dumps(history))
    out = level_loss_lora.section(
        [runs.Run("r5", "r5", folder=folder, live=True)]
    )
    widths = {int(w) for w in re.findall(r'viewBox="0 0 (\d+) ', out)}
    assert widths
    assert max(widths) <= level_loss_lora.CARD_WIDTH <= 440


def test_a_chart_s_note_is_small_print():
    out = svg.chart(
        "Loss", [("a", "#111", [(0, 1.0), (1, 0.5)], "pts")], note="n"
    )
    assert "<p class=cnote>n</p>" in out


def test_mark_labels_that_would_overlap_are_left_out():
    # n100, n200 and n400 printed on top of each other near the origin.
    out = svg.chart(
        "Loss",
        [("a", "#111", [(0, 1.0), (3000, 0.5)], "line")],
        marks=[(100, "n100"), (200, "n200"), (400, "n400"), (1600, "n1600")],
        width=420,
    )
    drawn = re.findall(r"class=ax>(n\d+)</text>", out)
    assert drawn[0] == "n100"
    assert "n1600" in drawn
    assert "n200" not in drawn
    # Every mark keeps its line and names itself on hover.
    assert out.count("class=mark") == 4
    assert "<title>n200</title>" in out


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_close_labels_are_thinned_left_to_right():
    got = _node(
        "const {thinLabels}=require(process.argv[1]);"
        "console.log(JSON.stringify(thinLabels([10,20,40,300,310],28)))"
    )
    assert got == [0, 2, 3]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_line_breaks_where_its_points_have_a_long_gap():
    # The validation subset ran straight from n100 to n1600 across steps
    # with no evaluation, as if it had been measured there.
    got = _node(
        "const {splitGaps}=require(process.argv[1]);"
        "const p=[[0,1],[10,1],[20,1],[30,1],[200,1],[210,1],[220,1]];"
        "console.log(JSON.stringify(splitGaps(p).map(s=>s.length)))"
    )
    assert got == [4, 3]


def test_the_loss_charts_draw_at_their_card_s_width():
    js = (PAGE / "loss.js").read_text()
    size = js[js.index("function size(") :]
    size = size[: size.index("}") + 1]
    assert "el.clientWidth" in size.split("if(!f)")[1].split(";")[0]
    assert "thinLabels(" in js
    assert "splitGaps(" in js


def test_the_scores_sit_below_the_plot_in_three_columns():
    # Six score panels shared a 540 px column beside the plot.
    css = (PAGE / "style.css").read_text()
    js = (PAGE / "nowcasts.js").read_text()
    rule = re.search(r"\.gscores\{([^}]*)\}", css)
    assert rule
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in rule[1]
    wrap = re.search(r"\.gwrap\{([^}]*)\}", css)
    assert wrap
    assert "grid-template-columns:1fr" in wrap[1]
    assert "function scoreCols(" in js


def test_bars_have_round_ticks_and_no_number_on_every_bar():
    js = (PAGE / "nowcasts.js").read_text()
    assert "rotate(-90)" not in js
    assert "niceTicks(gb.top" in js


def test_the_learning_rate_chart_is_a_titled_card_that_fills_it():
    out = page.lr_card(
        {
            "groups": {"lora": 1e-5},
            "lr_outbreaks": 3200,
            "logged": {},
            "current": {"lora": 5e-6},
        },
        10,
        "runa",
        lambda s: s * 10,
    )
    assert "<h3>Learning rate" in out
    assert 'width="100%"' in out


def test_the_checkpoints_table_scrolls_on_its_own_on_a_phone():
    css = (PAGE / "style.css").read_text()
    assert re.search(r"\.mode\{[^}]*overflow-x:auto", css)
