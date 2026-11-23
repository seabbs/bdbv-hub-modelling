"""The page's scripts and style name no study, place, run or host."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard" / "page"
NAMES = re.compile(
    r"""["'](?:germany|ebola|bdbv|DE|CD|rw|zero|complete|zero-shot|"""
    r"""complete-only|randomwalknowcast|slots|no slots|7day|daily|weekly|"""
    r"""national)["']""",
    re.I,
)


@pytest.mark.parametrize(
    "path", sorted(PAGE.glob("*.js")), ids=lambda p: p.name
)
def test_scripts_take_names_from_the_page_data(path):
    assert NAMES.findall(path.read_text()) == []


def test_the_old_single_dataset_scripts_are_gone():
    assert not (PAGE / "germany.js").exists()
    assert not (PAGE / "checkpoints.js").exists()


def test_the_nowcasts_view_and_its_controls_expand_as_one():
    html = (PAGE / "nowcasts.html").read_text()
    assert html.lstrip().startswith("<div class=expandable id=gview")
    assert "id=gctl" in html or "class=gctl" in html
    assert html.rstrip().endswith("</div>")


def test_the_expand_script_is_keyboard_accessible():
    js = (PAGE / "expand.js").read_text()
    # A real button (Tab, Enter, Space), a label, Esc to close, the
    # Fullscreen API with the overlay as fallback, and a redraw.
    for needle in (
        'createElement("button")',
        "aria-label",
        '"Escape"',
        "requestFullscreen",
        'new Event("resize")',
        ".focus()",
    ):
        assert needle in js


HEX = re.compile(r"(?<![\w&])#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
NEUTRALS = {
    "#fff",
    "#1d2127",
    "#6b7280",
    "#f6f7f9",
    "#d9dde3",
    "#14171c",
    "#e6e8eb",
    "#9aa3af",
    "#1d2128",
    "#343a44",
}
"""The page's neutral surfaces, text and lines (style.css)."""


def test_no_dashboard_file_holds_a_colour_except_neutrals_in_the_style():
    # Every data and accent colour comes from the shared map
    # (nowcasteval.labels): models, relations, layouts,
    # levels, runs, WIS parts and the page's accents.
    found = {}
    for path in sorted(PAGE.parent.rglob("*")):
        if path.suffix not in {".py", ".js", ".css", ".html", ".json"}:
            continue
        hexes = set(HEX.findall(path.read_text()))
        if path.name == "style.css":
            hexes -= NEUTRALS
        if hexes:
            found[str(path.relative_to(PAGE.parent))] = sorted(hexes)
    assert found == {}


def test_the_page_writes_its_accents_from_the_shared_map():
    from flowlib.dashboard import page

    from nowcasteval import labels

    css = page.ui_colours()
    for name, (light, dark) in labels.UI_COLOURS.items():
        assert f"--{name}:{light}" in css
        assert f"--{name}:{dark}" in css


def test_the_score_panels_sit_in_two_columns():
    # Five score panels one under another ran far below the plot.
    js = (PAGE / "nowcasts.js").read_text()
    css = (PAGE / "style.css").read_text()
    assert "<div class=gscores>" in js
    assert "<div class=gsc>" in js
    rule = re.search(r"\.gscores\{([^}]*)\}", css)
    assert rule
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in rule[1]


def test_the_horizon_control_filters_bands_and_scores():
    html = (PAGE / "nowcasts.html").read_text()
    js = (PAGE / "nowcasts.js").read_text()
    assert "id=ghz" in html
    # Bands, score panels and the WIS-by-horizon plot all read the choice.
    assert "inRange(" in js
    assert "d.ranges" in js
    assert "d.byday" in js
    assert "by day of the horizon" in js


def test_the_horizon_split_pools_the_common_nowcast_dates():
    # The split was pooled over the plot's date, the newest by default,
    # which a checkpoint still being scored lacks: its bars were empty and
    # it was named as having no forecast days.
    js = (PAGE / "nowcasts.js").read_text()
    pool = js[js.index("function pool(") :]
    pool = pool[: pool.index("return out}")]
    assert "(com||[]).forEach" in pool
    assert "(ds||[])" not in pool
    assert "nowcast date(s) shown" not in js


def test_nowcasts_over_time_draw_every_shown_model_over_any_span():
    # Over time, every ticked model is drawn (as one nowcast date draws
    # them), with no single-model choice, and the origins can be every
    # scored date.
    html = (PAGE / "nowcasts.html").read_text()
    js = (PAGE / "nowcasts.js").read_text()
    assert "id=gmodl" not in html
    assert "<option value=0>all</option>" in html
    eff = js[js.index("function effective(") :]
    assert eff[: eff.index("}") + 1] == "function effective(){return ticked()}"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_horizon_split_draws_a_visible_bar_for_every_value():
    # The split's bars were a few pixels wide; each value gets a bar of
    # visible width and a height that grows with it.
    script = (
        "const {groupedBars}=require(process.argv[1]);"
        "const g=['now','wk 1','wk 2','wk 3','wk 4'];"
        "const v={now:53,'wk 1':120,'wk 2':150,'wk 3':170,'wk 4':190};"
        "const ms=[1,2,3,4,5,6,7].map(i=>({id:'m'+i,values:v}));"
        "console.log(JSON.stringify(groupedBars(g,ms,520,170).bars))"
    )
    out = subprocess.run(
        ["node", "-e", script, str(PAGE / "bars.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    bars = json.loads(out.stdout)
    assert len(bars) == 35
    assert all(b["w"] >= 6 and b["h"] >= 10 for b in bars)
    now = [b["h"] for b in bars if b["g"] == "now"]
    wk4 = [b["h"] for b in bars if b["g"] == "wk 4"]
    assert min(wk4) > max(now)
    js = (PAGE / "nowcasts.js").read_text()
    assert "groupedBars(" in js


def test_the_audit_s_panel_fixes_are_in_place():
    # dash-audit rows 2, 3, 4, 6 and 7: a caption per target, one truth
    # caption, the legend only of models drawn, interim checkpoints marked
    # on their chips, and round tick steps on the WIS-by-day axis.
    import yaml

    config = yaml.safe_load((PAGE.parents[2] / "config.yaml").read_text())
    germany = next(
        d for d in config["dashboard"]["datasets"] if d["key"] == "germany"
    )
    assert set(germany["target"]) == {"7day", "daily"}
    assert "daily" in germany["target"]["daily"]
    assert "latest vintage" not in germany["truth"]
    js = (PAGE / "nowcasts.js").read_text()
    assert "id=gleg" in js
    assert "interim" in js
    assert "function niceTicks(" in js
