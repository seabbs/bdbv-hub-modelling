"""The forecast explorer shows every model's intervals in every view."""

from pathlib import Path

JS = Path(__file__).parents[3] / "docs" / "javascripts" / "explorer.js"


def test_trajectories_keep_each_model_s_intervals():
    # In the trajectories view a model with samples drew its draws and
    # median only; its 50% and 90% intervals are drawn there too.
    js = JS.read_text()
    slot = js.index("var slot = models.indexOf(r[0])")
    rows = js[js.rindex("rows.forEach(function (r) {", 0, slot) :]
    rows = rows[: rows.index("});")]
    assert "asTraj(r[0])" not in rows
    assert "Y(r[4]), y2: Y(r[8])" in rows
    assert "Y(r[5]), y2: Y(r[7])" in rows
