"""The forecast explorer shows every model's intervals in every view."""

from pathlib import Path

JS = Path(__file__).parents[2] / "docs" / "javascripts" / "explorer.js"


def test_trajectories_keep_each_model_s_intervals():
    # In the trajectories view a model with samples drew its draws and
    # median only; its 50% and 90% intervals are drawn there too.
    js = JS.read_text()
    rows = js[js.index("rows.forEach(function (r) {") :]
    rows = rows[: rows.index("});")]
    assert "asTraj(r[0])" not in rows
    assert "Y(r[4]), y2: Y(r[8])" in rows
    assert "Y(r[5]), y2: Y(r[7])" in rows
