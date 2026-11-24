"""The this-round page plots each location the pick's checks warned on."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
AS_OF, REF = "2026-11-24", "2026-11-28"
LOCS = ("CD", "CD-IT")


@pytest.fixture
def report():
    pytest.importorskip("bdbvhub")
    path = ROOT / "workflow" / "scripts" / "this_round_report.py"
    spec = importlib.util.spec_from_file_location("this_round_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hub_file(path: Path, ref: str) -> None:
    import pandas as pd

    weeks = pd.date_range(end=REF, periods=4, freq="7D")
    rows = [
        {"reference_date": ref, "target": "wk inc confirmed cases",
         "target_end_date": w.date().isoformat(), "location": loc,
         "output_type": "sample", "output_type_id": s, "value": s}
        for loc in LOCS for w in weeks for s in range(1, 6)
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(root: Path, warned: dict) -> Path:
    import pandas as pd

    run = root / "run"
    gate = run / "gate" / "gate"
    gate.mkdir(parents=True)
    (gate / "provenance.json").write_text(
        json.dumps({"selected": None, "attempts": [{"id": "bvd"}]})
    )
    inputs = run / "inputs" / "inputs"
    inputs.mkdir(parents=True)
    weeks = pd.date_range(end=AS_OF, periods=6, freq="7D")
    pd.DataFrame(
        [{"as_of": AS_OF, "target_end_date": w.date().isoformat(),
          "target": "wk inc confirmed cases", "location": loc,
          "observation": 3}
         for loc in LOCS for w in weeks]
    ).to_csv(inputs / "time-series.csv", index=False)  # fmt: skip
    _hub_file(inputs / "previous.csv", "2026-11-21")
    rw = run / "inputs" / "candidates" / "randomwalknowcast" / "submission"
    _hub_file(rw / f"{REF}-seabbs-rwnowcast.csv", REF)
    cand = run / "candidates" / "bvd"
    _hub_file(cand / "submission" / f"{REF}-seabbs-renewal.csv", REF)
    (cand / "run.json").write_text(json.dumps({"model": "renewalbvd"}))
    (cand / "checks.json").write_text(
        json.dumps(
            {"checks": [{"name": "previous", "status": "warn",
                         "warnings": warned}]}
        )
    )  # fmt: skip
    return run


def test_the_pick_s_warned_locations_are_plotted(report, tmp_path):
    run = _run(tmp_path, {"CD-IT": ["CD-IT: median 9 against 1"]})
    data = tmp_path / "data"
    data.mkdir()
    html = report.warnings_section(run, AS_OF, data)
    assert (data / "warnings.png").exists()
    assert "data/warnings.png" in html
    assert "bvd" in html


def test_no_warnings_no_plot(report, tmp_path):
    run = _run(tmp_path, {})
    data = tmp_path / "data"
    data.mkdir()
    html = report.warnings_section(run, AS_OF, data)
    assert not (data / "warnings.png").exists()
    assert "No check warned" in html
