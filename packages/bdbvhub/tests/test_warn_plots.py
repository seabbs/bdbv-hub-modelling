from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhub.reporting.warn_plots import (
    warned_locations,
    warnings_figure,
)

AS_OF = "2026-11-24"
REF = "2026-11-28"
TARGET = "wk inc confirmed cases"
LOCS = ("CD", "CD-IT", "CD-NK-29")


def _hub_file(path: Path, ref: str, scale: float) -> Path:
    weeks = pd.date_range(end=pd.Timestamp(ref) + pd.Timedelta(weeks=3),
                          periods=8, freq="7D")  # fmt: skip
    rows = [
        {
            "reference_date": ref,
            "target": TARGET,
            "target_end_date": w.date().isoformat(),
            "location": loc,
            "output_type": "sample",
            "output_type_id": s,
            "value": scale * (s + i),
        }
        for loc in LOCS
        for i, w in enumerate(weeks)
        for s in range(1, 11)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _observed(path: Path) -> Path:
    weeks = pd.date_range(end=AS_OF, periods=20, freq="7D")
    rows = [
        {
            "as_of": v,
            "target_end_date": w.date().isoformat(),
            "target": TARGET,
            "location": loc,
            "observation": 5 + i,
        }
        for v in ("2026-11-17", AS_OF)
        for loc in LOCS
        for i, w in enumerate(weeks)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _checks(path: Path, warned: dict[str, list[str]]) -> Path:
    checks = [
        {"name": "schema", "status": "pass", "summary": "ok",
         "warnings": {}},
        {"name": "previous", "status": "warn" if warned else "pass",
         "summary": "", "warnings": warned},
    ]  # fmt: skip
    path.write_text(json.dumps({"status": "warn", "checks": checks}))
    return path


def _figure(tmp_path: Path, warned: dict[str, list[str]]) -> object:
    return warnings_figure(
        _checks(tmp_path / "checks.json", warned),
        _hub_file(tmp_path / "sub.csv", REF, 2.0),
        _observed(tmp_path / "ts.csv"),
        AS_OF,
        previous=_hub_file(tmp_path / "prev.csv", "2026-11-21", 1.0),
        randomwalk=_hub_file(tmp_path / "rw.csv", REF, 1.5),
        label="renewalbvd",
    )


def test_warned_locations_gather_every_check() -> None:
    checks = {
        "checks": [
            {"name": "a", "status": "warn", "warnings": {"CD": ["CD: x"]}},
            {"name": "b", "status": "warn", "warnings": {"CD": ["CD: y"]}},
            {"name": "c", "status": "pass", "warnings": {}},
        ]
    }
    assert warned_locations(checks) == {"CD": ["a: CD: x", "b: CD: y"]}


def test_one_panel_per_warned_location(tmp_path: Path) -> None:
    warned = {
        "CD-IT": ["CD-IT: median 20 against last week's submission 3"],
        "CD-NK-29": ["CD-NK-29: median 1 against last week's submission 9"],
    }
    fig = _figure(tmp_path, warned)
    assert fig is not None
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 2
    titles = [ax.get_title(loc="left") for ax in axes]
    assert titles[0].startswith("CD-IT")
    assert "median 20" in titles[0]
    assert titles[1].startswith("CD-NK-29")
    # Data, the submission, last week's and the random walk nowcast.
    labels = axes[0].get_legend_handles_labels()[1]
    assert labels == [
        f"reported as of {AS_OF}",
        "renewalbvd",
        "last week's submission",
        "random walk nowcast",
    ]
    lo, _ = axes[0].get_xlim()
    assert pd.Timestamp(AS_OF) - pd.Timedelta(weeks=15) < pd.Timestamp(
        pd.to_datetime(lo, unit="D")
    )


def test_no_warnings_no_figure(tmp_path: Path) -> None:
    assert _figure(tmp_path, {}) is None
