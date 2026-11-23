"""The hub rounds report ranks models by nowcasteval's pairwise skill."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def report():
    pytest.importorskip("bdbvhub")
    path = ROOT / "workflow" / "scripts" / "hub_rounds_report.py"
    spec = importlib.util.spec_from_file_location("hub_rounds_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_is_pairwise_skill_with_each_models_cells(report):
    import pandas as pd

    from nowcasteval.scoring import pairwise_skill

    rows = []
    for i, model in enumerate(("ref", "a", "b")):
        for day in range(4):
            if model == "b" and day > 1:
                continue
            for place in ("x", "y"):
                rows.append(
                    {
                        "model": model,
                        "reference_date": f"2026-01-0{day + 1}",
                        "truth_vintage": "2026-02-01",
                        "location": place,
                        "target_end_date": f"2026-01-0{day + 1}",
                        "wis": 1.0 + i + day + (place == "y"),
                        "coverage_50": 0.5,
                        "coverage_90": 1.0,
                    }
                )
    scores = pd.DataFrame(rows)
    got = report.pairwise(scores, "ref").set_index("model")
    want = pairwise_skill(
        scores, "ref", metrics=("wis",), match_on=report.MATCH
    ).set_index("model")
    for m in ("ref", "a", "b"):
        assert got.loc[m, "relative_wis"] == pytest.approx(
            want.loc[m, "relative_wis"], rel=1e-12
        )
    assert got.loc["b", "cells"] == 4
    assert got.loc["b", "rounds"] == 2
    assert list(got.index) == list(got["relative_wis"].sort_values().index)
